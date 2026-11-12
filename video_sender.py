#!/usr/bin/env python3
"""Prepare MP4 videos as an ABR ladder and send them to three receivers.

This sender intentionally uses only the Python standard library.  FFmpeg and
FFprobe are external executables.  The wire protocol is compatible with the
existing ``abr_receiver.py`` used by ``web_controller.py``:

    chunk_offer -> bitrate_decision -> chunk_data + payload -> chunk_ack

The six renditions are prepared before a receiver is contacted.  That is
important because the current receiver has a short timeout between returning a
decision and waiting for the chunk header.  It also gives the policies exact
next-chunk sizes instead of estimates.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


BITRATES_KBPS = (300, 750, 1200, 1850, 2850, 4300)
ALGORITHM_NAMES = ("netllm", "bba", "mpc")
JSON_LENGTH_SIZE = 4
MAX_JSON_SIZE = 1024 * 1024
DEFAULT_SEND_BLOCK_SIZE = 256 * 1024
PROTOCOL_VERSION = 1
SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_SERVER_HOST = "192.168.0.128"
DEFAULT_TRAIN_TRACE = (
    SCRIPT_ROOT
    / "adaptive_bitrate_streaming"
    / "data"
    / "traces"
    / "train"
    / "fcc-train"
    / "tram.ljabru-jernbanetorget-report.2011-02-02_1345CET.log"
)


class SenderError(RuntimeError):
    """Expected sender failure with a user-facing message."""


def log(message: str) -> None:
    print(message, flush=True)


def resolve_program(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    resolved = shutil.which(value)
    if resolved:
        return resolved
    raise SenderError(
        f"找不到程序 {value!r}。请安装 FFmpeg，或通过 --ffmpeg/--ffprobe 指定完整路径。"
    )


def send_json(conn: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if not payload or len(payload) > MAX_JSON_SIZE:
        raise SenderError(f"JSON 消息长度非法：{len(payload)}")
    conn.sendall(struct.pack("!I", len(payload)))
    conn.sendall(payload)


def recv_exact(conn: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        block = conn.recv(size - len(data))
        if not block:
            raise ConnectionError("服务端提前关闭连接")
        data.extend(block)
    return bytes(data)


def recv_json(conn: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", recv_exact(conn, JSON_LENGTH_SIZE))[0]
    if size <= 0 or size > MAX_JSON_SIZE:
        raise SenderError(f"服务端 JSON 消息长度非法：{size}")
    value = json.loads(recv_exact(conn, size).decode("utf-8"))
    if not isinstance(value, dict):
        raise SenderError("服务端 JSON 消息不是对象")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def safe_slug(value: str) -> str:
    result = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return result[:60] or "video"


def natural_name_key(path: Path) -> list[tuple[int, Any]]:
    """Sort 视频2.mp4 before 视频10.mp4 while remaining Unicode-safe."""
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    ]


def choose_input_video(directory: Path = SCRIPT_ROOT) -> Path:
    """Interactively choose exactly one MP4 beside this script."""
    videos = sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp4"
        ),
        key=natural_name_key,
    )
    if not videos:
        raise SenderError(
            f"在 {directory} 中没有找到 MP4。请把视频放到该目录，"
            "或使用 --input 指定视频完整路径。"
        )

    log("\n请选择本次要传输的视频：")
    for index, path in enumerate(videos, start=1):
        size_mb = path.stat().st_size / 1024 / 1024
        log(f"  {index}. {path.name} ({size_mb:.1f} MB)")

    while True:
        try:
            answer = input(f"请输入编号 [1-{len(videos)}]（q 退出）：").strip()
        except EOFError as error:
            raise SenderError("当前终端无法交互选择视频，请使用 --input 指定文件。") from error
        if answer.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            selected = int(answer)
        except ValueError:
            log("输入无效，请输入视频前面的数字编号。")
            continue
        if 1 <= selected <= len(videos):
            path = videos[selected - 1]
            log(f"[已选择] {path.name}\n")
            return path
        log(f"编号超出范围，请输入 1 到 {len(videos)}。")


def cache_fingerprint(
    path: Path,
    chunk_seconds: float,
    max_width: int,
    preset: str,
) -> tuple[str, dict[str, Any]]:
    stat = path.stat()
    metadata = {
        "source": str(path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "chunk_seconds": chunk_seconds,
        "bitrates_kbps": list(BITRATES_KBPS),
        "max_width": max_width,
        "preset": preset,
        "video_codec": "libx264",
        "audio": False,
        "cache_format": 1,
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16], metadata


class BandwidthController:
    """A fixed limit or a piecewise-constant NetLLM trace in Mbps."""

    def __init__(self, fixed_mbps: float | None = None, trace_path: Path | None = None):
        self.fixed_mbps = fixed_mbps
        self.trace_path = trace_path
        self.times: list[float] = []
        self.rates: list[float] = []
        if trace_path is not None:
            self._load_trace(trace_path)

    def _load_trace(self, path: Path) -> None:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise SenderError(f"网络轨迹不存在：{path}")
        points: list[tuple[float, float]] = []
        with path.open("r", encoding="utf-8", errors="replace") as source:
            for line_number, raw in enumerate(source, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.replace(",", " ").split()
                if len(fields) < 2:
                    raise SenderError(f"网络轨迹第 {line_number} 行格式错误：{raw.rstrip()}")
                try:
                    timestamp, rate = float(fields[0]), float(fields[1])
                except ValueError as error:
                    raise SenderError(f"网络轨迹第 {line_number} 行不是数字") from error
                if timestamp < 0 or rate <= 0:
                    raise SenderError(f"网络轨迹第 {line_number} 行必须是非负时间和正带宽")
                points.append((timestamp, rate))
        if not points:
            raise SenderError(f"网络轨迹为空：{path}")
        points.sort()
        origin = points[0][0]
        self.times = [timestamp - origin for timestamp, _ in points]
        self.rates = [rate for _, rate in points]
        self.trace_path = path

    def rate_at(self, elapsed_seconds: float) -> float | None:
        if self.fixed_mbps is not None:
            return self.fixed_mbps if self.fixed_mbps > 0 else None
        if not self.times:
            return None
        index = bisect.bisect_right(self.times, max(0.0, elapsed_seconds)) - 1
        return self.rates[max(index, 0)]

    def describe(self) -> str:
        if self.fixed_mbps is not None:
            return "不限速" if self.fixed_mbps == 0 else f"固定 {self.fixed_mbps:g} Mbps/路"
        if not self.rates:
            return "不限速"
        mean = sum(self.rates) / len(self.rates)
        return (
            f"训练轨迹 {self.trace_path.name}: mean={mean:.3f}, "
            f"min={min(self.rates):.3f}, max={max(self.rates):.3f} Mbps"
        )


def run_checked(command: list[str], description: str) -> None:
    display = subprocess.list2cmdline(command)
    log(f"[执行] {description}\n{display}")
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.splitlines()[-20:])
        raise SenderError(f"{description}失败（退出码 {result.returncode}）：\n{tail}")


def probe_video(ffprobe: str, path: Path) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate:format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise SenderError(f"FFprobe 无法读取 {path.name}：{result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        duration = float(payload["format"]["duration"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SenderError(f"无法解析 {path.name} 的视频信息") from error
    if duration <= 0:
        raise SenderError(f"视频时长非法：{path}")
    return {
        "duration": duration,
        "codec_name": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
    }


@dataclass
class PreparedVideo:
    source: Path
    video_id: str
    duration: float
    chunk_seconds: float
    cache_dir: Path
    chunks_by_bitrate: dict[int, list[Path]]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks_by_bitrate[BITRATES_KBPS[0]])

    def paths_for_chunk(self, local_index: int) -> list[Path]:
        return [self.chunks_by_bitrate[bitrate][local_index] for bitrate in BITRATES_KBPS]

    def duration_for_chunk(self, local_index: int) -> float:
        remaining = self.duration - local_index * self.chunk_seconds
        return max(0.001, min(self.chunk_seconds, remaining))


class LadderPreparer:
    def __init__(self, args: argparse.Namespace, ffmpeg: str, ffprobe: str):
        self.args = args
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.cache_root = Path(args.work_dir).expanduser().resolve() / "cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def prepare(self, source: Path, ordinal: int) -> PreparedVideo:
        source = source.expanduser().resolve()
        if not source.is_file():
            raise SenderError(f"输入视频不存在：{source}")
        info = probe_video(self.ffprobe, source)
        fingerprint, metadata = cache_fingerprint(
            source, self.args.chunk_seconds, self.args.max_width, self.args.preset
        )
        video_id = f"{safe_slug(source.stem)}-{ordinal + 1}"
        cache_dir = self.cache_root / f"{safe_slug(source.stem)}-{fingerprint}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_dir / "cache_manifest.json"

        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = None
            if existing != metadata:
                raise SenderError(
                    f"缓存元数据不一致：{cache_dir}。可删除该具体缓存目录后重试。"
                )

        log(
            f"[视频] {source.name}: {info['duration']:.3f}s, "
            f"{info.get('width')}x{info.get('height')}, codec={info.get('codec_name')}"
        )
        workers = max(1, min(self.args.prepare_workers, len(BITRATES_KBPS)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(self._prepare_bitrate, source, cache_dir, bitrate): bitrate
                for bitrate in BITRATES_KBPS
            }
            for future in concurrent.futures.as_completed(future_map):
                bitrate = future_map[future]
                future.result()
                log(f"[缓存完成] {source.name} / {bitrate} kbps")

        chunks_by_bitrate = {
            bitrate: sorted((cache_dir / f"{bitrate}k").glob("chunk_*.ts"))
            for bitrate in BITRATES_KBPS
        }
        counts = {bitrate: len(paths) for bitrate, paths in chunks_by_bitrate.items()}
        if not counts or len(set(counts.values())) != 1 or next(iter(counts.values()), 0) <= 0:
            raise SenderError(f"六档切片数量不一致：{counts}")
        for bitrate, paths in chunks_by_bitrate.items():
            if any(path.stat().st_size <= 0 for path in paths):
                raise SenderError(f"{bitrate} kbps 缓存中存在空文件")

        expected = max(1, math.ceil(info["duration"] / self.args.chunk_seconds - 1e-6))
        actual = next(iter(counts.values()))
        if abs(actual - expected) > 1:
            raise SenderError(f"切片数量异常：期望约 {expected}，实际 {actual}")
        manifest_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return PreparedVideo(
            source=source,
            video_id=video_id,
            duration=info["duration"],
            chunk_seconds=self.args.chunk_seconds,
            cache_dir=cache_dir,
            chunks_by_bitrate=chunks_by_bitrate,
        )

    def _prepare_bitrate(self, source: Path, cache_dir: Path, bitrate: int) -> None:
        final_dir = cache_dir / f"{bitrate}k"
        completed = final_dir / ".complete"
        existing = sorted(final_dir.glob("chunk_*.ts")) if final_dir.exists() else []
        if completed.exists() and existing and all(path.stat().st_size > 0 for path in existing):
            return

        partial_dir = cache_dir / f".{bitrate}k-building"
        if partial_dir.exists():
            # This exact child is owned by the current cache and only stores an
            # incomplete rendition from an interrupted preparation.
            shutil.rmtree(partial_dir)
        partial_dir.mkdir(parents=True)
        output_pattern = partial_dir / "chunk_%010d.ts"

        command = [
            self.ffmpeg,
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(source),
            "-map", "0:v:0", "-an",
        ]
        if self.args.max_width > 0:
            command += [
                "-vf",
                f"scale='min(iw,{self.args.max_width})':-2:flags=fast_bilinear",
            ]
        command += [
            "-c:v", "libx264",
            "-preset", self.args.preset,
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-b:v", f"{bitrate}k",
            "-maxrate", f"{bitrate}k",
            "-bufsize", f"{bitrate * 2}k",
            "-threads", str(self.args.ffmpeg_threads),
            "-sc_threshold", "0",
            "-force_key_frames", f"expr:gte(t,n_forced*{self.args.chunk_seconds:g})",
            "-f", "segment",
            "-segment_format", "mpegts",
            "-segment_time", f"{self.args.chunk_seconds:g}",
            "-segment_time_delta", "0.05",
            "-reset_timestamps", "0",
            str(output_pattern),
        ]
        run_checked(command, f"准备 {source.name} 的 {bitrate} kbps 切片")
        generated = sorted(partial_dir.glob("chunk_*.ts"))
        if not generated or any(path.stat().st_size <= 0 for path in generated):
            raise SenderError(f"FFmpeg 未生成有效的 {bitrate} kbps 切片")

        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(partial_dir, final_dir)
        (final_dir / ".complete").write_text("ok\n", encoding="ascii")


@dataclass
class Route:
    name: str
    host: str
    port: int
    connect_timeout: float
    socket_timeout: float
    conn: socket.socket | None = None
    state: dict[str, Any] = field(default_factory=lambda: {
        "bitrate": BITRATES_KBPS[0],
        "size": 0,
        "throughput": 0.0,
        "buffer": 0.0,
        "rebuffer": 0.0,
    })
    lock: threading.Lock = field(default_factory=threading.Lock)

    def connect(self) -> None:
        self.close(send_close=False)
        self.conn = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout
        )
        self.conn.settimeout(self.socket_timeout)
        log(f"[已连接] {self.name}: {self.host}:{self.port}")

    def ensure_connected(self) -> socket.socket:
        if self.conn is None:
            self.connect()
        assert self.conn is not None
        return self.conn

    def close(self, send_close: bool = True) -> None:
        conn, self.conn = self.conn, None
        if conn is None:
            return
        if send_close:
            try:
                send_json(conn, {"type": "close"})
                recv_json(conn)
            except Exception:
                pass
        try:
            conn.close()
        except OSError:
            pass

    def request_decision(self, offer: dict[str, Any]) -> tuple[int, int]:
        with self.lock:
            conn = self.ensure_connected()
            send_json(conn, offer)
            message = recv_json(conn)
        if message.get("type") != "bitrate_decision":
            raise SenderError(f"{self.name} 返回错误决策：{message}")
        if str(message.get("chunk_id")) != str(offer["chunk_id"]):
            raise SenderError(f"{self.name} 返回了错误 chunk_id：{message}")
        action = int(message.get("action", -1))
        if action < 0 or action >= len(BITRATES_KBPS):
            raise SenderError(f"{self.name} action 越界：{action}")
        bitrate = BITRATES_KBPS[action]
        returned_bitrate = int(message.get("bitrate_kbps", bitrate))
        if returned_bitrate != bitrate:
            raise SenderError(f"{self.name} action 与 bitrate 不一致：{message}")
        return action, bitrate

    def send_payload(
        self,
        chunk_id: str,
        action: int,
        bitrate: int,
        path: Path,
        send_block_size: int,
        bandwidth: BandwidthController,
        network_started: float,
    ) -> dict[str, Any]:
        size = path.stat().st_size
        header = {
            "type": "chunk_data",
            "chunk_id": chunk_id,
            "action": action,
            "bitrate_kbps": bitrate,
            "filename": path.name,
            "size_bytes": size,
            "sha256": sha256_file(path),
        }
        with self.lock:
            conn = self.ensure_connected()
            send_json(conn, header)
            with path.open("rb") as source:
                while True:
                    block = source.read(send_block_size)
                    if not block:
                        break
                    limit_mbps = bandwidth.rate_at(time.monotonic() - network_started)
                    if limit_mbps is not None:
                        # Small blocks track the roughly one-second sampling
                        # interval of the original training traces closely.
                        time.sleep(len(block) * 8.0 / (limit_mbps * 1_000_000.0))
                    conn.sendall(block)
            ack = recv_json(conn)
        if ack.get("type") != "chunk_ack":
            raise SenderError(f"{self.name} ACK 错误：{ack}")
        if str(ack.get("chunk_id")) != chunk_id:
            raise SenderError(f"{self.name} ACK chunk_id 错误：{ack}")
        if int(ack.get("received", -1)) != size:
            raise SenderError(f"{self.name} ACK 文件大小不一致：{ack}")
        if ack.get("sha256") and ack["sha256"] != header["sha256"]:
            raise SenderError(f"{self.name} ACK SHA-256 不一致：{ack}")
        self.state.update({
            "bitrate": bitrate,
            "size": size,
            "throughput": float(ack.get("throughput_mbps", 0.0)),
            "buffer": float(ack.get("buffer_seconds", 0.0)),
            "rebuffer": float(ack.get("rebuffer_seconds", 0.0)),
        })
        return ack


class VideoSender:
    def __init__(self, args: argparse.Namespace, prepared: list[PreparedVideo]):
        self.args = args
        self.prepared = prepared
        self.routes = [
            Route(
                name=name,
                host=args.server_host,
                port=port,
                connect_timeout=args.connect_timeout,
                socket_timeout=args.socket_timeout,
            )
            for name, port in zip(ALGORITHM_NAMES, args.ports)
        ]
        self.stream_id = args.stream_id or time.strftime("video-%Y%m%d-%H%M%S")
        self.log_path = Path(args.work_dir).expanduser().resolve() / "sent_manifest.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if args.limit_mbps is not None:
            self.bandwidth = BandwidthController(fixed_mbps=args.limit_mbps)
        else:
            self.bandwidth = BandwidthController(trace_path=Path(args.trace_file))
        self.network_started = 0.0

    def connect_all(self) -> None:
        errors = []
        for route in self.routes:
            try:
                route.connect()
            except OSError as error:
                errors.append(f"{route.name} {route.host}:{route.port}: {error}")
        if errors:
            self.close_all()
            raise SenderError("无法连接全部三路 receiver：\n" + "\n".join(errors))

    def close_all(self) -> None:
        for route in self.routes:
            route.close()

    def make_offer(
        self,
        video: PreparedVideo,
        local_index: int,
        global_index: int,
        paths: list[Path],
    ) -> dict[str, Any]:
        # These compatibility fields are retained for the current receiver.
        # The receiver keeps its own per-policy runtime state.
        representative = self.routes[0].state
        return {
            "type": "chunk_offer",
            "protocol_version": PROTOCOL_VERSION,
            "stream_id": self.stream_id,
            "video_id": video.video_id,
            "source_name": video.source.name,
            "chunk_id": f"{global_index:010d}",
            "video_chunk_index": local_index,
            "total_video_chunks": video.chunk_count,
            "duration_seconds": video.duration_for_chunk(local_index),
            "bitrates_kbps": list(BITRATES_KBPS),
            "chunk_sizes_bytes": [path.stat().st_size for path in paths],
            "sizes_are_estimates": False,
            "last_selected_bitrate_kbps": representative["bitrate"],
            "last_chunk_size_bytes": representative["size"],
            "measured_throughput_mbps": representative["throughput"],
            "receiver_buffer_seconds": representative["buffer"],
            "last_rebuffer_seconds": representative["rebuffer"],
        }

    def run(self) -> None:
        self.connect_all()
        global_index = self.args.start_chunk_id
        schedule_started = time.monotonic()
        self.network_started = schedule_started
        scheduled_index = 0
        log(f"[网络模拟] {self.bandwidth.describe()}")
        try:
            for video in self.prepared:
                log(f"[开始视频] {video.source.name}, chunks={video.chunk_count}")
                for local_index in range(video.chunk_count):
                    if self.args.chunk_interval > 0:
                        due_at = schedule_started + scheduled_index * self.args.chunk_interval
                        delay = due_at - time.monotonic()
                        if delay > 0:
                            log(f"[播放节拍] 等待 {delay:.2f}s 后发布下一片")
                            time.sleep(delay)
                    paths = video.paths_for_chunk(local_index)
                    offer = self.make_offer(video, local_index, global_index, paths)
                    self._send_one(offer, paths)
                    global_index += 1
                    scheduled_index += 1
                log(f"[视频完成] {video.source.name}")
        finally:
            self.close_all()
        log(f"[全部完成] 共发送 {global_index - self.args.start_chunk_id} 个源 chunk")

    def _send_one(self, offer: dict[str, Any], paths: list[Path]) -> None:
        decisions: dict[str, tuple[int, int]] = {}
        results: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Each policy gets a complete independent transaction.  In
            # particular, NetLLM no longer waits for the BBA and MPC decision
            # round trips before its payload starts arriving.
            future_map = {
                executor.submit(self._route_transaction, route, offer, paths): route.name
                for route in self.routes
            }
            for future in concurrent.futures.as_completed(future_map):
                name = future_map[future]
                action, bitrate, ack = future.result()
                decisions[name] = (action, bitrate)
                results[name] = ack

        summary = ", ".join(
            f"{name}={decisions[name][1]}k" for name in ALGORITHM_NAMES
        )
        log(f"[决策] chunk={offer['chunk_id']} {summary}")

        record = {
            "sent_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stream_id": self.stream_id,
            "video_id": offer["video_id"],
            "source_name": offer["source_name"],
            "chunk_id": offer["chunk_id"],
            "video_chunk_index": offer["video_chunk_index"],
            "decisions": {
                name: {"action": value[0], "bitrate_kbps": value[1]}
                for name, value in decisions.items()
            },
            "acks": results,
        }
        with self.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            output.flush()
        ack_text = ", ".join(
            f"{name}:{float(ack.get('throughput_mbps', 0.0)):.3f}Mbps"
            for name, ack in results.items()
        )
        log(f"[发送完成] chunk={offer['chunk_id']} {ack_text}")

    def _route_transaction(
        self,
        route: Route,
        offer: dict[str, Any],
        paths: list[Path],
    ) -> tuple[int, int, dict[str, Any]]:
        action, bitrate = self._decision_with_retry(route, offer)
        ack = self._payload_with_retry(route, offer, action, bitrate, paths[action])
        return action, bitrate, ack

    def _decision_with_retry(
        self, route: Route, offer: dict[str, Any]
    ) -> tuple[int, int]:
        last_error: Exception | None = None
        for attempt in range(self.args.retries + 1):
            try:
                return route.request_decision(offer)
            except (OSError, ConnectionError, TimeoutError, SenderError, json.JSONDecodeError) as error:
                last_error = error
                route.close(send_close=False)
                if attempt < self.args.retries:
                    log(f"[重试决策] {route.name} 第 {attempt + 1} 次失败：{error}")
                    time.sleep(self.args.retry_delay)
        raise SenderError(f"{route.name} 获取决策失败：{last_error}")

    def _payload_with_retry(
        self,
        route: Route,
        offer: dict[str, Any],
        action: int,
        bitrate: int,
        path: Path,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        current_action, current_bitrate = action, bitrate
        for attempt in range(self.args.retries + 1):
            try:
                if attempt > 0:
                    # A broken connection loses the pending decision.  Start a
                    # complete transaction again on only this failed route.
                    current_action, current_bitrate = route.request_decision(offer)
                    path = self.prepared_path_for_action(offer, current_action)
                return route.send_payload(
                    str(offer["chunk_id"]),
                    current_action,
                    current_bitrate,
                    path,
                    self.args.send_block_kb * 1024,
                    self.bandwidth,
                    self.network_started,
                )
            except (OSError, ConnectionError, TimeoutError, SenderError, json.JSONDecodeError) as error:
                last_error = error
                route.close(send_close=False)
                if attempt < self.args.retries:
                    log(f"[重试发送] {route.name} 第 {attempt + 1} 次失败：{error}")
                    time.sleep(self.args.retry_delay)
        raise SenderError(f"{route.name} 发送失败：{last_error}")

    def prepared_path_for_action(self, offer: dict[str, Any], action: int) -> Path:
        video_id = offer["video_id"]
        local_index = int(offer["video_chunk_index"])
        for video in self.prepared:
            if video.video_id == video_id:
                return video.paths_for_chunk(local_index)[action]
        raise SenderError(f"找不到重试所需缓存：{video_id}/{local_index}")


def check_server(host: str, ports: Iterable[int], timeout: float) -> list[str]:
    results = []
    for name, port in zip(ALGORITHM_NAMES, ports):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                results.append(f"{name} {host}:{port} 可连接")
        except OSError as error:
            results.append(f"{name} {host}:{port} 不可连接：{error}")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 MP4 预切成六档 ABR TS，并发送给现有 NetLLM/BBA/MPC receiver。"
    )
    parser.add_argument(
        "--input",
        help="单个视频文件；不提供时列出脚本目录中的 MP4 供手动选择",
    )
    parser.add_argument(
        "--server-host", default=DEFAULT_SERVER_HOST,
        help=f"运行 web_controller.py 的服务器 IP（默认 {DEFAULT_SERVER_HOST}）",
    )
    parser.add_argument(
        "--ports", type=int, nargs=3, default=(5001, 5002, 5003),
        metavar=("NETLLM", "BBA", "MPC"), help="三个 receiver 端口",
    )
    parser.add_argument(
        "--work-dir", default=str(SCRIPT_ROOT / "sender_runtime"),
        help="六档缓存和发送日志目录",
    )
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--stream-id", default=None)
    parser.add_argument("--start-chunk-id", type=int, default=0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--max-width", type=int, default=1280,
        help="为降低本机负载，将宽度限制到该值；0 表示保持原分辨率",
    )
    parser.add_argument(
        "--preset", choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium"),
        default="ultrafast",
    )
    parser.add_argument("--ffmpeg-threads", type=int, default=2, help="每个 FFmpeg 进程使用的线程数")
    parser.add_argument("--prepare-workers", type=int, default=1, help="同时准备的码率数量；低性能机器保持 1")
    parser.add_argument(
        "--chunk-interval", type=float, default=4.0,
        help="源 chunk 的最短发布间隔；默认 4 秒以匹配播放速度，0 表示尽快发送",
    )
    parser.add_argument(
        "--trace-file", default=str(DEFAULT_TRAIN_TRACE),
        help="默认重放的两列式训练带宽轨迹（秒、Mbps）",
    )
    parser.add_argument(
        "--limit-mbps", type=float, default=None,
        help="覆盖训练轨迹并使用固定带宽；0 表示使用局域网原始速度",
    )
    parser.add_argument("--send-block-kb", type=int, default=64)
    parser.add_argument("--connect-timeout", type=float, default=8.0)
    parser.add_argument("--socket-timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument(
        "--prepare-only", action="store_true", help="只准备六档切片，不连接服务器"
    )
    parser.add_argument(
        "--check", action="store_true", help="只检查 FFmpeg，并测试三个服务端口"
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.chunk_seconds <= 0:
        parser.error("--chunk-seconds 必须大于 0")
    if args.max_width < 0:
        parser.error("--max-width 不能小于 0")
    if args.ffmpeg_threads <= 0 or args.prepare_workers <= 0:
        parser.error("线程数必须大于 0")
    if args.send_block_kb <= 0:
        parser.error("--send-block-kb 必须大于 0")
    if args.chunk_interval < 0 or (args.limit_mbps is not None and args.limit_mbps < 0):
        parser.error("--chunk-interval 和 --limit-mbps 不能小于 0")
    if args.start_chunk_id < 0 or args.retries < 0:
        parser.error("chunk id 和重试次数不能小于 0")
    if len(set(args.ports)) != 3 or any(port < 1 or port > 65535 for port in args.ports):
        parser.error("--ports 必须是三个互不相同的有效端口")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    try:
        ffmpeg = resolve_program(args.ffmpeg)
        try:
            ffprobe = resolve_program(args.ffprobe)
        except SenderError:
            # A manually supplied FFmpeg normally ships with FFprobe in the
            # same bin directory, even when that directory is not on PATH.
            ffprobe_name = "ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe"
            sibling = Path(ffmpeg).with_name(ffprobe_name)
            if not sibling.is_file():
                raise
            ffprobe = str(sibling.resolve())
        log(f"[环境] Python {sys.version.split()[0]}")
        log(f"[环境] FFmpeg: {ffmpeg}")
        log(f"[环境] FFprobe: {ffprobe}")

        if args.check:
            for line in check_server(args.server_host, args.ports, args.connect_timeout):
                log(f"[检查] {line}")
            return 0

        selected_input = Path(args.input) if args.input else choose_input_video()

        preparer = LadderPreparer(args, ffmpeg, ffprobe)
        prepared = [preparer.prepare(selected_input, 0)]
        total_chunks = sum(video.chunk_count for video in prepared)
        log(f"[准备完成] videos={len(prepared)}, source_chunks={total_chunks}")
        if args.prepare_only:
            return 0

        VideoSender(args, prepared).run()
        return 0
    except KeyboardInterrupt:
        log("\n[已停止] 用户中断")
        return 130
    except (SenderError, OSError, subprocess.SubprocessError) as error:
        print(f"[错误] {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
