import argparse
import json
import math
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "index.html"
RECEIVER_SCRIPT = ROOT / "abr_receiver.py"
HLS_ROOT = ROOT / "hls"
ALGORITHMS = ("netllm", "bba", "mpc")
BITRATES = [300, 750, 1200, 1850, 2850, 4300]

CONFIG = {
    "base_port": 5001,
    "results_port": 5173,
    "segment_duration": 4.0,
    "window_size": 10,
}
STATE = {
    "started_at": None,
    "algorithms": {
        name: {
            "process": None,
            "command": [],
            "stdout": deque(maxlen=160),
            "history": deque(maxlen=240),
            "manifest_offset": 0,
            "listening": False,
            "config": {},
        }
        for name in ALGORITHMS
    },
}
LOCK = threading.RLock()


def algorithm_port(name, base_port=None):
    return int(base_port or CONFIG["base_port"]) + ALGORITHMS.index(name)


def hls_dir(name):
    return (HLS_ROOT / name).resolve()


def rel_or_abs(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def safe_json(handler, status, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def serve_file(handler, file_path, content_type, cache_control="no-store"):
    try:
        size = file_path.stat().st_size
        source = file_path.open("rb")
    except (FileNotFoundError, IsADirectoryError, OSError):
        safe_json(handler, 404, {"error": "file not found"})
        return

    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(size))
    handler.send_header("Cache-Control", cache_control)
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    try:
        with source:
            while True:
                block = source.read(256 * 1024)
                if not block:
                    break
                handler.wfile.write(block)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass


def chunk_index(path):
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return None


def contiguous_chunks(name):
    indexed = []
    for path in hls_dir(name).glob("chunk_*.ts"):
        index = chunk_index(path)
        if index is not None and path.stat().st_size > 0:
            indexed.append((index, path))
    indexed.sort(key=lambda item: item[0])

    # HLS media sequence must not contain gaps. Keep the newest contiguous run.
    result = []
    for item in indexed:
        if not result or item[0] == result[-1][0] + 1:
            result.append(item)
        else:
            result = [item]
    return result


def serve_hls_playlist(handler, name):
    chunks = contiguous_chunks(name)
    if not chunks:
        safe_json(handler, 404, {
            "error": "waiting for the first completed video chunk",
            "algorithm": name,
        })
        return

    chunks = chunks[-int(CONFIG["window_size"]):]
    duration = float(CONFIG["segment_duration"])
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{math.ceil(duration)}",
        f"#EXT-X-MEDIA-SEQUENCE:{chunks[0][0]}",
        "#EXT-X-INDEPENDENT-SEGMENTS",
    ]
    for _, path in chunks:
        lines.extend([
            f"#EXTINF:{duration:.3f},",
            f"/hls/{name}/{path.name}",
        ])

    data = ("\n".join(lines) + "\n").encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "application/vnd.apple.mpegurl")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(data)


def serve_hls_segment(handler, name, filename):
    clean_name = Path(unquote(filename)).name
    if clean_name != filename or not clean_name.startswith("chunk_"):
        safe_json(handler, 403, {"error": "invalid HLS path"})
        return
    path = (hls_dir(name) / clean_name).resolve()
    try:
        path.relative_to(hls_dir(name))
    except ValueError:
        safe_json(handler, 403, {"error": "invalid HLS path"})
        return
    if path.suffix.lower() != ".ts":
        safe_json(handler, 415, {"error": "unsupported HLS asset"})
        return
    serve_file(handler, path, "video/mp2t")


def read_manifest(name, limit=240):
    path = hls_dir(name) / "received_manifest.jsonl"
    if not path.exists():
        with LOCK:
            return list(STATE["algorithms"][name]["history"])
    try:
        with LOCK:
            item_state = STATE["algorithms"][name]
            offset = item_state["manifest_offset"]
        if path.stat().st_size < offset:
            offset = 0
        with path.open("r", encoding="utf-8") as source:
            source.seek(offset)
            for raw in source:
                try:
                    item = json.loads(raw)
                    item["algorithm"] = name
                    item["chunk_idx"] = int(item.get("chunk_id", -1))
                    item["video_bitrate_kbps"] = item.get("bitrate_kbps")
                    item["send_elapsed_seconds"] = item.get("total_elapsed_seconds")
                    item["buffer_source"] = "receiver"
                    with LOCK:
                        STATE["algorithms"][name]["history"].append(item)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
            new_offset = source.tell()
        with LOCK:
            STATE["algorithms"][name]["manifest_offset"] = new_offset
    except OSError:
        pass
    with LOCK:
        return list(STATE["algorithms"][name]["history"])[-limit:]


def read_process_output(name, process):
    try:
        for line in process.stdout:
            with LOCK:
                STATE["algorithms"][name]["stdout"].append(line.rstrip())
                if line.startswith("[READY] "):
                    STATE["algorithms"][name]["listening"] = True
    except Exception as error:
        with LOCK:
            STATE["algorithms"][name]["stdout"].append(
                f"[controller] output reader stopped: {error}"
            )


def clear_stream(name):
    directory = hls_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    for pattern in ("chunk_*.ts", "chunk_*.ts.part", "received_manifest.jsonl"):
        for path in directory.glob(pattern):
            if path.is_file():
                path.unlink()


def receiver_command(name, config):
    base_port = int(config.get("base_port") or CONFIG["base_port"])
    command = [
        sys.executable,
        "-X", "utf8",
        str(RECEIVER_SCRIPT),
        "--host", config.get("listen_host") or "0.0.0.0",
        "--port", str(algorithm_port(name, base_port)),
        "--output-dir", str(hls_dir(name)),
        "--policy", name,
        "--chunk-seconds", str(float(config.get("chunk_seconds") or 4.0)),
        "--socket-timeout", str(float(config.get("socket_timeout") or 5.0)),
        "--payload-idle-timeout", str(
            float(config.get("payload_idle_timeout") or 300.0)
        ),
    ]
    if name == "netllm":
        command.extend([
            "--model-dir", str(rel_or_abs(config.get("model_dir") or "data/ft_plms/try_llama2_7b")),
            "--exp-pool-path", str(rel_or_abs(config.get("exp_pool_path") or "artifacts/exp_pools/exp_pool.pkl")),
            "--plm-type", config.get("plm_type") or "llama",
            "--plm-size", config.get("plm_size") or "base",
            "--rank", str(int(config.get("rank") or 128)),
            "--device", config.get("device") or "cuda:0",
            "--device-out", config.get("device_out") or config.get("device") or "cuda:0",
        ])
    return command


def start_receivers(config):
    if not RECEIVER_SCRIPT.exists():
        raise FileNotFoundError(f"Receiver script does not exist: {RECEIVER_SCRIPT}")
    with LOCK:
        running = [
            name for name, item in STATE["algorithms"].items()
            if item["process"] is not None and item["process"].poll() is None
        ]
    if running:
        raise RuntimeError("Receivers are already running: " + ", ".join(running))

    config = dict(config)
    config["base_port"] = int(config.get("base_port") or CONFIG["base_port"])
    processes = []
    try:
        for name in ALGORITHMS:
            clear_stream(name)
            command = receiver_command(name, config)
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            processes.append(process)
            with LOCK:
                item = STATE["algorithms"][name]
                item["process"] = process
                item["command"] = command
                item["config"] = dict(config)
                item["stdout"].clear()
                item["history"].clear()
                item["manifest_offset"] = 0
                item["listening"] = False
            threading.Thread(
                target=read_process_output,
                args=(name, process),
                daemon=True,
            ).start()
    except Exception:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise

    with LOCK:
        STATE["started_at"] = time.time()
    return {
        name: list(STATE["algorithms"][name]["command"])
        for name in ALGORITHMS
    }


def stop_receivers():
    with LOCK:
        processes = [
            item["process"] for item in STATE["algorithms"].values()
            if item["process"] is not None
        ]
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    with LOCK:
        for item in STATE["algorithms"].values():
            item["process"] = None
            item["listening"] = False


def model_inputs(history):
    latest = history[-1] if history else {}
    throughputs = [item.get("throughput_mbps", 0.0) for item in history[-6:]]
    downloads = [item.get("total_elapsed_seconds", 0.0) for item in history[-6:]]
    while len(throughputs) < 6:
        throughputs.insert(0, 0.0)
    while len(downloads) < 6:
        downloads.insert(0, 0.0)
    action = int(latest.get("action", 0)) if latest else 0
    return {
        "last_bitrate": {
            "value": BITRATES[action] if 0 <= action < len(BITRATES) else None,
            "normalized": BITRATES[action] / max(BITRATES) if 0 <= action < len(BITRATES) else None,
        },
        "buffer": {
            "seconds": latest.get("buffer_seconds"),
            "normalized": (
                float(latest["buffer_seconds"]) / 10.0
                if latest.get("buffer_seconds") is not None else None
            ),
            "source": "receiver",
        },
        "throughput_history_mbps": throughputs,
        "download_time_history_seconds": downloads,
        "next_chunk_sizes_bytes": latest.get("estimated_sizes_bytes", []),
        "remaining_chunks": None,
    }


def status_snapshot():
    with LOCK:
        process_state = {
            name: {
                "process": item["process"],
                "command": list(item["command"]),
                "stdout": list(item["stdout"]),
                "listening": item["listening"],
                "config": dict(item["config"]),
            }
            for name, item in STATE["algorithms"].items()
        }
        started_at = STATE["started_at"]

    algorithms = {}
    combined_stdout = []
    for name in ALGORITHMS:
        item = process_state[name]
        process = item["process"]
        history = read_manifest(name)
        port = algorithm_port(name, item["config"].get("base_port"))
        running = process is not None and process.poll() is None
        listening = running and item["listening"]
        combined_stdout.extend(f"[{name}] {line}" for line in item["stdout"][-50:])
        algorithms[name] = {
            "running": running,
            "listening": listening,
            "exit_code": process.poll() if process is not None else None,
            "port": port,
            "command": item["command"],
            "stdout": item["stdout"],
            "history": history,
            "latest": history[-1] if history else None,
            "playlist": f"/hls/{name}/index.m3u8",
        }

    history = algorithms["netllm"]["history"]
    return {
        "running": all(algorithms[name]["listening"] for name in ALGORITHMS),
        "started_at": started_at,
        "algorithms": algorithms,
        # Keep the original dashboard data contract; it displays NetLLM here.
        "command": algorithms["netllm"]["command"],
        "stdout": combined_stdout[-160:],
        "history": history,
        "latest": algorithms["netllm"]["latest"],
        "player_telemetry": None,
        "model_inputs": model_inputs(history),
    }


def results_health():
    url = f"http://127.0.0.1:{CONFIG['results_port']}/"
    try:
        request = Request(url, headers={"User-Agent": "NetLLM-WebController/1.0"})
        with urlopen(request, timeout=1.5) as response:
            return {"available": response.status < 500, "url": url}
    except (HTTPError, URLError, OSError, TimeoutError) as error:
        return {"available": False, "url": url, "error": str(error)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            try:
                html = HTML_PATH.read_bytes().replace(b'data-mode="demo"', b'data-mode="live"', 1)
            except OSError:
                safe_json(self, 404, {"error": "dashboard not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html)
            return
        if path in ("/hls.min.js", "/assets/hls.min.js"):
            serve_file(
                self,
                ROOT / "hls.min.js",
                "text/javascript; charset=utf-8",
                "public, max-age=31536000, immutable",
            )
            return
        if path == "/api/status":
            safe_json(self, 200, status_snapshot())
            return
        if path == "/api/assets":
            safe_json(self, 200, {
                "root": str(ROOT),
                "receiver_script": str(RECEIVER_SCRIPT),
                "hls_root": str(HLS_ROOT),
            })
            return
        if path == "/api/runtime-config":
            safe_json(self, 200, {
                "results_port": CONFIG["results_port"],
                "base_port": CONFIG["base_port"],
                "playlists": {
                    name: f"/hls/{name}/index.m3u8" for name in ALGORITHMS
                },
            })
            return
        if path == "/api/results-health":
            safe_json(self, 200, results_health())
            return
        if path == "/hls":
            serve_hls_playlist(self, "netllm")
            return
        if path.startswith("/hls/"):
            parts = path.strip("/").split("/")
            if len(parts) >= 2 and parts[1] in ALGORITHMS:
                name = parts[1]
                if len(parts) == 2 or (len(parts) == 3 and parts[2] == "index.m3u8"):
                    serve_hls_playlist(self, name)
                    return
                if len(parts) == 3:
                    serve_hls_segment(self, name, parts[2])
                    return
        safe_json(self, 404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            if path == "/api/start":
                commands = start_receivers(body)
                safe_json(self, 200, {"ok": True, "commands": commands})
                return
            if path == "/api/stop":
                stop_receivers()
                safe_json(self, 200, {"ok": True})
                return
            safe_json(self, 404, {"error": "not found"})
        except Exception as error:
            safe_json(self, 400, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(
        description="NetLLM three-stream receiver and Web controller"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--base-port", type=int, default=5001)
    parser.add_argument("--results-port", type=int, default=5173)
    parser.add_argument("--hls-segment-duration", type=float, default=4.0)
    parser.add_argument("--hls-window-size", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.base_port <= 65533:
        parser.error("--base-port must leave room for three receiver ports")
    if args.hls_segment_duration <= 0:
        parser.error("--hls-segment-duration must be greater than 0")
    if args.hls_window_size <= 0:
        parser.error("--hls-window-size must be greater than 0")

    CONFIG.update({
        "base_port": args.base_port,
        "results_port": args.results_port,
        "segment_duration": args.hls_segment_duration,
        "window_size": args.hls_window_size,
    })
    for name in ALGORITHMS:
        hls_dir(name).mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"NetLLM ABR Web: http://{args.host}:{args.port}")
    for name in ALGORITHMS:
        print(
            f"{name}: receiver port {algorithm_port(name)}, "
            f"playlist /hls/{name}/index.m3u8"
        )
    print(f"Experiment results service: http://127.0.0.1:{args.results_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_receivers()
        server.server_close()


if __name__ == "__main__":
    main()
