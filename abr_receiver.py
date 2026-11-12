"""与 board_gstreamer_abr_agent.py 配套的 ABR 控制与 chunk 接收服务。"""

import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import time
import signal
from pathlib import Path
import threading

STOP_EVENT = threading.Event()


def shutdown_handler(signum, frame):
    print("\\n收到退出信号，正在关闭 ABR receiver...", flush=True)
    STOP_EVENT.set()


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

BITRATES_KBPS = (300, 750, 1200, 1850, 2850, 4300)
JSON_LENGTH_SIZE = 4
RECEIVE_BLOCK_SIZE = 1024 * 1024
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
BITS_IN_BYTE = 8.0
CHUNK_TIL_VIDEO_END_CAP = 48.0


def netllm_throughput_feature(payload_bytes, receive_elapsed_seconds):
    """Return the MB/s feature used by the original training pipeline."""
    if payload_bytes <= 0 or receive_elapsed_seconds <= 0:
        return 0.0
    return float(payload_bytes) / float(receive_elapsed_seconds) / M_IN_K / M_IN_K


def remaining_chunk_feature(offer):
    """Return the normalized number of chunks including the offered chunk."""
    try:
        chunk_index = int(offer["video_chunk_index"])
        total_chunks = int(offer["total_video_chunks"])
    except (KeyError, TypeError, ValueError):
        # Compatibility with an older sender that did not provide video length.
        return 1.0
    remaining = max(total_chunks - chunk_index, 0)
    return min(remaining, CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP


def send_json(conn, message):
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)))
    conn.sendall(payload)


def recv_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        if STOP_EVENT.is_set():
            return None
        try:
            block = conn.recv(size - len(data))
        except socket.timeout:
            continue
        if not block:
            return None
        data.extend(block)
    return bytes(data)


def recv_json(conn):
    length_bytes = recv_exact(conn, JSON_LENGTH_SIZE)
    if length_bytes is None:
        return None
    size = struct.unpack("!I", length_bytes)[0]
    if size <= 0 or size > 1024 * 1024:
        raise ValueError("非法 JSON 消息长度：{}".format(size))
    payload = recv_exact(conn, size)
    if payload is None:
        raise ConnectionError("JSON 消息未接收完整")
    return json.loads(payload.decode("utf-8"))


class HeuristicPolicy:
    """用于先验证端到端协议的保守吞吐率策略。"""

    def choose(self, offer, runtime):
        throughput_mbps = runtime.throughput_mbps
        if throughput_mbps <= 0:
            throughput_mbps = float(offer.get("measured_throughput_mbps", 0.0))
        if throughput_mbps <= 0:
            return 0

        sizes = [int(value) for value in offer["chunk_sizes_bytes"]]
        safe_throughput_bps = throughput_mbps * 1_000_000.0 * 0.80
        available_seconds = max(runtime.buffer_seconds, runtime.chunk_seconds)
        selected = 0
        for action, size in enumerate(sizes):
            estimated_download = size * 8.0 / safe_throughput_bps
            if estimated_download <= available_seconds:
                selected = action
        return selected

class BBAPolicy:
    def choose(self, offer, runtime):

        buffer = runtime.buffer_seconds

        if buffer < 5:
            return 0
        elif buffer < 15:
            return 2
        elif buffer < 30:
            return 4
        else:
            return 5

class MPCPolicy:

    def choose(self, offer, runtime):

        throughput = runtime.throughput_mbps

        sizes = offer["chunk_sizes_bytes"]

        best = 0

        for i,size in enumerate(sizes):

            bitrate = BITRATES_KBPS[i]

            need = bitrate / 1000

            if need < throughput*0.85:
                best=i

        return best

class NetLLMPolicy:
    """延迟导入项目模型，避免启发式联调依赖 PyTorch/Transformers。"""

    def __init__(self, args):
        import pickle

        import numpy as np
        import torch
        from munch import Munch

        script_root = Path(__file__).resolve().parent
        if str(script_root) not in sys.path:
            sys.path.insert(0, str(script_root))

        from config import cfg
        from plm_special.data.dataset import ExperienceDataset
        from plm_special.models.low_rank import peft_model
        from plm_special.models.rl_policy import OfflineRLPolicy
        from plm_special.models.state_encoder import EncoderNetwork
        from plm_special.utils.plm_utils import load_plm
        from plm_special.utils.utils import set_random_seed

        set_random_seed(args.seed)
        model_path = Path(args.model_dir).expanduser()
        pool_path = Path(args.exp_pool_path).expanduser()
        if not model_path.is_absolute():
            model_path = script_root / model_path
        if not pool_path.is_absolute():
            pool_path = script_root / pool_path
        args.model_dir = str(model_path.resolve())
        args.exp_pool_path = str(pool_path.resolve())

        with open(args.exp_pool_path, "rb") as pool_file:
            exp_pool = pickle.load(pool_file)
        dataset = ExperienceDataset(
            exp_pool,
            gamma=args.gamma,
            scale=args.scale,
            max_length=args.w,
        )
        self.info = Munch(dataset.exp_dataset_info)
        plm, *_ = load_plm(
            args.plm_type,
            str(script_root.parent / "downloaded_plms" / args.plm_type / args.plm_size),
            device_input_side=args.device,
            device_output_side=args.device_out,
            device_middle_side=args.device_mid,
        )
        if args.plm_type != "llama":
            plm = plm.to(args.device)
        if args.rank != -1:
            plm = peft_model(plm, args.plm_type, rank=args.rank)

        state_encoder = EncoderNetwork(
            embed_dim=args.state_feature_dim
        ).to(args.device)
        self.model = OfflineRLPolicy(
            state_feature_dim=args.state_feature_dim,
            bitrate_levels=len(BITRATES_KBPS),
            state_encoder=state_encoder,
            plm=plm,
            plm_embed_size=cfg.plm_embed_sizes[args.plm_type][args.plm_size],
            max_length=args.w,
            max_ep_len=self.info.max_timestep + 1,
            device=args.device,
            device_out=args.device_out,
            which_layer=args.which_layer,
        )
        if args.rank > 0:
            self.model.plm.load_adapter(args.model_dir, adapter_name="default")
            modules_path = Path(args.model_dir) / "modules_except_plm.bin"
            self.model.modules_except_plm.load_state_dict(
                torch.load(str(modules_path), map_location=args.device)
            )
        else:
            model_path = Path(args.model_dir) / "model.bin"
            self.model.load_state_dict(
                torch.load(str(model_path), map_location=args.device)
            )
        self.model.eval()
        self.model.clear_dq()
        self.target_return = self.info.max_return * args.target_return_scale
        self.np = np
        self.torch = torch
        self.device = args.device
        self.scale = args.scale
        self.state = np.zeros((6, 6), dtype=np.float32)
        self.timestep = 0

    def choose(self, offer, runtime):
        state = self.state
        state[0, -1] = runtime.last_bitrate_kbps / float(max(BITRATES_KBPS))
        state[1, -1] = runtime.buffer_seconds / BUFFER_NORM_FACTOR
        # Training uses bytes / milliseconds / 1000, which is numerically MB/s,
        # not the Mbps value exposed by the live dashboard.
        state[2, -1] = netllm_throughput_feature(
            runtime.last_payload_bytes,
            runtime.last_receive_elapsed,
        )
        state[3, -1] = runtime.last_total_elapsed / BUFFER_NORM_FACTOR
        state[4, :] = self.np.asarray(
            offer["chunk_sizes_bytes"], dtype=self.np.float32
        ) / 1_000_000.0
        state[5, -1] = remaining_chunk_feature(offer)

        tensor = self.torch.as_tensor(
            state, dtype=self.torch.float32, device=self.device
        ).reshape(1, 1, 6, 6)
        with self.torch.no_grad():
            action = int(
                self.model.sample(
                    tensor,
                    target_return=self.target_return,
                    timestep=self.timestep,
                )
            )
        if action < 0 or action >= len(BITRATES_KBPS):
            raise RuntimeError("NetLLM 返回非法 action：{}".format(action))
        self.timestep += 1
        return action

    def observe_reward(self, reward):
        self.target_return -= reward / self.scale
        self.state = self.np.roll(self.state, -1, axis=1)
        self.state[:, -1] = 0.0


class RuntimeState:
    def __init__(self, chunk_seconds):
        self.chunk_seconds = chunk_seconds
        self.buffer_seconds = 0.0
        self.throughput_mbps = 0.0
        self.last_bitrate_kbps = BITRATES_KBPS[0]
        self.last_total_elapsed = 0.0
        self.last_rebuffer_seconds = 0.0
        self.last_payload_bytes = 0
        self.last_receive_elapsed = 0.0

    def update(self, bitrate, payload_bytes, receive_elapsed):
        # Match the offline ABR environment: only payload download time drains
        # the playback buffer. Model inference and protocol waiting are logged
        # separately and must not be treated as network stalls.
        download_elapsed = receive_elapsed
        rebuffer = max(download_elapsed - self.buffer_seconds, 0.0)
        self.buffer_seconds = min(
            max(self.buffer_seconds - download_elapsed, 0.0) + self.chunk_seconds,
            60.0,
        )
        self.throughput_mbps = (
            payload_bytes * BITS_IN_BYTE
            / max(receive_elapsed, 1e-9)
            / 1_000_000.0
        )
        previous_bitrate = self.last_bitrate_kbps
        self.last_bitrate_kbps = bitrate
        # Keep this attribute name because NetLLMPolicy already consumes it;
        # its value is now the training-compatible download duration.
        self.last_total_elapsed = download_elapsed
        self.last_rebuffer_seconds = rebuffer
        self.last_payload_bytes = payload_bytes
        self.last_receive_elapsed = receive_elapsed

        bitrate_reward = bitrate / 1000.0
        smooth_penalty = abs(bitrate - previous_bitrate) / 1000.0
        reward = bitrate_reward - 4.3 * rebuffer - smooth_penalty
        return rebuffer, reward


class ChunkReceiver:
    def __init__(self, args, policy):
        self.args = args
        self.policy = policy
        self.output_dir = Path(args.output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = RuntimeState(args.chunk_seconds)
        self.manifest_path = self.output_dir / "{}_manifest.jsonl".format(
            args.policy
        )
        self.compatibility_manifest_path = (
            self.output_dir / "received_manifest.jsonl"
        )
        # web_controller clears received_manifest.jsonl for a new experiment.
        # Use that as the signal to reset the tagged export as well, while
        # preserving append behaviour when the receiver is started manually.
        if not self.compatibility_manifest_path.exists():
            self.manifest_path.write_text("", encoding="utf-8")

    @staticmethod
    def validate_offer(offer):
        if offer.get("type") != "chunk_offer":
            raise ValueError("预期 chunk_offer，收到：{}".format(offer))
        bitrates = tuple(int(value) for value in offer.get("bitrates_kbps", []))
        sizes = offer.get("chunk_sizes_bytes", [])
        if bitrates != BITRATES_KBPS:
            raise ValueError("六档码率配置不一致：{}".format(bitrates))
        if len(sizes) != len(BITRATES_KBPS) or any(int(size) <= 0 for size in sizes):
            raise ValueError("chunk_sizes_bytes 必须包含六个正数")
        if "chunk_id" not in offer:
            raise ValueError("chunk_offer 缺少 chunk_id")

    def receive_payload(self, conn, header):
        size = int(header.get("size_bytes", header.get("size", -1)))
        if size <= 0:
            raise ValueError(
                "chunk_data 文件大小非法：size_bytes={} size={}".format(
                    header.get("size_bytes"), header.get("size")
                )
            )
        action = int(header.get("action", -1))
        if action < 0 or action >= len(BITRATES_KBPS):
            raise ValueError("chunk_data action 非法")
        bitrate = int(header.get("bitrate_kbps", -1))
        if bitrate != BITRATES_KBPS[action]:
            raise ValueError("chunk_data action 与码率不一致")

        chunk_id = str(header["chunk_id"])
#        bitrate_dir = self.output_dir / "chunks_{}k".format(bitrate)
#        bitrate_dir.mkdir(parents=True, exist_ok=True)
#        final_path = bitrate_dir / "chunk_{}.ts".format(chunk_id)
#        partial_path = final_path.with_suffix(final_path.suffix + ".part")
        final_path = self.output_dir / "chunk_{}.ts".format(chunk_id)
        partial_path = final_path.with_suffix(final_path.suffix + ".part")

        digest = hashlib.sha256()
        received = 0
        started = time.perf_counter()
        last_data_at = time.monotonic()
        last_wait_log_at = 0.0
        print(
            "[RECEIVE START] chunk={} bitrate={}k expected={} bytes part={}".format(
                chunk_id, bitrate, size, partial_path
            ),
            flush=True,
        )
        try:
            with partial_path.open("wb") as output_file:
                while received < size:
                    if STOP_EVENT.is_set():
                        raise ConnectionError(
                            "receiver 正在退出，chunk payload 未接收完整：{}/{} bytes".format(
                                received, size
                            )
                        )
                    try:
                        block = conn.recv(min(RECEIVE_BLOCK_SIZE, size - received))
                    except socket.timeout:
                        now = time.monotonic()
                        idle = now - last_data_at
                        if now - last_wait_log_at >= 1.0:
                            print(
                                "[RECEIVE WAIT] chunk={} received={}/{} bytes idle={:.1f}s".format(
                                    chunk_id, received, size, idle
                                ),
                                flush=True,
                            )
                            last_wait_log_at = now
                        if idle >= self.args.payload_idle_timeout:
                            raise TimeoutError(
                                "chunk {} 连续 {:.1f}s 未收到新数据，received={}/{}".format(
                                    chunk_id, idle, received, size
                                )
                            )
                        continue
                    if not block:
                        raise ConnectionError(
                            "chunk payload 未接收完整：{}/{} bytes".format(
                                received, size
                            )
                        )
                    output_file.write(block)
                    digest.update(block)
                    received += len(block)
                    last_data_at = time.monotonic()
                output_file.flush()
                os.fsync(output_file.fileno())
        except Exception as error:
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                "[RECEIVE FAILED] chunk={} received={}/{} bytes elapsed={:.3f}s "
                "error={}: {}".format(
                    chunk_id, received, size, elapsed, type(error).__name__, error
                ),
                file=sys.stderr,
                flush=True,
            )
            if partial_path.exists():
                try:
                    actual_part_size = partial_path.stat().st_size
                except OSError:
                    actual_part_size = -1
                print(
                    "[PART KEPT] {} size={} bytes".format(
                        partial_path, actual_part_size
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            raise

        receive_elapsed = max(time.perf_counter() - started, 1e-9)
        actual_digest = digest.hexdigest()
        expected_digest = header.get("sha256")
        if expected_digest and actual_digest != expected_digest:
            print(
                "[SHA256 FAILED] chunk={} expected={} actual={}，保留 .part".format(
                    chunk_id, expected_digest, actual_digest
                ),
                file=sys.stderr,
                flush=True,
            )
            send_json(
                conn,
                {
                    "type": "chunk_nack",
                    "chunk_id": chunk_id,
                    "error": "sha256_mismatch",
                    "received": received,
                    "sha256": actual_digest,
                },
            )
            raise RuntimeError("chunk {} SHA-256 校验失败".format(chunk_id))

        os.replace(str(partial_path), str(final_path))
        print(
            "[RECEIVE OK] chunk={} received={}/{} bytes elapsed={:.3f}s file={}".format(
                chunk_id, received, size, receive_elapsed, final_path
            ),
            flush=True,
        )
        return final_path, received, receive_elapsed, actual_digest

    def append_manifest(self, record):
        line = json.dumps(
            record, ensure_ascii=False, separators=(",", ":")
        ) + "\n"
        # The tagged file is convenient for exporting/comparing experiments.
        # The compatibility file keeps the current web_controller unchanged.
        for path in (self.manifest_path, self.compatibility_manifest_path):
            with path.open("a", encoding="utf-8") as output_file:
                output_file.write(line)
                output_file.flush()

    def handle_connection(self, conn, address):
        print("板端已连接：{}:{}".format(*address), flush=True)
        while True:
            offer = recv_json(conn)
            if offer is None:
                return
            if offer.get("type") == "close":
                send_json(conn, {"type": "closed"})
                return

            self.validate_offer(offer)
            chunk_id = str(offer["chunk_id"])
            transaction_started = time.perf_counter()
            decision_started = transaction_started
            action = self.policy.choose(offer, self.runtime)
            decision_elapsed = max(time.perf_counter() - decision_started, 0.0)
            bitrate = BITRATES_KBPS[action]
            send_json(
                conn,
                {
                    "type": "bitrate_decision",
                    "chunk_id": chunk_id,
                    "action": action,
                    "bitrate_kbps": bitrate,
                },
            )

            header = recv_json(conn)
            if header is None or header.get("type") != "chunk_data":
                raise ValueError("决策后未收到 chunk_data：{}".format(header))
            if str(header.get("chunk_id")) != chunk_id:
                raise ValueError("chunk_data chunk_id 与决策不一致")
            if int(header.get("action", -1)) != action:
                raise ValueError("板端发送的 action 与决策不一致")

            final_path, received, receive_elapsed, digest = self.receive_payload(
                conn, header
            )
            transaction_elapsed = max(
                time.perf_counter() - transaction_started, 1e-9
            )
            # total_elapsed_seconds remains in the existing ACK/data contract,
            # but now means effective download time, as in the training data.
            total_elapsed = receive_elapsed
            rebuffer, reward = self.runtime.update(
                bitrate, received, receive_elapsed
            )
            if hasattr(self.policy, "observe_reward"):
                self.policy.observe_reward(reward)

            ack = {
                "type": "chunk_ack",
                "chunk_id": chunk_id,
                "received": received,
                "sha256": digest,
                "buffer_seconds": self.runtime.buffer_seconds,
                "rebuffer_seconds": rebuffer,
                "receive_elapsed_seconds": receive_elapsed,
                "total_elapsed_seconds": total_elapsed,
                "decision_elapsed_seconds": decision_elapsed,
                "transaction_elapsed_seconds": transaction_elapsed,
                "throughput_mbps": self.runtime.throughput_mbps,
            }
            send_json(conn, ack)
            record = dict(ack)
            record.update(
                {
                    "algorithm": self.args.policy,
                    "action": action,
                    "bitrate_kbps": bitrate,
                    "reward": reward,
                    "path": str(final_path),
                    "sizes_are_estimates": bool(
                        offer.get("sizes_are_estimates", False)
                    ),
                    "estimated_sizes_bytes": offer["chunk_sizes_bytes"],
                    "received_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            self.append_manifest(record)
            print(
                "chunk={} action={} bitrate={}k bytes={} download={:.3f}s "
                "decision={:.3f}s transaction={:.3f}s "
                "throughput={:.3f}Mbps buffer={:.3f}s rebuffer={:.3f}s".format(
                    chunk_id,
                    action,
                    bitrate,
                    received,
                    total_elapsed,
                    decision_elapsed,
                    transaction_elapsed,
                    self.runtime.throughput_mbps,
                    self.runtime.buffer_seconds,
                    rebuffer,
                ),
                flush=True,
            )

    def close(self):
        STOP_EVENT.set()
        print("receiver shutdown requested", flush=True)


    def serve_forever(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.args.host, self.args.port))
            server.listen(4)
            print("[READY] {}:{}".format(self.args.host, self.args.port), flush=True)
            print("监听 {}:{}".format(self.args.host, self.args.port), flush=True)
            print("保存目录：{}".format(self.output_dir), flush=True)
            print("决策策略：{}".format(self.args.policy), flush=True)
            server.settimeout(1.0)

            while not STOP_EVENT.is_set():
                try:
                    conn, address = server.accept()
                except socket.timeout:
                    continue

                conn.settimeout(self.args.socket_timeout)
                try:
                    self.handle_connection(conn, address)
                except (
                    ConnectionError,
                    OSError,
                    socket.timeout,
                    ValueError,
                    RuntimeError,
                ) as error:
                    print("连接处理失败：{}".format(error), file=sys.stderr, flush=True)
                finally:
                    conn.close()
                    print("板端已断开：{}:{}".format(*address), flush=True)

            print("ABR receiver 已退出", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="PYNQ-Z2 ABR 模型控制与 chunk 接收端")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--output-dir", default="hls")
    parser.add_argument("--policy", choices=("netllm","bba","mpc"), default="netllm")
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--socket-timeout", type=float, default=5.0)
    parser.add_argument(
        "--payload-idle-timeout",
        type=float,
        default=300.0,
        help="chunk payload 连续无新数据的最大等待秒数",
    )

    # 以下参数只在 --policy model 时使用，与 abr_sender.py 保持一致。
    parser.add_argument("--model-dir", default="data/ft_plms/try_llama2_7b")
    parser.add_argument("--exp-pool-path", default="artifacts/exp_pools/exp_pool.pkl")
    parser.add_argument("--plm-type", default="llama")
    parser.add_argument("--plm-size", default="base")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-out", default="cuda:0")
    parser.add_argument("--device-mid", default=None)
    parser.add_argument("--state-feature-dim", type=int, default=256)
    parser.add_argument("--which-layer", type=int, default=-1)
    parser.add_argument("--w", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=1000.0)
    parser.add_argument("--target-return-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=100003)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds 必须大于0")
    if args.socket_timeout <= 0:
        raise ValueError("--socket-timeout 必须大于0")
    if args.payload_idle_timeout <= 0:
        raise ValueError("--payload-idle-timeout 必须大于0")
    if args.policy == "netllm":
        policy = NetLLMPolicy(args)
    elif args.policy == "bba":
        policy = BBAPolicy()
    elif args.policy == "mpc":
        policy = MPCPolicy()
    else:
        policy = HeuristicPolicy()
    receiver = ChunkReceiver(args, policy)

    try:
        receiver.serve_forever()
    except KeyboardInterrupt:
        shutdown_handler(None, None)
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
