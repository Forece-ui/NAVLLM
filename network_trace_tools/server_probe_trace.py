import argparse
import os
import socket
import struct
import time


HEADER_SIZE = 8
ACK = b"OK"


def recv_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def write_trace_line(output_path, timestamp, bandwidth_mbps):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="ascii") as trace_file:
        trace_file.write(f"{timestamp:.6f}\t{bandwidth_mbps:.6f}\n")
        trace_file.flush()


def run_probe(args):
    payload_size = int(args.payload_mb * 1024 * 1024)
    payload = b"\0" * payload_size
    header = struct.pack("!Q", payload_size)
    start_wall = time.time()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)

    with socket.create_connection((args.client_host, args.port), timeout=args.connect_timeout) as sock:
        sock.settimeout(args.socket_timeout)
        print(f"Connected to client {args.client_host}:{args.port}")
        print(f"Writing trace to {args.output}")
        print("timestamp_seconds\tthroughput_mbps")

        sample_idx = 0
        while args.samples <= 0 or sample_idx < args.samples:
            sample_started = time.perf_counter()
            sock.sendall(header)
            sock.sendall(payload)
            ack = recv_exact(sock, len(ACK))
            elapsed = max(time.perf_counter() - sample_started, 1e-9)

            if ack != ACK:
                raise RuntimeError(f"Client did not return ACK, got: {ack!r}")

            timestamp = time.time() - start_wall
            throughput_mbps = payload_size * 8 / elapsed / 1_000_000
            write_trace_line(args.output, timestamp, throughput_mbps)
            print(f"{timestamp:.3f}\t{throughput_mbps:.3f}")

            sample_idx += 1
            sleep_time = args.interval - (time.perf_counter() - sample_started)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    parser = argparse.ArgumentParser(
        description="Actively probe client throughput and write a NetLLM-style two-column trace."
    )
    parser.add_argument("--client-host", required=True, help="Client IP address or hostname.")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between probes.")
    parser.add_argument("--payload-mb", type=float, default=1.0, help="Probe payload size in MiB.")
    parser.add_argument("--samples", type=int, default=0, help="Number of samples. 0 means run forever.")
    parser.add_argument(
        "--output",
        default="data/traces/test/live-test/live_trace_0",
        help="Output trace file path. Use this default from adaptive_bitrate_streaming/.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output trace file first.")
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--socket-timeout", type=float, default=30.0)
    args = parser.parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be > 0")
    if args.payload_mb <= 0:
        raise ValueError("--payload-mb must be > 0")

    run_probe(args)


if __name__ == "__main__":
    main()
