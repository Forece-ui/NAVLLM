import argparse
import json
import os
import socket
import struct
import time


HEADER_LEN_SIZE = 4
DEFAULT_BUFFER_SIZE = 1024 * 1024


def send_json(conn, message):
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)))
    conn.sendall(payload)


def recv_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def recv_json(conn):
    header_len_bytes = recv_exact(conn, HEADER_LEN_SIZE)
    if header_len_bytes is None:
        return None
    header_len = struct.unpack("!I", header_len_bytes)[0]
    payload = recv_exact(conn, header_len)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def list_files(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    files = []
    for name in sorted(os.listdir(input_path)):
        path = os.path.join(input_path, name)
        if os.path.isfile(path):
            files.append(path)
    return files


def write_trace_line(output_path, timestamp, throughput_mbps):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "a", encoding="ascii") as trace_file:
        trace_file.write(f"{timestamp:.6f}\t{throughput_mbps:.6f}\n")
        trace_file.flush()


def load_limit_profile(profile_path):
    profile = []
    with open(profile_path, "r", encoding="utf-8") as profile_file:
        for line_no, raw_line in enumerate(profile_file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError(f"Bad profile line {line_no}: expected '<seconds> <mbps>', got {raw_line!r}")
            timestamp = float(parts[0])
            mbps = float(parts[1])
            if timestamp < 0:
                raise ValueError(f"Bad profile line {line_no}: timestamp must be >= 0")
            if mbps <= 0:
                raise ValueError(f"Bad profile line {line_no}: mbps must be > 0")
            profile.append((timestamp, mbps))

    if not profile:
        raise ValueError(f"Limit profile is empty: {profile_path}")

    profile.sort(key=lambda item: item[0])
    if profile[0][0] != 0:
        profile.insert(0, (0.0, profile[0][1]))
    return profile


def get_profile_limit_mbps(profile, elapsed_seconds):
    current_mbps = profile[0][1]
    for start_seconds, mbps in profile:
        if elapsed_seconds < start_seconds:
            break
        current_mbps = mbps
    return current_mbps


def get_current_limit_mbps(limit_profile, fixed_limit_mbps, transfer_started):
    if limit_profile is not None:
        elapsed = time.perf_counter() - transfer_started
        return get_profile_limit_mbps(limit_profile, elapsed)
    return fixed_limit_mbps


def send_file(
    conn,
    path,
    seq,
    transfer_started,
    limit_profile=None,
    limit_mbps=None,
    send_block_bytes=DEFAULT_BUFFER_SIZE,
    metadata=None,
):
    file_size = os.path.getsize(path)
    filename = os.path.basename(path)
    header = {"type": "chunk", "seq": seq, "filename": filename, "size": file_size}
    if metadata:
        header.update(metadata)
    send_json(conn, header)

    started = time.perf_counter()
    next_send_time = started
    with open(path, "rb") as input_file:
        while True:
            data = input_file.read(send_block_bytes)
            if not data:
                break
            conn.sendall(data)

            current_limit_mbps = get_current_limit_mbps(limit_profile, limit_mbps, transfer_started)
            if current_limit_mbps is not None:
                next_send_time += len(data) * 8 / (current_limit_mbps * 1_000_000)
                sleep_time = next_send_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    ack = recv_json(conn)
    elapsed = max(time.perf_counter() - started, 1e-9)
    if ack is None or ack.get("type") != "ack":
        raise RuntimeError(f"Did not receive ACK for {filename}, got: {ack!r}")
    if int(ack.get("received", -1)) != file_size:
        raise RuntimeError(f"Receiver ACK size mismatch for {filename}: {ack!r}")

    throughput_mbps = file_size * 8 / elapsed / 1_000_000
    return filename, file_size, elapsed, throughput_mbps, ack


def run(args):
    files = list_files(args.input)
    if not files:
        raise FileNotFoundError(f"No files found in {args.input}")

    limit_profile = load_limit_profile(args.limit_profile) if args.limit_profile else None

    if args.overwrite and os.path.exists(args.trace_output):
        os.remove(args.trace_output)

    start_wall = time.time()
    transfer_started = time.perf_counter()
    with socket.create_connection((args.client_host, args.port), timeout=args.connect_timeout) as conn:
        conn.settimeout(args.socket_timeout)
        print(f"Connected to receiver {args.client_host}:{args.port}")
        print(f"Sending {len(files)} file(s)")
        print(f"Writing trace to {args.trace_output}")
        if limit_profile is not None:
            print(f"Using limit profile: {args.limit_profile}")
        elif args.limit_mbps is not None:
            print(f"Using fixed limit: {args.limit_mbps} Mbps")
        print("filename\tbytes\tsend_elapsed_seconds\tthroughput_mbps")

        for seq, path in enumerate(files):
            filename, file_size, elapsed, throughput_mbps, ack = send_file(
                conn,
                path,
                seq,
                transfer_started,
                limit_profile=limit_profile,
                limit_mbps=args.limit_mbps,
                send_block_bytes=args.send_block_kb * 1024,
            )
            timestamp = time.time() - start_wall
            write_trace_line(args.trace_output, timestamp, throughput_mbps)
            print(f"{filename}\t{file_size}\t{elapsed:.3f}\t{throughput_mbps:.3f}")

            if args.interval > 0 and seq != len(files) - 1:
                time.sleep(args.interval)

        send_json(conn, {"type": "close"})
        recv_json(conn)


def main():
    parser = argparse.ArgumentParser(
        description="Send video chunks/files to a receiver and record real-time throughput trace."
    )
    parser.add_argument("--client-host", required=True, help="Receiver/client IP address.")
    parser.add_argument("--port", type=int, default=5002, help="Receiver TCP port.")
    parser.add_argument("--input", required=True, help="File or directory to send.")
    parser.add_argument(
        "--trace-output",
        default="data/traces/test/live-test/live_trace_0",
        help="Two-column NetLLM trace output: timestamp_seconds throughput_mbps.",
    )
    parser.add_argument("--interval", type=float, default=0.0, help="Sleep seconds between files.")
    parser.add_argument(
        "--limit-mbps",
        type=float,
        default=None,
        help="Optional application-level send rate limit in Mbps, for example 0.5 or 5.",
    )
    parser.add_argument(
        "--limit-profile",
        default=None,
        help="Optional two-column rate profile file: '<elapsed_seconds> <mbps>'. Overrides --limit-mbps.",
    )
    parser.add_argument(
        "--send-block-kb",
        type=int,
        default=64,
        help="Send block size in KiB. Smaller values make --limit-mbps smoother.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace trace output before sending.")
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--socket-timeout", type=float, default=60.0)
    args = parser.parse_args()

    if args.limit_mbps is not None and args.limit_mbps <= 0:
        raise ValueError("--limit-mbps must be > 0")
    if args.send_block_kb <= 0:
        raise ValueError("--send-block-kb must be > 0")

    run(args)


if __name__ == "__main__":
    main()
