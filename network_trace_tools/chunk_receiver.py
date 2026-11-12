import argparse
import json
import os
import socket
import struct
import time


HEADER_LEN_SIZE = 4
BUFFER_SIZE = 1024 * 1024


def recv_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def recv_json_header(conn):
    header_len_bytes = recv_exact(conn, HEADER_LEN_SIZE)
    if header_len_bytes is None:
        return None
    header_len = struct.unpack("!I", header_len_bytes)[0]
    header_bytes = recv_exact(conn, header_len)
    if header_bytes is None:
        return None
    return json.loads(header_bytes.decode("utf-8"))


def send_json(conn, message):
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)))
    conn.sendall(payload)


def safe_output_path(output_dir, name):
    clean_name = os.path.basename(name)
    return os.path.join(output_dir, clean_name)


def append_manifest(output_dir, header, received, elapsed, throughput_mbps):
    manifest_path = os.path.join(output_dir, "chunk_manifest.jsonl")
    record = {
        "seq": header.get("seq"),
        "filename": os.path.basename(header.get("filename", "")),
        "size": int(header.get("size", 0)),
        "received": received,
        "receive_elapsed": elapsed,
        "receive_throughput_mbps": throughput_mbps,
        "received_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for key in ["bitrate_level", "bitrate_kbps", "action", "chunk_duration"]:
        if key in header:
            record[key] = header[key]
    with open(manifest_path, "a", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def handle_connection(conn, addr, output_dir):
    print(f"Connected: {addr}")
    os.makedirs(output_dir, exist_ok=True)
    try:
        while True:
            header = recv_json_header(conn)
            if header is None:
                break
            if header.get("type") == "close":
                send_json(conn, {"type": "closed"})
                break
            if header.get("type") != "chunk":
                raise RuntimeError(f"Unknown message type: {header!r}")

            filename = header["filename"]
            size = int(header["size"])
            output_path = safe_output_path(output_dir, filename)

            received = 0
            started = time.perf_counter()
            with open(output_path, "wb") as output_file:
                while received < size:
                    chunk = conn.recv(min(BUFFER_SIZE, size - received))
                    if not chunk:
                        raise ConnectionError("Connection closed while receiving chunk payload.")
                    output_file.write(chunk)
                    received += len(chunk)

            elapsed = max(time.perf_counter() - started, 1e-9)
            throughput_mbps = received * 8 / elapsed / 1_000_000
            append_manifest(output_dir, header, received, elapsed, throughput_mbps)
            send_json(
                conn,
                {
                    "type": "ack",
                    "seq": header.get("seq"),
                    "filename": filename,
                    "received": received,
                    "receive_elapsed": elapsed,
                    "receive_throughput_mbps": throughput_mbps,
                },
            )
            print(f"received {filename} bytes={received} elapsed={elapsed:.3f}s throughput={throughput_mbps:.3f}Mbps")
    finally:
        conn.close()
        print(f"Disconnected: {addr}")


def main():
    parser = argparse.ArgumentParser(description="Receive video chunks/files from a server and send ACKs.")
    parser.add_argument("--host", default="0.0.0.0", help="Listening address.")
    parser.add_argument("--port", type=int, default=5002, help="Listening TCP port.")
    parser.add_argument("--output-dir", default="received_chunks", help="Directory used to save received files.")
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Listening on {args.host}:{args.port}")
        print(f"Saving chunks to {os.path.abspath(args.output_dir)}")
        while True:
            conn, addr = server.accept()
            handle_connection(conn, addr, args.output_dir)


if __name__ == "__main__":
    main()
