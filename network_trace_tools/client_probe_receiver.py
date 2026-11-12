import argparse
import socket
import struct
import time


HEADER_SIZE = 8
ACK = b"OK"


def recv_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def handle_client(conn, addr):
    print(f"Connected: {addr}")
    try:
        while True:
            header = recv_exact(conn, HEADER_SIZE)
            if header is None:
                break
            payload_size = struct.unpack("!Q", header)[0]
            remaining = payload_size
            received = 0
            started = time.perf_counter()
            while remaining > 0:
                chunk = conn.recv(min(remaining, 1024 * 1024))
                if not chunk:
                    return
                received += len(chunk)
                remaining -= len(chunk)
            elapsed = max(time.perf_counter() - started, 1e-9)
            mbps = received * 8 / elapsed / 1_000_000
            conn.sendall(ACK)
            print(f"probe bytes={received} elapsed={elapsed:.3f}s throughput={mbps:.3f}Mbps")
    finally:
        conn.close()
        print(f"Disconnected: {addr}")


def main():
    parser = argparse.ArgumentParser(description="Receive active throughput probes from the video server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"Listening on {args.host}:{args.port}")
        while True:
            conn, addr = server.accept()
            handle_client(conn, addr)


if __name__ == "__main__":
    main()
