"""Length-prefixed JSON messages shared by the coordinator and workers."""

import json
import socket
import struct

HEADER = "!Q"
HEADER_SIZE = struct.calcsize(HEADER)


def send_msg(sock: socket.socket, obj) -> None:
    payload = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack(HEADER, len(payload)) + payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed before expected bytes arrived")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket):
    (length,) = struct.unpack(HEADER, _recv_exact(sock, HEADER_SIZE))
    return json.loads(_recv_exact(sock, length).decode("utf-8"))
