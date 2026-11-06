"""Length-prefixed JSON messages shared by the master and workers."""

import json
import struct

SOCKET_TIMEOUT = 120


def send_msg(sock, message):
    payload = json.dumps(message, allow_nan=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)) + payload)


def recvall(sock, size):
    data = bytearray()
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            raise EOFError("Connection closed before the message was received.")
        data.extend(packet)
    return data


def recv_msg(sock):
    size = struct.unpack(">I", recvall(sock, 4))[0]
    return json.loads(recvall(sock, size))


def collect_workers(server, first_connection, first_host, count):
    """Both algorithms reserve the same number of workers for one request."""
    connections, hosts = [first_connection], [first_host]
    server.settimeout(SOCKET_TIMEOUT)
    try:
        while len(connections) < count:
            connection, address = server.accept()
            connection.settimeout(SOCKET_TIMEOUT)
            try:
                if recv_msg(connection) != "READY":
                    raise ValueError("Expected READY from worker.")
            except (OSError, EOFError, ValueError):
                connection.close()
                raise
            connections.append(connection)
            hosts.append(address[0])
        return connections, hosts
    except (OSError, EOFError, ValueError):
        for connection in connections[1:]:
            connection.close()
        raise
    finally:
        server.settimeout(None)