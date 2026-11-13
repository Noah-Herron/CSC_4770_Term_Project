"""Length-prefixed JSON messages shared by the master and workers."""

import json
import struct

import numpy as np

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


def send_array(sock, array, **meta):
    """Send a NumPy array as raw bytes (a JSON header, then the data). Much faster than tolist()."""
    array = np.ascontiguousarray(array)
    send_msg(sock, {"shape": list(array.shape), "dtype": array.dtype.str, **meta})
    if array.size:  # an empty array (a worker that owns no tiles) is just the header
        sock.sendall(memoryview(array).cast("B"))


def recv_array(sock):
    """Receive an array sent by send_array. Returns (array, header)."""
    header = recv_msg(sock)
    array = np.empty(header["shape"], dtype=np.dtype(header["dtype"]))
    if array.size == 0:
        return array, header
    view = memoryview(array).cast("B")
    while len(view):
        received = sock.recv_into(view)
        if received == 0:
            raise EOFError("Connection closed in the middle of an array.")
        view = view[received:]
    return array, header