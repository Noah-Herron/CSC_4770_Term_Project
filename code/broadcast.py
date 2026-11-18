"""Broadcast engine: the master (rank 0) sends a message to the workers along a tree or chain.

Workers open direct peer connections to each other on PEER_PORT, so the data really
travels worker to worker instead of through the master. Control messages (setup,
acknowledgements) still use each worker's existing connection to the master.

Message data is raw bytes, not JSON. Each node forwards a segment to its children as
soon as it arrives (on a separate thread), so a segmented chain is a real pipeline.
"""

import os
import queue
import selectors
import socket
import statistics
import threading
import time
import uuid
import zlib

from protocol import SOCKET_TIMEOUT, recv_msg, send_msg
from topology import build_children, depth, parents_from_children

PEER_PORT = int(os.environ.get("PEER_PORT", "7000"))
MAX_BROADCAST_BYTES = 256 * 1024 * 1024


def _tune(sock):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # no Nagle delay on small messages
    sock.settimeout(SOCKET_TIMEOUT)


def recv_exact_into(sock, view):
    while len(view):
        received = sock.recv_into(view)
        if received == 0:
            raise EOFError("Peer closed the connection mid-message.")
        view = view[received:]


def connect_peer(address, setup_id, my_rank):
    sock = socket.create_connection((address[0], int(address[1])), SOCKET_TIMEOUT)
    _tune(sock)
    send_msg(sock, {"setup_id": setup_id, "from": my_rank})
    return sock


class PeerListener:
    """Accepts peer connections in the background and hands them out by (setup_id, sender)."""

    def __init__(self, port=PEER_PORT, host="0.0.0.0"):
        self._server = socket.create_server((host, port))
        self.port = self._server.getsockname()[1]
        self._incoming = queue.Queue()
        self._early = {}  # (setup_id, from_rank) -> connection that arrived before it was asked for
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self):
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            try:
                _tune(conn)
                hello = recv_msg(conn)
            except (OSError, EOFError, ValueError):
                conn.close()
                continue
            self._incoming.put((hello, conn))

    def get(self, setup_id, from_rank, timeout=SOCKET_TIMEOUT):
        """The connection from `from_rank` for this setup. Connections from other ranks of the
        same setup can arrive first; they are kept until asked for. Ones from older setups are closed."""
        deadline = time.monotonic() + timeout
        while True:
            for key in list(self._early):
                if key[0] != setup_id:
                    self._early.pop(key).close()  # left over from an earlier setup that failed
            if (setup_id, from_rank) in self._early:
                return self._early.pop((setup_id, from_rank))
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    raise queue.Empty
                hello, conn = self._incoming.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"No peer connection from rank {from_rank} for setup {setup_id}.")
            if isinstance(hello, dict) and "setup_id" in hello and "from" in hello:
                self._early[(hello["setup_id"], hello["from"])] = conn
            else:
                conn.close()

    def close(self):
        self._server.close()


class Forwarder:
    """Sends queued segments to every child on its own thread."""

    def __init__(self, socks):
        self.socks = socks
        self.error = None
        self._queue = queue.Queue()
        self._thread = None
        if socks:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self):
        while True:
            segment = self._queue.get()
            if segment is None:
                return
            if self.error is not None:
                continue
            try:
                for sock in self.socks:
                    sock.sendall(segment)
            except OSError as error:
                self.error = error

    def send(self, segment):
        if self.error is not None:
            raise self.error
        if self._thread is not None:
            self._queue.put(segment)

    def close(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
        if self.error is not None:
            raise self.error


def _segments(view, segment_bytes):
    for offset in range(0, len(view), segment_bytes):
        yield view[offset:offset + segment_bytes]


# WORKER SIDE
def run_bcast_worker(control, job, listener):
    rank, setup_id = job["rank"], job["setup_id"]
    size, segment_bytes = job["size_bytes"], job["segment_bytes"]
    child_socks, parent = [], None
    try:
        # Connect down first, then wait for the parent: no node waits on its own child, so no deadlock
        child_socks = [connect_peer(child["addr"], setup_id, rank) for child in job["children"]]
        parent = listener.get(setup_id, job["parent"])
        send_msg(control, "SETUP_OK")

        forwarder = Forwarder(child_socks)
        for trial in range(job["trials"]):
            view = memoryview(bytearray(size))  # fresh buffer, so queued segments are never overwritten
            crc = 0
            for part in _segments(view, segment_bytes):
                recv_exact_into(parent, part)
                forwarder.send(part)
                crc = zlib.crc32(part, crc)
            send_msg(control, {"trial": trial, "crc": crc})
        forwarder.close()
    finally:
        for sock in child_socks + ([parent] if parent else []):
            sock.close()


def run_pingpong_worker(control, job, listener):
    sock = listener.get(job["setup_id"], 0)
    try:
        for size in job["sizes"]:
            buffer = memoryview(bytearray(size))
            for _ in range(job["rounds"]):
                recv_exact_into(sock, buffer)
                sock.sendall(buffer)
    finally:
        sock.close()


# MASTER (ROOT) SIDE
def run_bcast_root(control_conns, peer_addrs, topology, size_bytes, segment_bytes=None,
                   trials=10, warmup=2, k=4):
    """Broadcast `size_bytes` random bytes to every worker, `warmup + trials` times.

    control_conns[i] and peer_addrs[i] belong to rank i + 1. A trial's time runs from
    the root starting to send until the last worker acknowledges the whole message.
    """
    if not 1 <= size_bytes <= MAX_BROADCAST_BYTES:
        raise ValueError(f"size_bytes must be between 1 and {MAX_BROADCAST_BYTES}.")
    num_nodes = len(control_conns) + 1
    children = build_children(topology, num_nodes, k)
    parents = parents_from_children(children)
    segment_bytes = max(1, min(segment_bytes or size_bytes, size_bytes))
    setup_id = uuid.uuid4().hex
    total = warmup + trials

    for index, conn in enumerate(control_conns):
        rank = index + 1
        send_msg(conn, {
            "algorithm": "bcast", "setup_id": setup_id, "rank": rank, "parent": parents[rank],
            "children": [{"rank": c, "addr": list(peer_addrs[c - 1])} for c in children[rank]],
            "size_bytes": size_bytes, "segment_bytes": segment_bytes, "trials": total,
        })

    child_socks = []
    selector = selectors.DefaultSelector()
    try:
        child_socks = [connect_peer(peer_addrs[c - 1], setup_id, 0) for c in children[0]]
        for conn in control_conns:
            reply = recv_msg(conn)
            if reply != "SETUP_OK":
                raise RuntimeError(f"Worker setup failed: {reply!r}")

        payload = memoryview(os.urandom(size_bytes))
        expected_crc = zlib.crc32(payload)
        rank_of = {conn: index + 1 for index, conn in enumerate(control_conns)}

        forwarder = Forwarder(child_socks)
        trial_seconds = []
        arrivals = {rank: [] for rank in rank_of.values()}
        for trial in range(total):
            # Each worker sends one ack per trial; stop watching it once that ack is in
            for conn in control_conns:
                selector.register(conn, selectors.EVENT_READ)
            started = time.perf_counter()
            for part in _segments(payload, segment_bytes):
                forwarder.send(part)

            finished, deadline = {}, time.monotonic() + SOCKET_TIMEOUT
            while len(finished) < len(control_conns):
                if forwarder.error is not None:
                    raise forwarder.error
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Trial {trial}: only {len(finished)} of {len(control_conns)} workers finished.")
                for key, _ in selector.select(timeout=1.0):
                    arrived = time.perf_counter() - started
                    ack = recv_msg(key.fileobj)
                    rank = rank_of[key.fileobj]
                    if ack.get("trial") != trial or ack.get("crc") != expected_crc:
                        raise RuntimeError(f"Rank {rank} sent a bad acknowledgement: {ack}")
                    finished[rank] = arrived
                    selector.unregister(key.fileobj)

            if trial >= warmup:
                trial_seconds.append(max(finished.values()))
                for rank, seconds in finished.items():
                    arrivals[rank].append(seconds)
        forwarder.close()
    finally:
        selector.close()
        for sock in child_socks:
            sock.close()

    median = statistics.median(trial_seconds)
    return {
        "topology": topology,
        "k": k if topology == "knomial" else None,
        "num_workers": len(control_conns),
        "size_bytes": size_bytes,
        "segment_bytes": segment_bytes,
        "segments": -(-size_bytes // segment_bytes),
        "depth": depth(children),
        "root_fanout": len(children[0]),
        "trials": trials,
        "warmup": warmup,
        "trial_seconds": trial_seconds,
        "median_seconds": median,
        "mean_seconds": statistics.fmean(trial_seconds),
        "min_seconds": min(trial_seconds),
        "max_seconds": max(trial_seconds),
        "stdev_seconds": statistics.stdev(trial_seconds) if len(trial_seconds) > 1 else 0.0,
        "bandwidth_bytes_per_second": size_bytes / median if median > 0 else None,
        "per_rank_median_seconds": {str(r): statistics.median(v) for r, v in arrivals.items()},
        "verified": True,
    }


def run_pingpong_root(control_conn, peer_addr, sizes, repeats=20, warmup=3):
    """Round trips between the master and one worker; one-way time = round trip / 2."""
    setup_id = uuid.uuid4().hex
    send_msg(control_conn, {"algorithm": "pingpong", "setup_id": setup_id,
                            "sizes": list(sizes), "rounds": warmup + repeats})
    sock = connect_peer(peer_addr, setup_id, 0)
    samples = []
    try:
        for size in sizes:
            outgoing = memoryview(os.urandom(size))
            incoming = memoryview(bytearray(size))
            one_way = []
            for round_index in range(warmup + repeats):
                started = time.perf_counter()
                sock.sendall(outgoing)
                recv_exact_into(sock, incoming)
                if round_index >= warmup:
                    one_way.append((time.perf_counter() - started) / 2)
            if incoming != outgoing:
                raise RuntimeError(f"Ping-pong data mismatch at {size} bytes.")
            samples.append({"size_bytes": size, "one_way_seconds": statistics.median(one_way),
                            "min_seconds": min(one_way), "max_seconds": max(one_way)})
    finally:
        sock.close()
    return samples