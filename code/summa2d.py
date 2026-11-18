"""2D SUMMA, C-stationary, as in Agullo, Buttari, Guermouche, Herrmann & Jego (ACM TOMS 2023).

* Matrices are cut into b x b tiles laid out 2D block-cyclically on a p x q worker grid:
  tile (i, j) belongs to worker (i mod p, j mod q). The same rule places tiles of A, B and C.
* Owner of C computes: worker (r, c) keeps its C tiles for the whole run.
* For every step l along the inner dimension:
    - the owner of A's tile column l in each grid row broadcasts it along that row,
    - the owner of B's tile row l in each grid column broadcasts it down that column,
    - every worker adds A_panel @ B_panel to its C tiles.
* The broadcast shape is the variable under study. One topology is used for every broadcast:
    flat     - the owner sends to each worker itself (the paper's point-to-point baseline)
    binomial - the paper's dynamically detected collectives (binomial tree)
    knomial  - k-nomial tree
    chain    - each worker forwards to the next one: asynchronous pipelined SUMMA,
               which the paper leaves as future work
    auto     - selector.choose() picks per broadcast from the panel size and group size
* Workers talk to each other directly. A receiver thread runs up to `lookahead` steps ahead
  of the computation, so communication overlaps with computing (like the runtime's
  asynchronous execution in the paper).

The master only scatters the initial tiles, plans the broadcasts and gathers C. All data
moves as raw float64 bytes, never JSON.
"""

import math
import os
import queue
import threading
import time
import uuid

import numpy as np

from broadcast import Forwarder, connect_peer, recv_exact_into
from protocol import SOCKET_TIMEOUT, recv_array, recv_msg, send_array, send_msg
from selector import choose, default_segment_bytes
from summa import get_grid_dims
from topology import build_children

DEFAULT_BLOCK = int(os.environ.get("SUMMA2D_BLOCK", "256"))
DEFAULT_LOOKAHEAD = int(os.environ.get("SUMMA2D_LOOKAHEAD", "2"))
TOPOLOGIES = ("flat", "binomial", "knomial", "chain", "auto")
ALGORITHMS = tuple(f"summa2d_{t}" for t in TOPOLOGIES)
ITEM = 8  # bytes per float64


# LAYOUT
def plan_grid(num_workers, m, n, block_size=None):
    """Grid (p, q) and tile size b. By default b is 256, shrunk for small matrices so every worker gets tiles."""
    p0, q0 = get_grid_dims(num_workers, num_workers, num_workers)  # factor without capping
    b = block_size or min(DEFAULT_BLOCK, max(1, min(math.ceil(m / p0), math.ceil(n / q0))))
    p, q = min(p0, math.ceil(m / b)), min(q0, math.ceil(n / b))
    return p, q, b


def tiles_owned(total, b, parts, index):
    """Tile numbers t with t mod parts == index, and their (start, end) element ranges."""
    tiles = list(range(index, math.ceil(total / b), parts))
    return tiles, [(t * b, min((t + 1) * b, total)) for t in tiles]


def owned_indices(total, b, parts, index):
    _, ranges = tiles_owned(total, b, parts, index)
    return np.concatenate([np.arange(s, e) for s, e in ranges]) if ranges else np.empty(0, dtype=int)


def local_offsets(total, b, parts, index):
    """{tile number: (offset, width)} inside a worker's packed local array."""
    tiles, ranges = tiles_owned(total, b, parts, index)
    offsets, position = {}, 0
    for t, (s, e) in zip(tiles, ranges):
        offsets[t] = (position, e - s)
        position += e - s
    return offsets


def group_tree(members, root_position, topology, k):
    """Parent and children (as worker ranks) for a broadcast within `members` rooted at members[root_position]."""
    order = members[root_position:] + members[:root_position]  # the root goes first
    children = build_children(topology, len(order), k)
    parent = {}
    for position, kids in enumerate(children):
        for child in kids:
            parent[order[child]] = order[position]
    return parent, {order[i]: [order[c] for c in kids] for i, kids in enumerate(children)}


# MASTER
def plan_broadcasts(topology, m, n, K, p, q, b, k, calibration, table):
    """For each step l: a choice per grid row (A broadcast) and per grid column (B broadcast)."""
    rows_per_grid_row = [len(owned_indices(m, b, p, r)) for r in range(p)]
    cols_per_grid_col = [len(owned_indices(n, b, q, c)) for c in range(q)]

    def decide(size, group):
        receivers = group - 1
        if receivers < 1:
            return None  # nobody to send to
        if topology == "auto":
            pick = choose(size, receivers, calibration, table, k)
            return {"t": pick["topology"], "k": pick.get("k") or k, "seg": pick["segment_bytes"]}
        return {"t": topology, "k": k, "seg": default_segment_bytes(topology, size, receivers, calibration)}

    plan = []
    for l in range(math.ceil(K / b)):
        width = min(b, K - l * b)
        plan.append({
            "rows": [decide(rows_per_grid_row[r] * width * ITEM, q) for r in range(p)],
            "cols": [decide(cols_per_grid_col[c] * width * ITEM, p) for c in range(q)],
        })
    return plan


def run_summa2d_master(connections, peer_addrs, A, B, topology="auto", block_size=None,
                       k=4, lookahead=DEFAULT_LOOKAHEAD, calibration=None, table=None, num_workers=None):
    """Distribute tiles, let the workers run SUMMA among themselves, gather C.

    connections[i] / peer_addrs[i] belong to worker rank i, placed at grid (i // q, i % q).
    Returns (C, details).
    """
    if topology not in TOPOLOGIES:
        raise ValueError(f"topology must be one of {', '.join(TOPOLOGIES)}.")
    m, K = A.shape
    n = B.shape[1]
    # Plan from the requested worker count, exactly as the dispatcher did when collecting workers
    p, q, b = plan_grid(num_workers or len(connections), m, n, block_size)
    if len(connections) < p * q:
        raise ValueError(f"Need {p * q} workers for a {p}x{q} grid, got {len(connections)}.")
    workers = connections[: p * q]
    plan = plan_broadcasts(topology, m, n, K, p, q, b, k, calibration, table)
    setup_id = uuid.uuid4().hex

    rows = [owned_indices(m, b, p, r) for r in range(p)]
    cols = [owned_indices(n, b, q, c) for c in range(q)]
    started = time.perf_counter()
    for rank, conn in enumerate(workers):
        r, c = divmod(rank, q)
        send_msg(conn, {
            "algorithm": "summa2d", "setup_id": setup_id, "rank": rank, "grid": [p, q],
            "m": m, "n": n, "K": K, "b": b, "lookahead": lookahead, "plan": plan,
            "peers": [list(peer_addrs[i]) for i in range(p * q)],
        })
        # This worker's A tiles: its grid row's rows x the k-tiles with l mod q == c; B likewise
        send_array(conn, A[np.ix_(rows[r], owned_indices(K, b, q, c))])
        send_array(conn, B[np.ix_(owned_indices(K, b, p, r), cols[c])])
    distributed = time.perf_counter()

    C = np.empty((m, n))
    stats = []
    for rank, conn in enumerate(workers):
        r, c = divmod(rank, q)
        block, header = recv_array(conn)
        C[np.ix_(rows[r], cols[c])] = block
        stats.append(header)
    finished = time.perf_counter()

    used = {}
    for step in plan:
        for choice in step["rows"] + step["cols"]:
            if choice:
                used[choice["t"]] = used.get(choice["t"], 0) + 1
    details = {
        "algorithm_variant": f"summa2d_{topology}",
        "grid": [p, q], "block_size": b, "steps": len(plan), "lookahead": lookahead,
        "broadcasts_by_topology": used,
        "distribute_seconds": distributed - started,
        "compute_phase_seconds": finished - distributed,
        "max_worker_wait_seconds": max(s["wait_seconds"] for s in stats),
        "max_worker_gemm_seconds": max(s["gemm_seconds"] for s in stats),
        "workers": stats,
    }
    return C, details


# WORKER
class _Peers:
    """One connection (and one sending thread) per worker in my grid row and grid column."""

    def __init__(self, job, listener):
        self.rank, (p, q) = job["rank"], job["grid"]
        r, c = divmod(self.rank, q)
        self.row_members = [r * q + j for j in range(q)]
        self.col_members = [i * q + c for i in range(p)]
        neighbours = sorted(set(self.row_members + self.col_members) - {self.rank})
        self.socks, self.forwarders = {}, {}
        # The lower rank connects, the higher one accepts: each pair gets exactly one socket
        for other in neighbours:
            if other > self.rank:
                self.socks[other] = connect_peer(job["peers"][other], job["setup_id"], self.rank)
        for other in neighbours:
            if other < self.rank:
                self.socks[other] = listener.get(job["setup_id"], other)
        for other, sock in self.socks.items():
            self.forwarders[other] = Forwarder([sock])

    def close(self):
        try:
            for forwarder in self.forwarders.values():
                forwarder.close()  # waits until everything queued has been sent
        finally:
            for sock in self.socks.values():
                sock.close()


def _broadcast_step(peers, members, root_position, choice, local_panel, shape):
    """Take part in one broadcast; returns the full panel. local_panel is set only on the root."""
    if choice is None:  # group of one: I am the root and nobody needs it
        return local_panel
    parent, children = group_tree(members, root_position, choice["t"], choice["k"])
    me = peers.rank
    segment = max(1, int(choice["seg"]))
    if local_panel is not None:
        panel = np.ascontiguousarray(local_panel)
        view = memoryview(panel).cast("B")
        for offset in range(0, len(view), segment):
            for child in children[me]:
                peers.forwarders[child].send(view[offset:offset + segment])
        return panel
    panel = np.empty(shape)
    view = memoryview(panel).cast("B")
    sock = peers.socks[parent[me]]
    for offset in range(0, len(view), segment):
        part = view[offset:offset + segment]
        recv_exact_into(sock, part)
        for child in children[me]:  # forward each piece as soon as it arrives
            peers.forwarders[child].send(part)
    return panel


def run_summa2d_worker(control, job, listener):
    p, q = job["grid"]
    r, c = divmod(job["rank"], q)
    m, n, K, b = job["m"], job["n"], job["K"], job["b"]
    A_local, _ = recv_array(control)
    B_local, _ = recv_array(control)
    a_offsets = local_offsets(K, b, q, c)  # my A columns, by k-tile
    b_offsets = local_offsets(K, b, p, r)  # my B rows, by k-tile
    my_rows, my_cols = A_local.shape[0], B_local.shape[1]
    C_local = np.zeros((my_rows, my_cols))

    peers = _Peers(job, listener)
    ready = queue.Queue(maxsize=max(1, job["lookahead"]))
    timing = {"wait": 0.0, "gemm": 0.0}

    def receive_all():
        """Receive (and forward) every step's panels in order, staying `lookahead` steps ahead."""
        try:
            for l, step in enumerate(job["plan"]):
                width = min(b, K - l * b)
                a_root, b_root = l % q, l % p
                a_mine = A_local[:, a_offsets[l][0]:a_offsets[l][0] + width] if a_root == c else None
                b_mine = B_local[b_offsets[l][0]:b_offsets[l][0] + width, :] if b_root == r else None
                a_panel = _broadcast_step(peers, peers.row_members, a_root, step["rows"][r], a_mine, (my_rows, width))
                b_panel = _broadcast_step(peers, peers.col_members, b_root, step["cols"][c], b_mine, (width, my_cols))
                ready.put((a_panel, b_panel))
            ready.put(None)
        except BaseException as error:  # hand it to the computing thread
            ready.put(error)

    receiver = threading.Thread(target=receive_all, daemon=True)
    started = time.perf_counter()
    try:
        receiver.start()
        while True:
            waiting = time.perf_counter()
            item = ready.get(timeout=SOCKET_TIMEOUT)
            timing["wait"] += time.perf_counter() - waiting
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            computing = time.perf_counter()
            C_local += item[0] @ item[1]
            timing["gemm"] += time.perf_counter() - computing
        send_array(control, C_local, rank=job["rank"], grid_pos=[r, c],
                   wait_seconds=timing["wait"], gemm_seconds=timing["gemm"],
                   total_seconds=time.perf_counter() - started)
    finally:
        receiver.join(timeout=5)
        peers.close()