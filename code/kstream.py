"""Stream-K work splitting adapted from StreamK/StreamK/coordinator.py.

Flatten output tiles and their K iterations into one stream, then give each
worker a contiguous share. Shares can cross tile boundaries; partial products
for the same tile are summed by the master. Use the existing worker connections
and protocol, sending one assignment per worker per batch to bound payloads.
"""

import numpy as np

from protocol import recv_msg, send_msg

TILE_SIZE = 64


def build_tiles(M, N, tile_size):
    """Return (tile_row, tile_col, row_start, row_end, col_start, col_end) tiles."""
    tiles = []
    for ti, row_start in enumerate(range(0, M, tile_size)):
        row_end = min(row_start + tile_size, M)
        for tj, col_start in enumerate(range(0, N, tile_size)):
            col_end = min(col_start + tile_size, N)
            tiles.append((ti, tj, row_start, row_end, col_start, col_end))
    return tiles


def streamk_assignments(tiles, K, num_workers):
    """Split tile/K work into ceiling-sized contiguous shares, as in the prototype."""
    total_units = len(tiles) * K
    units_per_worker = -(-total_units // num_workers)
    worker_jobs = [[] for _ in range(num_workers)]
    cursor = 0
    for jobs in worker_jobs:
        remaining = min(units_per_worker, total_units - cursor)
        while remaining > 0:
            tile_index, k_start = divmod(cursor, K)
            take = min(remaining, K - k_start)
            jobs.append((tile_index, k_start, k_start + take))
            cursor += take
            remaining -= take
    return worker_jobs


def run_kstream_master(connections, A, B):
    tiles = build_tiles(A.shape[0], B.shape[1], TILE_SIZE)
    worker_jobs = streamk_assignments(tiles, A.shape[1], len(connections))
    pending = [iter(jobs) for jobs in worker_jobs]
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float64)

    while True:
        batch = []
        for connection, jobs in zip(connections, pending):
            assignment = next(jobs, None)
            if assignment is None:
                continue
            tile_index, k_start, k_end = assignment
            _, _, r0, r1, c0, c1 = tiles[tile_index]
            # Slice K here so partial-tile jobs do not transmit unused data.
            send_msg(connection, {
                "algorithm": "kstream",
                "A_block": A[r0:r1, k_start:k_end].tolist(),
                "B_block": B[k_start:k_end, c0:c1].tolist(),
            })
            batch.append((connection, tile_index))

        if not batch:
            break

        # Dispatch the entire batch before collecting, like the baseline.
        for connection, tile_index in batch:
            _, _, r0, r1, c0, c1 = tiles[tile_index]
            C[r0:r1, c0:c1] += np.asarray(recv_msg(connection)["partial"], dtype=np.float64)

    return C.tolist()


def run_kstream_worker(connection, job):
    A_block = np.asarray(job["A_block"], dtype=np.float64)
    B_block = np.asarray(job["B_block"], dtype=np.float64)
    send_msg(connection, {"partial": (A_block @ B_block).tolist()})
