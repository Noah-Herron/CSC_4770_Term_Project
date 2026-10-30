"""Blocked SUMMA on a 2D worker grid (P_rows x P_cols).

At each step k along the inner dimension, each worker (i, j) receives horizontal
panel A[i, k] and vertical panel B[k, j], then updates its local block:
    C[i, j] += A[i, k] @ B[k, j]
"""

import math
import numpy as np
from protocol import recv_msg, send_msg

BLOCK_SIZE = 64


def get_grid_dims(num_workers, m, n):
    """Factor num_workers into a (rows, cols) grid that fits an m x n output."""
    rows = int(math.isqrt(num_workers))
    while num_workers % rows != 0:
        rows -= 1
    cols = num_workers // rows
    return min(rows, m), min(cols, n)


def bounds(size, parts, index):
    """Return (start, end) slice indices for partition `index` of `parts`."""
    return index * size // parts, (index + 1) * size // parts


def run_summa_master(connections, A, B):
    rows, cols = get_grid_dims(len(connections), A.shape[0], B.shape[1])
    active_workers = connections[: rows * cols]

    # 1. Initialize each worker with its local C block dimensions
    for rank, conn in enumerate(active_workers):
        i, j = divmod(rank, cols)
        top, bottom = bounds(A.shape[0], rows, i)
        left, right = bounds(B.shape[1], cols, j)
        send_msg(conn, {
            "algorithm": "summa",
            "shape": [bottom - top, right - left],
        })

    # 2. Step through the inner dimension in panels of width BLOCK_SIZE
    for start in range(0, A.shape[1], BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, A.shape[1])

        # Slice the current step's row panels of A and column panels of B
        a_panels = [
            A[bounds(A.shape[0], rows, i)[0] : bounds(A.shape[0], rows, i)[1], start:end].tolist()
            for i in range(rows)
        ]
        b_panels = [
            B[start:end, bounds(B.shape[1], cols, j)[0] : bounds(B.shape[1], cols, j)[1]].tolist()
            for j in range(cols)
        ]

        # Broadcast panel A[i] across row i and panel B[j] down column j
        for rank, conn in enumerate(active_workers):
            i, j = divmod(rank, cols)
            send_msg(conn, {
                "command": "step",
                "A_panel": a_panels[i],
                "B_panel": b_panels[j],
            })

        # Wait for all workers to finish the outer-product update for this step
        for conn in active_workers:
            recv_msg(conn)

    # 3. Collect the completed C blocks and assemble the final matrix
    C = np.empty((A.shape[0], B.shape[1]))
    for rank, conn in enumerate(active_workers):
        send_msg(conn, {"command": "finish"})
        i, j = divmod(rank, cols)
        top, bottom = bounds(A.shape[0], rows, i)
        left, right = bounds(B.shape[1], cols, j)
        C[top:bottom, left:right] = recv_msg(conn)["C_chunk"]

    return C.tolist()


def run_summa_worker(connection, job):
    C = np.zeros(job["shape"], dtype=np.float64)
    while True:
        msg = recv_msg(connection)
        if msg["command"] == "step":
            A_panel = np.asarray(msg["A_panel"], dtype=np.float64)
            B_panel = np.asarray(msg["B_panel"], dtype=np.float64)
            C += A_panel @ B_panel
            send_msg(connection, "OK")
        elif msg["command"] == "finish":
            send_msg(connection, {"C_chunk": C.tolist()})
            return
        else:
            raise ValueError(f"Unknown SUMMA command: {msg['command']}")