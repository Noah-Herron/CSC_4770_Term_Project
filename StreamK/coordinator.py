"""Stream-K coordinator: splits GEMM work evenly by iteration count, not by tile.

Plays input + aggregation for this comparison harness (the real project
architecture keeps those as separate services -- see README).
"""

import argparse
import socket
import time

import numpy as np

from shared import send_msg, recv_msg


def build_tiles(M, N, tile_size):
    """Cuts the M x N output into tile_size x tile_size tiles, row by row."""
    tiles = []
    for ti, row_start in enumerate(range(0, M, tile_size)):
        row_end = min(row_start + tile_size, M)
        for tj, col_start in enumerate(range(0, N, tile_size)):
            col_end = min(col_start + tile_size, N)
            tiles.append((ti, tj, row_start, row_end, col_start, col_end))
    return tiles


def streamk_assignments(tiles, K, num_workers):
    """Splits all tiles' K-depth into one flat pool and hands out equal shares."""
    # 1. Lay every tile's MAC-loop iterations end to end into one pool
    total_units = len(tiles) * K
    iters_per_cta = -(-total_units // num_workers)

    # 2. Walk the pool, cutting it into one contiguous share per worker
    worker_jobs = [[] for _ in range(num_workers)]
    cursor = 0
    for w in range(num_workers):
        remaining = min(iters_per_cta, total_units - cursor)
        while remaining > 0:
            tile_index = cursor // K
            k_pos = cursor % K
            take = min(remaining, K - k_pos)
            worker_jobs[w].append((tile_index, k_pos, k_pos + take))
            cursor += take
            remaining -= take
    return worker_jobs


def send_to_worker(address, assignments, tiles, A, B):
    payload = {"assignments": []}
    for tile_index, k_start, k_end in assignments:
        ti, tj, r0, r1, c0, c1 = tiles[tile_index]
        payload["assignments"].append({
            "tile_row": ti, "tile_col": tj,
            "k_start": k_start, "k_end": k_end,
            "A_block": A[r0:r1, :].tolist(),
            "B_block": B[:, c0:c1].tolist(),
        })

    with socket.create_connection(address, timeout=30) as sock:
        send_msg(sock, payload)
        return recv_msg(sock)["results"]


def main():
    parser = argparse.ArgumentParser(description="Stream-K coordinator")
    parser.add_argument("--M", type=int, default=256)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", nargs="+", required=True,
                         help="host:port per worker, e.g. localhost:9001 localhost:9002")
    args = parser.parse_args()

    worker_addrs = [(w.split(":")[0], int(w.split(":")[1])) for w in args.workers]

    rng = np.random.default_rng(args.seed)
    A = rng.random((args.M, args.K))
    B = rng.random((args.K, args.N))

    tiles = build_tiles(args.M, args.N, args.tile_size)
    worker_jobs = streamk_assignments(tiles, args.K, len(worker_addrs))

    # 1. Print each worker's iteration count so the split can be checked
    print("[coordinator] MAC-loop-iteration distribution (should be ~equal across workers):")
    for w, jobs in enumerate(worker_jobs):
        units = sum(k_end - k_start for _, k_start, k_end in jobs)
        print(f"  worker {w} ({worker_addrs[w][0]}:{worker_addrs[w][1]}): "
              f"{units} units across {len(jobs)} sub-assignment(s)")

    tile_lookup = {(ti, tj): (r0, r1, c0, c1) for ti, tj, r0, r1, c0, c1 in tiles}

    # 2. Send each worker its share and sum the partials that land on the same tile
    C = np.zeros((args.M, args.N))
    start = time.perf_counter()
    for address, jobs in zip(worker_addrs, worker_jobs):
        if not jobs:
            continue
        results = send_to_worker(address, jobs, tiles, A, B)
        for r in results:
            r0, r1, c0, c1 = tile_lookup[(r["tile_row"], r["tile_col"])]
            C[r0:r1, c0:c1] += np.array(r["partial"])
    elapsed = time.perf_counter() - start

    # 3. Check the result against numpy

    expected = A @ B
    max_err = float(np.max(np.abs(C - expected)))
    print(f"\n[coordinator] wall-clock time: {elapsed:.4f}s")
    print(f"[coordinator] max abs error vs numpy A@B: {max_err:.2e}")
    print("[coordinator] PASS" if max_err < 1e-6 else "[coordinator] FAIL")


if __name__ == "__main__":
    main()
