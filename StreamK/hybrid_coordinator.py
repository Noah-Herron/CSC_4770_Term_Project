"""DP + two-tile Stream-K coordinator (paper Sec 5.2) -- for direct comparison
against coordinator.py (basic Stream-K). Same worker.py, same wire protocol.
"""

import argparse
import time

import numpy as np

from shared import send_msg, recv_msg
from coordinator import build_tiles, send_to_worker


def hybrid_assignments(tiles, K, num_workers):
    """Does all but one DP round whole-tile, then Stream-K-splits what's left."""
    total_tiles = len(tiles)
    naive_full_rounds, leftover = divmod(total_tiles, num_workers)
    full_rounds = max(0, naive_full_rounds - 1) if naive_full_rounds > 0 else 0

    worker_jobs = [[] for _ in range(num_workers)]
    tile_cursor = 0

    # 1. Hand out whole tiles round-robin, one round short of the naive count
    for _ in range(full_rounds):
        for w in range(num_workers):
            worker_jobs[w].append((tile_cursor, 0, K))
            tile_cursor += 1

    # 2. Pool the held-back round plus the leftover tiles, split evenly
    stream_k_tile_count = total_tiles - tile_cursor
    if stream_k_tile_count > 0:
        leftover_tiles = list(range(tile_cursor, total_tiles))
        total_units = stream_k_tile_count * K
        iters_per_cta = -(-total_units // num_workers)
        cursor = 0
        for w in range(num_workers):
            remaining = min(iters_per_cta, total_units - cursor)
            while remaining > 0:
                local_tile = cursor // K
                tile_index = leftover_tiles[local_tile]
                k_pos = cursor % K
                take = min(remaining, K - k_pos)
                worker_jobs[w].append((tile_index, k_pos, k_pos + take))
                cursor += take
                remaining -= take

    return worker_jobs, full_rounds, stream_k_tile_count


def main():
    parser = argparse.ArgumentParser(description="Hybrid DP + two-tile Stream-K coordinator")
    parser.add_argument("--M", type=int, default=256)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", nargs="+", required=True)
    args = parser.parse_args()

    worker_addrs = [(w.split(":")[0], int(w.split(":")[1])) for w in args.workers]

    rng = np.random.default_rng(args.seed)
    A = rng.random((args.M, args.K))
    B = rng.random((args.K, args.N))

    tiles = build_tiles(args.M, args.N, args.tile_size)
    worker_jobs, full_rounds, sk_tiles = hybrid_assignments(tiles, args.K, len(worker_addrs))

    print(f"[hybrid] {len(tiles)} tiles -> {full_rounds} DP round(s), "
          f"{sk_tiles} tile(s) via Stream-K split (held-back wave + leftover)")
    for w, jobs in enumerate(worker_jobs):
        units = sum(k_end - k_start for _, k_start, k_end in jobs)
        print(f"  worker {w} ({worker_addrs[w][0]}:{worker_addrs[w][1]}): "
              f"{units} units across {len(jobs)} sub-assignment(s)")

    tile_lookup = {(ti, tj): (r0, r1, c0, c1) for ti, tj, r0, r1, c0, c1 in tiles}

    # 1. Send each worker its share and sum the partials that land on the same tile
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

    # 2. Check the result against numpy
    expected = A @ B
    max_err = float(np.max(np.abs(C - expected)))
    print(f"\n[hybrid] wall-clock time: {elapsed:.4f}s")
    print(f"[hybrid] max abs error vs numpy A@B: {max_err:.2e}")
    print("[hybrid] PASS" if max_err < 1e-6 else "[hybrid] FAIL")


if __name__ == "__main__":
    main()
