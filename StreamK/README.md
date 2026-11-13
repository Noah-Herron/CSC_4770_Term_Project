# Stream-K: work-centric distributed matrix multiplication

Our variant for the comparison. Splits GEMM work by total iteration count
across containers instead of by whole tile, so container counts that don't
divide the tile grid evenly don't leave one container idle while others
finish.

From the repo root:

```bash
docker compose -f StreamK/docker-compose.yaml up --build worker1 worker2 worker3 worker4 -d
docker compose -f StreamK/docker-compose.yaml run --rm coordinator
```

Or locally without Docker:

```bash
pip install -r StreamK/requirements.txt
python3 StreamK/worker.py --port 9001 &
python3 StreamK/worker.py --port 9002 &
python3 StreamK/worker.py --port 9003 &
python3 StreamK/worker.py --port 9004 &
python3 StreamK/coordinator.py --M 512 --K 512 --N 512 --tile-size 128 \
    --workers localhost:9001 localhost:9002 localhost:9003 localhost:9004
```

## Code walkthrough

* `shared.py`: length-prefixed JSON over TCP, same shape as the other algorithms' protocol module.
* `worker.py`: takes a batch of (tile, k-slice) jobs, multiplies its assigned k-range, returns partials.
* `coordinator.py`: input + aggregation combined for this harness (basic Stream-K).
* `hybrid_coordinator.py`: the DP + two-tile Stream-K variant, run the same way, for direct comparison.

## How it works

Every output tile's full K-depth counts as a pool of MAC-loop iterations
(`total = tiles * K`). We split that flat pool into even shares across
however many workers we have, crossing tile boundaries if needed, and hand
each worker its share as a batch of `(tile, k_start, k_end)` jobs:

1. Coordinator computes `iters_per_cta = ceil(total / num_workers)` and walks the flat (tile, k) sequence, cutting it into one contiguous share per worker.
2. Each worker multiplies only its assigned k-slice of a tile and sends back a partial.
3. Coordinator sums partials that share a tile across different k-ranges to rebuild `C`.

`hybrid_coordinator.py` adds one wrinkle on top: hold back one full round of
whole-tile assignment and fold it into the Stream-K split, so Stream-K
workers get more than one tile's worth of work but less than two. That's
the variant the paper's own benchmarks actually use.

The reference is Osama, Merrill, Cecka, Garland, Owens, *Stream-K:
Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication
on the GPU*, arXiv:2301.03598. [Paper](https://arxiv.org/abs/2301.03598),
[PDF](https://arxiv.org/pdf/2301.03598). It splits MAC-loop iterations
across GPU CTAs sharing on-chip memory; we split the same way across
Docker containers over TCP sockets instead, which is the actual thing
being tested here -- does the even-split idea still win once coordination
costs a network round trip instead of being nearly free. Their fixup uses
wait/signal flags between CTAs; we don't have shared memory over TCP, so
the coordinator does the summing centrally -- same net effect, different
mechanism. See `NOTES_streamk_paper.md` for exactly what's confirmed from
the paper versus filled in to make it run over sockets.

## Comparing runs

Run the same matrix size and worker count against `coordinator.py`,
`hybrid_coordinator.py`, and the other algorithms, several times each.
Both scripts print per-worker MAC-loop-iteration counts (the evidence for
"even workload," not just an assumption) and a wall-clock time covering
assignment, dispatch, compute, and reassembly -- nothing before the
coordinator starts or after it prints PASS/FAIL.

## Known gaps

* Dense only -- no sparse path yet.
* Full row/column blocks go over the wire, not minimal slices. Fine at this scale; revisit if bandwidth bites at 10,000x10,000.
* No fault tolerance -- a dropped worker connection crashes the run.
* This is a standalone comparison harness, not the final 3-service split (input / multiplication / aggregation). Split `coordinator.py` back apart if this variant gets adopted into the real project.
