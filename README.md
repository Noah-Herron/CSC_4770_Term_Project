# Distributed matrix multiplication

Original master/worker structure, with SUMMA and K-stream (Stream-K) algorithms.

From the repository root, start the master and the configured worker pool:

```bash
docker compose -f code/docker-compose.yaml up --build
```

In another terminal, install the dependencies and choose an algorithm:

```bash
python3 -m venv code/venv
code/venv/bin/python -m pip install -r code/requirements.txt
code/venv/bin/python code/client.py --algorithm baseline --size 100 --verify
code/venv/bin/python code/client.py --algorithm summa --size 100 --verify
code/venv/bin/python code/client.py --algorithm kstream --size 100 --verify
```

The client accepts `--algorithm` (default baseline), `--size` (default 100), and optional `--verify`, as well as rectangular dimensions, worker count, seed, and host options. It sends one upload and one blocking result request, then prints server time, wait/download time, and the result shape. All algorithm commands generate identical matrices for a given size and seed (default 0). Verification checks the result against local NumPy after timing. The NumPy 2.3.5 dependency and the fix for verification at size 1000 are retained.

## Code walkthrough

* `code/client.py`: generate matrices, upload them, and wait for the result.
* `code/app.py`: Flask endpoints, the job queue, the TCP dispatcher, and the worker loop.
* `code/protocol.py`: send and receive length-prefixed JSON over TCP.
* `code/summa.py`: the SUMMA coordinator and worker functions.
* `code/kstream.py`: the Stream-K tile/K work splitter, coordinator, and worker function.

For **baseline**, the master divides A into two-row jobs, each paired with all of B. It reserves the same worker pool used by SUMMA and keeps their TCP connections open for the whole request. It sends one row job to each worker before collecting the batch of results, then repeats with the next rows. The worker processes compute independently; the master needs no extra threads. When all row jobs finish, it sorts and joins the output chunks. Workers do not sleep or reconnect between row jobs. The 0.5-second retry delay applies only after an error.

For **SUMMA**, the master collects worker connections and arranges them into a 2D grid (`rows x cols`, defaulting to 2x2 for 4 workers) where each worker `(i, j)` initializes its own local output block `C[i, j]`. At each step along the inner dimension:

1. The master slices the current horizontal panels `A[i, k]` (for each row `i`) and vertical panels `B[k, j]` (for each column `j`).
2. The master broadcasts `A[i, k]` across grid row `i` and `B[k, j]` down grid column `j` to each worker `(i, j)`.
3. Every worker updates its local output block in parallel (`C += A_panel @ B_panel`) and acknowledges completion (`OK`).

Repeat over the inner dimension in steps of `BLOCK_SIZE` (64 in `summa.py`; edge panels can be smaller), then collect and assemble the output blocks. Small matrices automatically shrink the grid dimensions so no worker owns an empty block.

For **K-stream** (`--algorithm kstream`), the master builds output tiles of up to 64x64 and flattens their K ranges into one stream of work. It assigns contiguous, ceiling-sized shares to the reserved workers, retaining the basic splitter from `StreamK.zip`'s `coordinator.py`. A share can cross tile boundaries or cover only part of a tile's K range. Workers compute partial products, and the master adds contributions to each output tile. Only the assigned K slices are transmitted. One assignment per worker is sent before collecting each batch, so workers compute concurrently without building a single enormous batch payload. Workers with empty shares wait for the existing `done` message. This integrates the basic Stream-K prototype; the separate hybrid coordinator is not selected by `kstream`.

K-stream uses the existing Flask API, queue, worker connections, four-byte JSON framing, timing, and Docker services. The standalone prototype's worker servers, eight-byte framing, matrix generator, and Docker configuration are not used. Workers are separate processes and perform local multiplication concurrently for all algorithms. `collect_workers` in `protocol.py` is the shared, blocking connection setup for each request.

The reference is van de Geijn and Watts, *SUMMA: Scalable Universal Matrix Multiplication Algorithm*, sections 5.1 and 5.3. [Original paper](https://www.google.com/search?q=https://sites.cs.ucsb.edu/~gilbert/cs140/old/cs140Win2009/assignments/lawn96-SUMMA.pdf). This version uses explicit master-coordinated blocking broadcasts over the existing worker TCP connections; it does not implement the paper's pipelined broadcasts.

## Comparing runs

Use the same matrix size and worker configuration, and run each algorithm several times. Compare the printed **Server time**: it includes worker collection, input distribution, computation, result collection, and assembly. It excludes the initial HTTP upload, queue waiting, final HTTP response/download, and local verification. The same timing boundaries apply to all algorithms. The result request waits for completion, so there is no two-second client polling interval in the measurement. The separate wait/download figure includes waiting for queued work and receiving the JSON result.

This removes the artificial worker pauses and serial baseline dispatch from the comparison. Baseline still sends all of B with every two-row job, whereas SUMMA broadcasts only the required row and column panels at each step over the same master-worker TCP connections. Those communication and block-size differences are part of the implementations being compared. Results are not a claim about SUMMA speedup in every implementation or at every size.

## API and tests

`POST /multiply` accepts `matrix_A`, `matrix_B`, and optional `algorithm` (`baseline`, `summa`, or `kstream`; default `baseline`) and `num_workers`. `GET /result/<request_id>` returns progress, a completed matrix, or an error. Adding `?wait=1` keeps the HTTP request open until completion; this is what the client uses. Requests and results are held in memory. Input and aggregation still share the original master container.

By default, all algorithms reserve up to `NUM_WORKERS=4` running workers (a 2x2 grid for SUMMA); small output dimensions can shrink the shared pool. The request times out if enough workers are unavailable. The current Compose file starts 16 workers; select a count with the client's `--workers` option or set the master's `NUM_WORKERS` default. Rebuild the Docker images after adding K-stream.

The experiment handler also accepts `--algorithm kstream` and includes it when scheduling all algorithms. It adds missing K-stream combinations when loading tracking; existing run counts are preserved.

Run the checks with:

```bash
code/venv/bin/python -m unittest discover -s code/tests -v
```
