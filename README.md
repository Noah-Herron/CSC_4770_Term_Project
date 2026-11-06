# Distributed matrix multiplication

Original master/worker structure, with SUMMA added as a second algorithm.

From the repository root, start the master and four workers:

```bash
docker compose -f code/docker-compose.yaml up --build
```

In another terminal, install the dependencies and choose an algorithm:

```bash
python3 -m venv code/venv
code/venv/bin/python -m pip install -r code/requirements.txt
code/venv/bin/python code/client.py --algorithm baseline --size 100 --verify
code/venv/bin/python code/client.py --algorithm summa --size 100 --verify
```

The client keeps three arguments: `--algorithm` (required), `--size` (default 100), and optional `--verify`. It sends one upload and one blocking result request, then prints server time, wait/download time, and the result shape. Both algorithm commands generate identical matrices for a given size using seed 0. Verification checks the result against local NumPy after timing. The NumPy 2.3.5 dependency and the fix for verification at size 1000 are retained.

## Code walkthrough

* `code/client.py`: generate matrices, upload them, and wait for the result.
* `code/app.py`: Flask endpoints, the job queue, the TCP dispatcher, and the worker loop.
* `code/protocol.py`: send and receive length-prefixed JSON over TCP.
* `code/summa.py`: the SUMMA coordinator and worker functions.

For **baseline**, the master divides A into two-row jobs, each paired with all of B. It reserves the same worker pool used by SUMMA and keeps their TCP connections open for the whole request. It sends one row job to each worker before collecting the batch of results, then repeats with the next rows. The worker processes compute independently; the master needs no extra threads. When all row jobs finish, it sorts and joins the output chunks. Workers do not sleep or reconnect between row jobs. The 0.5-second retry delay applies only after an error.

For **SUMMA**, the master collects worker connections and arranges them into a 2D grid (`rows x cols`, defaulting to 2x2 for 4 workers) where each worker `(i, j)` initializes its own local output block `C[i, j]`. At each step along the inner dimension:

1. The master slices the current horizontal panels `A[i, k]` (for each row `i`) and vertical panels `B[k, j]` (for each column `j`).
2. The master broadcasts `A[i, k]` across grid row `i` and `B[k, j]` down grid column `j` to each worker `(i, j)`.
3. Every worker updates its local output block in parallel (`C += A_panel @ B_panel`) and acknowledges completion (`OK`).

Repeat over the inner dimension in steps of `BLOCK_SIZE` (64 in `summa.py`; edge panels can be smaller), then collect and assemble the output blocks. Small matrices automatically shrink the grid dimensions so no worker owns an empty block.

The only application thread is the TCP server thread from the original app, which runs beside Flask. There are no thread pools, worker listener threads, locks, persistent worker registry, background request scheduler, or benchmark runner. Workers are separate processes and perform local multiplication concurrently for both algorithms. `collect_workers` in `protocol.py` is the shared, blocking connection setup for each request.

The reference is van de Geijn and Watts, *SUMMA: Scalable Universal Matrix Multiplication Algorithm*, sections 5.1 and 5.3. [Original paper](https://www.google.com/search?q=https://sites.cs.ucsb.edu/~gilbert/cs140/old/cs140Win2009/assignments/lawn96-SUMMA.pdf). This version uses explicit master-coordinated blocking broadcasts over the existing worker TCP connections; it does not implement the paper's pipelined broadcasts.

## Comparing runs

Use the same matrix size and worker configuration, and run each algorithm several times. Compare the printed **Server time**: it includes worker collection, input distribution, computation, result collection, and assembly. It excludes the initial HTTP upload, queue waiting, final HTTP response/download, and local verification. The same timing boundaries apply to both algorithms. The result request waits for completion, so there is no two-second client polling interval in the measurement. The separate wait/download figure includes waiting for queued work and receiving the JSON result.

This removes the artificial worker pauses and serial baseline dispatch from the comparison. Baseline still sends all of B with every two-row job, whereas SUMMA broadcasts only the required row and column panels at each step over the same master-worker TCP connections. Those communication and block-size differences are part of the implementations being compared. Results are not a claim about SUMMA speedup in every implementation or at every size.

## API and tests

`POST /multiply` accepts `matrix_A`, `matrix_B`, and optional `algorithm` (`baseline` by default). `GET /result/<request_id>` returns progress, a completed matrix, or an error. Adding `?wait=1` keeps the HTTP request open until completion; this is what the client uses. Requests and results are held in memory. Input and aggregation still share the original master container.

By default, both algorithms reserve `NUM_WORKERS=4` running workers (a 2x2 grid for SUMMA); the request times out if enough workers are unavailable. To scale horizontally, increase `replicas` in `docker-compose.yaml` and set `NUM_WORKERS` on the master container to match (e.g., `6` for a 2x3 grid or `9` for a 3x3 grid). Rebuild the Docker images when switching to this version.

Run the checks with:

```bash
code/venv/bin/python -m unittest discover -s code/tests -v
```