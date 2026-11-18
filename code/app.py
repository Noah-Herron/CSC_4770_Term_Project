import json
import os
import queue
import socket
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request

from broadcast import (MAX_BROADCAST_BYTES, PEER_PORT, PeerListener, run_bcast_root,
                       run_bcast_worker, run_pingpong_root, run_pingpong_worker)
from kstream import run_kstream_master, run_kstream_worker
from protocol import SOCKET_TIMEOUT, collect_workers, recv_msg, send_msg
from selector import DEFAULT_CALIBRATION, choose, default_segment_bytes, fit_calibration, load_json, predict
from summa import get_grid_dims, run_summa_master, run_summa_worker
from summa2d import ALGORITHMS as SUMMA2D_ALGORITHMS, plan_grid, run_summa2d_master, run_summa2d_worker
from topology import TOPOLOGIES

app = Flask(__name__)

# Shared state, touched by Flask threads and the dispatcher thread
state_lock = threading.Lock()
tasks_db = {}               # request_id -> tracking state
job_queue = queue.Queue()   # requests waiting for the dispatcher, oldest first

MATRIX_ALGORITHMS = ("baseline", "summa", "kstream") + SUMMA2D_ALGORITHMS
DEFAULT_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
BASELINE_CHUNK_ROWS = 2     # rows of A per baseline job
WAIT_TIMEOUT_SECONDS = 60   # cap for ?wait=1 so a stuck job can't hang the request forever

# Calibration and the measured selection table live here (bind-mounted to ./data on the host)
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
CALIBRATION_FILE = DATA_DIR / "calibration.json"
TABLE_FILE = DATA_DIR / "bcast_table.json"
DEFAULT_CALIBRATION_SIZES = [8, 1024, 8192, 65536, 262144, 1048576, 4194304, 16777216]


# FLASK API (INPUT & RESULTS)
@app.post("/multiply")
def receive_matrices():
    data = request.get_json(silent=True)
    try:
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")

        A = np.asarray(data["matrix_A"], dtype=np.float64)
        B = np.asarray(data["matrix_B"], dtype=np.float64)
        if A.ndim != 2 or B.ndim != 2 or 0 in A.shape or 0 in B.shape:
            raise ValueError("Matrices must be nonempty 2D arrays.")
        if A.shape[1] != B.shape[0]:
            raise ValueError("Incompatible matrix dimensions.")
        if not np.isfinite(A).all() or not np.isfinite(B).all():
            raise ValueError("Matrices must contain finite numbers.")

        algorithm = data.get("algorithm", "baseline")
        if algorithm not in MATRIX_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {', '.join(MATRIX_ALGORITHMS)}.")
        block_size = data.get("block_size")
        block_size = None if block_size is None else int(block_size)
        if block_size is not None and block_size < 1:
            raise ValueError("block_size must be at least 1.")

        num_workers = int(data.get("num_workers", DEFAULT_WORKERS))
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1.")
    except KeyError as error:
        return jsonify(error=f"Missing field: {error.args[0]}."), 400
    except (TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400

    request_id = str(uuid.uuid4())
    if algorithm == "baseline":
        job_count = -(-A.shape[0] // BASELINE_CHUNK_ROWS)  # ceiling division
    else:
        job_count = 1

    with state_lock:
        tasks_db[request_id] = {
            "algorithm": algorithm,
            "num_workers": num_workers,
            "total_jobs": job_count,
            "results": [],
            "workers": [],
            "status": "queued",
            "final_matrix": None,
            "error": None,
            "server_seconds": None,
        }
    job_queue.put({
        "request_id": request_id, "algorithm": algorithm,
        "num_workers": num_workers, "A": A, "B": B, "block_size": block_size,
    })
    return jsonify(
        request_id=request_id,
        message=f"Queued {algorithm} request: {job_count} job(s) across up to {num_workers} worker(s).",
    ), 202


@app.get("/result/<request_id>")
def get_result(request_id):
    # ?wait=1 lets the client block until completion instead of polling
    wait = request.args.get("wait") == "1"
    deadline = time.monotonic() + WAIT_TIMEOUT_SECONDS

    while True:
        with state_lock:
            task = tasks_db.get(request_id)
            if task is None:
                return jsonify(error="Request ID not found."), 404

            status = task["status"]
            if status == "completed":
                matrix = task["final_matrix"]
                response = jsonify(
                    status="completed",
                    final_matrix=None if matrix is None else matrix.tolist(),
                    server_seconds=task["server_seconds"],
                    workers_used=task["workers"],
                    result=task.get("result"),
                )
                # The client fetches a result once; free it so finished runs don't pile up in memory
                task["final_matrix"] = None
                return response, 200
            if status == "failed":
                return jsonify(status="failed", error=task["error"]), 200

            progress = f"{len(task['results'])}/{task['total_jobs']} jobs completed."

        # Sleep outside the lock so the dispatcher can keep updating the task
        if not wait or time.monotonic() >= deadline:
            return jsonify(status=status, progress=progress), 202
        time.sleep(0.01)


def _int_field(data, name, default, low, high=None):
    value = int(data.get(name, default))
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be between {low} and {high}." if high else f"{name} must be at least {low}.")
    return value


def _queue_task(kind, job):
    request_id = str(uuid.uuid4())
    with state_lock:
        tasks_db[request_id] = {
            "algorithm": kind, "num_workers": job.get("num_workers", 1), "total_jobs": 1,
            "results": [], "workers": [], "status": "queued", "final_matrix": None,
            "error": None, "server_seconds": None, "result": None,
        }
    job_queue.put({"request_id": request_id, "kind": kind, **job})
    return request_id


@app.post("/broadcast")
def receive_broadcast():
    """Time a broadcast of size_bytes from the master to num_workers workers.

    topology: binomial | knomial | chain | auto (auto asks the selector).
    """
    data = request.get_json(silent=True)
    try:
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")
        topology = data.get("topology", "auto")
        if topology not in TOPOLOGIES + ("auto",):
            raise ValueError(f"topology must be one of {', '.join(TOPOLOGIES + ('auto',))}.")
        segment = data.get("segment_bytes")
        job = {
            "num_workers": _int_field(data, "num_workers", DEFAULT_WORKERS, 1),
            "size_bytes": _int_field(data, "size_bytes", 1048576, 1, MAX_BROADCAST_BYTES),
            "topology": topology,
            "segment_bytes": None if segment is None else _int_field(data, "segment_bytes", 0, 1),
            "k": _int_field(data, "k", 4, 2, 64),
            "trials": _int_field(data, "trials", 10, 1, 1000),
            "warmup": _int_field(data, "warmup", 2, 0, 100),
        }
    except (TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400
    request_id = _queue_task("bcast", job)
    return jsonify(request_id=request_id,
                   message=f"Queued {topology} broadcast of {job['size_bytes']} bytes to {job['num_workers']} worker(s)."), 202


@app.post("/calibrate")
def receive_calibrate():
    """Ping-pong between the master and one worker to measure alpha and beta."""
    data = request.get_json(silent=True) or {}
    try:
        sizes = [int(size) for size in data.get("sizes", DEFAULT_CALIBRATION_SIZES)]
        if len(sizes) < 2 or min(sizes) < 1 or max(sizes) > MAX_BROADCAST_BYTES:
            raise ValueError("sizes needs at least two values between 1 and the broadcast limit.")
        job = {"num_workers": 1, "sizes": sorted(set(sizes)),
               "repeats": _int_field(data, "repeats", 20, 1, 1000)}
    except (TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400
    request_id = _queue_task("calibrate", job)
    return jsonify(request_id=request_id, message="Queued ping-pong calibration."), 202


@app.get("/selector")
def get_selection():
    """What the selector would pick, without running anything."""
    try:
        size_bytes = int(request.args.get("size_bytes", "1048576"))
        num_workers = int(request.args.get("num_workers", str(DEFAULT_WORKERS)))
        k = int(request.args.get("k", "4"))
        if size_bytes < 1 or num_workers < 1 or k < 2:
            raise ValueError("size_bytes and num_workers must be positive, k at least 2.")
    except ValueError as error:
        return jsonify(error=str(error)), 400
    calibration = load_json(CALIBRATION_FILE)
    return jsonify(choice=choose(size_bytes, num_workers, calibration, load_json(TABLE_FILE), k),
                   calibration=calibration or DEFAULT_CALIBRATION), 200


def run_broadcast_job(connections, hosts, job):
    """Resolve auto/segment choices with the latest calibration and table, then run the broadcast."""
    calibration = load_json(CALIBRATION_FILE)
    size, workers, k = job["size_bytes"], job["num_workers"], job["k"]
    selection = None
    if job["topology"] == "auto":
        selection = choose(size, workers, calibration, load_json(TABLE_FILE), k)
        topology, segment = selection["topology"], selection["segment_bytes"]
        k = selection.get("k") or k
    else:
        topology = job["topology"]
        segment = job["segment_bytes"] or default_segment_bytes(topology, size, workers, calibration)

    result = run_bcast_root(connections, [(host, PEER_PORT) for host in hosts], topology, size,
                            segment, job["trials"], job["warmup"], k)
    params = calibration or DEFAULT_CALIBRATION
    result.update(
        requested_topology=job["topology"],
        requested_segment_bytes=job["segment_bytes"],
        selection=selection,
        model_seconds=predict(topology, size, workers + 1, params["alpha"], params["beta"],
                              result["segments"], k),
        calibrated=calibration is not None,
    )
    return result


def run_calibration_job(connections, hosts, job):
    samples = run_pingpong_root(connections[0], (hosts[0], PEER_PORT), job["sizes"], job["repeats"])
    calibration = {**fit_calibration(samples), "samples": samples, "source": "ping-pong",
                   "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "worker": hosts[0]}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CALIBRATION_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(calibration, indent=2))
    tmp.replace(CALIBRATION_FILE)
    return calibration


# TCP SERVER (MASTER DISPATCHER & AGGREGATOR)
def run_baseline_master(connections, A, B, task):
    """Send a batch first, then collect it. Worker processes compute in parallel."""
    B_matrix = B.tolist()
    batch_size = BASELINE_CHUNK_ROWS * len(connections)
    for batch_start in range(0, A.shape[0], batch_size):
        starts = range(batch_start, min(batch_start + batch_size, A.shape[0]), BASELINE_CHUNK_ROWS)
        batch = list(zip(connections, starts))
        for connection, start_row in batch:
            send_msg(connection, {
                "algorithm": "baseline", "start_row": start_row,
                "A_chunk": A[start_row:start_row + BASELINE_CHUNK_ROWS].tolist(),
                "B_matrix": B_matrix,
            })
        for connection, _ in batch:
            task["results"].append(recv_msg(connection))
    chunks = sorted(task["results"], key=lambda chunk: chunk["start_row"])
    return [row for chunk in chunks for row in chunk["C_chunk"]]


def run_job(server_socket, first_connection, first_host):
    """Wait for the next request, then run it on this worker plus enough others to fill the grid."""
    connections = [first_connection]
    job = None
    try:
        print(f"Dispatcher: worker connected from {first_host}, waiting for READY", flush=True)
        first_connection.settimeout(SOCKET_TIMEOUT)
        hello = recv_msg(first_connection)
        if hello != "READY":
            print(f"Dispatcher: expected READY, got {hello!r}; dropping connection", flush=True)
            return

        print(f"Dispatcher: {first_host} is ready; waiting for a request "
              f"({job_queue.qsize()} queued)", flush=True)

        job = job_queue.get()  # The first worker waits here until a request arrives.
        with state_lock:
            task = tasks_db[job["request_id"]]
            task["status"] = "processing"
        started = time.perf_counter()
        kind = job.get("kind", "matrix")

        if kind in ("bcast", "calibrate"):
            print(f"Dispatcher: starting {kind} request {job['request_id']} "
                  f"({job['num_workers']} workers)", flush=True)
            connections, hosts = collect_workers(
                server_socket, first_connection, first_host, job["num_workers"])
            print(f"Dispatcher: collected {len(connections)} workers; running", flush=True)
            if kind == "bcast":
                result = run_broadcast_job(connections, hosts, job)
            else:
                result = run_calibration_job(connections, hosts, job)
            final_matrix = None
        else:
            A, B = job["A"], job["B"]
            print(f"Dispatcher: starting {job['algorithm']} request {job['request_id']} "
                  f"({A.shape[0]}x{A.shape[1]} @ {B.shape[0]}x{B.shape[1]}, {job['num_workers']} workers)", flush=True)

            if job["algorithm"] in SUMMA2D_ALGORITHMS:
                grid_rows, grid_cols, _ = plan_grid(job["num_workers"], A.shape[0], B.shape[1],
                                                    job.get("block_size"))
            else:
                grid_rows, grid_cols = get_grid_dims(job["num_workers"], A.shape[0], B.shape[1])
            print(f"Dispatcher: grid {grid_rows}x{grid_cols}, collecting {grid_rows * grid_cols} workers", flush=True)
            connections, hosts = collect_workers(
                server_socket, first_connection, first_host, grid_rows * grid_cols)
            print(f"Dispatcher: collected {len(connections)} workers; running", flush=True)

            if job["algorithm"] in SUMMA2D_ALGORITHMS:
                final_matrix, result = run_summa2d_master(
                    connections, [(host, PEER_PORT) for host in hosts], A, B,
                    topology=job["algorithm"].removeprefix("summa2d_"),
                    block_size=job.get("block_size"),
                    calibration=load_json(CALIBRATION_FILE), table=load_json(TABLE_FILE),
                    num_workers=job["num_workers"])
            elif job["algorithm"] == "summa":
                final_matrix, result = run_summa_master(connections, A, B), None
            elif job["algorithm"] == "kstream":
                final_matrix, result = run_kstream_master(connections, A, B), None
            else:
                final_matrix, result = run_baseline_master(connections, A, B, task), None
            # Keep it as a NumPy array (8 bytes a number, not ~32 as Python lists) until fetched
            final_matrix = np.asarray(final_matrix, dtype=np.float64)
            task["results"] = []  # baseline's row chunks are already assembled

        for worker in connections:
            send_msg(worker, {"algorithm": "done"})

        with state_lock:
            task.update(
                final_matrix=final_matrix,
                result=result,
                server_seconds=time.perf_counter() - started,
                workers=sorted(hosts),
                status="completed",
            )
        print(f"Dispatcher: request {job['request_id']} completed", flush=True)
    except Exception as error:  # Keep the dispatcher alive no matter what one request does.
        if job is not None:
            with state_lock:
                tasks_db[job["request_id"]].update(status="failed", error=str(error))
        print(f"Job failed: {error!r}", flush=True)
    finally:
        for worker in connections:
            worker.close()


def run_tcp_server():
    port = int(os.environ.get("MASTER_PORT", "6000"))
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(("0.0.0.0", port))
    server_socket.listen()
    print(f"Master TCP dispatcher listening on port {port}...", flush=True)

    # Requests run one at a time; other workers wait in the listen backlog.
    while True:
        print("Dispatcher: waiting for a worker to connect", flush=True)
        connection, address = server_socket.accept()
        run_job(server_socket, connection, address[0])


def run_master():
    threading.Thread(target=run_tcp_server, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("API_PORT", "5000")),
            threaded=True, use_reloader=False)


# WORKER
# A connection stays open for all jobs in the current multiplication.
def run_worker():
    host = os.environ.get("MASTER_HOST", "master")
    port = int(os.environ.get("MASTER_PORT", "6000"))
    listener = PeerListener(PEER_PORT)  # other workers connect here to forward broadcast data
    print(f"Worker: peer listener on port {listener.port}", flush=True)

    while True:
        try:
            print(f"Worker: connecting to {host}:{port}", flush=True)
            with socket.create_connection((host, port), SOCKET_TIMEOUT) as connection:
                connection.settimeout(None)  # Idle workers wait for a request.
                send_msg(connection, "READY")
                print("Worker: connected and sent READY", flush=True)
                while True:
                    job = recv_msg(connection)
                    connection.settimeout(SOCKET_TIMEOUT)
                    if job["algorithm"] == "done":
                        break
                    if job["algorithm"] == "summa":
                        run_summa_worker(connection, job)
                    elif job["algorithm"] == "kstream":
                        run_kstream_worker(connection, job)
                    elif job["algorithm"] == "summa2d":
                        run_summa2d_worker(connection, job, listener)
                    elif job["algorithm"] == "bcast":
                        run_bcast_worker(connection, job, listener)
                    elif job["algorithm"] == "pingpong":
                        run_pingpong_worker(connection, job, listener)
                    else:
                        result = np.asarray(job["A_chunk"]) @ np.asarray(job["B_matrix"])
                        send_msg(connection, {"start_row": job["start_row"], "C_chunk": result.tolist()})
                    connection.settimeout(None)  # Back to waiting for the next job.
        except (OSError, EOFError, ValueError, RuntimeError) as error:
            print(f"Worker: {error}", flush=True)
            time.sleep(0.5)  # Retry only after a connection or job error.


if __name__ == "__main__":
    role = os.environ.get("NODE_ROLE")
    if role == "master":
        run_master()
    elif role == "worker":
        run_worker()
    else:
        raise SystemExit("Set NODE_ROLE to master or worker.")
