import os
import queue
import socket
import threading
import time
import uuid

import numpy as np
from flask import Flask, jsonify, request

from protocol import SOCKET_TIMEOUT, collect_workers, recv_msg, send_msg
from summa import get_grid_dims, run_summa_master, run_summa_worker

app = Flask(__name__)

# Shared state, touched by Flask threads and the dispatcher thread
state_lock = threading.Lock()
tasks_db = {}               # request_id -> tracking state
job_queue = queue.Queue()   # requests waiting for the dispatcher, oldest first

DEFAULT_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
BASELINE_CHUNK_ROWS = 2     # rows of A per baseline job
WAIT_TIMEOUT_SECONDS = 60   # cap for ?wait=1 so a stuck job can't hang the request forever


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
        if algorithm not in ("baseline", "summa"):
            raise ValueError("algorithm must be baseline or summa.")

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
        "num_workers": num_workers, "A": A, "B": B,
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
                return jsonify(
                    status="completed",
                    final_matrix=task["final_matrix"],
                    server_seconds=task["server_seconds"],
                    workers_used=task["workers"],
                ), 200
            if status == "failed":
                return jsonify(status="failed", error=task["error"]), 200

            progress = f"{len(task['results'])}/{task['total_jobs']} jobs completed."

        # Sleep outside the lock so the dispatcher can keep updating the task
        if not wait or time.monotonic() >= deadline:
            return jsonify(status=status, progress=progress), 202
        time.sleep(0.01)


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
        A, B = job["A"], job["B"]
        started = time.perf_counter()
        print(f"Dispatcher: starting {job['algorithm']} request {job['request_id']} "
              f"({A.shape[0]}x{A.shape[1]} @ {B.shape[0]}x{B.shape[1]}, {job['num_workers']} workers)", flush=True)

        grid_rows, grid_cols = get_grid_dims(job["num_workers"], A.shape[0], B.shape[1])
        print(f"Dispatcher: grid {grid_rows}x{grid_cols}, collecting {grid_rows * grid_cols} workers", flush=True)
        connections, hosts = collect_workers(
            server_socket, first_connection, first_host, grid_rows * grid_cols)
        print(f"Dispatcher: collected {len(connections)} workers; running", flush=True)

        if job["algorithm"] == "summa":
            final_matrix = run_summa_master(connections, A, B)
        else:
            final_matrix = run_baseline_master(connections, A, B, task)

        for worker in connections:
            send_msg(worker, {"algorithm": "done"})

        with state_lock:
            task.update(
                final_matrix=np.asarray(final_matrix).tolist(),
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
