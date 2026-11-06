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
job_queue = queue.Queue()
tasks_db = {}


@app.post("/multiply")
def receive_matrices():
    data = request.get_json(silent=True)
    try:
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
    except (KeyError, TypeError, ValueError) as error:
        return jsonify(error=str(error)), 400

    request_id = str(uuid.uuid4())
    job_count = (A.shape[0] + 1) // 2 if algorithm == "baseline" else 1
    tasks_db[request_id] = {
        "total_jobs": job_count, "results": [],
        "status": "processing", "final_matrix": None,
    }
    job_queue.put({"request_id": request_id, "algorithm": algorithm, "A": A, "B": B})
    return jsonify(request_id=request_id, message=f"Created {job_count} {algorithm} job(s)."), 202


@app.get("/result/<request_id>")
def get_result(request_id):
    task = tasks_db.get(request_id)
    if task is None:
        return jsonify(error="Request ID not found."), 404
    # The client can wait for completion instead of measuring polling delays.
    if request.args.get("wait") == "1":
        while task["status"] == "processing":
            time.sleep(0.01)
    if task["status"] == "completed":
        return jsonify(status="completed", final_matrix=task["final_matrix"],
                       server_seconds=task["server_seconds"])
    if task["status"] == "failed":
        return jsonify(status="failed", error=task["error"])
    return jsonify(status="processing", progress=f"{len(task['results'])}/{task['total_jobs']} jobs completed."), 202


def run_baseline_master(connections, A, B, task):
    """Send a batch first, then collect it. Worker processes compute in parallel."""
    B_matrix = B.tolist()
    chunk_size = 2
    batch_size = chunk_size * len(connections)
    for batch_start in range(0, A.shape[0], batch_size):
        batch = list(zip(connections, range(batch_start, min(batch_start + batch_size, A.shape[0]), chunk_size)))
        for connection, start_row in batch:
            send_msg(connection, {
                "algorithm": "baseline", "start_row": start_row,
                "A_chunk": A[start_row:start_row + chunk_size].tolist(),
                "B_matrix": B_matrix,
            })
        for connection, _ in batch:
            task["results"].append(recv_msg(connection))
    chunks = sorted(task["results"], key=lambda chunk: chunk["start_row"])
    return [row for chunk in chunks for row in chunk["C_chunk"]]


# One dispatcher reserves a group of workers for each complete request.
def run_tcp_server():
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("0.0.0.0", int(os.environ.get("MASTER_PORT", "6000"))))
        server.listen()
        print("Master TCP dispatcher listening...", flush=True)
        while True:
            connection, address = server.accept()
            connections = [connection]
            job = None
            try:
                connection.settimeout(SOCKET_TIMEOUT)
                if recv_msg(connection) != "READY":
                    continue
                # An idle worker waits here; there is no successful-job sleep.
                job = job_queue.get()
                task = tasks_db[job["request_id"]]
                A, B = job["A"], job["B"]
                started = time.perf_counter()

                num_workers = int(os.environ.get("NUM_WORKERS", "4"))
                rows, cols = get_grid_dims(num_workers, A.shape[0], B.shape[1])
                count = rows * cols
                connections, _ = collect_workers(server, connection, address[0], count)
                if job["algorithm"] == "summa":
                    task["final_matrix"] = run_summa_master(connections, A, B)
                else:
                    task["final_matrix"] = run_baseline_master(connections, A, B, task)

                for worker in connections:
                    send_msg(worker, {"algorithm": "done"})
                task["server_seconds"] = time.perf_counter() - started
                task["status"] = "completed"
            except (OSError, EOFError, ValueError, RuntimeError) as error:
                if job is not None:
                    tasks_db[job["request_id"]].update(status="failed", error=str(error))
                print(f"Job failed: {error}", flush=True)
            finally:
                for worker in connections:
                    worker.close()


def run_master():
    # This is the same TCP-server thread the original Flask app used.
    threading.Thread(target=run_tcp_server, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("API_PORT", "5000")), use_reloader=False)


# A connection stays open for all jobs in the current multiplication.
def run_worker():
    host = os.environ.get("MASTER_HOST", "master")
    port = int(os.environ.get("MASTER_PORT", "6000"))
    while True:
        try:
            with socket.create_connection((host, port), SOCKET_TIMEOUT) as connection:
                connection.settimeout(None)  # Idle workers wait for a request.
                send_msg(connection, "READY")
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
