import socket
import os
import time
import threading
import json
import math
import struct
import uuid
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Shared state, touched by Flask threads and one thread per worker connection
state_lock = threading.Lock()
pending_jobs = []   # jobs waiting for a worker, oldest first
tasks_db = {}       # request_id -> tracking / aggregation state

# SOCKET BUFFER LOOP UTILITIES
def send_msg(sock, msg):
    msg_bytes = msg.encode('utf-8')
    sock.sendall(struct.pack('>I', len(msg_bytes)) + msg_bytes)

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def recv_msg(sock):
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    msg_bytes = recvall(sock, msglen)
    if msg_bytes is None:
        return None
    return msg_bytes.decode('utf-8')

# FLASK API (INPUT & AGGREGATION MICROSERVICE)
@app.route('/multiply', methods=['POST'])
def receive_matrices():
    data = request.get_json()
    if not data or 'matrix_A' not in data or 'matrix_B' not in data:
        return jsonify({"error": "Please provide matrix_A and matrix_B."}), 400

    A = np.array(data['matrix_A'])
    B = np.array(data['matrix_B'])

    if A.shape[1] != B.shape[0]:
        return jsonify({"error": "Incompatible dimensions."}), 400

    num_workers = int(data.get('num_workers', 1))
    if num_workers < 1:
        return jsonify({"error": "num_workers must be at least 1."}), 400

    request_id = str(uuid.uuid4())
    total_rows = A.shape[0]

    # One chunk of rows per worker
    chunk_size = math.ceil(total_rows / num_workers)
    starts = list(range(0, total_rows, chunk_size))
    B_list = B.tolist()

    with state_lock:
        tasks_db[request_id] = {
            "total_jobs": len(starts),
            "max_workers": num_workers,
            "workers": set(),       # distinct workers allowed to take this request's jobs
            "results": [],
            "status": "processing",
            "final_matrix": None,
        }
        for job_id, start_row in enumerate(starts):
            end_row = min(start_row + chunk_size, total_rows)
            pending_jobs.append({
                "request_id": request_id,
                "job_id": job_id,
                "start_row": start_row,
                "A_chunk": A[start_row:end_row].tolist(),
                "B_matrix": B_list,
            })

    return jsonify({
        "message": f"Matrices divided into {len(starts)} jobs across up to {num_workers} workers.",
        "request_id": request_id
    }), 202

@app.route('/result/<request_id>', methods=['GET'])
def get_result(request_id):
    with state_lock:
        task = tasks_db.get(request_id)
        if not task:
            return jsonify({"error": "Request ID not found."}), 404

        if task["status"] == "completed":
            return jsonify({
                "status": "completed",
                "final_matrix": task["final_matrix"],
                "workers_used": sorted(task["workers"]),
            }), 200
        return jsonify({
            "status": "processing",
            "progress": f"{len(task['results'])}/{task['total_jobs']} jobs completed."
        }), 202

# TCP SERVER (MASTER DISPATCHER & AGGREGATOR)
def take_job(worker_id):
    """Give this worker the oldest job it's allowed to run, or None.

    Until a request has num_workers distinct workers, its jobs only go to new workers,
    so the work is spread across exactly num_workers containers. Once that set is full,
    only workers in the set can take its remaining jobs (e.g. a requeued one).
    """
    with state_lock:
        for i, job in enumerate(pending_jobs):
            task = tasks_db[job["request_id"]]
            is_member = worker_id in task["workers"]
            is_full = len(task["workers"]) >= task["max_workers"]
            if is_member == is_full:
                task["workers"].add(worker_id)
                return pending_jobs.pop(i)
    return None

def record_result(result):
    req_id = result['request_id']
    with state_lock:
        task = tasks_db[req_id]
        task["results"].append(result)
        print(f"Master received Job {result['job_id']} for Request {req_id}", flush=True)

        # AGGREGATION LOGIC
        if len(task["results"]) == task["total_jobs"]:
            print(f"All jobs finished for {req_id}. Aggregating...", flush=True)
            # Sort the sub-matrices by their starting row to reconstruct correctly
            sorted_chunks = sorted(task["results"], key=lambda x: x['start_row'])
            final_matrix = []
            for chunk in sorted_chunks:
                final_matrix.extend(chunk['C_chunk'])
            task["final_matrix"] = final_matrix
            task["status"] = "completed"
            print(f"Aggregation complete for Request {req_id} "
                  f"using {len(task['workers'])} workers.", flush=True)

def handle_worker(client_socket, client_address):
    try:
        worker_signal = recv_msg(client_socket)
        if not worker_signal or not worker_signal.startswith("READY"):
            return
        parts = worker_signal.split(maxsplit=1)
        worker_id = parts[1] if len(parts) > 1 else client_address[0]

        job = take_job(worker_id)
        if job is None:
            send_msg(client_socket, "EMPTY")
            return

        try:
            send_msg(client_socket, json.dumps(job))
            result_data = recv_msg(client_socket)
        except OSError:
            result_data = None

        if not result_data:
            # Worker dropped mid-job: put the job back at the front of the queue
            with state_lock:
                pending_jobs.insert(0, job)
            return

        record_result(json.loads(result_data))
    finally:
        client_socket.close()

def run_tcp_server():
    host = '0.0.0.0'
    port = 6000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(32)
    print(f"Master TCP dispatcher listening on port {port}...", flush=True)

    while True:
        client_socket, client_address = server_socket.accept()
        # Each worker gets its own thread so workers compute in parallel
        threading.Thread(target=handle_worker, args=(client_socket, client_address),
                         daemon=True).start()

def run_master():
    tcp_thread = threading.Thread(target=run_tcp_server, daemon=True)
    tcp_thread.start()
    app.run(host='0.0.0.0', port=5000, use_reloader=False, threaded=True)

# WORKER COMPUTATION
def run_worker():
    time.sleep(3)
    master_host = os.environ.get('MASTER_HOST', 'master')
    master_port = int(os.environ.get('MASTER_PORT', 6000))
    worker_id = socket.gethostname()

    while True:
        got_job = False
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((master_host, master_port))
            send_msg(client_socket, f"READY {worker_id}")

            response = recv_msg(client_socket)
            if response and response != "EMPTY":
                got_job = True
                job = json.loads(response)

                # Computation Phase
                A_chunk = np.array(job['A_chunk'])
                B_matrix = np.array(job['B_matrix'])
                C_chunk = np.dot(A_chunk, B_matrix)

                # Return Phase
                result_payload = {
                    "request_id": job['request_id'],
                    "job_id": job['job_id'],
                    "start_row": job['start_row'],
                    "C_chunk": C_chunk.tolist()
                }
                send_msg(client_socket, json.dumps(result_payload))

        except (ConnectionRefusedError, ConnectionResetError, OSError):
            pass
        finally:
            client_socket.close()

        # Ask again right away after finishing a job; back off a little when idle
        if not got_job:
            time.sleep(0.5)

if __name__ == "__main__":
    role = os.environ.get('NODE_ROLE', 'unassigned')
    if role == 'master': run_master()
    elif role == 'worker': run_worker()