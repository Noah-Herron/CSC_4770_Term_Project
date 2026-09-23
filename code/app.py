import socket
import os
import time
import threading
import queue
import json
import struct
import uuid
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)
job_queue = queue.Queue()

# In-memory database to track multiplication requests and their aggregated results
tasks_db = {} 

# --- SOCKET BUFFER LOOP UTILITIES ---
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

# --- FLASK API (INPUT & AGGREGATION MICROSERVICE) ---
@app.route('/multiply', methods=['POST'])
def receive_matrices():
    data = request.get_json()
    if not data or 'matrix_A' not in data or 'matrix_B' not in data:
        return jsonify({"error": "Please provide matrix_A and matrix_B."}), 400
    
    A = np.array(data['matrix_A'])
    B = np.array(data['matrix_B'])
    
    if A.shape[1] != B.shape[0]:
        return jsonify({"error": f"Incompatible dimensions."}), 400
    
    request_id = str(uuid.uuid4())
    chunk_size = 2 # Lowered to 2 so a 3x3 matrix forces multiple jobs for testing
    total_rows = A.shape[0]
    
    # Calculate how many jobs this matrix will generate
    job_count = len(range(0, total_rows, chunk_size))
    
    # Initialize the aggregation tracking state
    tasks_db[request_id] = {
        "total_jobs": job_count,
        "results": [],
        "status": "processing",
        "final_matrix": None
    }
    
    job_id = 0
    for start_row in range(0, total_rows, chunk_size):
        end_row = min(start_row + chunk_size, total_rows)
        A_chunk = A[start_row:end_row].tolist()
        
        job = {
            "request_id": request_id,
            "job_id": job_id,
            "start_row": start_row,
            "A_chunk": A_chunk,
            "B_matrix": B.tolist() 
        }
        job_queue.put(job)
        job_id += 1
        
    return jsonify({
        "message": f"Matrices divided into {job_count} jobs.",
        "request_id": request_id
    }), 202

@app.route('/result/<request_id>', methods=['GET'])
def get_result(request_id):
    task = tasks_db.get(request_id)
    if not task:
        return jsonify({"error": "Request ID not found."}), 404
        
    if task["status"] == "completed":
        return jsonify({
            "status": "completed",
            "final_matrix": task["final_matrix"]
        }), 200
    else:
        return jsonify({
            "status": "processing",
            "progress": f"{len(task['results'])}/{task['total_jobs']} jobs completed."
        }), 202

# --- TCP SERVER (MASTER DISPATCHER & AGGREGATOR) ---
def run_tcp_server():
    host = '0.0.0.0'
    port = 6000
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Master TCP dispatcher listening on port {port}...", flush=True)

    while True:
        client_socket, client_address = server_socket.accept()
        try:
            worker_signal = recv_msg(client_socket)
            
            if worker_signal == "READY":
                if not job_queue.empty():
                    job = job_queue.get()
                    send_msg(client_socket, json.dumps(job))
                    
                    # Wait for worker computation
                    result_data = recv_msg(client_socket)
                    if result_data:
                        result = json.loads(result_data)
                        req_id = result['request_id']
                        
                        print(f"Master received Job {result['job_id']} for Request {req_id}", flush=True)
                        tasks_db[req_id]["results"].append(result)
                        
                        # --- AGGREGATION LOGIC ---
                        if len(tasks_db[req_id]["results"]) == tasks_db[req_id]["total_jobs"]:
                            print(f"All jobs finished for {req_id}. Aggregating...", flush=True)
                            
                            # Sort the sub-matrices by their starting row to reconstruct correctly
                            sorted_chunks = sorted(tasks_db[req_id]["results"], key=lambda x: x['start_row'])
                            
                            # Vertically stack the chunks to form the final matrix
                            final_matrix = []
                            for chunk in sorted_chunks:
                                final_matrix.extend(chunk['C_chunk'])
                                
                            tasks_db[req_id]["final_matrix"] = final_matrix
                            tasks_db[req_id]["status"] = "completed"
                            print(f"Aggregation complete for Request {req_id}.", flush=True)
                else:
                    send_msg(client_socket, "EMPTY")
        finally:
            client_socket.close()

def run_master():
    tcp_thread = threading.Thread(target=run_tcp_server, daemon=True)
    tcp_thread.start()
    app.run(host='0.0.0.0', port=5000, use_reloader=False) 

# --- WORKER COMPUTATION ---
def run_worker():
    time.sleep(3)
    master_host = os.environ.get('MASTER_HOST', 'master')
    master_port = int(os.environ.get('MASTER_PORT', 6000))
    worker_id = socket.gethostname()
    
    while True:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((master_host, master_port))
            send_msg(client_socket, "READY")
            
            response = recv_msg(client_socket)
            if response and response != "EMPTY":
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
                
        except ConnectionRefusedError:
            pass 
        finally:
            client_socket.close()
            
        time.sleep(2) 

if __name__ == "__main__":
    role = os.environ.get('NODE_ROLE', 'unassigned')
    if role == 'master': run_master()
    elif role == 'worker': run_worker()