import socket
import os
import time

def run_master():
    host = '0.0.0.0'
    port = 6000
    
    # Create a TCP/IP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the port
    server_socket.bind((host, port))
    
    # Listen for incoming connections (max 5 queued connections)
    server_socket.listen(5)
    print(f"Master node listening for TCP connections on port {port}...", flush=True)

    while True:
        # Wait for a connection
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with worker at {client_address}", flush=True)
        
        try:
            # Receive the simple text string (buffer size 1024 bytes)
            data = client_socket.recv(1024).decode('utf-8')
            if data:
                print(f"Master received: '{data}'", flush=True)
                
                # Send the acknowledgment back to the worker
                ack_message = "ACK: Master has received your message."
                client_socket.sendall(ack_message.encode('utf-8'))
        finally:
            # Clean up the connection
            client_socket.close()

def run_worker():
    # Brief pause to ensure the master node socket is fully bound before connecting
    time.sleep(3)
    
    master_host = os.environ.get('MASTER_HOST', 'master')
    master_port = int(os.environ.get('MASTER_PORT', 6000))
    worker_id = socket.gethostname()
    
    # Create a TCP/IP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Connect the socket to the master's port
        client_socket.connect((master_host, master_port))
        print(f"Worker {worker_id} connected to Master at {master_host}:{master_port}", flush=True)
        
        # Transmit a simple text string
        message = f"Hello from Worker {worker_id}!"
        client_socket.sendall(message.encode('utf-8'))
        print(f"Worker {worker_id} sent: '{message}'", flush=True)
        
        # Wait for the acknowledgment from the master
        ack_data = client_socket.recv(1024).decode('utf-8')
        print(f"Worker {worker_id} received: '{ack_data}'", flush=True)
        
    except ConnectionRefusedError:
        print(f"Worker {worker_id} failed to connect. Is the master node running?", flush=True)
    finally:
        client_socket.close()
        
    # Keep the worker container alive after transmission for logging observation
    while True:
        time.sleep(10)

if __name__ == "__main__":
    # Route the application logic based on the environment variable
    role = os.environ.get('NODE_ROLE', 'unassigned')
    
    if role == 'master':
        run_master()
    elif role == 'worker':
        run_worker()
    else:
        print(f"Unknown NODE_ROLE '{role}'. Exiting.", flush=True)