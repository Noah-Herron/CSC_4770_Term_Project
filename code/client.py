import numpy as np
import requests
import time
import sys

# Master node API endpoints
MULTIPLY_URL = "http://localhost:5000/multiply"
RESULT_URL = "http://localhost:5000/result/"

def generate_and_send(size):
    print(f"Generating two {size}x{size} matrices...")
    
    # Generate random integers between 1 and 10
    # matrix_A = np.random.randint(1, 10, size=(size, size))
    matrix_B = np.random.randint(1, 10, size=(size, size))

    # Generate two simple 3x3 matrices
    # matrix_A = np.array([
    #     [1, 2, 3],
    #     [4, 5, 6],
    #     [7, 8, 9]
    # ])
    
    # matrix_B = np.array([
    #     [9, 8, 7],
    #     [6, 5, 4],
    #     [3, 2, 1]
    # ])
    
    print("Converting to JSON payload and uploading to master node...")
    payload = {
        "matrix_A": matrix_A.tolist(),
        "matrix_B": matrix_B.tolist()
    }
    
    # Send the matrices to the master node
    try:
        response = requests.post(MULTIPLY_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        request_id = data['request_id']
        print(f"Upload successful! Request ID: {request_id}")
        print(f"Server message: {data['message']}")
        
        return request_id
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to the Master API: {e}")
        sys.exit(1)

def poll_for_result(request_id):
    print("\nPolling for results...")
    start_time = time.time()
    
    while True:
        response = requests.get(RESULT_URL + request_id)
        data = response.json()
        
        if data.get('status') == 'completed':
            elapsed = time.time() - start_time
            # Reconstruct the NumPy array from the result list to verify dimensions
            final_matrix = np.array(data['final_matrix'])
            print(f"\n--- SUCCESS ---")
            print(f"Time elapsed: {elapsed:.2f} seconds")
            print(f"Final Matrix Shape: {final_matrix.shape}")
            break
        else:
            print(f"Status: {data.get('progress', 'Processing...')}")
            time.sleep(2) # Wait 2 seconds before checking again

if __name__ == "__main__":
    # Starting with the 100x100
    matrix_size = 3 
    
    req_id = generate_and_send(matrix_size)
    if req_id:
        poll_for_result(req_id)