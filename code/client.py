import argparse
import json
import sys
import time

import numpy as np
import requests


def generate_and_send(base_url, a_rows, a_cols, b_cols, num_workers, seed):
    print(f"Generating A ({a_rows}x{a_cols}) and B ({a_cols}x{b_cols}) with seed {seed}...")
    rng = np.random.default_rng(seed)

    # Random integers between 1 and 10
    matrix_A = rng.integers(1, 10, size=(a_rows, a_cols))
    matrix_B = rng.integers(1, 10, size=(a_cols, b_cols))

    print(f"Uploading to master node (using {num_workers} workers)...")
    payload = {
        "matrix_A": matrix_A.tolist(),
        "matrix_B": matrix_B.tolist(),
        "num_workers": num_workers,
    }

    try:
        response = requests.post(f"{base_url}/multiply", json=payload)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to the Master API: {e}")
        sys.exit(1)

    print(f"Upload successful! Request ID: {data['request_id']}")
    print(f"Server message: {data['message']}")
    return data["request_id"]


def poll_for_result(base_url, request_id, poll_interval):
    print("Polling for results...")
    start_time = time.time()

    while True:
        response = requests.get(f"{base_url}/result/{request_id}")
        data = response.json()

        if data.get("status") == "completed":
            elapsed = time.time() - start_time
            final_matrix = np.array(data["final_matrix"])
            print("--- SUCCESS ---")
            print(f"Time elapsed: {elapsed:.2f} seconds")
            print(f"Final Matrix Shape: {final_matrix.shape}")
            return elapsed, list(final_matrix.shape), data.get("workers_used", [])

        print(f"Status: {data.get('progress', 'Processing...')}")
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Send A @ B to the master node and time it")
    parser.add_argument("--a-rows", type=int, default=3, help="Rows of A (x1)")
    parser.add_argument("--a-cols", type=int, default=3, help="Columns of A = rows of B (y1 = x2)")
    parser.add_argument("--b-cols", type=int, default=3, help="Columns of B (y2)")
    parser.add_argument("--workers", type=int, default=2, help="How many worker containers to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="http://localhost:5000")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    args = parser.parse_args()

    upload_start = time.time()
    req_id = generate_and_send(args.host, args.a_rows, args.a_cols, args.b_cols, args.workers, args.seed)
    upload_seconds = time.time() - upload_start

    compute_seconds, shape, workers_used = poll_for_result(args.host, req_id, args.poll_interval)

    # Machine-readable last line for experiment_handler.py
    print("RESULT_JSON " + json.dumps({
        "request_id": req_id,
        "upload_seconds": upload_seconds,
        "compute_seconds": compute_seconds,
        "result_shape": shape,
        "workers_used": workers_used,
    }))


if __name__ == "__main__":
    main()