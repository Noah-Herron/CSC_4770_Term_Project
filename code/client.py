import argparse
import time

import numpy as np
import requests

MULTIPLY_URL = "http://localhost:5000/multiply"
RESULT_URL = "http://localhost:5000/result/"


def generate_and_send(size, algorithm):
    print(f"Generating two {size}x{size} matrices...")
    rng = np.random.default_rng(0)  # Same input for both algorithm commands.
    A = rng.integers(1, 10, size=(size, size))
    B = rng.integers(1, 10, size=(size, size))
    response = requests.post(MULTIPLY_URL, json={
        "matrix_A": A.tolist(), "matrix_B": B.tolist(), "algorithm": algorithm,
    })
    response.raise_for_status()
    data = response.json()
    print(data["message"])
    return data["request_id"], A, B


def wait_for_result(request_id):
    print("Waiting for results...")
    started = time.perf_counter()
    while True:
        response = requests.get(RESULT_URL + request_id, params={"wait": "1"})
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "failed":
            raise RuntimeError(data.get("error", "Multiplication failed."))
        if data.get("status") == "completed":
            if data.get("final_matrix") is None:
                raise RuntimeError("Master reported completion without a matrix. Rebuild the containers.")
            break
        if data.get("status") not in ("queued", "processing"):
            raise RuntimeError("Unexpected result response from master.")
        time.sleep(0.05)
    result = np.asarray(data["final_matrix"])
    if "server_seconds" in data:
        print(f"Server time: {data['server_seconds']:.4f} seconds")
    else:
        print("Server time unavailable; rebuild the containers to enable timing.")
    print(f"Wait and download time: {time.perf_counter() - started:.4f} seconds")
    print(f"Final matrix shape: {result.shape}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Distributed matrix multiplication")
    parser.add_argument("--algorithm", required=True, choices=("baseline", "summa"))
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    try:
        request_id, A, B = generate_and_send(args.size, args.algorithm)
        result = wait_for_result(request_id)
        if args.verify:
            expected = A.astype(np.float64) @ B
            if result.shape != expected.shape or not np.allclose(result, expected, rtol=1e-10, atol=1e-10):
                raise RuntimeError("NumPy verification failed.")
            print("NumPy verification passed.")
    except (requests.RequestException, RuntimeError) as error:
        parser.exit(1, f"Failed: {error}\n")


if __name__ == "__main__":
    main()
