"""Send A @ B to the master node, wait for the result and time it."""

import argparse
import json
import time

import numpy as np
import requests

MATRIX_ALGORITHMS = ("baseline", "summa", "summa2d_flat", "summa2d_binomial",
                     "summa2d_knomial", "summa2d_chain", "summa2d_auto")


def read_json(response):
    """Return the response body, or raise with the server's error message."""
    if response.status_code >= 400:
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {message}")
    return response.json()


def generate_and_send(base_url, a_rows, a_cols, b_cols, algorithm, num_workers, seed, block_size=None):
    print(f"Generating A ({a_rows}x{a_cols}) and B ({a_cols}x{b_cols}) with seed {seed}...")
    rng = np.random.default_rng(seed)

    # Random integers between 1 and 9
    A = rng.integers(1, 10, size=(a_rows, a_cols))
    B = rng.integers(1, 10, size=(a_cols, b_cols))

    payload = {"matrix_A": A.tolist(), "matrix_B": B.tolist(), "algorithm": algorithm}
    if block_size is not None:
        payload["block_size"] = block_size
    if num_workers is not None:
        payload["num_workers"] = num_workers
        print(f"Uploading to master node ({algorithm}, {num_workers} workers)...")
    else:
        print(f"Uploading to master node ({algorithm}, master's default worker count)...")

    data = read_json(requests.post(f"{base_url}/multiply", json=payload))
    print(f"Upload successful! Request ID: {data['request_id']}")
    print(f"Server message: {data['message']}")
    return data["request_id"], A, B


def wait_for_result(base_url, request_id, poll_interval):
    print("Waiting for results...")
    started = time.perf_counter()

    while True:
        # wait=1 makes the master hold the request until the job finishes (or its wait cap)
        data = read_json(requests.get(f"{base_url}/result/{request_id}", params={"wait": "1"}))
        status = data.get("status")
        if status == "failed":
            raise RuntimeError(data.get("error") or "Multiplication failed.")
        if status == "completed":
            break
        if status not in ("queued", "processing"):
            raise RuntimeError(f"Unexpected result response from master: {data}")
        print(f"Status: {data.get('progress', status)}")
        time.sleep(poll_interval)

    elapsed = time.perf_counter() - started
    if data.get("final_matrix") is None:
        raise RuntimeError("Master reported completion without a matrix. Rebuild the containers.")

    result = np.asarray(data["final_matrix"])
    server_seconds = data.get("server_seconds")
    print("--- SUCCESS ---")
    if server_seconds is not None:
        print(f"Server time: {server_seconds:.4f} seconds")
    else:
        print("Server time unavailable; rebuild the containers to enable timing.")
    print(f"Wait and download time: {elapsed:.4f} seconds")
    print(f"Final matrix shape: {result.shape}")
    if data.get("result"):
        details = data["result"]
        print(f"Grid {details['grid'][0]}x{details['grid'][1]}, tiles {details['block_size']}, "
              f"{details['steps']} steps, broadcasts: {details['broadcasts_by_topology']}")
    return result, elapsed, server_seconds, data.get("workers_used", []), data.get("result")


def main():
    parser = argparse.ArgumentParser(description="Send A @ B to the master node and time it")
    parser.add_argument("--algorithm", default="baseline", choices=MATRIX_ALGORITHMS)
    parser.add_argument("--block-size", type=int, help="Tile size for summa2d_* (default: 256, smaller for small matrices)")
    parser.add_argument("--size", type=int, default=100,
                        help="Square size; fills in any of --a-rows/--a-cols/--b-cols not given")
    parser.add_argument("--a-rows", type=int, help="Rows of A (x1)")
    parser.add_argument("--a-cols", type=int, help="Columns of A = rows of B (y1 = x2)")
    parser.add_argument("--b-cols", type=int, help="Columns of B (y2)")
    parser.add_argument("--workers", type=int,
                        help="How many worker containers to use (default: master's NUM_WORKERS)")
    parser.add_argument("--seed", type=int, default=0, help="Same seed gives the same input for both algorithms")
    parser.add_argument("--host", default="http://localhost:5000")
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--verify", action="store_true", help="Check the result against NumPy")
    args = parser.parse_args()

    a_rows = args.size if args.a_rows is None else args.a_rows
    a_cols = args.size if args.a_cols is None else args.a_cols
    b_cols = args.size if args.b_cols is None else args.b_cols
    if min(a_rows, a_cols, b_cols) <= 0:
        parser.error("matrix dimensions must be positive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    host = args.host.rstrip("/")

    try:
        upload_start = time.perf_counter()
        request_id, A, B = generate_and_send(host, a_rows, a_cols, b_cols,
                                             args.algorithm, args.workers, args.seed, args.block_size)
        upload_seconds = time.perf_counter() - upload_start

        result, compute_seconds, server_seconds, workers_used, details = wait_for_result(
            host, request_id, args.poll_interval)

        verified = None
        if args.verify:
            expected = A.astype(np.float64) @ B
            if result.shape != expected.shape or not np.allclose(result, expected, rtol=1e-10, atol=1e-10):
                raise RuntimeError("NumPy verification failed.")
            print("NumPy verification passed.")
            verified = True
    except (requests.RequestException, RuntimeError) as error:
        parser.exit(1, f"Failed: {error}\n")

    # Machine-readable last line for experiment_handler.py
    print("RESULT_JSON " + json.dumps({
        "request_id": request_id,
        "algorithm": args.algorithm,
        "upload_seconds": upload_seconds,
        "compute_seconds": compute_seconds,
        "server_seconds": server_seconds,
        "result_shape": list(result.shape),
        "workers_used": workers_used,
        "verified": verified,
        "details": details,
    }))


if __name__ == "__main__":
    main()