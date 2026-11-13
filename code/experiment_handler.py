import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path

# Files live next to this script, so it works no matter where you run it from
BASE_DIR = Path(__file__).resolve().parent
TRACKING_FILE = BASE_DIR / "experiment_tracking.json"
RESULTS_FILE = BASE_DIR / "experiment_results.jsonl"
CLIENT_SCRIPT = BASE_DIR / "client.py"

# values = [100, 1000, 2500, 5000, 7500, 10000]
MATRIX_VALUES = [25, 50, 75, 100]
CONTAINER_COUNTS = [2, 4, 8, 16]
ALGORITHMS = ["baseline", "summa", "kstream"]
START_SEED = 42


# Generate keys for experiment_tracking.json
def make_key(algorithm, num_containers, x1, y1, x2, y2):
    return str((algorithm, num_containers, (x1, y1, x2, y2)))


def parse_key(key):
    """Return (algorithm, num_containers, x1, y1, x2, y2).

    Keys from before algorithms were tracked look like (n, (x1, y1, x2, y2));
    those runs were all baseline.
    """
    parsed = ast.literal_eval(key)
    if len(parsed) == 2:
        num_containers, (x1, y1, x2, y2) = parsed
        return "baseline", num_containers, x1, y1, x2, y2
    algorithm, num_containers, (x1, y1, x2, y2) = parsed
    return algorithm, num_containers, x1, y1, x2, y2


def build_tracking():
    """Every (algorithm, num_containers, (x1, y1, x2, y2)) combination where y1 == x2, with a run count of 0."""
    tracking = {}
    xy_tuples = [(x, y) for x in MATRIX_VALUES for y in MATRIX_VALUES]
    for algorithm in ALGORITHMS:
        for n in CONTAINER_COUNTS:
            for x1, y1 in xy_tuples:
                for x2, y2 in xy_tuples:
                    if y1 == x2:
                        tracking[make_key(algorithm, n, x1, y1, x2, y2)] = 0
    return tracking


def save_tracking(tracking):
    # Write to a temp file first so a crash mid-write can't corrupt the tracking file
    tmp = TRACKING_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(tracking, f, indent=2)
    tmp.replace(TRACKING_FILE)


def load_tracking():
    fresh = build_tracking()
    if TRACKING_FILE.exists():
        print("Config file exists, loading previous run counts")
        with open(TRACKING_FILE) as f:
            saved = json.load(f)
        # Keep saved counts (converting old-format keys); add any combinations new since the file was made
        for old_key, count in saved.items():
            key = make_key(*parse_key(old_key))
            if key in fresh:
                fresh[key] = max(fresh[key], count)
    else:
        print(f"Creating config file with {len(fresh)} combinations")
    save_tracking(fresh)
    return fresh


def next_experiment(tracking, algorithm=None):
    """The combination that has been run the fewest times (first one wins ties)."""
    candidates = [k for k in tracking if algorithm is None or parse_key(k)[0] == algorithm]
    key = min(candidates, key=tracking.get)
    return key, tracking[key]


def run_client(algorithm, num_containers, x1, y1, y2, seed):
    """Run client.py for an (x1 x y1) @ (y1 x y2) multiply. Returns its result dict, or None on failure."""
    cmd = [
        sys.executable, "-u", str(CLIENT_SCRIPT),  # -u so the client's output streams live and in order
        "--algorithm", algorithm,
        "--a-rows", str(x1),
        "--a-cols", str(y1),
        "--b-cols", str(y2),
        "--workers", str(num_containers),
        "--seed", str(seed),
    ]
    result = None
    # Stream client output live and pick out its final RESULT_JSON line
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1) as proc:
        for line in proc.stdout:
            if line.startswith("RESULT_JSON "):
                result = json.loads(line[len("RESULT_JSON "):])
            else:
                print("  " + line, end="")
    if proc.returncode != 0:
        return None
    return result


def main():
    parser = argparse.ArgumentParser(description="Run the next matrix multiplication experiments")
    parser.add_argument("--runs", type=int, default=1,
                        help="How many experiments to run back to back (default 1)")
    parser.add_argument("--algorithm", choices=ALGORITHMS,
                        help="Only run experiments for this algorithm (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show which experiment would run next")
    args = parser.parse_args()

    tracking = load_tracking()

    for i in range(args.runs):
        key, count = next_experiment(tracking, args.algorithm)
        algorithm, num_containers, x1, y1, x2, y2 = parse_key(key)
        print(f"\n[{i + 1}/{args.runs}] {algorithm}, {num_containers} containers: "
              f"({x1}x{y1}) @ ({x2}x{y2})  (run #{count + 1} of this combination)")

        if args.dry_run:
            return

        # Different seed for each repeat of the same combination, but reproducible
        seed = START_SEED + count
        result = run_client(algorithm, num_containers, x1, y1, y2, seed)
        if result is None:
            print("Client failed; run count not updated. Stopping.")
            sys.exit(1)

        tracking[key] += 1
        save_tracking(tracking)

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_containers": num_containers,
            "dims": [x1, y1, x2, y2],
            "seed": seed,
            **result,
            "algorithm": algorithm,
        }
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        server = result.get("server_seconds")
        server_text = f"{server:.4f}s server, " if server is not None else ""
        print(f"Recorded: {server_text}{result['compute_seconds']:.4f}s compute, "
              f"workers used: {len(result['workers_used'])}")


if __name__ == "__main__":
    main()