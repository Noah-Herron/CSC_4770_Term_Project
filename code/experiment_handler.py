"""Run the next experiments, tracking how many times each (algorithm, parameters) combination ran.

Every algorithm is registered in ALGORITHMS below. An algorithm says which parameter
combinations it has (its grid) and how to run one. To add a new one, write (or reuse) a
class with `combinations()`, `describe(params)` and `run(params, seed)`, then add an
instance to ALGORITHMS. Its combinations show up in experiment_tracking.json automatically.

  python experiment_handler.py --list                          # algorithms and progress
  python experiment_handler.py --runs 20                       # fewest-run combinations, any algorithm
  python experiment_handler.py --runs 10 --algorithm summa
  python experiment_handler.py --runs 30 --family broadcast    # all bcast_* algorithms
"""

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
HOST = "http://localhost:5000"

# values = [100, 1000, 2500, 5000, 7500, 10000]
MATRIX_VALUES = [1000, 5000, 10000]
CONTAINER_COUNTS = [2, 4, 8, 16]
BROADCAST_SIZES = ["1K", "16K", "256K", "4M", "16M"]
BROADCAST_TRIALS = 10
BROADCAST_WARMUP = 2
START_SEED = 42


# ALGORITHM TYPES
class MatrixAlgorithm:
    """A matrix multiplication algorithm the master runs through /multiply (via client.py)."""

    family = "matrix"

    def __init__(self, name, matrix_values=MATRIX_VALUES, container_counts=CONTAINER_COUNTS):
        self.name = name
        self.matrix_values = matrix_values
        self.container_counts = container_counts

    def combinations(self):
        """Every (x1 x y1) @ (x2 x y2) with y1 == x2, for each container count."""
        xy = [(x, y) for x in self.matrix_values for y in self.matrix_values]
        for n in self.container_counts:
            for x1, y1 in xy:
                for x2, y2 in xy:
                    if y1 == x2:
                        yield {"num_containers": n, "dims": [x1, y1, x2, y2]}

    def describe(self, params):
        x1, y1, x2, y2 = params["dims"]
        return f"{params['num_containers']} containers: ({x1}x{y1}) @ ({x2}x{y2})"

    def run(self, params, seed):
        x1, y1, _, y2 = params["dims"]
        cmd = [
            sys.executable, "-u", str(CLIENT_SCRIPT),  # -u so the client's output streams live and in order
            "--host", HOST,
            "--algorithm", self.name,
            "--a-rows", str(x1),
            "--a-cols", str(y1),
            "--b-cols", str(y2),
            "--workers", str(params["num_containers"]),
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

    def summary(self, result):
        details = result.get("details") or {}
        extra = ""
        if details.get("grid"):
            extra = (f", grid {details['grid'][0]}x{details['grid'][1]}, "
                     f"broadcasts {details['broadcasts_by_topology']}")
        server = result.get("server_seconds")
        server_text = f"{server:.4f}s server, " if server is not None else ""
        return (f"{server_text}{result['compute_seconds']:.4f}s compute, "
                f"workers used: {len(result['workers_used'])}{extra}")


class BroadcastAlgorithm:
    """A broadcast topology the master runs through /broadcast (see broadcast.py)."""

    family = "broadcast"

    def __init__(self, name, topology, sizes=BROADCAST_SIZES, container_counts=CONTAINER_COUNTS,
                 segment=None, k=4, trials=BROADCAST_TRIALS, warmup=BROADCAST_WARMUP):
        self.name = name
        self.topology = topology  # binomial | knomial | chain | auto
        self.sizes = sizes
        self.container_counts = container_counts
        self.segment = segment    # None = master's default (model's best segment size for chain)
        self.k = k
        self.trials = trials
        self.warmup = warmup

    def combinations(self):
        for n in self.container_counts:
            for size in self.sizes:
                yield {"num_containers": n, "size": size}

    def describe(self, params):
        """e.g. '8 containers: 256K auto (= 32768 numbers, one 512x64 SUMMA panel)'.

        No matrix is multiplied here: the master sends this many bytes of test data to every
        worker. The note translates the size into float64 values and the SUMMA panel
        (BLOCK_SIZE columns wide) that would carry the same amount of data.
        """
        import bcast_bench
        from summa import BLOCK_SIZE

        values = bcast_bench.parse_size(params["size"]) // 8  # float64 = 8 bytes
        rows = values // BLOCK_SIZE
        panel = (f"one {rows}x{BLOCK_SIZE} SUMMA panel" if rows >= 1
                 else f"{values} numbers, less than one {BLOCK_SIZE}-wide row")
        segment = f", segment {self.segment}" if self.segment else ""
        return (f"{params['num_containers']} containers: {params['size']} {self.topology}{segment} "
                f"(= {values} numbers, {panel})")

    def run(self, params, seed):
        import bcast_bench  # only needed for broadcast runs

        payload = {"num_workers": params["num_containers"],
                   "size_bytes": bcast_bench.parse_size(params["size"]),
                   "topology": self.topology, "k": self.k,
                   "trials": self.trials, "warmup": self.warmup}
        if self.segment:
            payload["segment_bytes"] = bcast_bench.parse_size(self.segment)
        try:
            result = bcast_bench.submit_and_wait(HOST, "/broadcast", payload)
        except Exception as error:  # requests or server errors
            print(f"  Failed: {error}")
            return None
        bcast_bench.print_result(result)
        # Also log it where `bcast_bench.py analyze` looks, so the crossover analysis sees it
        bcast_bench.save_record(result, self.segment or ("auto" if self.topology == "chain" else "default"))
        return result

    def summary(self, result):
        selected = f" (auto picked {result['topology']})" if self.topology == "auto" else ""
        return f"{result['median_seconds'] * 1e3:.3f} ms median over {result['trials']} trials{selected}"


# REGISTRY: add new algorithms here. The name is what gets tracked and recorded.
ALGORITHMS = {algorithm.name: algorithm for algorithm in [
    MatrixAlgorithm("baseline"),
    MatrixAlgorithm("summa"),
    # The paper's 2D SUMMA (block-cyclic, owner computes); only the broadcast shape differs
    MatrixAlgorithm("summa2d_flat"),       # point-to-point from the owner (the paper's baseline)
    MatrixAlgorithm("summa2d_binomial"),   # binomial-tree collectives (the paper's experiments)
    MatrixAlgorithm("summa2d_knomial"),
    MatrixAlgorithm("summa2d_chain"),      # pipelined SUMMA (the paper's future work)
    MatrixAlgorithm("summa2d_auto"),       # selector picks per broadcast
    BroadcastAlgorithm("bcast_binomial", "binomial"),
    BroadcastAlgorithm("bcast_knomial", "knomial"),
    BroadcastAlgorithm("bcast_chain", "chain"),
    BroadcastAlgorithm("bcast_auto", "auto"),
]}


# TRACKING
def make_key(algorithm, params):
    return json.dumps({"algorithm": algorithm, **params}, sort_keys=True)


def parse_key(key):
    """Return (algorithm, params). Also reads the older tuple-style keys:
    (n, (x1, y1, x2, y2)) from before algorithms were tracked (all baseline), and
    (algorithm, n, (x1, y1, x2, y2)).
    """
    if key.startswith("{"):
        params = json.loads(key)
        return params.pop("algorithm"), params
    parsed = ast.literal_eval(key)
    if len(parsed) == 2:
        algorithm, (n, dims) = "baseline", parsed
    else:
        algorithm, n, dims = parsed
    return algorithm, {"num_containers": n, "dims": list(dims)}


def build_tracking():
    """Every combination of every registered algorithm, with a run count of 0."""
    return {make_key(name, params): 0
            for name, algorithm in ALGORITHMS.items()
            for params in algorithm.combinations()}


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
        # Keep saved counts (converting old-format keys). Combinations that are no longer in a
        # grid are kept too, so changing a grid never loses history.
        for old_key, count in saved.items():
            key = make_key(*parse_key(old_key))
            fresh[key] = max(fresh.get(key, 0), count)
    else:
        print(f"Creating config file with {len(fresh)} combinations")
    save_tracking(fresh)
    return fresh


def active_keys(tracking, names):
    """Keys that belong to a selected algorithm and are still in its current grid."""
    current = {make_key(name, params) for name in names for params in ALGORITHMS[name].combinations()}
    return [key for key in tracking if key in current]


def next_experiment(tracking, names, session_runs):
    """Strict rotation between the selected algorithms, then that algorithm's least-run combination.

    `--algorithm baseline --algorithm summa --algorithm bcast_auto` runs baseline, summa,
    bcast_auto, baseline, ... however different their grid sizes or past run counts are.
    Within one algorithm, the combination run the fewest times goes next (first one wins ties).
    """
    order = {name: index for index, name in enumerate(names)}
    name = min(names, key=lambda n: (session_runs[n], order[n]))
    key = min(active_keys(tracking, [name]), key=tracking.get)
    return key, tracking[key]


def selected_algorithms(args):
    names = list(ALGORITHMS)
    if args.family:
        names = [n for n in names if ALGORITHMS[n].family in args.family]
    if args.algorithm:
        names = [n for n in names if n in args.algorithm]
    if not names:
        sys.exit("No registered algorithm matches --algorithm/--family.")
    return names


def print_progress(tracking, names):
    print(f"{'algorithm':16s} {'family':10s} {'combos':>6s} {'run':>5s} {'min runs':>8s} {'total runs':>10s}")
    for name in names:
        counts = [tracking[k] for k in active_keys(tracking, [name])]
        print(f"{name:16s} {ALGORITHMS[name].family:10s} {len(counts):>6d} "
              f"{sum(1 for c in counts if c):>5d} {min(counts):>8d} {sum(counts):>10d}")


def backfill_results():
    """Label old result lines that have no algorithm (they were all baseline runs)."""
    if not RESULTS_FILE.exists():
        return 0
    lines = RESULTS_FILE.read_text().splitlines()
    fixed = 0
    out = []
    for line in lines:
        if line.strip():
            record = json.loads(line)
            if "algorithm" not in record:
                record["algorithm"] = "baseline"
                fixed += 1
            known = ALGORITHMS.get(record["algorithm"])
            record.setdefault("family", known.family if known else "matrix")
            line = json.dumps(record)
        out.append(line)
    RESULTS_FILE.with_suffix(".jsonl.bak").write_text("\n".join(lines) + "\n")
    RESULTS_FILE.write_text("\n".join(out) + "\n")
    return fixed


def main():
    global HOST
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=1,
                        help="How many experiments to run back to back (default 1)")
    parser.add_argument("--algorithm", action="append", choices=list(ALGORITHMS),
                        help="Only run this algorithm; repeat to pick several (default: all)")
    parser.add_argument("--family", action="append",
                        choices=sorted({a.family for a in ALGORITHMS.values()}),
                        help="Only run algorithms of this family (matrix, broadcast)")
    parser.add_argument("--list", action="store_true", help="Show algorithms and progress, then exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only show which experiment would run next")
    parser.add_argument("--backfill-results", action="store_true",
                        help="Add algorithm/family to old result lines (keeps a .bak copy), then exit")
    parser.add_argument("--host", default=HOST)
    args = parser.parse_args()
    HOST = args.host.rstrip("/")

    if args.backfill_results:
        print(f"Labelled {backfill_results()} old result line(s) as baseline; backup in "
              f"{RESULTS_FILE.with_suffix('.jsonl.bak').name}")
        return

    names = selected_algorithms(args)
    tracking = load_tracking()
    if args.list:
        print_progress(tracking, names)
        return

    session_runs = {name: 0 for name in names}
    for i in range(args.runs):
        key, count = next_experiment(tracking, names, session_runs)
        name, params = parse_key(key)
        algorithm = ALGORITHMS[name]
        print(f"\n[{i + 1}/{args.runs}] {name}, {algorithm.describe(params)}  "
              f"(run #{count + 1} of this combination)")

        if args.dry_run:
            return

        # Different seed for each repeat of the same combination, but reproducible
        seed = START_SEED + count
        result = algorithm.run(params, seed)
        if result is None:
            print("Run failed; run count not updated. Stopping.")
            sys.exit(1)

        tracking[key] += 1
        session_runs[name] += 1
        save_tracking(tracking)

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **params,
            "seed": seed,
            **result,
            "algorithm": name,       # last, so a result field can never overwrite these
            "family": algorithm.family,
        }
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"Recorded: {algorithm.summary(result)}")


if __name__ == "__main__":
    main()