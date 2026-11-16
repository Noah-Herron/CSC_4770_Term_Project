"""Tree-vs-chain broadcast experiments against the running master.

  python bcast_bench.py calibrate                    # measure alpha/beta (do this first)
  python bcast_bench.py choose --workers 8 --size 4M # what would the selector pick?
  python bcast_bench.py run --workers 8 --size 4M --topology chain
  python bcast_bench.py sweep                        # resumable grid of workers x sizes x topologies
  python bcast_bench.py analyze                      # winners, crossover per worker count, table for 'auto'

Results go to data/bcast_results.jsonl. `analyze` writes data/bcast_table.json, which
the master (via the ./data volume) uses for topology=auto.
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

from selector import load_json, model_choices

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_FILE = DATA_DIR / "bcast_results.jsonl"
TABLE_FILE = DATA_DIR / "bcast_table.json"
SUMMARY_FILE = DATA_DIR / "bcast_summary.csv"
CALIBRATION_FILE = DATA_DIR / "calibration.json"

DEFAULT_WORKERS = "2,4,8,16"
DEFAULT_SIZES = "1K,4K,16K,64K,256K,1M,4M,16M,64M"
DEFAULT_TOPOLOGIES = "binomial,knomial,chain"
DEFAULT_CHAIN_SEGMENTS = "auto,64K"
CLOSE_CALL = 0.05  # winners within 5% are flagged as too close to call
UNITS = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}


def parse_size(text):
    text = text.strip().upper().removesuffix("B")
    unit = text[-1] if text and text[-1] in UNITS else ""
    return int(float(text[:-1] if unit else text) * UNITS[unit])


def fmt_size(n):
    for unit in ("G", "M", "K"):
        if n >= UNITS[unit] and n % UNITS[unit] == 0:
            return f"{n // UNITS[unit]}{unit}"
    return str(n)


def fmt_time(seconds):
    return f"{seconds * 1e3:.3f} ms" if seconds < 1 else f"{seconds:.3f} s"


def read_json(response):
    if response.status_code >= 400:
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {message}")
    return response.json()


def submit_and_wait(host, path, payload):
    request_id = read_json(requests.post(f"{host}{path}", json=payload))["request_id"]
    while True:
        data = read_json(requests.get(f"{host}/result/{request_id}", params={"wait": "1"}))
        if data.get("status") == "failed":
            raise RuntimeError(data.get("error") or "Request failed.")
        if data.get("status") == "completed":
            if data.get("result") is None:
                raise RuntimeError("Master returned no result; rebuild the containers.")
            return data["result"]
        time.sleep(0.2)


def label_of(record):
    label = record["topology"]
    if record["topology"] == "knomial":
        label += f"(k={record['k']})"
    if record["segments"] > 1:
        label += f" seg={fmt_size(record['segment_bytes'])}"
    return label


def save_record(result, requested_segment):
    DATA_DIR.mkdir(exist_ok=True)
    record = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "requested_segment": requested_segment, **result}
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def load_records(path=RESULTS_FILE):
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def print_result(result):
    print(f"  {label_of(result):24s} workers={result['num_workers']:<3d} size={fmt_size(result['size_bytes']):>5s}  "
          f"median {fmt_time(result['median_seconds'])}  (min {fmt_time(result['min_seconds'])}, "
          f"model {fmt_time(result['model_seconds'])}, depth {result['depth']})")


# SUBCOMMANDS
def cmd_calibrate(args):
    payload = {"repeats": args.repeats}
    if args.sizes:
        payload["sizes"] = [parse_size(s) for s in args.sizes.split(",")]
    result = submit_and_wait(args.host, "/calibrate", payload)
    print(f"alpha (latency)  = {result['alpha'] * 1e6:.1f} us")
    print(f"beta             = {result['beta'] * 1e9:.3f} ns/byte  "
          f"(~{result['bandwidth_bytes_per_second'] / 1e6:.0f} MB/s)")
    for sample in result["samples"]:
        print(f"  {fmt_size(sample['size_bytes']):>5s}: one-way {fmt_time(sample['one_way_seconds'])}")
    print("Saved on the master as data/calibration.json")


def cmd_choose(args):
    data = read_json(requests.get(f"{args.host}/selector", params={
        "size_bytes": parse_size(args.size), "num_workers": args.workers, "k": args.k}))
    choice = data["choice"]
    print(f"Pick: {choice['topology']}  segment={fmt_size(choice['segment_bytes'])}  "
          f"({choice['segments']} segment(s))  source: {choice['source']}")
    for option in choice["model_options"]:
        print(f"  model {option['topology']:9s} {fmt_time(option['predicted_seconds'])}")


def broadcast_payload(args_or_cfg, workers, size, topology, segment):
    payload = {"num_workers": workers, "size_bytes": size, "topology": topology,
               "k": args_or_cfg.k, "trials": args_or_cfg.trials, "warmup": args_or_cfg.warmup}
    if segment is not None:
        payload["segment_bytes"] = segment
    return payload


def cmd_run(args):
    segment = parse_size(args.segment) if args.segment else None
    result = submit_and_wait(args.host, "/broadcast",
                             broadcast_payload(args, args.workers, parse_size(args.size), args.topology, segment))
    print_result(result)
    if result.get("selection"):
        print(f"  selected by: {result['selection']['source']}")
    if not args.no_save:
        save_record(result, args.segment or "default")


def sweep_configs(args):
    for workers in [int(w) for w in args.workers.split(",")]:
        for size in [parse_size(s) for s in args.sizes.split(",")]:
            for topology in args.topologies.split(","):
                if topology != "chain":
                    yield workers, size, topology, None, "default"
                    continue
                for seg_text in args.chain_segments.split(","):
                    if seg_text == "auto":
                        yield workers, size, topology, None, "auto"
                    elif parse_size(seg_text) < size:  # a segment >= size is just the unsegmented chain
                        yield workers, size, topology, parse_size(seg_text), fmt_size(parse_size(seg_text))


def cmd_sweep(args):
    done = defaultdict(int)
    for r in load_records():
        done[(r["num_workers"], r["size_bytes"], r["requested_topology"], r["requested_segment"],
              r["trials"], r["warmup"])] += 1
    configs = list(sweep_configs(args))
    print(f"{len(configs)} configurations x {args.repeat} repeat(s)")
    for index, (workers, size, topology, segment, seg_label) in enumerate(configs, 1):
        key = (workers, size, topology, seg_label, args.trials, args.warmup)
        while done[key] < args.repeat:
            print(f"[{index}/{len(configs)}] {topology} seg={seg_label} workers={workers} size={fmt_size(size)}")
            try:
                result = submit_and_wait(args.host, "/broadcast",
                                         broadcast_payload(args, workers, size, topology, segment))
            except (requests.RequestException, RuntimeError) as error:
                sys.exit(f"Failed: {error}. Re-run the same command to resume.")
            print_result(result)
            save_record(result, seg_label)
            done[key] += 1


def cmd_analyze(args):
    records = [r for r in load_records(args.results) if r.get("requested_topology") != "auto"]
    if not records:
        sys.exit(f"No fixed-topology results in {args.results}; run sweep first.")
    calibration = load_json(CALIBRATION_FILE)

    # (workers, size) -> label -> list of per-run medians, and one representative record per label
    times = defaultdict(lambda: defaultdict(list))
    example = {}
    for r in records:
        label = label_of(r)
        times[(r["num_workers"], r["size_bytes"])][label].append(r["median_seconds"])
        example[(r["num_workers"], r["size_bytes"], label)] = r

    rows, entries = [], []
    for (workers, size) in sorted(times):
        configs = {label: statistics.median(v) for label, v in times[(workers, size)].items()}
        trees = {l: t for l, t in configs.items() if not l.startswith("chain")}
        chains = {l: t for l, t in configs.items() if l.startswith("chain")}
        best_label = min(configs, key=configs.get)
        best = example[(workers, size, best_label)]
        entries.append({"num_workers": workers, "size_bytes": size, "topology": best["topology"],
                        "k": best["k"], "segments": best["segments"], "segment_bytes": best["segment_bytes"],
                        "label": best_label, "median_seconds": configs[best_label]})
        row = {"num_workers": workers, "size_bytes": size, "winner": best_label,
               "best_tree": None, "tree_seconds": None, "best_chain": None, "chain_seconds": None,
               "chain_speedup": None, "verdict": "", "model_pick": None}
        if trees:
            row["best_tree"] = min(trees, key=trees.get)
            row["tree_seconds"] = trees[row["best_tree"]]
        if chains:
            row["best_chain"] = min(chains, key=chains.get)
            row["chain_seconds"] = chains[row["best_chain"]]
        if trees and chains:
            row["chain_speedup"] = row["tree_seconds"] / row["chain_seconds"]
            if abs(row["chain_speedup"] - 1) <= CLOSE_CALL:
                row["verdict"] = "close"
            else:
                row["verdict"] = "chain" if row["chain_speedup"] > 1 else "tree"
        if calibration:
            row["model_pick"] = model_choices(size, workers, calibration)[0]["topology"]
        rows.append(row)

    # Print the comparison
    print(f"{'workers':>7} {'size':>6}  {'best tree':22s} {'time':>11}  {'best chain':22s} {'time':>11}  "
          f"{'tree/chain':>10}  verdict  model")
    for row in rows:
        speedup = f"{row['chain_speedup']:.2f}x" if row["chain_speedup"] else "-"
        print(f"{row['num_workers']:>7} {fmt_size(row['size_bytes']):>6}  {row['best_tree'] or '-':22s} "
              f"{fmt_time(row['tree_seconds']) if row['tree_seconds'] else '-':>11}  {row['best_chain'] or '-':22s} "
              f"{fmt_time(row['chain_seconds']) if row['chain_seconds'] else '-':>11}  {speedup:>10}  "
              f"{row['verdict']:7s}  {row['model_pick'] or '-'}")

    # Crossover per worker count
    print("\nCrossover (smallest size where chain beats the best tree by more than "
          f"{CLOSE_CALL:.0%}):")
    for workers in sorted({r["num_workers"] for r in rows}):
        mine = [r for r in rows if r["num_workers"] == workers and r["verdict"]]
        first = next((r["size_bytes"] for r in mine if r["verdict"] == "chain"), None)
        stable = None
        for r in reversed(mine):  # largest sizes first: how far down does chain keep winning?
            if r["verdict"] != "chain":
                break
            stable = r["size_bytes"]
        model_first = None
        if calibration:
            model_first = next((r["size_bytes"] for r in mine if r["model_pick"] == "chain"), None)
        print(f"  {workers:>3} workers: first chain win {fmt_size(first) if first else 'none'}, "
              f"chain wins from {fmt_size(stable) + ' up' if stable else 'never consistently'}"
              + (f"; model predicts chain from {fmt_size(model_first) if model_first else 'never'}"
                 if calibration else ""))
    if calibration:
        judged = [r for r in rows if r["verdict"] in ("chain", "tree")]
        agree = sum(1 for r in judged if (r["model_pick"] == "chain") == (r["verdict"] == "chain"))
        print(f"\nModel agrees with measurement on {agree}/{len(judged)} clear-cut cases.")
    else:
        print("\nNo data/calibration.json; run calibrate to compare with the model.")

    # Outputs
    TABLE_FILE.parent.mkdir(exist_ok=True)
    with open(TABLE_FILE, "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "calibration": calibration,
                   "entries": entries}, f, indent=2)
    with open(SUMMARY_FILE, "w") as f:
        columns = list(rows[0])
        f.write(",".join(columns) + "\n")
        for row in rows:
            f.write(",".join("" if row[c] is None else str(row[c]) for c in columns) + "\n")
    print(f"\nWrote {TABLE_FILE.relative_to(BASE_DIR)} (used by topology=auto) and "
          f"{SUMMARY_FILE.relative_to(BASE_DIR)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="http://localhost:5000")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="Ping-pong to measure alpha and beta")
    p.add_argument("--sizes", help="Comma list, e.g. 8,1K,64K,1M,16M")
    p.add_argument("--repeats", type=int, default=20)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("choose", help="Ask the selector without running")
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--size", required=True)
    p.add_argument("--k", type=int, default=4)
    p.set_defaults(func=cmd_choose)

    for name, func in (("run", cmd_run), ("sweep", cmd_sweep)):
        p = sub.add_parser(name)
        p.add_argument("--k", type=int, default=4, help="Fan-out for knomial")
        p.add_argument("--trials", type=int, default=10)
        p.add_argument("--warmup", type=int, default=2)
        p.set_defaults(func=func)
        if name == "run":
            p.add_argument("--workers", type=int, required=True)
            p.add_argument("--size", required=True, help="e.g. 64K, 4M")
            p.add_argument("--topology", default="auto", choices=("auto", "binomial", "knomial", "chain"))
            p.add_argument("--segment", help="Segment size, e.g. 64K (default: model's best for chain)")
            p.add_argument("--no-save", action="store_true")
        else:
            p.add_argument("--workers", default=DEFAULT_WORKERS)
            p.add_argument("--sizes", default=DEFAULT_SIZES)
            p.add_argument("--topologies", default=DEFAULT_TOPOLOGIES)
            p.add_argument("--chain-segments", default=DEFAULT_CHAIN_SEGMENTS,
                           help="'auto' = model's best segment size; others are fixed sizes")
            p.add_argument("--repeat", type=int, default=1, help="Runs per configuration")

    p = sub.add_parser("analyze", help="Winners, crossover, and the table for topology=auto")
    p.add_argument("--results", default=str(RESULTS_FILE))
    p.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.host = args.host.rstrip("/")
    try:
        args.func(args)
    except (requests.RequestException, RuntimeError) as error:
        sys.exit(f"Failed: {error}")


if __name__ == "__main__":
    main()