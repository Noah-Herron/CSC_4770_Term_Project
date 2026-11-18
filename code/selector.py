"""Choose a broadcast topology (tree vs chain) for a message size and node count.

Two layers:
1. A cost model in the latency/bandwidth form, using alpha (seconds per message)
   and beta (seconds per byte) measured by ping-pong between containers.
2. A lookup table built from measured sweeps (bcast_bench.py analyze). When the
   table has a measurement close to the request it wins over the model.

Cost model, with P = nodes including the root and n = message bytes:
    flat              alpha + (P-1)*n*beta  (the root sends every copy itself)
    binomial          ceil(log2 P) * (alpha + n*beta)
    k-nomial          ceil(log_k P) * (alpha + (k-1)*n*beta)
                      (a node's sends to its k-1 children in one round overlap their
                      latency but share the sender's bandwidth)
    chain, s pieces   (P - 2 + s) * (alpha + n*beta/s)
                      best s* = sqrt((P-2) * n*beta / alpha)

Trees are modeled unsegmented. The engine can segment them too, but in this model
segmenting a tree only adds per-message latency at the root.
"""

import json
import math
from pathlib import Path

DEFAULT_CALIBRATION = {"alpha": 1e-4, "beta": 1e-9, "source": "built-in default (run calibrate)"}
MIN_SEGMENT_BYTES = 4096
DEFAULT_K = 4
CANDIDATES = ("binomial", "knomial", "chain")
TABLE_MATCH_FACTOR = 2.0  # use a table entry only if its size is within 2x of the request


def tree_rounds(num_nodes, k):
    rounds, reach = 0, 1
    while reach < num_nodes:
        reach *= k
        rounds += 1
    return rounds


def best_chain_segments(size_bytes, num_nodes, alpha, beta):
    if num_nodes <= 2 or size_bytes <= MIN_SEGMENT_BYTES or alpha <= 0:
        return 1
    ideal = math.sqrt((num_nodes - 2) * size_bytes * beta / alpha)
    most = max(1, size_bytes // MIN_SEGMENT_BYTES)
    return int(min(max(1, round(ideal)), most))


def segment_bytes_for(size_bytes, segments):
    return max(1, -(-size_bytes // max(1, segments)))  # ceiling division


def predict(topology, size_bytes, num_nodes, alpha, beta, segments=1, k=DEFAULT_K):
    """Model time (seconds) until the last node has the whole message."""
    if num_nodes <= 1:
        return 0.0
    if topology == "chain":
        s = max(1, segments)
        return (num_nodes - 2 + s) * (alpha + size_bytes * beta / s)
    if topology == "flat":
        return alpha + (num_nodes - 1) * size_bytes * beta  # root's link carries every copy
    if topology == "binomial":
        k = 2
    elif topology != "knomial":
        raise ValueError(f"Unknown topology {topology!r}.")
    return tree_rounds(num_nodes, k) * (alpha + (k - 1) * size_bytes * beta)


def calibration_params(calibration):
    calibration = calibration or DEFAULT_CALIBRATION
    return float(calibration["alpha"]), float(calibration["beta"])


def default_segment_bytes(topology, size_bytes, num_workers, calibration=None):
    """Chains get the model's best segment size; trees send the whole message at once."""
    if topology != "chain":
        return size_bytes
    alpha, beta = calibration_params(calibration)
    return segment_bytes_for(size_bytes, best_chain_segments(size_bytes, num_workers + 1, alpha, beta))


def model_choices(size_bytes, num_workers, calibration=None, k=DEFAULT_K, candidates=CANDIDATES):
    """Predicted time for each candidate, cheapest first (ties keep candidate order)."""
    alpha, beta = calibration_params(calibration)
    num_nodes = num_workers + 1
    options = []
    for topology in candidates:
        segments = best_chain_segments(size_bytes, num_nodes, alpha, beta) if topology == "chain" else 1
        options.append({
            "topology": topology,
            "k": k if topology == "knomial" else None,
            "segments": segments,
            "segment_bytes": segment_bytes_for(size_bytes, segments),
            "predicted_seconds": predict(topology, size_bytes, num_nodes, alpha, beta, segments, k),
        })
    return sorted(options, key=lambda option: option["predicted_seconds"])


def lookup_table(table, size_bytes, num_workers):
    """Nearest measured entry (by log size) for this worker count, or None."""
    if not table:
        return None
    entries = [e for e in table.get("entries", []) if e["num_workers"] == num_workers]
    if not entries:
        return None
    nearest = min(entries, key=lambda e: abs(math.log2(e["size_bytes"]) - math.log2(size_bytes)))
    ratio = max(nearest["size_bytes"], size_bytes) / min(nearest["size_bytes"], size_bytes)
    return nearest if ratio <= TABLE_MATCH_FACTOR else None


def choose(size_bytes, num_workers, calibration=None, table=None, k=DEFAULT_K, candidates=CANDIDATES):
    """Pick topology and segment size. Measured table first, cost model otherwise."""
    options = model_choices(size_bytes, num_workers, calibration, k, candidates)
    model_pick = options[0]
    entry = lookup_table(table, size_bytes, num_workers)
    if entry is not None and entry["topology"] in candidates:
        segments = entry.get("segments", 1)
        return {
            "topology": entry["topology"],
            "k": entry.get("k") or (k if entry["topology"] == "knomial" else None),
            "segments": segments,
            "segment_bytes": segment_bytes_for(size_bytes, segments),
            "source": "table",
            "table_entry": entry,
            "model_pick": model_pick["topology"],
            "model_options": options,
        }
    return {
        **model_pick,
        "source": "model" if calibration else "model (default alpha/beta, not calibrated)",
        "model_pick": model_pick["topology"],
        "model_options": options,
    }


def fit_calibration(samples):
    """Fit alpha and beta from ping-pong samples [{size_bytes, one_way_seconds}].

    alpha comes from the smallest message; beta is the least-squares slope over the
    larger messages, where bandwidth dominates. A plain fit over all sizes lets the
    big messages swamp alpha (it can even come out negative).
    """
    points = sorted((s["size_bytes"], s["one_way_seconds"]) for s in samples)
    if len(points) < 2:
        raise ValueError("Need at least two message sizes to calibrate.")
    alpha = max(points[0][1], 1e-7)
    large = [p for p in points if p[0] >= 65536] or points[len(points) // 2:]
    if len(large) < 2:
        large = points[-2:]
    mean_x = sum(x for x, _ in large) / len(large)
    mean_y = sum(y for _, y in large) / len(large)
    var_x = sum((x - mean_x) ** 2 for x, _ in large)
    beta = sum((x - mean_x) * (y - mean_y) for x, y in large) / var_x if var_x else 0.0
    if beta <= 0:  # fall back to the biggest message's throughput
        size, seconds = points[-1]
        beta = max(seconds - alpha, 1e-12) / size
    return {"alpha": alpha, "beta": beta, "bandwidth_bytes_per_second": 1.0 / beta}


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None