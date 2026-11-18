"""Run from the code folder:  python -m unittest discover tests -v"""

import math
import socket
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broadcast import PeerListener, run_bcast_root, run_bcast_worker, run_pingpong_root, run_pingpong_worker  # noqa: E402
from protocol import recv_msg  # noqa: E402
from selector import tree_rounds, best_chain_segments, choose, fit_calibration, model_choices, predict  # noqa: E402
from topology import TOPOLOGIES, build_children, depth, parents_from_children  # noqa: E402


class TopologyTests(unittest.TestCase):
    def test_every_rank_reached_once(self):
        for topology in TOPOLOGIES:
            for k in (2, 3, 4, 5):
                for p in range(1, 41):
                    children = build_children(topology, p, k)
                    parents_from_children(children)  # raises if not a tree
                    self.assertEqual(sum(map(len, children)), p - 1)

    def test_depths(self):
        for p in range(2, 41):
            self.assertLessEqual(depth(build_children("binomial", p)), math.ceil(math.log2(p)))
            self.assertEqual(depth(build_children("chain", p)), p - 1)
            self.assertLessEqual(depth(build_children("knomial", p, 4)), tree_rounds(p, 4))

    def test_binomial_sends_largest_subtree_first(self):
        self.assertEqual(build_children("binomial", 8)[0], [1, 2, 4])
        self.assertEqual(build_children("binomial", 8)[1], [3, 5])


class ModelTests(unittest.TestCase):
    alpha, beta = 1e-4, 1e-9  # 100 us latency, ~1 GB/s

    def test_small_messages_pick_a_tree(self):
        self.assertNotEqual(choose(1024, 16, {"alpha": self.alpha, "beta": self.beta})["topology"], "chain")

    def test_large_messages_pick_chain(self):
        self.assertEqual(choose(64 << 20, 16, {"alpha": self.alpha, "beta": self.beta})["topology"], "chain")

    def test_segments(self):
        self.assertEqual(best_chain_segments(64 << 20, 2, self.alpha, self.beta), 1)  # one hop: nothing to pipeline
        s = best_chain_segments(64 << 20, 17, self.alpha, self.beta)
        expected = math.sqrt(15 * (64 << 20) * self.beta / self.alpha)
        self.assertAlmostEqual(s, expected, delta=1)
        # the chosen s is at least as good as its neighbours
        t = lambda n: predict("chain", 64 << 20, 17, self.alpha, self.beta, n)
        self.assertLessEqual(t(s), min(t(s - 1), t(s + 1)))

    def test_model_options_sorted(self):
        options = model_choices(1 << 20, 8, {"alpha": self.alpha, "beta": self.beta})
        times = [o["predicted_seconds"] for o in options]
        self.assertEqual(times, sorted(times))

    def test_table_overrides_model(self):
        table = {"entries": [{"num_workers": 8, "size_bytes": 1 << 20, "topology": "binomial",
                              "k": None, "segments": 1, "segment_bytes": 1 << 20}]}
        cal = {"alpha": 1e-6, "beta": 1e-8}  # a model that would pick chain
        self.assertEqual(choose(1 << 20, 8, cal)["topology"], "chain")
        pick = choose(1 << 20, 8, cal, table)
        self.assertEqual((pick["topology"], pick["source"]), ("binomial", "table"))
        self.assertEqual(choose(16 << 20, 8, cal, table)["source"], "model")  # too far from the entry

    def test_fit_recovers_parameters(self):
        samples = [{"size_bytes": n, "one_way_seconds": self.alpha + n * self.beta}
                   for n in (8, 1024, 65536, 1 << 20, 16 << 20)]
        fit = fit_calibration(samples)
        self.assertAlmostEqual(fit["alpha"], self.alpha, delta=1e-6)
        self.assertAlmostEqual(fit["beta"], self.beta, delta=1e-12)


class LoopbackTests(unittest.TestCase):
    """Real sockets on localhost: worker threads play the containers."""

    def start_workers(self, count, target):
        listeners = [PeerListener(0, "127.0.0.1") for _ in range(count)]
        pairs = [socket.socketpair() for _ in range(count)]
        errors = []

        def worker(control, listener):
            try:
                job = recv_msg(control)
                target(control, job, listener)
            except Exception as error:  # surfaced through the assertion below
                errors.append(error)

        threads = [threading.Thread(target=worker, args=(pair[1], listener), daemon=True)
                   for pair, listener in zip(pairs, listeners)]
        for thread in threads:
            thread.start()
        return [p[0] for p in pairs], [("127.0.0.1", l.port) for l in listeners], threads, errors, listeners

    def finish(self, threads, errors, listeners):
        for thread in threads:
            thread.join(timeout=30)
        for listener in listeners:
            listener.close()
        self.assertEqual(errors, [])

    def test_broadcast_all_topologies(self):
        cases = [("binomial", 7, 300_000, None), ("knomial", 9, 50_000, None),
                 ("chain", 6, 1_000_000, 65536), ("chain", 3, 5, None), ("binomial", 1, 1234, 100)]
        for topology, workers, size, segment in cases:
            with self.subTest(topology=topology, workers=workers):
                controls, addrs, threads, errors, listeners = self.start_workers(workers, run_bcast_worker)
                result = run_bcast_root(controls, addrs, topology, size, segment, trials=3, warmup=1)
                self.finish(threads, errors, listeners)
                self.assertEqual(len(result["trial_seconds"]), 3)
                self.assertEqual(len(result["per_rank_median_seconds"]), workers)
                self.assertEqual(result["segments"], -(-size // (segment or size)))

    def test_pingpong(self):
        controls, addrs, threads, errors, listeners = self.start_workers(1, run_pingpong_worker)
        samples = run_pingpong_root(controls[0], addrs[0], [8, 4096, 1 << 20], repeats=3, warmup=1)
        self.finish(threads, errors, listeners)
        self.assertEqual([s["size_bytes"] for s in samples], [8, 4096, 1 << 20])
        fit = fit_calibration(samples)
        self.assertGreater(fit["alpha"], 0)
        self.assertGreater(fit["beta"], 0)


if __name__ == "__main__":
    unittest.main()
