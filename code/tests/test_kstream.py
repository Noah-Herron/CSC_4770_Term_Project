"""K-stream coverage and compatibility with main's algorithm registry."""

import queue
import socket
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as server
import client
import experiment_handler as handler
import kstream
from protocol import recv_msg, send_msg


class AssignmentTests(unittest.TestCase):
    def test_edge_tiles_cover_every_output_element_once(self):
        coverage = np.zeros((5, 7), dtype=int)
        for _, _, r0, r1, c0, c1 in kstream.build_tiles(5, 7, 3):
            self.assertTrue(0 < r1 - r0 <= 3)
            self.assertTrue(0 < c1 - c0 <= 3)
            coverage[r0:r1, c0:c1] += 1
        np.testing.assert_array_equal(coverage, np.ones((5, 7)))

    def test_shares_cover_each_tile_k_range_once_without_gaps(self):
        for tiles, K, workers in ((1, 7, 4), (5, 7, 3), (2, 1, 8), (6, 4, 1)):
            with self.subTest(tiles=tiles, K=K, workers=workers):
                assignments = kstream.streamk_assignments([None] * tiles, K, workers)
                coverage = np.zeros((tiles, K), dtype=int)
                cursor = 0
                self.assertEqual(len(assignments), workers)
                for jobs in assignments:
                    work = 0
                    for tile, start, end in jobs:
                        self.assertEqual(tile * K + start, cursor)
                        self.assertTrue(0 <= start < end <= K)
                        coverage[tile, start:end] += 1
                        work += end - start
                        cursor += end - start
                    self.assertLessEqual(work, -(-(tiles * K) // workers))
                np.testing.assert_array_equal(coverage, np.ones((tiles, K)))


class MultiplicationTests(unittest.TestCase):
    def multiply(self, A, B, workers, synchronize_start=False):
        connections, threads, errors = [], [], queue.Queue()
        barrier = threading.Barrier(workers) if synchronize_start else None

        def worker(connection):
            try:
                with connection:
                    first = True
                    while True:
                        job = recv_msg(connection)
                        if job["algorithm"] == "done":
                            return
                        self.assertEqual(job["algorithm"], "kstream")
                        if first and barrier is not None:
                            barrier.wait(timeout=5)
                        first = False
                        kstream.run_kstream_worker(connection, job)
            except Exception as error:
                errors.put(error)

        for _ in range(workers):
            master, peer = socket.socketpair()
            master.settimeout(10)
            peer.settimeout(10)
            connections.append(master)
            thread = threading.Thread(target=worker, args=(peer,), daemon=True)
            threads.append(thread)
            thread.start()
        try:
            with patch.object(kstream, "TILE_SIZE", 3):
                result = kstream.run_kstream_master(connections, A, B)
            for connection in connections:
                send_msg(connection, {"algorithm": "done"})
        finally:
            for connection in connections:
                connection.close()
            for thread in threads:
                thread.join(timeout=11)
                self.assertFalse(thread.is_alive(), "Worker did not finish")
        if not errors.empty():
            raise errors.get()
        return result

    def test_partial_reductions_rectangular_edges_and_idle_workers(self):
        rng = np.random.default_rng(42)
        for M, K, N, workers in ((1, 1, 1, 8), (2, 11, 2, 4), (7, 11, 8, 4),
                                 (8, 3, 7, 3), (5, 1, 7, 2), (7, 11, 8, 1)):
            with self.subTest(shape=(M, K, N), workers=workers):
                A, B = rng.normal(size=(M, K)), rng.normal(size=(K, N))
                np.testing.assert_allclose(self.multiply(A, B, workers), A @ B, rtol=1e-10, atol=1e-10)

    def test_dispatches_to_every_worker_before_waiting_for_results(self):
        A, B = np.ones((7, 11)), np.ones((11, 8))
        np.testing.assert_allclose(self.multiply(A, B, 4, synchronize_start=True), A @ B)

    def test_disconnected_worker_raises(self):
        master, peer = socket.socketpair()
        peer.close()
        with master:
            master.settimeout(5)
            with self.assertRaises((OSError, EOFError)):
                kstream.run_kstream_master([master], np.ones((2, 3)), np.ones((3, 2)))


class IntegrationTests(unittest.TestCase):
    def test_matrix_algorithm_registries_agree(self):
        matrix_names = {name for name, algorithm in handler.ALGORITHMS.items()
                        if algorithm.family == "matrix"}
        self.assertEqual(set(server.MATRIX_ALGORITHMS), set(client.MATRIX_ALGORITHMS))
        self.assertEqual(set(server.MATRIX_ALGORITHMS), matrix_names)
        self.assertIsInstance(handler.ALGORITHMS["kstream"], handler.MatrixAlgorithm)
        self.assertEqual(handler.ALGORITHMS["kstream"].matrix_values, handler.MATRIX_VALUES)
        self.assertEqual(handler.ALGORITHMS["kstream"].container_counts, handler.CONTAINER_COUNTS)

    def test_api_accepts_every_matrix_algorithm_and_keeps_baseline_default(self):
        with patch.object(server, "tasks_db", {}), patch.object(server, "job_queue", queue.Queue()):
            api = server.app.test_client()
            for algorithm in (None, *client.MATRIX_ALGORITHMS):
                with self.subTest(algorithm=algorithm):
                    payload = {"matrix_A": [[1, 2]], "matrix_B": [[3], [4]]}
                    if algorithm is not None:
                        payload["algorithm"] = algorithm
                    response = api.post("/multiply", json=payload)
                    self.assertEqual(response.status_code, 202)
                    self.assertEqual(server.job_queue.get_nowait()["algorithm"], algorithm or "baseline")


if __name__ == "__main__":
    unittest.main()
