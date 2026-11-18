"""Tracking/history regression checks for the main/Nol merge."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import experiment_handler as handler


class TrackingTests(unittest.TestCase):
    def test_repository_experiment_files_are_valid(self):
        tracking = json.loads((CODE_DIR / "experiment_tracking.json").read_text())
        for key, count in tracking.items():
            handler.parse_key(key)
            self.assertIsInstance(count, int)
            self.assertGreaterEqual(count, 0)
        results = [json.loads(line) for line in (CODE_DIR / "experiment_results.jsonl").read_text().splitlines()
                   if line.strip()]
        ids = [result["request_id"] for result in results]
        self.assertEqual(len(ids), len(set(ids)), "Result history has duplicate request IDs")

    def test_loading_migrates_keys_and_preserves_history_outside_current_grid(self):
        baseline = handler.MatrixAlgorithm("baseline", matrix_values=[2], container_counts=[2])
        current_params = {"num_containers": 2, "dims": [2, 2, 2, 2]}
        history_params = {"num_containers": 2, "dims": [25, 25, 25, 25]}
        current_key = handler.make_key("baseline", current_params)
        history_key = handler.make_key("baseline", history_params)
        saved = {
            "(2, (25, 25, 25, 25))": 3,
            "('baseline', 2, (25, 25, 25, 25))": 4,
            history_key: 2,
            current_key: 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            tracking_file = Path(directory) / "tracking.json"
            tracking_file.write_text(json.dumps(saved))
            with patch.object(handler, "TRACKING_FILE", tracking_file), \
                    patch.object(handler, "ALGORITHMS", {"baseline": baseline}):
                tracking = handler.load_tracking()
                self.assertEqual(tracking, {current_key: 1, history_key: 4})
                self.assertEqual(handler.active_keys(tracking, ["baseline"]), [current_key])
                self.assertEqual(handler.load_tracking(), tracking)


if __name__ == "__main__":
    unittest.main()
