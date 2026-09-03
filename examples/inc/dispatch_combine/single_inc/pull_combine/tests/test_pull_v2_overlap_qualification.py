#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).with_name("pull_v2_overlap_qualification.py")
SPEC = importlib.util.spec_from_file_location("pull_v2_overlap", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PullV2OverlapQualificationTest(unittest.TestCase):
    def test_timeline_overlap_and_speedup(self):
        dispatch = {"cycle_kernel_start": 100, "cycle_kernel_done": 500}
        combine = {"cycle_kernel_start": 200, "cycle_kernel_done": 600}
        result = MODULE.timeline_metrics(dispatch, combine, 400, 400)
        self.assertTrue(result["overlapped"])
        self.assertAlmostEqual(result["overlap_us"], 6.0)
        self.assertAlmostEqual(result["theoretical_speedup"], 2.0)
        self.assertAlmostEqual(result["real_speedup"], 1.6)
        self.assertAlmostEqual(result["overlap_geometry_speedup"], 1.6)
        self.assertAlmostEqual(result["dispatch_contention_inflation"], 1.0)
        self.assertAlmostEqual(result["combine_contention_inflation"], 1.0)

    def test_timeline_disjoint(self):
        dispatch = {"cycle_kernel_start": 100, "cycle_kernel_done": 200}
        combine = {"cycle_kernel_start": 250, "cycle_kernel_done": 400}
        result = MODULE.timeline_metrics(dispatch, combine, 100, 150)
        self.assertFalse(result["overlapped"])
        self.assertEqual(result["overlap_us"], 0.0)

    def test_real_speedup_uses_solo_not_contended_durations(self):
        dispatch = {"cycle_kernel_start": 100, "cycle_kernel_done": 600}
        combine = {"cycle_kernel_start": 200, "cycle_kernel_done": 700}
        result = MODULE.timeline_metrics(dispatch, combine, 400, 250)
        self.assertAlmostEqual(result["theoretical_speedup"], 1.625)
        self.assertAlmostEqual(result["real_speedup"], 650 / 600)
        self.assertAlmostEqual(result["overlap_geometry_speedup"], 1000 / 600)
        self.assertAlmostEqual(result["dispatch_contention_inflation"], 1.25)
        self.assertAlmostEqual(result["combine_contention_inflation"], 2.0)

    def test_iteration_parser_rejects_failed_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pe.log"
            path.write_text(json.dumps({
                "test": "pull_combine_v2_npu_e2e",
                "iteration": 0,
                "status": 10,
                "correct": False,
            }) + "\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                MODULE.read_iteration(path, "pull_combine_v2_npu_e2e")

    def test_schedule_set_has_both_orders_and_random_jitter(self):
        class Args:
            lead_us = 500
            random_cases = 2
            rank_jitter_us = 100
            seed = 7

        rows = MODULE.schedules(Args())
        self.assertEqual(rows[0].name, "simultaneous")
        self.assertLess(rows[1].dispatch_offset_us,
                        rows[1].combine_offset_us)
        self.assertGreater(rows[2].dispatch_offset_us,
                           rows[2].combine_offset_us)
        self.assertTrue(all(row.rank_jitter_us == 100 for row in rows[3:]))

    def test_placement_guard_is_profile_driven(self):
        self.assertTrue(MODULE.placement_is_one_plane(8, 4, 16, 8))
        self.assertTrue(MODULE.placement_is_one_plane(0, 2, 8, 4))
        self.assertFalse(MODULE.placement_is_one_plane(2, 2, 8, 4))
        self.assertFalse(MODULE.placement_is_one_plane(6, 2, 8, 8))

    def test_teardown_waits_until_every_device_is_idle(self):
        with mock.patch.object(
                MODULE, "all_npus_idle",
                side_effect=(False, False, True)) as idle, \
                mock.patch.object(MODULE.time, "sleep") as sleep:
            MODULE.wait_all_npus_idle(16, 1.0)
        self.assertEqual(idle.call_count, 3)
        self.assertEqual(sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
