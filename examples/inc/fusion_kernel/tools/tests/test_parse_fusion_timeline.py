#!/usr/bin/env python3

import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
SCRIPT = TOOLS_DIR / "parse_fusion_timeline.py"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_case.log"
SPEC = importlib.util.spec_from_file_location("parse_fusion_timeline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FusionTimelineParserTest(unittest.TestCase):
    def setUp(self):
        self.report = MODULE.parse_inputs([str(FIXTURE)])
        self.case = self.report["cases"][0]

    def test_parses_all_supported_trace_families(self):
        counts = {}
        for event in self.case["events"]:
            counts[event["type"]] = counts.get(event["type"], 0) + 1
        self.assertEqual(counts["worker_trace"], 1)
        self.assertEqual(counts["inc_dispatch_checkpoints"], 1)
        self.assertEqual(counts["inc_combine_checkpoints"], 1)
        self.assertEqual(counts["inc_trace"], 1)
        self.assertEqual(counts["device_trace"], 2)

    def test_recomputes_dc_window_metrics(self):
        event = next(item for item in self.case["events"] if item["type"] == "inc_trace")
        metrics = event["metrics"]
        self.assertEqual(metrics["serial_cycles"], 28832)
        self.assertEqual(metrics["concurrent_window_cycles"], 17955)
        self.assertAlmostEqual(metrics["overlap_realized"], 10877 / 11192)
        self.assertAlmostEqual(metrics["actual_window_speedup"], 28832 / 17955)
        self.assertAlmostEqual(metrics["theoretical_max_speedup"], 28832 / 17640)
        self.assertTrue(metrics["trace_valid"])
        self.assertEqual(self.report["warnings"], [])

    def test_parses_concatenated_python_literal_device_records(self):
        events = [item for item in self.case["events"] if item["type"] == "device_trace"]
        self.assertEqual([item["rank"] for item in events], [0, 1])
        self.assertEqual(len(events[0]["records"]), 2)
        self.assertEqual(events[0]["records"][1]["role_name"], "Worker Combine")
        self.assertEqual(events[1]["records"][0]["checkpoints"][-1], 34658)

    def test_markdown_states_observability_limit(self):
        rendered = MODULE.render_markdown(self.report)
        self.assertIn("不能据此还原或伪造逐 wave timeline", rendered)
        self.assertIn("97.19%", rendered)
        self.assertIn("1.6058×", rendered)
        self.assertIn("vLLM `INC_FUSION_DEVICE_TRACE`", rendered)

    def test_csv_contains_flattened_metrics_and_roles(self):
        rows = list(csv.DictReader(io.StringIO(MODULE.render_csv(self.report))))
        inc_row = next(row for row in rows if row["event_type"] == "inc_trace")
        self.assertEqual(inc_row["concurrent_window_cycles"], "17955")
        self.assertEqual(inc_row["aggregation"], "whole_kernel_dc_window")
        self.assertIn("不能据此还原", inc_row["aggregation_notice"])
        self.assertTrue(any(row["role_name"] == "Worker FFN 计算" for row in rows))

    def test_cli_runs_from_arbitrary_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "nested" / "timeline.json"
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(FIXTURE), "--format", "json", "-o", str(output)],
                cwd=temporary_directory,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            parsed = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(parsed["schema_version"], 1)
            self.assertEqual(len(parsed["cases"]), 1)

    def test_parent_directory_grouping_and_case_override(self):
        fixture_text = FIXTURE.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "case_a" / "pe0.log"
            second = root / "case_b" / "pe1.log"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_text(fixture_text, encoding="utf-8")
            second.write_text(fixture_text, encoding="utf-8")

            grouped = MODULE.parse_inputs([str(first), str(second)])
            self.assertEqual([case["name"] for case in grouped["cases"]], ["case_a", "case_b"])

            merged = MODULE.parse_inputs(
                [str(first), str(second)], case_name="merged_case"
            )
            self.assertEqual(len(merged["cases"]), 1)
            self.assertEqual(merged["cases"][0]["name"], "merged_case")
            self.assertEqual(len(merged["cases"][0]["files"]), 2)

    def test_strict_mode_rejects_malformed_trace(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            malformed = Path(temporary_directory) / "bad.log"
            malformed.write_text("FUSION_INC_TRACE d_span_cycles=1\n", encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(malformed), "--strict"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 2)
            self.assertIn("缺少字段", completed.stderr)


if __name__ == "__main__":
    unittest.main()
