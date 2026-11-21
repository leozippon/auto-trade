import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from autotrade.environment.data.quality import (
    FINDING_KEYS,
    QUALITY_REPORT_KEYS,
    QUALITY_REPORT_SCHEMA_VERSION,
    build_quality_report,
    read_quality_report,
    validate_quality_report,
    write_quality_report,
)


class DataQualityContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.findings = [
            {
                "severity": "warning",
                "check": "daily_partitions",
                "message": "daily partitions checked",
                "details": {"missing": 1},
            }
        ]
        self.scope = {
            "data_root": "/data/raw",
            "start_date": "20200101",
            "end_date": "20200131",
            "datasets": ["daily", "clean_dataset"],
        }

    def _report(self, **overrides):
        return build_quality_report(
            report_type="base_research",
            scope=self.scope,
            findings=self.findings,
            created_at="2026-07-20T00:00:00+00:00",
            **overrides,
        )

    def test_report_and_nested_records_have_one_exact_v2_envelope(self) -> None:
        report = self._report()

        self.assertEqual(report["schema_version"], QUALITY_REPORT_SCHEMA_VERSION)
        self.assertEqual(set(report), QUALITY_REPORT_KEYS)
        self.assertEqual(set(report["findings"][0]), FINDING_KEYS)
        self.assertEqual(report["finding_counts"], {"error": 0, "warning": 1, "info": 0})
        self.assertEqual(set(report["datasets"]), set(report["scope"]["datasets"]))
        self.assertEqual(report["datasets"]["daily"]["status"], "warning")
        self.assertEqual(report["datasets"]["clean_dataset"]["status"], "ok")
        self.assertEqual(report["datasets"]["clean_dataset"]["checks"], [])

    def test_invalid_finding_shape_is_rejected(self) -> None:
        invalid = [{**self.findings[0], "extra": True}]
        with self.assertRaisesRegex(ValueError, "require exactly"):
            build_quality_report(
                report_type="base_research",
                scope=self.scope,
                findings=invalid,
            )

    def test_consumer_rejects_v1_and_inconsistent_dataset_keys(self) -> None:
        report = self._report()
        legacy = {**report, "schema_version": 1}
        with self.assertRaisesRegex(ValueError, "regenerate it as schema v2"):
            validate_quality_report(legacy)

        inconsistent = {**report, "datasets": {"daily": report["datasets"]["daily"]}}
        with self.assertRaisesRegex(ValueError, "exactly match scope.datasets"):
            validate_quality_report(inconsistent)

    def test_reader_validates_schema_and_report_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "status.json"
            output.write_text(json.dumps(self._report()), encoding="utf-8")
            loaded = read_quality_report(output, expected_report_type="base_research")
            self.assertEqual(loaded["report_type"], "base_research")
            with self.assertRaisesRegex(ValueError, "report_type mismatch"):
                read_quality_report(output, expected_report_type="event_flow")

    def test_writer_publishes_parseable_contract_without_fixed_temp_name(self) -> None:
        report = self._report()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "status.json"
            write_quality_report(output, report)

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])

    def test_concurrent_writers_use_independent_temporaries(self) -> None:
        first = self._report(metadata={"publisher": "first"})
        second = self._report(metadata={"publisher": "second"})
        barrier = threading.Barrier(2)
        real_replace = os.replace

        def synchronized_replace(source, destination):
            barrier.wait(timeout=5)
            real_replace(source, destination)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "status.json"
            with patch("os.replace", side_effect=synchronized_replace):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(write_quality_report, output, report)
                        for report in (first, second)
                    ]
                    for future in futures:
                        future.result(timeout=10)

            validate_quality_report(json.loads(output.read_text(encoding="utf-8")))
            self.assertEqual(list(output.parent.glob(f".{output.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
