"""Small, shared contract for machine-readable data-quality reports."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

QUALITY_REPORT_SCHEMA_VERSION = 2
QUALITY_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "report_type",
        "created_at",
        "status",
        "scope",
        "finding_counts",
        "datasets",
        "findings",
        "metadata",
    }
)
FINDING_KEYS = frozenset({"severity", "check", "message", "details"})
DATASET_SUMMARY_KEYS = frozenset({"status", "finding_counts", "checks"})
SCOPE_KEYS = frozenset({"data_root", "start_date", "end_date", "datasets"})
SEVERITIES = ("error", "warning", "info")
STATUSES = frozenset({"ok", "warning", "error"})


def count_findings(findings: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {severity: 0 for severity in SEVERITIES}
    for finding in findings:
        severity = finding.get("severity")
        if severity not in counts:
            raise ValueError(f"unknown data-quality severity: {severity!r}")
        counts[str(severity)] += 1
    return counts


def quality_status(counts: Mapping[str, int]) -> str:
    if int(counts.get("error", 0)):
        return "error"
    if int(counts.get("warning", 0)):
        return "warning"
    return "ok"


def summarize_datasets(
    findings: Iterable[Mapping[str, Any]], dataset_names: Iterable[str]
) -> dict[str, dict[str, Any]]:
    """Map check-name prefixes to a compact summary for every scoped dataset."""
    unique_names = set(dataset_names)
    match_order = sorted(unique_names, key=len, reverse=True)
    summary: dict[str, dict[str, Any]] = {
        name: {
            "status": "ok",
            "finding_counts": {severity: 0 for severity in SEVERITIES},
            "checks": [],
        }
        for name in sorted(unique_names)
    }
    for finding in findings:
        check = str(finding.get("check") or "")
        dataset = next(
            (
                name
                for name in match_order
                if check == name
                or check.startswith(f"{name}_")
                or check.startswith(f"source_{name}")
            ),
            None,
        )
        if dataset is None:
            continue
        item = summary[dataset]
        severity = str(finding.get("severity") or "info")
        if severity not in SEVERITIES:
            raise ValueError(f"unknown data-quality severity: {severity!r}")
        item["finding_counts"][severity] += 1
        item["checks"].append(check)
    for item in summary.values():
        item["status"] = quality_status(item["finding_counts"])
        item["checks"] = sorted(set(item["checks"]))
    return dict(sorted(summary.items()))


def _validate_counts(value: Any, *, context: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(SEVERITIES):
        raise ValueError(f"{context} requires exactly the counts {sorted(SEVERITIES)}")
    counts: dict[str, int] = {}
    for severity in SEVERITIES:
        count = value[severity]
        if type(count) is not int or count < 0:
            raise ValueError(f"{context}.{severity} must be a non-negative integer")
        counts[severity] = count
    return counts


def _validate_created_at(value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("quality report created_at must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("quality report created_at must be a timezone-aware ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("quality report created_at must include a timezone")


def _validate_findings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError("data-quality findings must be an array")
    findings: list[dict[str, Any]] = []
    for index, raw_finding in enumerate(value):
        if not isinstance(raw_finding, Mapping) or set(raw_finding) != FINDING_KEYS:
            keys = sorted(raw_finding) if isinstance(raw_finding, Mapping) else []
            raise ValueError(
                f"data-quality findings[{index}] require exactly {sorted(FINDING_KEYS)}, got {keys}"
            )
        finding = dict(raw_finding)
        if finding["severity"] not in SEVERITIES:
            raise ValueError(
                f"data-quality findings[{index}] has invalid severity {finding['severity']!r}"
            )
        if not isinstance(finding["check"], str) or not finding["check"].strip():
            raise ValueError(f"data-quality findings[{index}].check must be a non-empty string")
        if not isinstance(finding["message"], str) or not finding["message"].strip():
            raise ValueError(f"data-quality findings[{index}].message must be a non-empty string")
        if not isinstance(finding["details"], dict):
            raise TypeError(f"data-quality findings[{index}].details must be an object")
        findings.append(finding)
    return findings


def validate_quality_report(
    report: Mapping[str, Any], *, expected_report_type: str | None = None
) -> None:
    """Validate the complete v2 contract and its internal consistency.

    Schema v1 is intentionally rejected. It was already used by the legacy
    revision summary, so accepting it as a unified report would be ambiguous.
    """
    if not isinstance(report, Mapping):
        raise TypeError("quality report must be an object")
    version = report.get("schema_version")
    if type(version) is not int or version != QUALITY_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported quality report schema_version={version!r}; "
            f"regenerate it as schema v{QUALITY_REPORT_SCHEMA_VERSION}"
        )
    if set(report) != QUALITY_REPORT_KEYS:
        raise ValueError(
            f"quality report requires exactly {sorted(QUALITY_REPORT_KEYS)}, got {sorted(report)}"
        )

    report_type = report["report_type"]
    if not isinstance(report_type, str) or not report_type.strip():
        raise ValueError("quality report_type must be a non-empty string")
    if expected_report_type is not None and report_type != expected_report_type:
        raise ValueError(
            f"quality report_type mismatch: expected {expected_report_type!r}, got {report_type!r}"
        )
    _validate_created_at(report["created_at"])

    findings = _validate_findings(report["findings"])
    counts = _validate_counts(report["finding_counts"], context="quality report finding_counts")
    expected_counts = count_findings(findings)
    if counts != expected_counts:
        raise ValueError(
            f"quality report finding_counts do not match findings: {counts} != {expected_counts}"
        )
    status = report["status"]
    expected_status = quality_status(counts)
    if status not in STATUSES or status != expected_status:
        raise ValueError(f"quality report status {status!r} does not match counts ({expected_status!r})")

    scope = report["scope"]
    if not isinstance(scope, Mapping):
        raise TypeError("quality report scope must be an object")
    missing_scope = SCOPE_KEYS - set(scope)
    if missing_scope:
        raise ValueError(f"data-quality scope is missing {sorted(missing_scope)}")
    for key in ("data_root", "start_date", "end_date"):
        if not isinstance(scope[key], str):
            raise TypeError(f"data-quality scope.{key} must be a string")
    dataset_names = scope["datasets"]
    if not isinstance(dataset_names, list):
        raise TypeError("data-quality scope.datasets must be an array")
    if any(not isinstance(name, str) or not name.strip() for name in dataset_names):
        raise ValueError("data-quality scope.datasets must contain non-empty strings")
    if len(dataset_names) != len(set(dataset_names)):
        raise ValueError("data-quality scope.datasets must not contain duplicates")

    dataset_records = report["datasets"]
    if not isinstance(dataset_records, Mapping):
        raise TypeError("quality report datasets must be an object")
    if set(dataset_records) != set(dataset_names):
        raise ValueError(
            "quality report datasets keys must exactly match scope.datasets: "
            f"summaries={sorted(dataset_records)} scope={sorted(dataset_names)}"
        )
    for name, summary in dataset_records.items():
        if not isinstance(summary, Mapping) or set(summary) != DATASET_SUMMARY_KEYS:
            raise ValueError(
                f"dataset summary {name!r} requires exactly {sorted(DATASET_SUMMARY_KEYS)}"
            )
        dataset_counts = _validate_counts(
            summary["finding_counts"], context=f"dataset summary {name!r} finding_counts"
        )
        if summary["status"] != quality_status(dataset_counts):
            raise ValueError(f"dataset summary {name!r} status does not match its finding counts")
        checks = summary["checks"]
        if not isinstance(checks, list) or any(
            not isinstance(check, str) or not check.strip() for check in checks
        ):
            raise ValueError(f"dataset summary {name!r} checks must be an array of strings")
        if len(checks) != len(set(checks)):
            raise ValueError(f"dataset summary {name!r} checks must not contain duplicates")
    if dict(dataset_records) != summarize_datasets(findings, dataset_names):
        raise ValueError("quality report dataset summaries do not match its findings and scope")
    if not isinstance(report["metadata"], Mapping):
        raise TypeError("quality report metadata must be an object")


def build_quality_report(
    *,
    report_type: str,
    scope: Mapping[str, Any],
    findings: Iterable[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the exact envelope shared by every retained quality report."""
    records = _validate_findings(list(findings))
    scope_record = dict(scope)
    missing_scope = SCOPE_KEYS - set(scope_record)
    if missing_scope:
        raise ValueError(f"data-quality scope is missing {sorted(missing_scope)}")
    raw_names = scope_record["datasets"]
    if not isinstance(raw_names, (list, tuple)):
        raise TypeError("data-quality scope.datasets must be an array")
    dataset_names = list(raw_names)
    scope_record["datasets"] = dataset_names
    counts = count_findings(records)
    report = {
        "schema_version": QUALITY_REPORT_SCHEMA_VERSION,
        "report_type": str(report_type),
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "status": quality_status(counts),
        "scope": scope_record,
        "finding_counts": counts,
        "datasets": summarize_datasets(records, dataset_names),
        "findings": records,
        "metadata": dict(metadata or {}),
    }
    validate_quality_report(report)
    return report


def read_quality_report(
    path: str | Path, *, expected_report_type: str | None = None
) -> dict[str, Any]:
    """Read and fail-closed validate one v2 quality report."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid quality report JSON: {source}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"quality report must be an object: {source}")
    validate_quality_report(payload, expected_report_type=expected_report_type)
    return payload


def write_quality_report(path: str | Path, report: Mapping[str, Any]) -> None:
    """Publish one validated report via a unique same-directory temporary."""
    validate_quality_report(report)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
