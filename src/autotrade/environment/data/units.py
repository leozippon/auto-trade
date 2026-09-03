"""Column-level unit registry and snapshot unit normalization.

``FIELD_RULES`` is the single source of truth for the unit and semantic type
of every column that can appear in a snapshot file. A rule targets explicit
column names or fnmatch globs under an exact ``(file, dataset)`` key — never a
free-text field family — so coverage and overlap are machine-checkable.

Resolution order for one ``(file, dataset, column)``:

1. the dataset's (or, for single-schema files, the file's) explicit rules —
   at most one may match, otherwise the registry is broken;
2. ``COMMON_FIELD_SEMANTICS`` — shared identifier/date/text/categorical
   classifiers that mean the same thing in every dataset;
3. the dataset's default rule (``columns=("*",)``), allowed only where the
   vendor contract is uniform (financial-statement amounts);
4. otherwise ``UnresolvedUnitError`` — an unclassified column fails the
   snapshot build instead of shipping without unit metadata.

Projections (never maintained by hand elsewhere): the snapshot conversion
tables derive from factor rules; audit
report metadata selects by each domain's dataset list; ``unit_reference.json``
(written next to data_summary.json) enumerates every column of the live
snapshot; ``docs/units-reference.md`` renders the registry for humans.

Rule declarations live in ``unit_rules.py``; this module owns resolution,
schema validation, unit conversion, and Agent artifact generation.
"""

from __future__ import annotations

from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from autotrade.environment.data.unit_rules import (  # noqa: F401  (re-exported)
    COMMON_FIELD_SEMANTICS,
    FIELD_RULES,
    FieldRule,
    NO_NUMERIC_DATASETS,
)


class UnresolvedUnitError(ValueError):
    """A snapshot column has no unit/semantic classification."""


@lru_cache(maxsize=1)
def _rules_index() -> dict[tuple[str, str | None], tuple[tuple[FieldRule, ...], FieldRule | None]]:
    index: dict[tuple[str, str | None], tuple[list[FieldRule], list[FieldRule]]] = {}
    for rule in FIELD_RULES:
        explicit, defaults = index.setdefault((rule.file, rule.dataset), ([], []))
        (defaults if rule.columns == ("*",) else explicit).append(rule)
    out: dict[tuple[str, str | None], tuple[tuple[FieldRule, ...], FieldRule | None]] = {}
    for key, (explicit, defaults) in index.items():
        if len(defaults) > 1:
            raise ValueError(f"multiple default unit rules for {key}")
        out[key] = (tuple(explicit), defaults[0] if defaults else None)
    return out


def resolve_field(file: str, dataset: str | None, column: str) -> dict[str, object]:
    """Classify one snapshot column; raises UnresolvedUnitError if unregistered."""
    explicit, default = _rules_index().get((file, dataset), ((), None))
    matches = [
        rule for rule in explicit
        if any(fnmatchcase(column, pattern) for pattern in rule.columns)
    ]
    if len(matches) > 1:
        raise ValueError(
            f"unit registry overlap for {file}:{dataset}:{column}: "
            f"{[rule.key() for rule in matches]}"
        )
    rule = matches[0] if matches else None
    semantic = source_unit = factor = normalized = status = note = None
    if rule is None:
        for pattern, common_semantic in COMMON_FIELD_SEMANTICS:
            if fnmatchcase(column, pattern):
                semantic, status = common_semantic, "official"
                break
        else:
            rule = default
    if rule is not None:
        semantic, source_unit, status = rule.semantic, rule.source_unit, rule.status
        factor, normalized, note = rule.factor, rule.normalized_unit, rule.note or None
    if semantic is None:
        raise UnresolvedUnitError(
            f"no unit rule or common classifier for {file}:{dataset}:{column}"
        )
    record: dict[str, object] = {
        "file": file,
        "dataset": dataset,
        "column": column,
        "semantic_type": semantic,
        "source_unit": source_unit,
        "status": status,
    }
    if factor is not None:
        record["factor"] = factor
        record["normalized_unit"] = normalized
    if note:
        record["note"] = note
    return record


def build_unit_reference(
    column_map: dict[tuple[str, str | None], list[str]],
) -> list[dict[str, object]]:
    """Per-column records for every (file, dataset, column) in the map.

    Raises UnresolvedUnitError listing ALL unclassified columns at once.
    """
    records: list[dict[str, object]] = []
    problems: list[str] = []
    for (file, dataset), columns in sorted(
        column_map.items(), key=lambda item: (item[0][0], item[0][1] or "")
    ):
        for column in columns:
            try:
                records.append(resolve_field(file, dataset, column))
            except UnresolvedUnitError as exc:
                problems.append(str(exc))
    if problems:
        raise UnresolvedUnitError(
            "unclassified snapshot columns:\n" + "\n".join(sorted(problems))
        )
    return records


# Union files whose per-dataset column attribution lives in the snapshot
# manifest (captured at build time); all other snapshot parquets are
# single-schema and classified from their footer directly.
UNION_DOMAIN_BY_FILE: dict[str, str] = {
    "events.parquet": "events",
    "macro.parquet": "macro",
    "fundamentals.parquet": "fundamentals",
}
# Domains attributed from the vendor schema rather than window content: they
# may declare columns not yet materialized in a given window. All other union
# domains must match their physical schema exactly.
SCHEMA_FORWARD_DOMAINS = frozenset({"fundamentals"})


def snapshot_column_map(view_dir, manifest: dict[str, object]) -> dict[tuple[str, str | None], list[str]]:
    """(file, dataset) -> columns for every parquet in one snapshot view.

    Union files require ``dataset_columns`` in the manifest domain metadata
    (written by the snapshot builder; external snapshots must supply it
    explicitly — ownership is never inferred from file content), and are
    reconciled against the physical schema: an unattributed physical column
    always fails, a declared-but-unmaterialized column fails outside
    ``SCHEMA_FORWARD_DOMAINS``.
    """
    view_dir = Path(view_dir)
    domains = manifest.get("domains", {})
    column_map: dict[tuple[str, str | None], list[str]] = {}
    for path in sorted(view_dir.glob("*.parquet")):
        try:
            physical = set(pq.read_schema(path).names)
        except Exception as exc:
            raise ValueError(
                f"unreadable parquet footer for snapshot file {path.name}: "
                f"{type(exc).__name__}"
            ) from exc
        domain = UNION_DOMAIN_BY_FILE.get(path.name)
        if domain is None:
            column_map[(path.name, None)] = sorted(physical)
            continue
        meta = domains.get(domain)
        if not isinstance(meta, dict) or "dataset_columns" not in meta:
            raise ValueError(
                f"snapshot manifest domain '{domain}' lacks dataset_columns for {path.name}; "
                "incompatible snapshot format — rebuild the snapshot"
            )
        declared: set[str] = set()
        for dataset, columns in meta["dataset_columns"].items():
            column_map[(path.name, dataset)] = list(columns)
            declared.update(columns)
        unattributed = sorted(physical - declared)
        if unattributed:
            raise ValueError(
                f"snapshot manifest domain '{domain}' does not attribute physical "
                f"columns of {path.name} to any dataset: {unattributed}"
            )
        undeclared = sorted(declared - physical)
        if undeclared and domain not in SCHEMA_FORWARD_DOMAINS:
            raise ValueError(
                f"snapshot manifest domain '{domain}' declares columns absent from "
                f"{path.name}: {undeclared}"
            )
    return column_map


def validate_snapshot_units(view_dir, manifest: dict[str, object]) -> None:
    """Fail-fast: every column of every snapshot file must classify."""
    build_unit_reference(snapshot_column_map(view_dir, manifest))


def rules_for(file: str | None = None, datasets: tuple[str, ...] | None = None) -> tuple[FieldRule, ...]:
    """Registry selection by file and/or exact dataset ids."""
    selected = FIELD_RULES
    if file is not None:
        selected = tuple(rule for rule in selected if rule.file == file)
    if datasets is not None:
        wanted = set(datasets)
        selected = tuple(rule for rule in selected if rule.dataset in wanted)
    return selected




def dataset_rules_records(datasets: tuple[str, ...]) -> dict[str, list[dict[str, object]]]:
    """Audit projection: registry records grouped by dataset, fail-fast on ids
    that neither carry a rule nor are known no-numeric datasets (typo guard)."""
    ruled = {rule.dataset for rule in FIELD_RULES if rule.dataset}
    unknown = sorted(set(datasets) - ruled - NO_NUMERIC_DATASETS)
    if unknown:
        raise KeyError(f"unit registry has no rules for datasets: {unknown}")
    by_dataset: dict[str, list[dict[str, object]]] = {dataset: [] for dataset in datasets}
    for rule in rules_for(datasets=datasets):
        by_dataset[rule.dataset].append(rule.to_record())
    return by_dataset


def column_source_units(file: str) -> dict[str, str]:
    """Exact column -> source unit for a single-schema file's numeric rules."""
    out: dict[str, str] = {}
    for rule in rules_for(file):
        if rule.dataset is not None or rule.source_unit is None:
            continue
        for column in rule.columns:
            if not any(ch in column for ch in "*?["):
                out[column] = rule.source_unit
    return out


def _conversions(file: str) -> tuple[tuple[str, float, str], ...]:
    conversions: list[tuple[str, float, str]] = []
    for rule in rules_for(file):
        if rule.factor is None:
            continue
        for column in rule.columns:
            if any(ch in column for ch in "*?["):
                raise ValueError(f"conversion rule {rule.key()} must use exact column names")
            conversions.append((column, rule.factor, f"{rule.source_unit}->{rule.normalized_unit}"))
    return tuple(conversions)


# Derived byte-conversion tables (single-sourced from the registry).
DAILY_UNIT_CONVERSIONS: tuple[tuple[str, float, str], ...] = _conversions("daily.parquet")
AUCTION_UNIT_CONVERSIONS: tuple[tuple[str, float, str], ...] = _conversions("auction.parquet")

# Minimal Agent-facing contract inside data_summary.json: the full per-column
# table ships as its own artifact next to it, loaded by the Agent on demand.
AGENT_UNIT_CONTRACT: dict[str, str] = {
    "identity_rule": "interpret units by file + dataset + column; never by column name alone",
    "unit_reference": "/mnt/artifacts/unit_reference.json",
    "normalized_files": (
        "daily/intraday_1min/auction/corporate_actions files store normalized values; "
        "records carrying a factor show the applied source->normalized conversion"
    ),
    # Scope clause. The table enumerates snapshot-file columns; data_summary.json
    # separately names each domain's vendor source datasets. A Fold read the two as
    # one namespace, looked up a source dataset (suspend_d) that contributes only a
    # derived column, found no record, and filed the missing entry as a broken
    # contract. Stating the scope is the only thing that separates "not a snapshot
    # column" from the real defect the next clause defines.
    "coverage": (
        "this table lists every column of every snapshot file in the view. The "
        "`datasets` list of a domain in data_summary.json names the vendor sources "
        "joined to build those files, not readable tables: a source such as suspend_d "
        "reaches the Agent only through derived columns and has no entry of its own"
    ),
    "unknown_unit_policy": (
        "status 'unknown' columns may be used only for scale-agnostic operations inside "
        "their own dataset, such as ranking or quantiles; absolute thresholds, unit "
        "conversion, and cross-dataset arithmetic require explicitly resolving the unit "
        "first. A snapshot column absent from this table is a broken data contract: do "
        "not use it at all and report it"
    ),
}


def normalize_daily_units(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Apply the daily unit contract and return conversion metadata."""
    return _normalize_units(frame, DAILY_UNIT_CONVERSIONS)


def normalize_auction_units(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Apply the opening-auction unit contract and return conversion metadata."""
    return _normalize_units(frame, AUCTION_UNIT_CONVERSIONS)


def _normalize_units(
    frame: pd.DataFrame,
    conversions_spec: tuple[tuple[str, float, str], ...],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frame = frame.copy()
    conversions: list[dict[str, object]] = []
    for column, factor, rule in conversions_spec:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * factor
            conversions.append({"column": column, "factor": factor, "rule": rule})
    return frame, conversions
