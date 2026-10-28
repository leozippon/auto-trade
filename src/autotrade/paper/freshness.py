"""Whether the release a Paper run pinned holds every dataset it reads, fresh.

Every mutating job commits a generation, a failed one included: the
partitions it wrote are complete versions. The newest committed release can
therefore hold the evening's daily bars beside event or financial tables a
failed later step left a session behind, or a dataset whose newest partition
the vendor answered defectively (recorded and skipped while the job went on).
A decision reads all of them, so the run is refused, before any view is
built, unless the pinned generation records for every dataset the book mounts

- a refresh at or after the launch of the last run of the dataset's landing
  job that was due before the decision's inference time (a refresh is a run
  whose whole pull of the dataset completed without a defective or staged
  partition), and
- no open defect and no staged version: either means the visible partition
  is not the vendor's current answer.

A pending history re-pull does not refuse. It follows a restatement the run
accepted, so the dataset's window months are current and its refresh has
advanced; only months outside the window await convergence, as they did for
every run before versioning. It stays an open issue for console health.

A dataset's landing jobs are the refresh nodes the replay gates it on
(``contracts``), with each node's launch time. A run of one is due on the
days its window moves: every calendar day for a job whose window is the
calendar day, and only SSE sessions for one whose window ends on a session
(``end_date_mode: sse_open_on_or_before`` in the schedule), whose runs on
other days are skipped as already done. So an 08:30 decision needs the
evening job of the previous session and the 09:45 disclosure job of the
previous day, never the run of its own morning after 08:30.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from autotrade.data_sources.tushare.common import STK_MINS_BY_DATE_DATASET, TEXT_SPECS
from autotrade.environment.data.contracts import (
    DOMAIN_REFRESH_NODES,
    EVENING_NODE,
    EVENT_DATASET_REFRESH_NODES,
    GENERATION_SCHEMA_VERSION,
    LOCALLY_DERIVED_DATASETS,
    MACRO_DATASET_REFRESH_NODES,
    REFRESH_NODES,
    TEXT_DATASET_REFRESH_NODES,
    TEXT_NODE,
)
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.strategy import CN_TZ
from autotrade.pipelines.pit_backend import required_release_raw_datasets

from .engine import PaperDataNotReady

SESSION_WINDOW = "sse_open_on_or_before"
# The open issues that make a mounted dataset unready (see the docstring).
REFUSING_ISSUES = frozenset({"defect", "staged"})


def landing_nodes(config: SnapshotConfig) -> dict[str, tuple[str, ...]]:
    """The refresh nodes that land each raw dataset ``config`` mounts and
    records a refresh of; frozen text sources and the minute-derived tables
    have none."""

    nodes: dict[str, tuple[str, ...]] = {}
    for dataset in required_release_raw_datasets(config):
        if dataset in LOCALLY_DERIVED_DATASETS or (dataset in TEXT_SPECS and TEXT_SPECS[dataset].frozen_through):
            continue
        if dataset in config.events_datasets:
            nodes[dataset] = EVENT_DATASET_REFRESH_NODES.get(dataset, (EVENING_NODE,))
        elif dataset in config.macro_datasets:
            nodes[dataset] = MACRO_DATASET_REFRESH_NODES.get(dataset, (EVENING_NODE,))
        elif dataset in config.text_datasets:
            nodes[dataset] = TEXT_DATASET_REFRESH_NODES.get(dataset, (TEXT_NODE,))
        elif dataset == STK_MINS_BY_DATE_DATASET:
            nodes[dataset] = DOMAIN_REFRESH_NODES["intraday_1min"]
        else:
            # The daily core, and the raw financial tables the evening job's
            # fundamental step lands for the 03:35 PIT build.
            nodes[dataset] = (EVENING_NODE,)
    return nodes


def due_launch(node: str, schedule: Mapping[str, object], sessions: frozenset[str], due_at: datetime) -> datetime | None:
    """The launch of the last run of ``node``'s job due before ``due_at``."""

    spec = schedule["jobs"][node]
    by_session = spec.get("end_date_mode") == SESSION_WINDOW
    refresh = REFRESH_NODES[node]
    day = due_at.astimezone(CN_TZ).date()
    # The longest exchange closure (Spring Festival, National Day) is nine days.
    for _ in range(31):
        if (not by_session or day.strftime("%Y%m%d") in sessions) and refresh.ready_at(day) <= due_at:
            return refresh.start_at(day)
        day -= timedelta(days=1)
    return None


def unready_datasets(
    generation: Mapping[str, object],
    config: SnapshotConfig,
    schedule: Mapping[str, object],
    sessions: Sequence[str],
    due_at: datetime,
) -> list[str]:
    """One line per reason the pinned generation is not fresh for the
    decision at ``due_at``; empty when it is."""

    if generation.get("schema_version") != GENERATION_SCHEMA_VERSION:
        return [
            (
                f"its record (schema {generation.get('schema_version')}) predates per-dataset refresh times; "
                "rebuild the generation"
            )
        ]
    entries = generation.get("datasets") or {}
    open_days = frozenset(sessions)
    lines: list[str] = []
    for dataset, nodes in sorted(landing_nodes(config).items()):
        launches = [(launch, node) for node in nodes if (launch := due_launch(node, schedule, open_days, due_at))]
        if not launches:
            continue
        launch, node = max(launches)
        refreshed = str((entries.get(dataset) or {}).get("refreshed_at") or "")
        if not refreshed or datetime.fromisoformat(refreshed) < launch:
            when = datetime.fromisoformat(refreshed).astimezone(CN_TZ).strftime("%Y-%m-%d %H:%M") if refreshed else "never"
            lines.append(f"{dataset} last refreshed {when}, before the {node} run of {launch:%Y-%m-%d %H:%M}")
    mounted = set(required_release_raw_datasets(config))
    for issue in generation.get("issues") or []:
        if isinstance(issue, dict) and issue.get("dataset") in mounted and issue.get("kind") in REFUSING_ISSUES:
            where = f" in {issue['partition']}" if issue.get("partition") else ""
            detail = issue.get("detail") or issue.get("version") or ""
            lines.append(f"{issue['dataset']} has an open {issue.get('kind')}{where}" + (f" ({detail})" if detail else ""))
    return lines


def require_fresh_release(
    generation: Mapping[str, object],
    config: SnapshotConfig,
    schedule: Mapping[str, object],
    sessions: Sequence[str],
    due_at: datetime,
) -> None:
    """Raise ``PaperDataNotReady`` naming every stale dataset and every open
    defect or staged version of a mounted dataset (see the module docstring)."""

    lines = unready_datasets(generation, config, schedule, sessions, due_at)
    if lines:
        raise PaperDataNotReady(
            f"generation {generation.get('generation_id')} is not ready for the decision at "
            f"{due_at.astimezone(CN_TZ):%Y-%m-%d %H:%M}: " + "; ".join(lines)
        )


__all__ = ["due_launch", "landing_nodes", "require_fresh_release", "unready_datasets"]
