"""Lineage: the earlier arms whose trials join an arm's freeze-gate family.

docs/pipeline-design.md §2.3. An arm that continues a family earlier arms
explored on the same research period is created with ``lineage_arms``; their
non-control research Validations and declared offline screens are trials of
the selection this arm's nominee comes out of, so the deflated Sharpe counts
them with the arm's own (``experiment.freeze_gate_for``).

The lineage is read from the earlier arms once, when the console creates the
arm, into a compact series file beside the arm's ledger holding the per-arm
counts and, for each measurable non-control revision, the daily neutralised
graded series the gate correlates (``verdict.neutral_daily``) -- the same
series and the same representative Validation (full span when there is one)
the gate reads for the arm's own trials. From then on nothing reads the
lineage arms' directories. Only their research sessions' Validations are read,
never a forward or Held-out replay, and every extracted day must lie inside the
research period. The ``lineage`` ledger record is the pipeline's to write, from
this file, before the research session (``experiment.lineage_ledger_record``).

Which earlier arms informed a new one is the round author's declaration; the
host cannot know it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from .config import DEFAULT_RESEARCH_GEOMETRY
from .experiment import (
    _recorded_steps,
    _style_analysis,
    trial_family,
    trial_representatives,
)
from .hitl_state import read_json
from .ledger import LINEAGE_SERIES_NAME, ExperimentLedger
from .verdict import neutral_daily

_ARM_ID = re.compile(r"[A-Za-z0-9_-]+")


def lineage_arm_ids(value: object, experiment_id: str) -> tuple[str, ...]:
    """The ``lineage_arms`` parameter as experiment ids, or ``ValueError``."""

    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError("lineage_arms must be a list of experiment ids")
    arms = tuple(value)
    malformed = [arm for arm in arms if not _ARM_ID.fullmatch(arm)]
    if malformed:
        raise ValueError(f"lineage_arms holds names that are not experiment ids: {malformed}")
    repeated = sorted({arm for arm in arms if arms.count(arm) > 1})
    if repeated:
        raise ValueError(f"lineage_arms lists {', '.join(repeated)} more than once")
    if experiment_id in arms:
        raise ValueError(f"lineage_arms lists the arm itself ({experiment_id})")
    return arms


def extract_lineage(
    experiments_root: str | Path,
    arms: Sequence[str],
    *,
    research_start: str,
    research_end: str,
) -> dict[str, object]:
    """What the freeze gate needs from each lineage arm, or ``ValueError``
    naming the arm that cannot be one.

    A lineage arm must exist, have researched exactly this research period and
    have recorded at least one non-control trial. Per arm: its trial family
    (``experiment.trial_family`` over its research session's Validations, so
    controls are left out and declared offline screens count, exactly as for
    the arm's own). Per measurable non-control revision: its daily series. A
    revision whose span cannot be measured is a trial with no series, as in
    the arm's own family.
    """

    root = Path(experiments_root)
    extracted: list[dict[str, object]] = []
    series: list[dict[str, object]] = []
    for arm in arms:
        directory = root / arm
        params_path = directory / "hitl" / "params.json"
        if not params_path.is_file():
            raise ValueError(f"lineage arm {arm} does not exist")
        params = read_json(params_path)
        period = (
            str(params.get("research_start", DEFAULT_RESEARCH_GEOMETRY.research_start)),
            str(params.get("research_end", DEFAULT_RESEARCH_GEOMETRY.research_end)),
        )
        if period != (research_start, research_end):
            raise ValueError(
                f"lineage arm {arm} researched {period[0]}..{period[1]}, not this arm's "
                f"{research_start}..{research_end}"
            )
        rows = _recorded_steps(ExperimentLedger(directory / "ledgers" / "experiment_ledger.jsonl").read())
        family = trial_family(rows)
        revisions = list(family["revisions"])  # type: ignore[call-overload]
        if not revisions:
            raise ValueError(f"lineage arm {arm} has no recorded non-control trial")
        representative = trial_representatives(rows)
        for revision in revisions:
            try:
                daily = neutral_daily(_style_analysis(representative[revision]))
            except ValueError:
                continue
            outside = sorted(day for day in daily if not research_start <= day <= research_end)
            if outside:
                raise ValueError(
                    f"lineage arm {arm} revision {revision} has days outside the research "
                    f"period ({outside[0]}..{outside[-1]})"
                )
            series.append(
                {
                    "experiment_id": arm,
                    "revision_id": revision,
                    "span": representative[revision].get("span"),
                    "daily": [[day, value] for day, value in daily.items()],
                }
            )
        extracted.append(
            {
                "experiment_id": arm,
                "host_trials": len(revisions),
                "offline_trials": family["offline_trials"],
                "controls": family["controls"],
                "undeclared_offline_validations": family["undeclared_offline_validations"],
            }
        )
    return {"arms": extracted, "series": series}


def write_lineage(experiment_dir: str | Path, extraction: Mapping[str, object]) -> Path:
    """Write the extracted lineage beside the arm's ledger, once, while the
    console creates the arm; returns the file.

    Only this file: the ledger stays empty until the arm runs, since the
    release pin and the identity store read a ledger with records and no pin
    as a legacy arm that already ran.
    """

    path = Path(experiment_dir) / "ledgers" / LINEAGE_SERIES_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compact: indented, the per-day pairs would triple the file.
    staging = path.with_name(f".{path.name}.tmp")
    staging.write_text(
        json.dumps(extraction, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    staging.replace(path)
    return path


__all__ = [
    "extract_lineage",
    "lineage_arm_ids",
    "write_lineage",
]
