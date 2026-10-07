#!/usr/bin/env python3
"""Judge every stored reading of the current-rule arms again and compare.

The consistency check of the gate code (``pipelines/verdict.py``,
``config.py``, ``experiment.py``). An arm keeps everything its readings were
read from -- ``hitl/params.json``, the ledger's Steps with their style
sidecars and revisions, the lineage series, the replays' results -- so each
reading of an arm created under the current rules is judged again by this
code, CPU only, and compared with the ledger key for key:

- a research session's freeze gate (``experiment.freeze_gate_for``),
- a completed forward record's slices and verdict (``experiment.judged_replay``),
- an incubation record's entry gate (``experiment.incubation_entry``) and the
  slices and verdict of its replay,
- a lineage record's figures (``experiment.lineage_summary``).

An arm created before ``CURRENT_RULES_SINCE`` was judged under rules since
retired and is never judged again: its readings are what its ledger says. A
reading written before the switches were retired states the two boolean
ones in its thresholds (``config.RETIRED_SWITCHES``); those keys are dropped
from the stored side and nothing else is.

Run it by hand before landing a change to the gate code, from a checkout
whose ``experiments/`` holds the arms, or with ``--experiments``:

    python scripts/dev/check_verdicts.py

Exit status 1 on any difference. Nothing is written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.environment.broker import BrokerProfile
from autotrade.pipelines.calendar import GEOMETRY_PARAMETERS, ResearchGeometry
from autotrade.pipelines.config import (
    DEFAULT_RESEARCH_GEOMETRY,
    RETIRED_SWITCHES,
    AcceptanceRules,
    RollingExperimentConfig,
    acceptance_for,
)
from autotrade.pipelines.experiment import (
    freeze_gate_for,
    incubation_entry,
    judged_replay,
    lineage_summary,
    replay_and_analysis,
)
from autotrade.pipelines.hitl_state import HITL_DIR_NAME, PARAMS_NAME

# When the current rules took effect: the last of the switches since folded
# into them (pricing declared offline screens as independent, bca4f5e) was
# stamped on every arm created from this instant on.
CURRENT_RULES_SINCE = datetime.fromisoformat("2026-10-06T02:21:56+00:00")


class Judgement(NamedTuple):
    """One stored reading judged again: where it is, and the paths at which
    the stored reading and today's differ (none when it reproduces)."""

    arm: str
    reading: str
    differences: list[str]


def current_arms(experiments: Path) -> list[Path]:
    """The arms under ``experiments`` created under the current rules."""

    arms = []
    for path in sorted(Path(experiments).iterdir()):
        params = path / HITL_DIR_NAME / PARAMS_NAME
        if not params.is_file():
            continue
        created = json.loads(params.read_text(encoding="utf-8")).get("_created_at")
        if created and datetime.fromisoformat(str(created)) >= CURRENT_RULES_SINCE:
            arms.append(path)
    return arms


def _stored(value: object) -> Any:
    """``value`` as a ledger line stores it: through JSON."""

    return json.loads(json.dumps(value))


def _without_retired(value: object) -> object:
    """A stored reading without the retired switches its thresholds state."""

    if isinstance(value, Mapping):
        return {
            key: (
                {name: item for name, item in inner.items() if name not in RETIRED_SWITCHES}
                if key == "thresholds" and isinstance(inner, Mapping)
                else _without_retired(inner)
            )
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [_without_retired(item) for item in value]
    return value


def differences(stored: object, now: object, path: str = "") -> list[str]:
    """The paths at which ``now`` does not hold ``stored``: a key on one side
    only, a value of another type, or another value (NaN equals NaN)."""

    def at(key: object) -> str:
        return f"{path}.{key}" if path else str(key)

    if isinstance(stored, dict) and isinstance(now, dict):
        return [
            item
            for key in sorted({*stored, *now}, key=str)
            for item in (
                differences(stored[key], now[key], at(key))
                if key in stored and key in now
                else [at(key)]
            )
        ]
    if isinstance(stored, list) and isinstance(now, list) and len(stored) == len(now):
        return [
            item
            for index, pair in enumerate(zip(stored, now, strict=True))
            for item in differences(*pair, f"{path}[{index}]")
        ]
    nan = isinstance(stored, float) and isinstance(now, float) and math.isnan(stored) and math.isnan(now)
    return [] if type(stored) is type(now) and (stored == now or nan) else [path or "the reading"]


def _replayed(
    record: Mapping[str, Any], rules: AcceptanceRules, slippage_bps: float
) -> dict[str, object]:
    """The slices and verdict of one stored forward replay (a forward record
    or an incubation's reading), judged again."""

    replay, analysis = replay_and_analysis(str(record["result_ref"]))
    spans = record["replay"]
    return judged_replay(
        replay,
        analysis,
        rules=rules,
        forward=(spans["start"], spans["forward_end"]),
        heldout=(spans["heldout_start"], spans["replay_end"]),
        seed_key=str(record["artifact_id"]),
        slippage_bps=slippage_bps,
        seed_replicates=[
            (
                {key: row[key] for key in ("artifact_id", "source_step_id")},
                replay_and_analysis(str(row["result_ref"]))[1],
            )
            for row in record.get("seed_replicates") or ()
        ],
    )


def check_arm(arm: Path) -> list[Judgement]:
    """Every stored reading of one arm against today's reading of it."""

    params = json.loads((arm / HITL_DIR_NAME / PARAMS_NAME).read_text(encoding="utf-8"))
    ledger = arm / "ledgers" / "experiment_ledger.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines() if ledger.is_file() else []
    records = [json.loads(line) for line in lines if line]
    rules = acceptance_for(params)
    geometry = ResearchGeometry(
        **{name: params.get(name, getattr(DEFAULT_RESEARCH_GEOMETRY, name)) for name in GEOMETRY_PARAMETERS}
    )
    years = [(slot.start, slot.end) for slot in geometry.research_years]
    slippage = float(params.get("slippage_bps", BrokerProfile().slippage_bps))
    found: list[Judgement] = []

    def judge(reading: str, stored: object, now: object) -> None:
        found.append(Judgement(arm.name, reading, differences(_without_retired(stored), _stored(now))))

    for index, record in enumerate(records):
        kind = record.get("record_type")
        if kind == "research_session" and isinstance(record.get("freeze_gate"), dict):
            steps = {str(row["step_id"]): row for row in record["steps"]}
            nominee = steps[str(record["nominated_step_id"])]
            named = (record["freeze_gate"].get("seed_replicates") or {}).get("replicates") or ()
            judge(
                "freeze_gate",
                record["freeze_gate"],
                freeze_gate_for(
                    records[:index],
                    list(steps.values()),
                    nominee,
                    experiment_dir=arm,
                    acceptance=rules,
                    hard_reasons=rules.evaluate(dict(nominee["summary"])),
                    years=years,
                    seed_replicates=[steps[str(entry["step_id"])] for entry in named],
                ),
            )
        elif kind == "forward" and record.get("status") == "ok":
            judge(
                "forward",
                {key: record[key] for key in ("slices", "verdict")},
                _replayed(record, rules, slippage),
            )
        elif kind == "incubation":
            config = RollingExperimentConfig(
                experiment_id=arm.name,
                experiments_root=arm.parent,
                geometry=geometry,
                acceptance=rules,
            )
            replicates = [
                str(item["source_step_id"]) for item in record["frozen"].get("seed_replicates") or ()
            ]
            entry = incubation_entry(records, str(record["source_step_id"]), replicates, config=config)
            judge("incubation.entry", record["entry"], entry["entry"])
            reading = record["forward_reading"]
            if reading.get("status") == "ok":
                judge(
                    "incubation.forward_reading",
                    {key: reading[key] for key in ("slices", "verdict")},
                    _replayed(reading, rules, slippage),
                )
        elif kind == "lineage":
            summary = lineage_summary(json.loads(Path(str(record["series_ref"])).read_text(encoding="utf-8")))
            judge("lineage", {key: record[key] for key in summary}, summary)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--experiments", type=Path, default=REPO_ROOT / "experiments")
    args = parser.parse_args(argv)
    arms = current_arms(args.experiments)
    failed = 0
    count = 0
    for arm in arms:
        judgements = check_arm(arm)
        if not judgements:
            print(f"{'-':<10} {arm.name}: no stored reading")
        for item in judgements:
            count += 1
            failed += bool(item.differences)
            detail = f": differs at {item.differences[:20]}" if item.differences else ""
            print(f"{'different' if item.differences else 'identical':<10} {item.reading:<27} {item.arm}{detail}")
    print(
        f"{len(arms)} arms created since {CURRENT_RULES_SINCE.isoformat()}: "
        f"{count} stored readings, {count - failed} identical, {failed} different"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
