#!/usr/bin/env python3
"""Judge every stored reading of the current-rule arms again and compare.

The consistency check of the gate code (``pipelines/verdict.py``,
``config.py``, ``experiment.py``). An arm keeps everything its readings were
read from -- ``hitl/params.json``, the ledger's Steps with their style
sidecars and revisions, the replays' results -- so each reading of an arm
created under the current rules is judged again by this code, CPU only, and
compared with the ledger key for key:

- a research session's freeze registration (``experiment.freeze_registration``),
- a forward record's sealed reading (``experiment.judged_sealed``), its
  ablation's increment where it froze one (``experiment.judged_ablation``)
  and, where its forward replay completed, its slices and verdict
  (``experiment.judged_replay``).

An arm whose rules state no ``sealed_years`` was judged under rules since
retired and is never judged again: its readings are what its ledger says.

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
from pathlib import Path
from typing import Any, NamedTuple

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.common.records import read_json_object
from autotrade.environment.broker import BrokerProfile
from autotrade.pipelines.config import AcceptanceRules, acceptance_for
from autotrade.pipelines.experiment import (
    freeze_registration,
    judged_ablation,
    judged_replay,
    judged_sealed,
    replay_and_analysis,
)
from autotrade.pipelines.hitl_state import HITL_DIR_NAME, PARAMS_NAME
from autotrade.pipelines.ledger import recorded_path


class Judgement(NamedTuple):
    """One stored reading judged again: where it is, and the paths at which
    the stored reading and today's differ (none when it reproduces)."""

    arm: str
    reading: str
    differences: list[str]


def current_arms(experiments: Path) -> list[Path]:
    """The arms under ``experiments`` judged under the current rules: those
    whose rules state ``sealed_years`` (the sealed gate's, which the session
    sees the visible research years under, a freeze registers and the
    frozen artifact's sealed reading judges). An arm created before then was
    judged under rules since retired, so its stored readings differ by the
    rule, not by a defect."""

    arms = []
    for path in sorted(Path(experiments).iterdir()):
        params = path / HITL_DIR_NAME / PARAMS_NAME
        if params.is_file() and read_json_object(params).get("sealed_years") is not None:
            arms.append(path)
    return arms


def _stored(value: object) -> Any:
    """``value`` as a ledger line stores it: through JSON."""

    return json.loads(json.dumps(value))


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


def _sealed(arm: Path, record: Mapping[str, Any], rules: AcceptanceRules) -> dict[str, object]:
    """The sealed block of one stored forward record of ``arm``, judged again."""

    sealed = record["sealed"]
    return judged_sealed(
        str(recorded_path(arm, str(sealed["result_ref"]))),
        rules=rules,
        start=str(sealed["start"]),
        end=str(sealed["end"]),
        seed_replicates=[
            (
                {key: row[key] for key in ("artifact_id", "source_step_id")},
                str(recorded_path(arm, str(row["result_ref"]))),
            )
            for row in sealed.get("seed_replicates") or ()
            if row.get("result_ref")
        ],
    )


def _replayed(
    arm: Path,
    record: Mapping[str, Any],
    rules: AcceptanceRules,
    slippage_bps: float,
    sealed: Mapping[str, object],
) -> dict[str, object]:
    """The slices and verdict of one stored forward replay of ``arm``,
    judged again with its ``sealed`` block."""

    replay, analysis = replay_and_analysis(str(recorded_path(arm, str(record["result_ref"]))))
    spans = record["replay"]
    return judged_replay(
        replay,
        analysis,
        rules=rules,
        sealed=sealed,
        forward=(spans["start"], spans["forward_end"]),
        heldout=(spans["heldout_start"], spans["replay_end"]),
        seed_key=str(record["artifact_id"]),
        slippage_bps=slippage_bps,
        seed_replicates=[
            (
                {key: row[key] for key in ("artifact_id", "source_step_id")},
                replay_and_analysis(str(recorded_path(arm, str(row["result_ref"]))))[1],
            )
            for row in record.get("seed_replicates") or ()
            if row.get("result_ref")
        ],
        strategy_errors=(
            ("replicate",)
            if any(
                row.get("strategy_error")
                for row in (*(record["sealed"].get("seed_replicates") or ()), *(record.get("seed_replicates") or ()))
            )
            else ()
        ),
    )


def check_arm(arm: Path) -> list[Judgement]:
    """Every stored reading of one arm against today's reading of it."""

    params = read_json_object(arm / HITL_DIR_NAME / PARAMS_NAME)
    ledger = arm / "ledgers" / "experiment_ledger.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines() if ledger.is_file() else []
    records = [json.loads(line) for line in lines if line]
    rules = acceptance_for(params)
    slippage = float(params.get("slippage_bps", BrokerProfile().slippage_bps))
    found: list[Judgement] = []

    def judge(reading: str, stored: object, now: object) -> None:
        found.append(Judgement(arm.name, reading, differences(stored, _stored(now))))

    for record in records:
        kind = record.get("record_type")
        if kind == "research_session" and isinstance(record.get("freeze_gate"), dict):
            steps = {str(row["step_id"]): row for row in record["steps"]}
            nominee = steps[str(record["nominated_step_id"])]
            seeds = record["freeze_gate"].get("seed_replicates") or {}
            ablation = record["freeze_gate"].get("ablation") or {}
            judge(
                "freeze_registration",
                record["freeze_gate"],
                freeze_registration(
                    nominee,
                    experiment_dir=arm,
                    acceptance=rules,
                    seed_replicates=[steps[str(entry["step_id"])] for entry in seeds.get("replicates") or ()],
                    deterministic=bool(seeds.get("deterministic")),
                    ablation=steps[str(ablation["step_id"])] if "step_id" in ablation else None,
                ),
            )
        elif kind == "forward" and isinstance(record.get("sealed"), Mapping):
            stored = record["sealed"]["slice"]
            sealed = stored if stored.get("strategy_error") else _sealed(arm, record, rules)
            judge("sealed", stored, sealed)
            control = record["sealed"].get("ablation") or {}
            if control.get("result_ref"):
                judge(
                    "ablation",
                    control["increment"],
                    judged_ablation(
                        str(recorded_path(arm, str(record["sealed"]["result_ref"]))),
                        str(recorded_path(arm, str(control["result_ref"]))),
                        rules=rules,
                        start=str(record["sealed"]["start"]),
                        end=str(record["sealed"]["end"]),
                    )["increment"],
                )
            if record.get("status") == "ok":
                judge(
                    "forward",
                    {key: record[key] for key in ("slices", "verdict")},
                    _replayed(arm, record, rules, slippage, sealed),
                )
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
        f"{len(arms)} arms under the sealed gate: "
        f"{count} stored readings, {count - failed} identical, {failed} different"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
