#!/usr/bin/env python3
"""Judge every stored reading of the current-rule experiments again and compare.

The consistency check of the reading code (``pipelines/verdict.py``,
``validation_record.py``, ``config.py``, ``experiment.py``). An experiment keeps everything its readings were
read from -- ``hitl/params.json``, the ledger's Steps with their style
sidecars and revisions, the replays' results -- so each reading of an experiment
created under the current rules is judged again by this code, CPU only, and
compared with the ledger key for key:

- a research session's final registration (``experiment.check_registration``),
- a forward record's slices, where its forward replay completed
  (``experiment.judged_replay``), and its readout off the ledger's rolling
  record and that forward slice (``validation_record.readout``).

An experiment whose rules state no ``research_years`` was judged under rules since
retired (the sealed gate) and is never judged again: its readings are what
its ledger says.

Run it by hand before landing a change to the reading code, from a checkout
whose ``experiments/`` holds the experiments, or with ``--experiments``:

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
from autotrade.pipelines.config import AcceptanceRules, acceptance_for
from autotrade.pipelines.experiment import (
    check_registration,
    judged_replay,
    replay_and_analysis,
)
from autotrade.pipelines.experiment_params import read_param
from autotrade.pipelines.hitl_state import HITL_DIR_NAME, PARAMS_NAME
from autotrade.pipelines.ledger import recorded_path
from autotrade.pipelines.validation_record import readout


class Judgement(NamedTuple):
    """One stored reading judged again: where it is, and the paths at which
    the stored reading and today's differ (none when it reproduces)."""

    experiment: str
    reading: str
    differences: list[str]


def current_experiments(root: Path) -> list[Path]:
    """The experiments under ``experiments`` judged under the current rules: those
    whose rules state ``research_years`` (the rolling validation's, which the
    session sees the research years under, every test registers against and
    the readout reads). An experiment created before then was judged under rules
    since retired, so its stored readings differ by the rule, not by a
    defect."""

    experiments = []
    for path in sorted(Path(root).iterdir()):
        params = path / HITL_DIR_NAME / PARAMS_NAME
        if params.is_file() and read_json_object(params).get("research_years") is not None:
            experiments.append(path)
    return experiments


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


def _replayed(
    experiment: Path, record: Mapping[str, Any], rules: AcceptanceRules, slippage_bps: float
) -> dict[str, object]:
    """The slices of one stored forward replay of ``experiment``, read again."""

    replay, analysis = replay_and_analysis(str(recorded_path(experiment, str(record["result_ref"]))))
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
                replay_and_analysis(str(recorded_path(experiment, str(row["result_ref"]))))[1],
            )
            for row in record.get("seed_replicates") or ()
            if row.get("result_ref")
        ],
    )["slices"]


def check_experiment(experiment: Path) -> list[Judgement]:
    """Every stored reading of one experiment against today's reading of it."""

    params = read_json_object(experiment / HITL_DIR_NAME / PARAMS_NAME)
    ledger = experiment / "ledgers" / "experiment_ledger.jsonl"
    lines = ledger.read_text(encoding="utf-8").splitlines() if ledger.is_file() else []
    records = [json.loads(line) for line in lines if line]
    rules = acceptance_for(params)
    slippage = float(read_param(params, "slippage_bps"))  # type: ignore[arg-type]
    found: list[Judgement] = []

    def judge(reading: str, stored: object, now: object) -> None:
        found.append(Judgement(experiment.name, reading, differences(stored, _stored(now))))

    for record in records:
        kind = record.get("record_type")
        if kind == "research_session" and isinstance(record.get("freeze_gate"), dict):
            steps = {str(row["step_id"]): row for row in record["steps"]}
            nominee = steps[str(record["nominated_step_id"])]
            seeds = record["freeze_gate"].get("seed_replicates") or {}
            judge(
                "check_registration",
                record["freeze_gate"],
                check_registration(
                    nominee,
                    experiment_dir=experiment,
                    acceptance=rules,
                    seed_replicates=[steps[str(entry["step_id"])] for entry in seeds.get("replicates") or ()],
                    deterministic=bool(seeds.get("deterministic")),
                ),
            )
        elif kind == "forward" and isinstance(record.get("readout"), Mapping):
            # The readout reads the ledger as it stood when the record was
            # written: the test steps before it, and the record's own slice.
            before = records[: records.index(record)]
            if record.get("status") == "ok":
                slices = _replayed(experiment, record, rules, slippage)
                judge("forward", record["slices"], slices)
                now = readout(before, alpha=rules.alpha, forward=slices["forward"])
            else:
                now = readout(before, alpha=rules.alpha, strategy_error=record["readout"]["forward"]["strategy_error"])
            judge("readout", record["readout"], now)
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--experiments", type=Path, default=REPO_ROOT / "experiments")
    args = parser.parse_args(argv)
    experiments = current_experiments(args.experiments)
    failed = 0
    count = 0
    for experiment in experiments:
        judgements = check_experiment(experiment)
        if not judgements:
            print(f"{'-':<10} {experiment.name}: no stored reading")
        for item in judgements:
            count += 1
            failed += bool(item.differences)
            detail = f": differs at {item.differences[:20]}" if item.differences else ""
            print(f"{'different' if item.differences else 'identical':<10} {item.reading:<27} {item.experiment}{detail}")
    print(
        f"{len(experiments)} experiments under the rolling validation: "
        f"{count} stored readings, {count - failed} identical, {failed} different"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
