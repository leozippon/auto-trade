#!/usr/bin/env python3
"""Judge every stored reading of the experiments again and compare.

The consistency check of the reading code (``pipelines/verdict.py``,
``validation_record.py``, ``config.py``, ``experiment.py``). An experiment keeps everything its readings were
read from -- ``hitl/params.json``, the ledger's Steps with their style
sidecars and revisions, the replays' results -- so each reading is judged again
by this code, CPU only, and compared with the ledger key for key (floats within
``REL_TOL``):

- a research session's final registration (``experiment.check_registration``),
- a forward record's slices, where its forward replay completed
  (``experiment.judged_replay``), and its readout off the ledger's
  validation record and that forward slice (``validation_record.readout``).

Run it by hand before landing a change to the reading code, from a checkout
whose ``experiments/`` holds the experiments, or with ``--experiments``:

    python scripts/dev/check_verdicts.py

Exit status 1 on any difference, each printed with its stored and today's
value. Nothing is written.
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
from autotrade.pipelines.ledger import ExperimentLedger, recorded_path
from autotrade.pipelines.validation_record import readout

# Floats are equal within this relative tolerance. A reading recorded on the
# remote GPU instance and judged again on this host differs in the last bits
# of its floats: at most 1.6e-14 relative (139 ulps) over the eight such
# experiments of 2026-11, while those recorded on this host reproduce bit for
# bit. A real change of a reading is many orders of magnitude larger.
REL_TOL = 1e-12

# The seed-replicate gate c76a687d retired (2026-11-07): a registration stored
# before it keeps two flags beside its replicates that no reading emits now and
# no pass condition reads. Only a stored key that today's reading lacks is
# ignored.
RETIRED_KEYS = frozenset({"seed_replicates.deterministic", "seed_replicates.trains_a_model"})


class _Absent:
    """The side of a key that only the other side holds."""

    def __repr__(self) -> str:
        return "absent"


_ABSENT = _Absent()


class Judgement(NamedTuple):
    """One stored reading judged again: where it is, and the leaves at which
    the stored reading and today's differ (none when it reproduces)."""

    experiment: str
    reading: str
    differences: list[tuple[str, object, object]]


def current_experiments(root: Path) -> list[Path]:
    """The experiments under ``root``: every directory holding its params."""

    return [path for path in sorted(Path(root).iterdir()) if (path / HITL_DIR_NAME / PARAMS_NAME).is_file()]


def _stored(value: object) -> Any:
    """``value`` as a ledger line stores it: through JSON."""

    return json.loads(json.dumps(value))


def differences(stored: object, now: object, path: str = "") -> list[tuple[str, object, object]]:
    """The leaves at which ``now`` does not hold ``stored``, each as ``(path,
    stored, now)``: a key on one side only (``absent`` on the other), a value of
    another type, or another value. Floats are equal within ``REL_TOL`` and NaN
    equals NaN; every other value must be equal exactly."""

    def at(key: object) -> str:
        return f"{path}.{key}" if path else str(key)

    if isinstance(stored, dict) and isinstance(now, dict):
        return [
            item
            for key in sorted({*stored, *now}, key=str)
            for item in differences(stored.get(key, _ABSENT), now.get(key, _ABSENT), at(key))
        ]
    if isinstance(stored, list) and isinstance(now, list) and len(stored) == len(now):
        return [
            item
            for index, pair in enumerate(zip(stored, now, strict=True))
            for item in differences(*pair, f"{path}[{index}]")
        ]
    if isinstance(stored, float) and isinstance(now, float):
        same = math.isclose(stored, now, rel_tol=REL_TOL) or (math.isnan(stored) and math.isnan(now))
    else:
        same = type(stored) is type(now) and stored == now
    return [] if same else [(path or "the reading", stored, now)]


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
    records = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl").read()
    rules = acceptance_for(params)
    slippage = float(read_param(params, "slippage_bps"))  # type: ignore[arg-type]
    found: list[Judgement] = []

    def judge(reading: str, stored: object, now: object) -> None:
        kept = [
            item
            for item in differences(stored, _stored(now))
            if not (item[0] in RETIRED_KEYS and item[2] is _ABSENT)
        ]
        found.append(Judgement(experiment.name, reading, kept))

    for record in records:
        kind = record.get("record_type")
        if kind == "research_session" and isinstance(record.get("registration"), dict):
            steps = {str(row["step_id"]): row for row in record["steps"]}
            nominee = steps[str(record["nominated_step_id"])]
            seeds = record["registration"].get("seed_replicates") or {}
            judge(
                "check_registration",
                record["registration"],
                check_registration(
                    nominee,
                    experiment_dir=experiment,
                    acceptance=rules,
                    seed_replicates=[steps[str(entry["step_id"])] for entry in seeds.get("replicates") or ()],
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
            detail = "".join(
                f"\n    {path}: stored {stored!r}, now {now!r}" for path, stored, now in item.differences[:20]
            )
            print(f"{'different' if item.differences else 'identical':<10} {item.reading:<27} {item.experiment}{detail}")
    print(
        f"{len(experiments)} experiments: "
        f"{count} stored readings, {count - failed} identical, {failed} different"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
