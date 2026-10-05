#!/usr/bin/env python3
"""Recompute every recorded freeze gate and forward verdict from its stored inputs.

The standing regression check of the gate code (``pipelines/verdict.py``,
``config.py``, ``experiment.py``). An arm keeps everything its judgements were
read from -- ``hitl/params.json``, the ledger's Steps with their style
sidecars and revisions, the lineage series, the forward replay's result -- so
every judgement can be read again without a replay: CPU only, a few minutes
over seventy arms.

Records mode (the default) recomputes each recorded ``freeze_gate`` and each
recorded forward ``verdict`` and ``slices`` under the rules its own record
states and compares every recorded key. A key the code has added since is
listed, not counted. A judgement the code no longer reproduces fails unless
``DECLARED_EXCEPTIONS`` names it with its reason and the exact keys that
differ; a declared exception that stops matching fails too.

Differential mode (``--baseline REV``) answers whether a code change moves any
judgement. REV's tree is extracted to a temporary directory, where its own
copy of this script dumps everything it recomputes; this tree dumps the same;
the two dumps must be equal key for key, key order included. The dump is
wider than the records: every recorded Step taken as the nominee and every
forward replay judged again under every rule era (``ERAS``, so an era no
record has reached yet is covered), the full-span bar after every session,
every arm's resolved rules and the facts its Agent reads from them, and the
creation contract (the console's creation defaults, the accepted parameters).
The arms are live: one whose parameters or ledger were not the same bytes in
both readings is named and left out, and the run fails so it is made again.

Run both by hand before landing a change to the gate code, from a checkout
whose ``experiments/`` holds the arms, or with ``--experiments``:

    python scripts/dev/recompute_verdicts.py
    python scripts/dev/recompute_verdicts.py --baseline main

Exit status 1 on any difference. Nothing is written outside a temporary
directory (and ``--dump``'s file). ``tests/unit/test_recorded_verdicts.py``
runs both modes on a small synthetic arm.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import tarfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, NamedTuple

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.replay.stats import window_activity
from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME
from autotrade.pipelines.calendar import GEOMETRY_PARAMETERS, ResearchGeometry
from autotrade.pipelines.config import AcceptanceRules, acceptance_for
from autotrade.pipelines.experiment import freeze_gate_for, full_span_bar
from autotrade.pipelines.hitl_state import (
    HITL_DIR_NAME,
    PARAMS_NAME,
    WEB_CREATE_DEFAULTS,
)
from autotrade.pipelines.verdict import (
    forward_slice,
    graduation_verdict,
    heldout_slice,
    seed_replicate_slice,
)
from autotrade.pipelines.worker import _ALLOWED_PARAMS

# What a baseline's tree needs to dump: the code, and its own copy of this
# script, which knows that code's interfaces.
SCRIPT = "scripts/dev/recompute_verdicts.py"
BASELINE_PATHS = ("pyproject.toml", "src", "scripts/_bootstrap.py", SCRIPT)

_RAW = "require_raw_excess_at_cost_stress"
_PLAIN = "require_forward_plain_selection"
_SEEDS = "require_seed_replicates"
# The optional conditions each rule era holds. An arm carries the era of its
# creation in ``params.json``; the differential judges every stored input
# under each of them.
ERAS: dict[str, dict[str, bool]] = {
    "R0": {_RAW: False, _PLAIN: False, _SEEDS: False},
    "R1": {_RAW: True, _PLAIN: False, _SEEDS: False},
    "R2": {_RAW: True, _PLAIN: True, _SEEDS: False},
    "R3": {_RAW: True, _PLAIN: True, _SEEDS: True},
}
# R3 once more with seed replicates named, which no record holds yet: at the
# freeze every other Step of the session (most are refused as replicates, and
# that refusal is the path read), forward the book's own slice as its one
# replicate.
WITH_REPLICATES = "R3+replicates"

_BY_REVISION = (
    "judged under the by-revision trial rule: recorded before a096fa2 (a trial is "
    "its bytes, not its revision; 2026-10-02 04:38 UTC) or, after it, by a worker "
    "that had started before it and still held the old module"
)
_FAMILY_KEYS = frozenset(
    f"deflated_sharpe.{key}"
    for key in (
        "controls",
        "deflated_sharpe_probability",
        "effective_trials",
        "host_trials",
        "information_ratio_bar",
        "sharpe_star",
        "trial_correlation",
        "trial_correlation_pairs",
        "trials",
    )
)
# Recorded judgements today's code does not reproduce: ``(arm, record) ->
# (reason, the recorded keys that differ)``. All three arms are finished.
# pslage's recorded pass would be refused today (probability 0.957 against
# 0.975).
DECLARED_EXCEPTIONS: dict[tuple[str, str], tuple[str, frozenset[str]]] = {
    ("census_gbdt_alla_8y_20261001", "freeze_gate"): (_BY_REVISION, _FAMILY_KEYS),
    ("pslage_alla_100k_20260927", "freeze_gate"): (
        _BY_REVISION,
        _FAMILY_KEYS | {"passed", "reasons"},
    ),
    ("seqbag_alla_8y_20261001", "freeze_gate"): (
        _BY_REVISION,
        frozenset({"deflated_sharpe.controls"}),
    ),
}


class Judgement(NamedTuple):
    """One recorded judgement and whether today's code reproduces it."""

    record: str  # freeze_gate | forward
    arm: str
    outcome: str  # reproduced | declared | failed
    detail: str


def _read(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stored(value: object) -> Any:
    """``value`` as a ledger line stores it: through JSON."""

    return json.loads(json.dumps(value))


def read_arm(arm: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """An arm's create-time parameters and its ledger lines, as written, and a
    digest of the two files' bytes: an arm that is still running appends to
    its ledger, and two readings compare only where they read the same."""

    ledger = arm / "ledgers" / "experiment_ledger.jsonl"
    params = (arm / HITL_DIR_NAME / PARAMS_NAME).read_bytes()
    lines = ledger.read_bytes() if ledger.is_file() else b""
    return (
        json.loads(params),
        [json.loads(line) for line in lines.split(b"\n") if line],
        hashlib.sha256(params + b"\0" + lines).hexdigest(),
    )


def _steps(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [row for row in record.get("steps") or () if isinstance(row, dict)]


def _gate(
    arm: Path,
    records: Sequence[dict[str, Any]],
    index: int,
    nominee: Mapping[str, Any],
    rules: AcceptanceRules,
    years: Sequence[tuple[str, str]],
    replicates: Sequence[Mapping[str, Any]] = (),
) -> dict[str, object]:
    """The freeze gate of one Step of session ``records[index]``, read as the
    Pipeline read it: over the ledger as it stood before that session."""

    return freeze_gate_for(
        records[:index],
        _steps(records[index]),
        nominee,
        experiment_dir=arm,
        hard_reasons=rules.evaluate(dict(nominee["summary"])),
        acceptance=rules,
        years=years,
        seed_replicates=replicates,
    )


def _replay_and_analysis(result_ref: object) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(str(result_ref))
    return _read(path), _read(path.parent / STYLE_ARTIFACT_NAME)


def _slots(record: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    """The forward and Held-out bounds one forward record replayed."""

    spans = record["replay"]
    return {
        "forward": (spans["start"], spans["forward_end"]),
        "heldout": (spans["heldout_start"], spans["replay_end"]),
    }


def _replicate_slices(
    record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, object]]]:
    """The slices of the seed replicates a forward record replayed (``rows``:
    its ``seed_replicates``), as the Pipeline handed them to the verdict."""

    slices: dict[str, list[dict[str, object]]] = {"forward": [], "heldout": []}
    for row in rows:
        _replay, analysis = _replay_and_analysis(row["result_ref"])
        identity = {key: row[key] for key in ("artifact_id", "source_step_id")}
        for name, (start, end) in _slots(record).items():
            slices[name].append({**identity, **seed_replicate_slice(analysis, start=start, end=end)})
    return slices


def _judged(
    replay: Mapping[str, Any],
    analysis: Mapping[str, Any],
    record: Mapping[str, Any],
    rules: AcceptanceRules,
    *,
    slippage_bps: float,
    replicates: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    """One forward record's ``slices`` and ``verdict`` from its stored replay."""

    slots = _slots(record)
    activity = {
        name: window_activity(replay["equity_curve"], replay["executions"], start=start, end=end)
        for name, (start, end) in slots.items()
    }
    forward = forward_slice(
        analysis,
        rules=rules,
        start=slots["forward"][0],
        end=slots["forward"][1],
        seed_key=str(record["artifact_id"]),
        slippage_bps=slippage_bps,
        turnover=float(activity["forward"]["turnover"]),
        round_trips=int(activity["forward"]["round_trips"]),
        mean_gross=float(activity["forward"]["mean_gross"]),
        seed_replicates=replicates["forward"],
    )
    heldout = heldout_slice(
        analysis,
        rules=rules,
        start=slots["heldout"][0],
        end=slots["heldout"][1],
        forward_tracking_error=float(forward["tracking_error"]),  # type: ignore[arg-type]
        mean_gross=float(activity["heldout"]["mean_gross"]),
        seed_replicates=replicates["heldout"],
    )
    return {
        "verdict": graduation_verdict(forward=forward, heldout=heldout),
        "slices": {
            "forward": {**forward, "activity": activity["forward"]},
            "heldout": {**heldout, "activity": activity["heldout"]},
        },
    }


def _nominee_gates(
    arm: Path,
    records: Sequence[dict[str, Any]],
    index: int,
    nominee: Mapping[str, Any],
    variants: Mapping[str, AcceptanceRules],
    years: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """One Step of a session taken as its nominee, under every rule variant.
    Variants that state the same rules share one reading."""

    read: dict[str, dict[str, object]] = {}
    gates: dict[str, object] = {}
    for name, rules in variants.items():
        key = json.dumps(rules.to_record(), sort_keys=True)
        if key not in read:
            read[key] = _gate(arm, records, index, nominee, rules, years)
        gates[name] = read[key]
    others = [row for row in _steps(records[index]) if row["step_id"] != nominee["step_id"]]
    gates[WITH_REPLICATES] = _gate(arm, records, index, nominee, variants["R3"], years, others)
    return gates


def recompute_arm(arm: Path) -> dict[str, Any]:
    """One arm's part of the dump: everything this code reads off its stored
    inputs, and ``inputs``, the digest of what was read."""

    params, records, digest = read_arm(arm)
    return {"inputs": digest, **_recomputed(arm, params, records, extended=True)}


def recorded_judgements(arm: Path) -> list[tuple[str, Any, Any]]:
    """``(record, as recorded, as recomputed)`` of every judgement one arm's
    ledger records: its freeze gates and its completed forward verdicts."""

    params, records, _digest = read_arm(arm)
    now = _recomputed(arm, params, records, extended=False)
    found: list[tuple[str, Any, Any]] = []
    for record in records:
        run_id = str(record.get("run_id"))
        if record.get("record_type") == "research_session":
            if "freeze_gate" in now["sessions"][run_id]:
                found.append(
                    ("freeze_gate", record["freeze_gate"], now["sessions"][run_id]["freeze_gate"])
                )
        elif run_id in now["forward"]:
            stored = {key: record[key] for key in ("verdict", "slices")}
            found.append(("forward", stored, now["forward"][run_id]["recorded"]))
    return found


def _recomputed(
    arm: Path, params: Mapping[str, Any], records: Sequence[dict[str, Any]], *, extended: bool
) -> dict[str, Any]:
    """What this code reads off one arm's parameters and ledger.

    ``sessions[run_id].freeze_gate`` and ``forward[run_id].recorded`` are the
    recorded judgements again, under the rules and with the seed replicates
    their own records state. ``extended`` adds what only the differential
    compares (the module docstring lists it).
    """

    geometry = ResearchGeometry(**{name: params[name] for name in GEOMETRY_PARAMETERS})
    years = [(slot.start, slot.end) for slot in geometry.research_years]
    own = acceptance_for(params)
    variants = {
        "own": own,
        **{
            era: AcceptanceRules.from_record({**own.to_record(), **flags})
            for era, flags in ERAS.items()
        },
    }
    sessions: dict[str, object] = {}
    forward: dict[str, object] = {}
    out: dict[str, object] = {"sessions": sessions, "forward": forward}
    if extended:
        out["rules"] = {name: rules.to_record() for name, rules in variants.items()}
        out["facts"] = {
            name: rules.agent_facts(research_years=len(years)) for name, rules in variants.items()
        }
    for index, record in enumerate(records):
        run_id = str(record.get("run_id"))
        if record.get("record_type") == "research_session":
            sessions[run_id] = _session(arm, records, index, variants, years, extended=extended)
        elif record.get("record_type") == "forward" and record.get("status") == "ok":
            # Absent only from a parameter file no console wrote: the Broker's.
            slippage = float(params.get("slippage_bps", BrokerProfile().slippage_bps))
            forward[run_id] = _forward(record, slippage_bps=slippage, extended=extended)
    return _stored(out)


def _session(
    arm: Path,
    records: Sequence[dict[str, Any]],
    index: int,
    variants: Mapping[str, AcceptanceRules],
    years: Sequence[tuple[str, str]],
    *,
    extended: bool,
) -> dict[str, object]:
    """One research session: its recorded gate again and, ``extended``, the
    full-span bar after it and every one of its Steps as the nominee."""

    record = records[index]
    steps = _steps(record)
    by_step = {str(row["step_id"]): row for row in steps}
    session: dict[str, object] = {}
    recorded = record.get("freeze_gate")
    if isinstance(recorded, dict):
        named = (recorded.get("seed_replicates") or {}).get("replicates") or ()
        session["freeze_gate"] = _gate(
            arm,
            records,
            index,
            by_step[str(record["nominated_step_id"])],
            AcceptanceRules.from_record(record["acceptance_rules"]),
            years,
            [by_step[str(entry["step_id"])] for entry in named],
        )
    if extended:
        session["full_span_bar"] = full_span_bar(
            records[:index],
            steps,
            experiment_dir=arm,
            research_years=len(years),
            acceptance=variants["own"],
        )
        session["nominees"] = {
            step_id: _nominee_gates(arm, records, index, row, variants, years)
            for step_id, row in by_step.items()
        }
    return session


def _forward(
    record: Mapping[str, Any], *, slippage_bps: float, extended: bool
) -> dict[str, object]:
    """One completed forward record: its recorded verdict again and,
    ``extended``, the verdict under every era's conditions."""

    replay, analysis = _replay_and_analysis(record["result_ref"])
    stated: Mapping[str, object] = record["acceptance_rules"]

    def judged(
        rules: Mapping[str, object], replicates: Sequence[Mapping[str, Any]] = ()
    ) -> dict[str, object]:
        return _judged(
            replay,
            analysis,
            record,
            AcceptanceRules.from_record(rules),
            slippage_bps=slippage_bps,
            replicates=_replicate_slices(record, replicates),
        )

    verdicts = {"recorded": judged(stated, record.get("seed_replicates") or ())}
    if extended:
        for era, flags in ERAS.items():
            verdicts[era] = judged({**stated, **flags})
        book = {key: "book" for key in ("artifact_id", "source_step_id")}
        verdicts[WITH_REPLICATES] = judged(
            {**stated, **ERAS["R3"]}, [{**book, "result_ref": record["result_ref"]}]
        )
    return verdicts


def _named(function: Callable[[Path], Any], arm: Path) -> Any:
    try:
        return function(arm)
    except Exception as exc:
        raise RuntimeError(f"{arm.name}: {type(exc).__name__}: {exc}") from exc


def over_arms(function: Callable[[Path], Any], experiments: Path, *, jobs: int) -> dict[str, Any]:
    """``function`` of every arm under ``experiments``, by arm id."""

    arms = sorted(
        path
        for path in Path(experiments).iterdir()
        if (path / HITL_DIR_NAME / PARAMS_NAME).is_file()
    )
    if jobs <= 1:
        results = [_named(function, arm) for arm in arms]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(partial(_named, function), arms))
    return {arm.name: result for arm, result in zip(arms, results)}


def dump(experiments: Path, *, jobs: int) -> dict[str, Any]:
    """What the differential compares between two trees."""

    return _stored(
        {
            "creation": {
                "defaults": list(WEB_CREATE_DEFAULTS.items()),
                "accepted": sorted(_ALLOWED_PARAMS),
            },
            "arms": over_arms(recompute_arm, experiments, jobs=jobs),
        }
    )


def _same(first: object, second: object) -> bool:
    if type(first) is not type(second):
        return False
    if isinstance(first, float) and math.isnan(first):
        return math.isnan(second)  # type: ignore[arg-type]
    return first == second


def _at(path: str, key: object) -> str:
    return f"{path}.{key}" if path else str(key)


def unreproduced(recorded: object, recomputed: object, path: str = "") -> list[str]:
    """Paths of the recorded keys whose value ``recomputed`` does not hold."""

    if isinstance(recorded, dict) and isinstance(recomputed, dict):
        return [
            item
            for key, value in recorded.items()
            for item in (
                unreproduced(value, recomputed[key], _at(path, key))
                if key in recomputed
                else [_at(path, key)]
            )
        ]
    if (
        isinstance(recorded, list)
        and isinstance(recomputed, list)
        and len(recorded) == len(recomputed)
        and any(isinstance(item, dict) for item in recorded)
    ):
        return [
            item
            for position, pair in enumerate(zip(recorded, recomputed))
            for item in unreproduced(*pair, f"{path}[{position}]")
        ]
    return [] if _same(recorded, recomputed) else [path]


def added(recorded: object, recomputed: object, path: str = "") -> list[str]:
    """Paths of the keys ``recomputed`` holds and the record does not."""

    if not (isinstance(recorded, dict) and isinstance(recomputed, dict)):
        return []
    return [
        item
        for key, value in recomputed.items()
        for item in (
            added(recorded[key], value, _at(path, key)) if key in recorded else [_at(path, key)]
        )
    ]


def changed(baseline: object, current: object, path: str = "") -> list[str]:
    """Where two dumps differ: a value, a key on one side only, or key order."""

    if isinstance(baseline, dict) and isinstance(current, dict):
        common = [key for key in baseline if key in current]
        found = [
            f"{_at(path, key)} (one side only)"
            for key in (*baseline, *current)
            if key not in common
        ]
        if not found and list(baseline) != list(current):
            found.append(f"{path or 'the dump'} (key order)")
        return found + [
            item for key in common for item in changed(baseline[key], current[key], _at(path, key))
        ]
    if isinstance(baseline, list) and isinstance(current, list) and len(baseline) == len(current):
        return [
            item
            for position, pair in enumerate(zip(baseline, current))
            for item in changed(*pair, f"{path}[{position}]")
        ]
    return [] if _same(baseline, current) else [path]


def check_records(
    experiments: Path,
    *,
    jobs: int,
    exceptions: Mapping[tuple[str, str], tuple[str, frozenset[str]]] = DECLARED_EXCEPTIONS,
) -> list[Judgement]:
    """Records mode: every recorded judgement against its recomputation."""

    judgements: list[Judgement] = []
    for name, found in over_arms(recorded_judgements, experiments, jobs=jobs).items():
        for kind, stored, now in found:
            differing = unreproduced(stored, now)
            declared = exceptions.get((name, kind))
            if declared is not None:
                reason, paths = declared
                if set(differing) == paths:
                    judgements.append(
                        Judgement(kind, name, "declared", f"{len(paths)} keys: {reason}")
                    )
                else:
                    judgements.append(
                        Judgement(
                            kind,
                            name,
                            "failed",
                            "the declared exception no longer matches: differs at "
                            f"{sorted(differing)}, declared {sorted(paths)}",
                        )
                    )
            elif differing:
                judgements.append(Judgement(kind, name, "failed", f"differs at {differing}"))
            else:
                extra = added(stored, now)
                judgements.append(
                    Judgement(
                        kind, name, "reproduced", f"added: {', '.join(extra)}" if extra else ""
                    )
                )
    met = {(item.arm, item.record) for item in judgements}
    judgements.extend(
        Judgement(kind, name, "failed", "the declared exception names no recorded judgement")
        for name, kind in sorted(set(exceptions) - met)
    )
    return judgements


def differential(baseline: dict[str, Any], current: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Where two trees' dumps differ, and the arms left out of that reading
    because their stored inputs were not the same bytes in both (a ledger
    that grew, an arm created or deleted in between)."""

    first, second = baseline["arms"], current["arms"]
    moved = sorted(
        name
        for name in {*first, *second}
        if first.get(name, {}).get("inputs") != second.get(name, {}).get("inputs")
    )
    kept = [{name: arm for name, arm in arms.items() if name not in moved} for arms in (first, second)]
    return (
        changed({**baseline, "arms": kept[0]}, {**current, "arms": kept[1]}),
        moved,
    )


def baseline_dump(
    revision: str, experiments: Path, *, jobs: int, repo: Path = REPO_ROOT
) -> dict[str, Any]:
    """``revision``'s own dump: its tree extracted, its own script run on it."""

    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", revision, "--", *BASELINE_PATHS],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with TemporaryDirectory(prefix="recompute_verdicts_") as raw:
        tree = Path(raw)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(tree, filter="data")
        target = tree / "dump.json"
        subprocess.run(
            [
                sys.executable,
                str(tree / SCRIPT),
                "--experiments",
                str(Path(experiments).resolve()),
                "--jobs",
                str(jobs),
                "--dump",
                str(target),
            ],
            check=True,
            cwd=tree,
        )
        return _read(target)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--experiments", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--jobs", type=int, default=8, help="Arms recomputed in parallel.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", metavar="REV", help="Differential against this revision.")
    mode.add_argument("--dump", type=Path, metavar="FILE", help="Write this tree's dump.")
    args = parser.parse_args(argv)
    if args.dump is not None:
        args.dump.write_text(json.dumps(dump(args.experiments, jobs=args.jobs)), encoding="utf-8")
        return 0
    if args.baseline is not None:
        baseline = baseline_dump(args.baseline, args.experiments, jobs=args.jobs)
        current = dump(args.experiments, jobs=args.jobs)
        differing, moved = differential(baseline, current)
        for path in differing[:200]:
            print(f"differs: {path}")
        compared = len(set(current["arms"]) - set(moved))
        print(
            f"differential against {args.baseline} over {compared} arms: "
            + (f"{len(differing)} differences" if differing else "identical")
        )
        if moved:
            print(
                f"not compared, their stored inputs changed between the two readings "
                f"(run again): {', '.join(moved)}"
            )
        return 1 if differing or moved else 0
    judgements = check_records(args.experiments, jobs=args.jobs)
    for item in judgements:
        detail = f": {item.detail}" if item.detail else ""
        print(f"{item.outcome:<10} {item.record:<11} {item.arm}{detail}")
    count = {
        outcome: sum(item.outcome == outcome for item in judgements)
        for outcome in ("reproduced", "declared", "failed")
    }
    print(
        f"{len(judgements)} recorded judgements: {count['reproduced']} reproduced, "
        f"{count['declared']} declared exceptions, {count['failed']} failed"
    )
    return 1 if count["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
