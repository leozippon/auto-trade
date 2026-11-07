"""A synthetic research experiment on disk, shaped as the pipeline writes it.

The console reads experiments only through their files: ``hitl/`` control
plane, the ledger and the result artifacts it names. ``build_experiment`` writes those
for one experiment at a chosen stage — researching, registered with the forward replay
running, or with its readout — from synthetic daily series, and computes
every recorded statistic with the pipeline's own functions
(``check_registration``, ``validation_record.step_entry``, ``forward_slice``,
``heldout_slice``, ``validation_record.readout``), so a projection test reads
the same numbers a real experiment would carry.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from autotrade.common.records import write_json_atomic
from autotrade.environment.artifacts import REVISION_MANIFEST_FILE
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.replay.style import (
    BENCHMARK_LABEL,
    BENCHMARK_TS_CODE,
    STYLE_ARTIFACT_NAME,
    STYLE_SCHEMA_VERSION,
)
from autotrade.pipelines import validation_record
from autotrade.pipelines.calendar import FULL_SPAN
from autotrade.pipelines.config import AcceptanceRules
from autotrade.pipelines.experiment import check_registration
from autotrade.pipelines.hitl_state import (
    ControlState,
    build_session_plan,
    proc_start_ticks,
    write_control,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.session_resume import REVISIONS_DIR
from autotrade.pipelines.verdict import (
    forward_mde,
    forward_slice,
    graded_statistics,
    heldout_slice,
    residual_reading,
)
from tests.unit.acceptance_fixture import REQUIRED_RULES, TEST_GEOMETRY

STAGES = (
    "created",
    "research",
    "forward",
    "registered_with_record",
    "deadline",
    "environment_blocked",
    "broken",
)
# The reason the host writes for an experiment whose session it ended on the
# environment (``session_tools.EnvironmentBlocked``).
ENVIRONMENT_BLOCKED_REASON = (
    "span full: 2 batches failed on the environment in every candidate; last error: "
    "daily Validation failed: ArrowTypeError: Unable to merge: Field dataset has "
    "incompatible types: large_string vs string"
)
GEOMETRY = TEST_GEOMETRY
# Worker-accepted parameters of a console-created experiment.
PARAMS: dict[str, object] = {
    **GEOMETRY.to_record(),
    "strategy_path": "configs/agent_output_template/main.py",
    "data_backend": "pit",
    "execution_mode": "sandbox",
    "developer_mode": "llm",
    "strategy_period": "day",
    "inference_time": "08:30",
    # No fundamentals: opening the Paper account of an experiment that reads them asks
    # the live PIT audit, which a fixture repository does not have
    # (test_paper_books covers that check).
    "include_fundamentals": False,
    # No prebuilt view cache, which a fixture repository does not have either.
    "pit_views_seed": "",
    **REQUIRED_RULES,
}
REPLAY = {
    "start": "20250701",
    "forward_end": "20260630",
    "heldout_start": "20260701",
    "replay_end": "20260911",
    "requested_end": "20260930",
    "truncation_reason": "release_ends_20260911",
}
# The rules every recorded statistic of the experiment is judged under: drawdown
# limits wide enough that the synthetic series' edge alone decides.
RULES = AcceptanceRules(**REQUIRED_RULES, max_drawdown=0.9, active_max_drawdown=0.9)
# The validation years of the default geometry under the fixture's rules.
VALIDATION = (
    GEOMETRY.split_research(RULES.research_years)[1][0].start,
    GEOMETRY.split_research(RULES.research_years)[1][-1].end,
)


def _business_day_count(start: str, end: str) -> int:
    day = date(int(start[:4]), int(start[4:6]), int(start[6:]))
    last = date(int(end[:4]), int(end[4:6]), int(end[6:]))
    count = 0
    while day <= last:
        count += day.weekday() < 5
        day += timedelta(days=1)
    return count


def _business_days(start: str, count: int) -> list[str]:
    day = date(int(start[:4]), int(start[4:6]), int(start[6:]))
    days: list[str] = []
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day.strftime("%Y%m%d"))
        day += timedelta(days=1)
    return days


def write_result(
    experiment_dir: Path, mode: str, *, start: str, days: int, edge: float, seed: int
) -> str:
    """One replay result and its style sidecar; returns the ``result.json`` path."""

    rng = np.random.default_rng(seed)
    dates = _business_days(start, days)
    benchmark = rng.normal(0.0003, 0.01, days)
    size = rng.normal(0.0, 0.005, days)
    strategy = edge + 0.9 * benchmark + 0.3 * size + rng.normal(0.0, 0.004, days)
    result_dir = experiment_dir / "artifacts/results" / f"{mode}_{uuid.uuid4().hex}"
    result_dir.mkdir(parents=True)
    equity = 1_000_000 * np.cumprod(1.0 + strategy)
    write_json_atomic(
        result_dir / "result.json",
        {
            "initial_cash": 1_000_000.0,
            "equity_curve": [
                {
                    "trade_date": day,
                    "initial_equity": 1_000_000.0,
                    "equity": float(value),
                    "cash": float(value) * 0.1,
                }
                for day, value in zip(dates, equity, strict=True)
            ],
            "executions": [
                {
                    "symbol": "000001.SZ",
                    "action": "buy",
                    "quantity": 500,
                    "execute_at": f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]}T09:30:00+08:00",
                    "status": "filled",
                    "price": 10.0,
                },
                {
                    "symbol": "600000.SH",
                    "action": "buy",
                    "quantity": 200,
                    "execute_at": f"{dates[1][:4]}-{dates[1][4:6]}-{dates[1][6:]}T09:30:00+08:00",
                    "status": "rejected",
                    "reason": "limit_up_blocked_buy",
                },
            ],
        },
    )
    write_json_atomic(
        result_dir / STYLE_ARTIFACT_NAME,
        {
            "schema_version": STYLE_SCHEMA_VERSION,
            "mode": mode,
            "benchmark": {"ts_code": BENCHMARK_TS_CODE, "label": BENCHMARK_LABEL},
            "benchmark_regression": {"available": True, "beta": 0.9, "alpha_annualized": 0.05, "r2": 0.8, "n_days": days},
            "style": {"available": False, "reason": "no_holdings"},
            "strategy_daily": [[day, float(value)] for day, value in zip(dates, strategy, strict=True)],
            "benchmark_daily": [[day, float(value)] for day, value in zip(dates, benchmark, strict=True)],
            "size_factor_daily": [[day, float(value)] for day, value in zip(dates, size, strict=True)],
            # Every replay carries its random baseline; a zero one leaves the
            # active figures the strategy's own (``_with_baselines`` sets others).
            "baseline_daily": [[day, 0.0] for day in dates],
            "value_factor_daily": [
                [day, float(value)]
                for day, value in zip(
                    dates, np.random.default_rng(len(dates)).normal(0.0, 0.003, len(dates)), strict=True
                )
            ],
        },
    )
    return str(result_dir / "result.json")


def _analysis(result_ref: str) -> dict[str, object]:
    return json.loads((Path(result_ref).parent / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8"))


def _step(experiment_dir: Path, session: str, index: int, *, edge: float, seed: int, span: str = FULL_SPAN) -> dict[str, object]:
    ref = write_result(experiment_dir, "valid", start="20240701", days=120, edge=edge, seed=seed)
    revision, fingerprint = f"revision_{session}_{index}", f"fingerprint_{session}_{index}"
    # The manifest the artifact store writes for the revision; a host sidecar
    # names only the revision, and its bytes are read off this. The registration
    # gate reads a nominee's strategy from beside it.
    write_json_atomic(
        experiment_dir / REVISIONS_DIR / revision / REVISION_MANIFEST_FILE,
        {"revision_id": revision, "fingerprint": fingerprint},
    )
    output = experiment_dir / REVISIONS_DIR / revision / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "main.py").write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    return {
        "step_id": f"research__session_ref_{session}__run_ref_{session}__valid_{index:03d}",
        "revision_id": revision,
        "fingerprint": fingerprint,
        "span": span,
        "summary": {
            "total_return": 0.1 + edge,
            "sharpe": 1.0,
            "max_drawdown": 0.05,
            # Above the benchmark well clear of the stressed slippage.
            "turnover": 1.0,
            "benchmark": {"excess_return": 0.05 + edge},
            "cost_sensitivity": {"slippage_bps": 5.0, "cost_per_bp_per_side": 1e-4},
        },
        "validation_result_ref": ref,
        "graded": graded_statistics(_analysis(ref)),
    }


def _session_record(experiment_id: str, **fields: object) -> dict[str, object]:
    return {
        "record_type": "research_session",
        "experiment_id": experiment_id,
        "epoch_id": "research",
        "fold_id": "research",
        "run_id": "run_research",
        "session_key": "research",
        "finish_reason": "llm_agent_finish_session",
        "reason": "the session's reason",
        "nominated_step_id": None,
        "registration": None,
        "registered": None,
        "experiment_end": None,
        "attempts": 1,
        "budget_used": {
            "inference_seconds": 600.0, "llm_calls": 40, "main_calls": 30, "subagent_calls": 10, "compact_calls": 0,
            "test_replays": 8, "null_controls": 0, "api_spend_usd": 1.25,
        },
        "run_wall_seconds": 600.0,
        **fields,
    }


def _with_baselines(result_ref: str, scale: float, *, series: str = "strategy_daily") -> None:
    """Give a replay's sidecar the random baseline series every replay
    carries: ``scale`` times its ``series``."""

    path = Path(result_ref).parent / STYLE_ARTIFACT_NAME
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    sidecar["baseline_daily"] = [[day, scale * value] for day, value in sidecar[series]]
    write_json_atomic(path, sidecar)


def build_experiment(
    root: Path, experiment_id: str, stage: str, *, alive: bool = False, seeded: bool = False
) -> Path:
    """Write one experiment at ``stage`` (one of :data:`STAGES`) under ``root``.

    ``seeded`` makes its nominee one that trains a model and registered with one
    seed replicate and, past the registration, was replayed with it and judged on
    the two together, in the shapes ``pipelines.experiment`` records
    (tests/unit/test_seed_replicates.py): the gate's ``seed_replicates``
    block, the registered block's, each slice's ``seed_replicates`` and
    ``seed_mean``, and the record's own rows.
    """

    if stage not in STAGES:
        raise ValueError(stage)
    directory = Path(root) / experiment_id
    AgentRefStore(directory)
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    write_json_atomic(
        hitl / "params.json",
        {
            "experiment_id": experiment_id,
            **PARAMS,
            "_created_at": "2026-09-13T00:00:00+00:00",
        },
    )
    write_control(hitl / "control.json", ControlState())
    if stage == "created":
        write_json_atomic(hitl / "status.json", {"schema_version": 1, "state": "created"})
        return directory
    write_json_atomic(hitl / "schedule.json", build_session_plan(forward=REPLAY))
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    records: list[dict[str, object]] = []
    # The one research session is in flight while the experiment researches: it has
    # no ledger record until it ends.
    current = "research"
    if stage in ("forward", "registered_with_record"):
        steps = [
            _step(directory, "research", 0, edge=0.0005, seed=1),
            _step(directory, "research", 1, edge=0.002, seed=3),
            _step(directory, "research", 2, edge=0.001, seed=4, span="Y3"),
        ]
        nominee = steps[1]
        gate = check_registration(nominee, experiment_dir=directory, acceptance=RULES)
        statistics = nominee["graded"]
        # The seed replicate: the full-span step other than the nominee.
        replicate = steps[0]
        if seeded:
            gate = {
                **gate,
                "seed_replicates": {
                    "replicates": [
                        {
                            "step_id": replicate["step_id"],
                            "seed_line": "main.py: SEED_BASE = 2000",
                            "information_ratio": replicate["graded"]["information_ratio"],
                        }
                    ],
                },
            }
        output = directory / "artifacts/strategy/registered/strategy_research_abc/output"
        output.mkdir(parents=True)
        (output / "main.py").write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
        registered = {
            "artifact_id": "strategy_research_abc",
            "output_path": str(output),
            "models_path": None,
            "source_step_id": nominee["step_id"],
            "revision_id": nominee["revision_id"],
            "research_result_ref": nominee["validation_result_ref"],
            "series": statistics["series"],
            "mandate": gate["mandate"],
            "days": statistics["days"],
            "plain_excess": statistics["plain_excess"],
            "tracking_error": statistics["tracking_error"],
            "information_ratio": statistics["information_ratio"],
            "residual_reading": residual_reading(
                _analysis(nominee["validation_result_ref"]), alpha=RULES.alpha
            ),
            "blocks": [
                {"kind": "year", "label": "202407-202506", "start": "20240701", "end": "20241217", "trade_days": 120, "partial": True, "return": 0.2, "excess_return": 0.1, "sharpe": 1.2, "max_drawdown": 0.05, "turnover": 1.0, "trade_count": 3}
            ],
            "null_control": {"excess_percentile": 0.81},
            "forward_mde": forward_mde(float(statistics["tracking_error"]), 242),
            "fit_plan": {"fit": seeded, "refit_period": "quarter" if seeded else None},
            **(
                {
                    "seed_replicates": [
                        {
                            "artifact_id": "strategy_research_seed",
                            "output_path": str(output.parent.parent / "strategy_research_seed/output"),
                            "models_path": None,
                            "source_step_id": replicate["step_id"],
                            "revision_id": replicate["revision_id"],
                            "research_result_ref": replicate["validation_result_ref"],
                            "information_ratio": replicate["graded"]["information_ratio"],
                        }
                    ]
                }
                if seeded
                else {}
            ),
        }
        records.append(
            _session_record(
                experiment_id,
                outcome="completed",
                finish_reason="validation_period_complete",
                steps=steps,
                nominated_step_id=nominee["step_id"],
                registration=gate,
                registered=registered,
            )
        )
        current = "forward"
    elif stage == "deadline":
        records.append(
            _session_record(
                experiment_id,
                outcome="deadline",
                finish_reason="llm_call_budget_exhausted",
                steps=[],
                reason=None,
                experiment_end={
                    "status": "no_deliverable",
                    "reason": (
                        "research budget exhausted before any test, with no registrable "
                        "full-span candidate (llm_call_budget_exhausted)"
                    ),
                },
            )
        )
        current = ""
    elif stage == "environment_blocked":
        records.append(
            _session_record(
                experiment_id,
                outcome="environment_blocked",
                finish_reason="environment_blocked",
                steps=[],
                reason=ENVIRONMENT_BLOCKED_REASON,
                experiment_end={"status": "environment_blocked", "reason": ENVIRONMENT_BLOCKED_REASON},
            )
        )
        current = ""
    elif stage == "broken":
        # The worker exhausted its attempts: the ledger records the failures,
        # status.json carries the error that stopped the run.
        records.extend(
            {
                "record_type": "attempt_failed",
                "experiment_id": experiment_id,
                "epoch_id": "research",
                "fold_id": "research",
                "run_id": f"run_research_{attempt}",
                "session_key": "research",
                "phase": "research",
                "error": f"RuntimeError: sandbox image is gone\n  attempt {attempt}",
            }
            for attempt in (1, 2, 3)
        )
        current = "research"
    for record in records:
        ledger.append(record)
    if stage == "forward":
        # The replay is running: its result directory is already on disk, and
        # no ledger record names it.
        write_result(directory, "heldout", start=REPLAY["start"], days=300, edge=0.002, seed=9)
    if stage in ("forward", "registered_with_record"):
        # The validation record: every step of the validation period, read off
        # one replay of the registered strategy, as the host's completion writes it.
        validation = GEOMETRY.split_research(RULES.research_years)[1]
        plan = validation_record.ValidationSteps.of(validation, RULES.test_step_months)
        record_ref = write_result(
            directory, "valid", start=VALIDATION[0], days=_business_day_count(VALIDATION[0], VALIDATION[1]), edge=0.002, seed=21
        )
        _with_baselines(record_ref, 0.9, series="benchmark_daily")
        record_replay = json.loads(Path(record_ref).read_text(encoding="utf-8"))
        record_analysis = _analysis(record_ref)
        for step in range(1, plan.count + 1):
            first, last = plan.window(step)
            step_record = validation_record.test_step_record(
                    head={
                        "experiment_id": experiment_id,
                        "epoch_id": "research",
                        "fold_id": "research",
                        "run_id": "run_research",
                        "session_key": "research",
                    },
                    steps=plan,
                    step=step,
                    candidate={
                        "step_id": nominee["step_id"],
                        "revision_id": nominee["revision_id"],
                        "fingerprint": nominee["fingerprint"],
                        "seed_replicates": [],
                    },
                    replay={"start": plan.start, "end": plan.end, "result_ref": record_ref},
                    entry=validation_record.step_entry(record_replay, record_analysis, start=first, end=last),
                    registration=gate,
                    source=validation_record.SOURCE_HOST_COMPLETION,
                )
            # The session record is already in the ledger: the steps follow it.
            records.append(step_record)
            ledger.append(step_record)
    if stage == "registered_with_record":
        edge = 0.002
        ref = write_result(directory, "heldout", start=REPLAY["start"], days=300, edge=edge, seed=7)
        replicates: list[tuple[dict[str, object], dict[str, object]]] = []
        replicate_rows: list[dict[str, object]] = []
        # Every forward replay carries its random baseline: random names on
        # the book's own skeleton, which ride the market as the book does.
        if seeded:
            _with_baselines(ref, 0.6)
        else:
            _with_baselines(ref, 0.9, series="benchmark_daily")
        if seeded:
            replicate_ref = write_result(
                directory, "heldout", start=REPLAY["start"], days=300, edge=edge / 4, seed=8
            )
            _with_baselines(replicate_ref, 0.9)
            identity = {"artifact_id": "strategy_research_seed", "source_step_id": replicate["step_id"]}
            replicates.append((identity, _analysis(replicate_ref)))
            replicate_rows.append(
                {
                    **identity,
                    "revision_id": replicate["revision_id"],
                    "result_ref": replicate_ref,
                    "refits_executed": {"forward": 0, "heldout": 0},
                }
            )
        analysis = _analysis(ref)
        forward = forward_slice(
            analysis,
            rules=RULES,
            start=REPLAY["start"],
            end=REPLAY["forward_end"],
            seed_key="strategy_research_abc",
            slippage_bps=5.0,
            turnover=2.0,
            round_trips=24,
            mean_gross=0.9,
            seed_replicates=replicates,
        )
        heldout = heldout_slice(
            analysis,
            rules=RULES,
            start=REPLAY["heldout_start"],
            end=REPLAY["replay_end"],
            mean_gross=0.9,
            seed_replicates=replicates,
        )
        ledger.append(
            {
                "record_type": "forward",
                "experiment_id": experiment_id,
                "epoch_id": "forward",
                "fold_id": "forward",
                "run_id": "run_forward",
                "session_key": "forward",
                "artifact_id": "strategy_research_abc",
                "replay": REPLAY,
                "status": "ok",
                "error": None,
                "result_ref": ref,
                "slices": {"forward": forward, "heldout": heldout},
                "refits_executed": {"forward": 0, "heldout": 0},
                **({"seed_replicates": replicate_rows} if replicate_rows else {}),
                "null_control": {"k": 500, "excess_percentile": 0.9, "step": {"excess_percentile": 0.77}},
                "readout": validation_record.readout(records, alpha=RULES.alpha, forward=forward),
            }
        )
        current = ""
    status: dict[str, object] = {"schema_version": 1, "pid": 999_999_999, "state": "stopped"}
    if stage in ("registered_with_record", "deadline", "environment_blocked"):
        status["state"] = "completed"
    elif stage == "broken":
        status["state"] = "failed"
        status["error"] = "RuntimeError: sandbox image is gone\n  attempt 3"
    elif alive:
        status = {
            "schema_version": 1,
            "pid": os.getpid(),
            "pid_start_ticks": proc_start_ticks(os.getpid()),
            "state": "running_session",
            "session_key": current,
            "run_id": f"run_{current}_live",
            "environment_stage": "forward_replay" if current == "forward" else "llm_call",
        }
    write_json_atomic(hitl / "status.json", status)
    return directory
