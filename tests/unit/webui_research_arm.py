"""A synthetic research arm on disk, shaped as the pipeline writes it.

The console reads experiments only through their files: ``hitl/`` control
plane, the ledger and the result artifacts it names. ``build_arm`` writes those
for one arm at a chosen stage — researching, frozen with the forward replay
sealed, or with its verdict — from synthetic daily series, and computes every
recorded statistic with the pipeline's own functions (``freeze_gate_for``,
``forward_slice``, ``heldout_slice``, ``graduation_verdict``), so a projection
test reads the same numbers a real arm would carry.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from autotrade.environment.artifacts import REVISION_MANIFEST_FILE
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME, STYLE_SCHEMA_VERSION
from autotrade.environment.runtime import write_json_atomic
from autotrade.pipelines.calendar import FULL_SPAN
from autotrade.pipelines.config import DEFAULT_RESEARCH_GEOMETRY, AcceptanceRules
from autotrade.pipelines.experiment import freeze_gate_for
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
    graduation_verdict,
    heldout_slice,
    neutralized_statistics,
    seed_replicate_slice,
)

STAGES = (
    "created",
    "research",
    "sealed",
    "graduated",
    "discarded",
    "no_deliverable",
    "deadline",
    "broken",
)
GEOMETRY = DEFAULT_RESEARCH_GEOMETRY
# Worker-accepted parameters of a console-created arm.
PARAMS: dict[str, object] = {
    **GEOMETRY.to_record(),
    "strategy_path": "configs/agent_output_template/main.py",
    "data_backend": "pit",
    "execution_mode": "sandbox",
    "developer_mode": "llm",
    "strategy_period": "day",
    "inference_time": "08:30",
}
REPLAY = {
    "start": "20250701",
    "forward_end": "20260630",
    "heldout_start": "20260701",
    "replay_end": "20260911",
    "requested_end": "20260930",
    "truncation_reason": "release_ends_20260911",
}
# The rules every recorded statistic of the arm is judged under: drawdown
# limits wide enough that the synthetic series' edge alone decides.
RULES = AcceptanceRules(max_drawdown=0.9, active_max_drawdown=0.9)


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
            "benchmark_regression": {"available": True, "beta": 0.9, "alpha_annualized": 0.05, "r2": 0.8, "n_days": days},
            "style": {"available": False, "reason": "no_holdings"},
            "strategy_daily": [[day, float(value)] for day, value in zip(dates, strategy, strict=True)],
            "benchmark_daily": [[day, float(value)] for day, value in zip(dates, benchmark, strict=True)],
            "size_factor_daily": [[day, float(value)] for day, value in zip(dates, size, strict=True)],
        },
    )
    return str(result_dir / "result.json")


def _analysis(result_ref: str) -> dict[str, object]:
    return json.loads((Path(result_ref).parent / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8"))


def _step(experiment_dir: Path, session: str, index: int, *, edge: float, seed: int, span: str = FULL_SPAN) -> dict[str, object]:
    ref = write_result(experiment_dir, "valid", start="20240701", days=120, edge=edge, seed=seed)
    revision, fingerprint = f"revision_{session}_{index}", f"fingerprint_{session}_{index}"
    # The manifest the artifact store writes for the revision; a host sidecar
    # names only the revision, and its bytes are read off this.
    write_json_atomic(
        experiment_dir / REVISIONS_DIR / revision / REVISION_MANIFEST_FILE,
        {"revision_id": revision, "fingerprint": fingerprint},
    )
    return {
        "step_id": f"research__session_ref_{session}__run_ref_{session}__valid_{index:03d}",
        "revision_id": revision,
        "fingerprint": fingerprint,
        "span": span,
        "summary": {"total_return": 0.1 + edge, "sharpe": 1.0, "max_drawdown": 0.05},
        "validation_result_ref": ref,
        "neutralized": neutralized_statistics(_analysis(ref)),
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
        "freeze_gate": None,
        "frozen": None,
        "arm_end": None,
        "attempts": 1,
        "budget_used": {"inference_seconds": 600.0, "llm_calls": 40, "replay_years": 8, "null_controls": 0},
        "run_wall_seconds": 600.0,
        **fields,
    }


def _with_panel(result_ref: str, scale: float) -> None:
    """Give a replay's sidecar the zero-skill panel series a panel-era replay
    carries, so its slices measure the plain selection."""

    path = Path(result_ref).parent / STYLE_ARTIFACT_NAME
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    sidecar["panel_daily"] = [[day, scale * value] for day, value in sidecar["strategy_daily"]]
    write_json_atomic(path, sidecar)


def build_arm(
    root: Path, experiment_id: str, stage: str, *, alive: bool = False, seeded: bool = False
) -> Path:
    """Write one arm at ``stage`` (one of :data:`STAGES`) under ``root``.

    ``seeded`` makes it an arm under ``require_seed_replicates`` whose nominee
    froze with one seed replicate and, past the freeze, was replayed with it
    and judged on their mean, in the shapes ``pipelines.experiment`` records
    (tests/unit/test_seed_replicates.py): the gate's ``seed_replicates``
    block, the frozen block's, each slice's ``seed_replicates`` and
    ``seed_mean`` and the record's own rows.
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
            **({"require_seed_replicates": True} if seeded else {}),
            "_created_at": "2026-09-13T00:00:00+00:00",
        },
    )
    write_control(hitl / "control.json", ControlState(mode="auto"))
    if stage == "created":
        write_json_atomic(hitl / "status.json", {"schema_version": 1, "state": "created"})
        return directory
    write_json_atomic(hitl / "schedule.json", build_session_plan(forward=REPLAY))
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    records: list[dict[str, object]] = []
    # The one research session is in flight while the arm researches: it has
    # no ledger record until it ends.
    current = "research"
    if stage in ("sealed", "graduated", "discarded"):
        steps = [
            _step(directory, "research", 0, edge=0.0005, seed=1),
            _step(directory, "research", 1, edge=0.002, seed=3),
            _step(directory, "research", 2, edge=0.001, seed=4, span="Y3"),
        ]
        nominee = steps[1]
        gate = freeze_gate_for([], steps, nominee, experiment_dir=directory, acceptance=RULES)
        # The seed replicate: the full-span step other than the nominee.
        replicate = steps[0]
        if seeded:
            ratios = [gate["information_ratio"], replicate["neutralized"]["information_ratio"]]
            gate = {
                **gate,
                "seed_replicates": {
                    "trains_a_model": True,
                    "replicates": [
                        {
                            "step_id": replicate["step_id"],
                            "seed_line": "main.py: SEED_BASE = 2000",
                            "information_ratio": ratios[1],
                        }
                    ],
                    "mean_information_ratio": sum(ratios) / len(ratios),
                    "information_ratio_bar": gate["deflated_sharpe"]["information_ratio_bar"],
                },
                "thresholds": {**gate["thresholds"], "require_seed_replicates": True},
            }
        output = directory / "artifacts/strategy/frozen/strategy_research_abc/output"
        output.mkdir(parents=True)
        (output / "main.py").write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
        frozen = {
            "artifact_id": "strategy_research_abc",
            "output_path": str(output),
            "models_path": None,
            "source_step_id": nominee["step_id"],
            "revision_id": nominee["revision_id"],
            "research_result_ref": nominee["validation_result_ref"],
            "days": gate["days"],
            "neutralized_excess": gate["neutralized_excess"],
            "tracking_error": gate["tracking_error"],
            "information_ratio": gate["information_ratio"],
            "blocks": [
                {"kind": "year", "label": "202407-202506", "start": "20240701", "end": "20241217", "trade_days": 120, "partial": True, "return": 0.2, "excess_return": 0.1, "sharpe": 1.2, "max_drawdown": 0.05, "turnover": 1.0, "trade_count": 3}
            ],
            "null_control": {"excess_percentile": 0.81},
            "deflated_sharpe": gate["deflated_sharpe"],
            "full_span_validations": gate["full_span_validations"],
            "forward_mde": forward_mde(float(gate["tracking_error"]), 242),
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
                            "information_ratio": replicate["neutralized"]["information_ratio"],
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
                outcome="freeze",
                steps=steps,
                trials_to_date=3,
                nominated_step_id=nominee["step_id"],
                freeze_gate=gate,
                frozen=frozen,
            )
        )
        current = "forward"
    elif stage == "no_deliverable":
        records.append(
            _session_record(
                experiment_id,
                outcome="no_edge",
                steps=[],
                trials_to_date=0,
                reason="没有候选值得冻结。三年都测过了，最好的一个也只有 +0.3%/年。",
                arm_end={"status": "no_deliverable", "reason": "no_edge: nothing survived"},
            )
        )
        current = ""
    elif stage == "deadline":
        records.append(
            _session_record(
                experiment_id,
                outcome="deadline",
                finish_reason="llm_call_budget_exhausted",
                steps=[],
                trials_to_date=0,
                reason=None,
                arm_end={
                    "status": "no_deliverable",
                    "reason": "research budget exhausted without a freeze (llm_call_budget_exhausted)",
                },
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
    if stage == "sealed":
        # The replay is running: its result directory is already on disk, and
        # no ledger record names it.
        write_result(directory, "heldout", start=REPLAY["start"], days=300, edge=0.002, seed=9)
    if stage in ("graduated", "discarded"):
        edge = 0.002 if stage == "graduated" else -0.002
        ref = write_result(directory, "heldout", start=REPLAY["start"], days=300, edge=edge, seed=7)
        seeds: dict[str, dict[str, object]] = {"forward": {}, "heldout": {}}
        replicate_rows: list[dict[str, object]] = []
        if seeded:
            replicate_ref = write_result(
                directory, "heldout", start=REPLAY["start"], days=300, edge=edge / 4, seed=8
            )
            _with_panel(ref, 0.6)
            _with_panel(replicate_ref, 0.9)
            identity = {"artifact_id": "strategy_research_seed", "source_step_id": replicate["step_id"]}
            replicate_analysis = _analysis(replicate_ref)
            for name, (start, end) in (
                ("forward", (REPLAY["start"], REPLAY["forward_end"])),
                ("heldout", (REPLAY["heldout_start"], REPLAY["replay_end"])),
            ):
                seeds[name] = {
                    "seed_replicates": [
                        {**identity, **seed_replicate_slice(replicate_analysis, start=start, end=end)}
                    ]
                }
            replicate_rows.append(
                {
                    **identity,
                    "revision_id": replicate["revision_id"],
                    "result_ref": replicate_ref,
                    "refits_executed": {"forward": 0, "heldout": 0},
                }
            )
        analysis = _analysis(ref)
        rules = replace(RULES, require_seed_replicates=seeded)
        forward = forward_slice(
            analysis,
            rules=rules,
            start=REPLAY["start"],
            end=REPLAY["forward_end"],
            seed_key="strategy_research_abc",
            slippage_bps=5.0,
            turnover=2.0,
            round_trips=24,
            mean_gross=0.9,
            **seeds["forward"],  # type: ignore[arg-type]
        )
        heldout = heldout_slice(
            analysis,
            rules=rules,
            start=REPLAY["heldout_start"],
            end=REPLAY["replay_end"],
            forward_tracking_error=float(forward["tracking_error"]),
            mean_gross=0.9,
            **seeds["heldout"],  # type: ignore[arg-type]
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
                "verdict": graduation_verdict(forward=forward, heldout=heldout),
            }
        )
        current = ""
    status: dict[str, object] = {"schema_version": 1, "pid": 999_999_999, "state": "stopped"}
    if stage in ("graduated", "discarded", "no_deliverable", "deadline"):
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
