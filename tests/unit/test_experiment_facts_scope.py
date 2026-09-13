"""The run facts a session is told: development window, universe policy and
strategy call cadence — one sentence each, with Held-out invisible as ever —
plus the budgets and decision-input windows the same object publishes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from autotrade.agent.experiment_facts import (
    BATCH_VALIDATE_FIT_TIMEOUT_NOTE,
    build_experiment_facts,
)
from autotrade.agent.prompts import (
    build_meta_learning_prompt,
    build_system_prompt,
)
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.identity import AgentRefStore


def _facts(
    *,
    data_summary: dict[str, object] | None = None,
    **manifest_overrides: object,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "experiment_id": "exp",
        "run_id": "run_x",
        "epoch_id": "epoch_001",
        "fold_id": "fold_2022",
        "kind": "fold",
        "fold": {
            "input_window": "20200101..20211231",
            "validation_period": "20220101..20221231",
            "valid_decision_time": "2021-12-31T23:59:59+08:00",
        },
        "fold_period": "year",
        "test_stage": False,
        "schedule": {"period": "day", "inference_time": "08:30"},
        "snapshot_config": SnapshotConfig().to_record(),
    }
    manifest.update(manifest_overrides)
    with tempfile.TemporaryDirectory() as tmp:
        return build_experiment_facts(
            manifest=manifest,
            ref_store=AgentRefStore(Path(tmp) / "experiment"),
            data_summary=data_summary,
        )


def _data_summary(rows: dict[str, int]) -> dict[str, object]:
    """A snapshot view whose files carry the given row counts."""

    return {
        "views": {
            "snapshot": {
                "mount_path": "/mnt/snapshot",
                "files": [
                    {"path": name, "mount_path": f"/mnt/snapshot/{name}", "rows": count}
                    for name, count in rows.items()
                ],
            }
        }
    }


def test_every_fold_states_whether_it_is_a_confirmation_fold() -> None:
    """False is stated, not implied by absence: an Agent read a missing flag as
    a confirmation Fold and left most of its budget unspent (XR1 A6)."""

    assert _facts()["identity"]["confirmation_fold"] is False
    assert _facts(confirmation_fold=True)["identity"]["confirmation_fold"] is True
    # The flag belongs to development Folds only.
    for kind in ("deployment_adjustment", "meta_learning"):
        assert "confirmation_fold" not in _facts(kind=kind)["identity"]


def test_regular_fold_facts_name_the_yearly_folds_and_the_meta_between_them() -> None:
    facts = _facts()
    scope = facts["research_scope"]
    assert scope["development_window"].startswith(
        "This Fold's validation period is 20220101..20221231."
    )
    assert "one Fold per year" in scope["development_window"]
    assert "Meta-learning session between Folds" in scope["development_window"]
    assert "no frozen Test stage" in scope["development_window"]
    assert "Held-out" in scope["development_window"]
    assert scope["universe"].startswith("The universe is unfiltered")
    assert "ST names included" in scope["universe"]
    assert "every trading day at 08:30" in scope["strategy_cadence"]
    assert "own rebalance cadence" in scope["strategy_cadence"]
    # The cadence is public research scope; Held-out stays invisible.
    assert facts["visible_timeline"]["fold_period"] == "year"
    assert facts["visibility_policy"]["heldout_visible"] is False
    assert facts["visibility_policy"]["test_visible"] is False
    rendered = json.dumps(facts, ensure_ascii=False)
    assert "2026" not in rendered
    assert "fold_2022" not in rendered


def test_the_signal_screen_path_is_a_fact_only_where_the_mount_exists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = AgentRefStore(Path(tmp) / "experiment")
        docker_fold = build_experiment_facts(
            manifest={"kind": "fold", "experiment_id": "exp", "run_id": "run_x"},
            ref_store=store,
            runtime_env={"mode": "docker"},
        )
        local_fold = build_experiment_facts(
            manifest={"kind": "fold", "experiment_id": "exp", "run_id": "run_x"},
            ref_store=store,
            runtime_env={"mode": "local"},
        )
        meta = build_experiment_facts(
            manifest={"kind": "meta_learning", "experiment_id": "exp"},
            ref_store=store,
            runtime_env={"mode": "docker"},
        )
    screen = docker_fold["source_refs"]["signal_screen_ref"]
    assert screen["path"] == "/mnt/tools/screen.py"
    # Sub-agents kept handing the bare path to read_file; the fact now carries
    # the argv contract and says which tools cannot open it.
    assert '["python", "/mnt/tools/screen.py", "--help"]' in screen["usage"]
    assert "shell only" in screen["usage"]
    assert "read_file" in screen["usage"]
    assert "signal_screen_ref" not in local_fold["source_refs"]
    assert "signal_screen_ref" not in meta["source_refs"]


def test_rolling_facts_keep_the_cadence_and_a_screened_universe_is_described() -> None:
    screened = SnapshotConfig(
        screen_exclude_st=True, screen_exclude_new_listed_days=180, screen_boards=("main",)
    ).to_record()
    facts = _facts(
        test_stage=True,
        fold={
            "input_window": "20200101..20211231",
            "validation_period": "20220101..20221231",
            "valid_decision_time": "2021-12-31T23:59:59+08:00",
        },
        snapshot_config=screened,
        schedule={"period": "month", "inference_time": "09:00"},
    )
    scope = facts["research_scope"]
    assert facts["visible_timeline"]["fold_period"] == "year"
    assert "rolls period by period" in scope["development_window"]
    assert scope["universe"].startswith("The universe is screened")
    assert "exclude_st=True" in scope["universe"]
    assert "boards=['main']" in scope["universe"]
    assert "first available trading day of each month at 09:00" in scope["strategy_cadence"]


def test_meta_facts_carry_universe_and_cadence_but_no_fold_window() -> None:
    facts = _facts(
        kind="meta_learning",
        meta_learning_id="epoch_001",
        experiment_parameters={
            "fold_period": "quarter",
            "validation_periods": 4,
            "schedule": {"period": "day", "inference_time": "08:30"},
            "snapshot_config": SnapshotConfig().to_record(),
        },
        fold={},
        schedule={},
    )
    scope = facts["research_scope"]
    assert "development_window" not in scope
    assert scope["universe"].startswith("The universe is unfiltered")
    assert "every trading day at 08:30" in scope["strategy_cadence"]
    # The geometry a Meta reviews Folds against comes from the same block; a
    # live Meta once read "every trading day at None" and no fold_period at all
    # because the pipeline never wrote it.
    assert facts["visible_timeline"]["fold_period"] == "quarter"
    assert facts["visible_timeline"]["validation_periods"] == 4


def test_the_validation_window_length_is_a_fold_fact() -> None:
    assert _facts(validation_periods=4)["visible_timeline"]["validation_periods"] == 4
    assert "validation_periods" not in _facts()["visible_timeline"]


def test_both_strategy_wall_clocks_reach_the_session() -> None:
    """The prompts tell the agent to read both, so both must be projected."""

    budgets = _facts(
        budgets={
            "max_llm_calls": 1600,
            "deadline_seconds": 43200,
            "strategy_inference_timeout_seconds": 30.0,
            "strategy_fit_timeout_seconds": 1800,
        }
    )["budgets"]
    assert budgets["strategy_inference_timeout_seconds"] == 30.0
    assert budgets["strategy_fit_timeout_seconds"] == 1800


def test_the_session_deadline_names_the_wrap_up_grace_inside_it() -> None:
    """``deadline_seconds`` is main deadline PLUS grace.

    Without the split the session plans against a wall clock ten minutes later
    than the one its directive names and the one hard finalization uses. Meta
    has no wrap-up window, so it must not be told it has one.
    """

    fold = _facts(
        budgets={"deadline_seconds": 43800.0, "deadline_grace_seconds": 600.0}
    )["budgets"]
    assert fold["deadline_seconds"] == 43800.0
    assert fold["deadline_grace_seconds"] == 600.0

    meta = _facts(kind="meta_learning", budgets={"deadline_seconds": 43200.0})["budgets"]
    assert meta["deadline_seconds"] == 43200.0
    assert "deadline_grace_seconds" not in meta


def test_fold_and_meta_are_told_deadline_seconds_is_pausable_effective_time() -> None:
    """Both sessions must see the pause clock next to ``deadline_seconds``.

    Every replay tool -- ``smoke_backtest`` and ``run_null_control`` included
    -- pauses the budget; shell and sub-agent waits do not, and must not be
    written as exemptions in the sentence the sessions actually read.
    """

    expected = (
        "`deadline_seconds` 统计可暂停的有效推理时间；"
        "`smoke_backtest`、`batch_validate` 和 `run_null_control` "
        "调用期间暂停计时，"
        "因此会话总墙钟可能更长。"
    )
    fold = _facts(
        budgets={"deadline_seconds": 43800.0, "deadline_grace_seconds": 600.0}
    )
    meta = _facts(kind="meta_learning", budgets={"deadline_seconds": 43200.0})
    assert fold["budgets"]["deadline_seconds_note"] == expected
    assert meta["budgets"]["deadline_seconds_note"] == expected
    for name in ("shell", "subagent", "sub-agent", "子代理"):
        assert name not in fold["budgets"]["deadline_seconds_note"]
    assert expected in build_system_prompt(mode="fold", experiment_facts=fold)
    assert expected in build_system_prompt(mode="meta", experiment_facts=meta)
    assert expected in build_meta_learning_prompt(experiment_facts=meta)


def test_the_facts_say_whether_a_parent_control_baseline_exists() -> None:
    """An initial template is a mounted starting point, not a parent artifact.

    The host seeds a ``parent_control`` node only when the pre-session parent
    replay produced a result, and records that outcome on the run manifest.
    Left implicit, four first-Fold sessions read the missing block as a fault
    and either spent a backtest reproducing the template or silently redefined
    their baseline, so the absence is a stated fact.
    """

    inherited = _facts(
        is_initial_artifact=False,
        parent_control_available=True,
        parent_strategy_artifact_id="strategy_epoch_001_fold_2022",
    )["artifact_contract"]["parent"]
    assert inherited["kind"] == "frozen_artifact"
    assert inherited["parent_control_available"] is True

    template = _facts(
        is_initial_artifact=True,
        parent_control_available=False,
        template_ref="agent_output_template",
    )["artifact_contract"]["parent"]
    assert template["kind"] == "initial_template"
    # False, not absent: compact_mapping drops empty values, so the fact has to
    # survive as a bool for the submit contract's clause to have a referent.
    assert template["parent_control_available"] is False

    # A parent whose pre-session control replay failed: the artifact is still
    # inherited (kind stays frozen_artifact) but no parent_control node exists,
    # and the submit contract tells the Agent to select that node by id.
    # Because that contract also makes this the one case worth re-replaying the
    # parent on the session's own budget, the reason is published with it --
    # the confirm arm spent three slots re-replaying a parent it could not see
    # the failure of.
    failed = _facts(
        is_initial_artifact=False,
        parent_control_available=False,
        parent_control_error="BacktestError: window shape cannot be larger",
        parent_strategy_artifact_id="strategy_epoch_001_fold_2022",
    )["artifact_contract"]["parent"]
    assert failed["kind"] == "frozen_artifact"
    assert failed["parent_control_available"] is False
    assert failed["parent_control_error"] == (
        "BacktestError: window shape cannot be larger"
    )
    # Nothing failed, nothing to explain.
    assert "parent_control_error" not in inherited

    # Manifests written before the field, and Meta sessions, fall back to
    # "an inherited parent exists".
    legacy = _facts(is_initial_artifact=False)["artifact_contract"]["parent"]
    assert legacy["parent_control_available"] is True


def test_the_intraday_lookback_is_named_only_when_minutes_are_built() -> None:
    """Without minute bars the execution policy reports none available, so no
    minute lookback window may be advertised either."""

    timeline = _facts()["visible_timeline"]
    assert "intraday_trade_days" not in timeline["snapshot_windows"]
    assert "decision_snapshot_intraday_lookback_trade_days" not in timeline
    assert timeline["execution_policy"]["historical_minutes_available"] is False

    with_minutes = _facts(
        snapshot_config=SnapshotConfig(include_intraday=True).to_record()
    )["visible_timeline"]
    assert with_minutes["snapshot_windows"]["intraday_trade_days"] == 21
    assert with_minutes["decision_snapshot_intraday_lookback_trade_days"] == 21


def test_a_zero_row_domain_file_is_reported_as_unavailable() -> None:
    """A switched-off domain (minutes) and a domain with nothing in the visible
    window (the auction before 2025) are still written as zero-row Parquet
    files. Reading availability off file presence told the session it could
    price orders at an exact minute or at the auction, and every such order was
    rejected as ``missing_execution_price``."""

    empty = _facts(
        data_summary=_data_summary(
            {
                "intraday_1min.parquet": 0,
                "auction.parquet": 0,
                "events.parquet": 13_770_524,
                "text_index.parquet": 0,
            }
        )
    )["visible_timeline"]["execution_policy"]
    assert empty["historical_minutes_available"] is False
    assert empty["auction_available"] is False
    assert empty["text_available"] is False
    # A populated domain in the same summary still reports available.
    assert empty["events_available"] is True

    populated = _facts(
        data_summary=_data_summary(
            {"intraday_1min.parquet": 4_800_000, "auction.parquet": 5_067}
        )
    )["visible_timeline"]["execution_policy"]
    assert populated["historical_minutes_available"] is True
    assert populated["auction_available"] is True
    assert populated["events_available"] is False


def test_a_cpu_only_experiment_still_states_its_strategy_gpu_count() -> None:
    """0 is the answer "every formal replay runs on CPU", not a missing fact:
    dropping it would leave the session unable to tell a CPU-only arm from an
    older manifest that never published the field."""

    budgets = _facts(
        budgets={
            "strategy_fit_timeout_seconds": 3600.0,
            "strategy_gpu_count": 0,
        }
    )["budgets"]
    assert budgets["strategy_gpu_count"] == 0
    assert "strategy_gpu_count" not in _facts(budgets={})["budgets"]


def test_the_facts_publish_the_strategy_containers_cpu_quota_and_batch_width() -> None:
    """A session sizing its own thread pools must not have to guess.

    The formal strategy container is started with a fixed CPU quota and has its
    OMP/MKL/OPENBLAS/NUMEXPR variables set from it; one session guessed half of
    it and lost two backtest slots to fit timeouts. Both this quota and how many
    replays a batch runs at once are read from the code that enforces them, so
    the fact cannot drift from the container.
    """

    from autotrade.environment.sandbox import SandboxLimits
    from autotrade.pipelines.local_backend import BATCH_VALIDATE_MAX_CONCURRENCY

    fold = _facts()["budgets"]
    assert fold["strategy_cpus"] == SandboxLimits().cpus
    assert fold["batch_validate_max_concurrency"] == BATCH_VALIDATE_MAX_CONCURRENCY
    # The width alone would mislead: the fit clock the batch is judged against
    # scales with it, so the rule travels with the number.
    assert fold["batch_validate_fit_timeout_note"] == BATCH_VALIDATE_FIT_TIMEOUT_NOTE
    assert "strategy_cpus" in build_system_prompt(mode="fold", experiment_facts=_facts())
    # Meta runs no replay of its own, so the batch width is not its fact; the
    # container quota still is, because fit(context) runs in that container.
    meta = _facts(kind="meta_learning")["budgets"]
    assert meta["strategy_cpus"] == SandboxLimits().cpus
    assert "batch_validate_max_concurrency" not in meta
    assert "batch_validate_fit_timeout_note" not in meta
