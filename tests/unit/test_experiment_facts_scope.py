"""The run facts a session is told: development window, universe policy and
strategy call cadence — one sentence each, with Held-out invisible as ever —
plus the budgets and decision-input windows the same object publishes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.agent.prompts import (
    build_meta_learning_prompt,
    build_system_prompt,
)
from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm import ProviderResponse, ScriptedLLM, ToolCall
from autotrade.environment.replay.style import NEUTRALIZATION_METHOD
from autotrade.environment.runtime import RunManifest, write_json_atomic
from autotrade.pipelines.agent_views import (
    compact_fold_history,
    fold_development_summary,
)
from autotrade.pipelines.local_backend import LLMMetaLearner


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
            "fold_period": "year",
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

    ``daily_backtest``, ``batch_validate`` and ``ask_user`` pause the budget;
    ``smoke_backtest``, shell and sub-agent waits do not, and must not be
    written as exemptions in the sentence the sessions actually read.
    """

    expected = (
        "`deadline_seconds` 统计可暂停的有效推理时间；"
        "`daily_backtest`、`batch_validate` 和 `ask_user` 调用期间暂停计时，"
        "因此会话总墙钟可能更长。"
    )
    fold = _facts(
        budgets={"deadline_seconds": 43800.0, "deadline_grace_seconds": 600.0}
    )
    meta = _facts(kind="meta_learning", budgets={"deadline_seconds": 43200.0})
    assert fold["budgets"]["deadline_seconds_note"] == expected
    assert meta["budgets"]["deadline_seconds_note"] == expected
    for name in ("smoke_backtest", "shell", "subagent", "sub-agent", "子代理"):
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
    failed = _facts(
        is_initial_artifact=False,
        parent_control_available=False,
        parent_strategy_artifact_id="strategy_epoch_001_fold_2022",
    )["artifact_contract"]["parent"]
    assert failed["kind"] == "frozen_artifact"
    assert failed["parent_control_available"] is False

    # Manifests written before the field, and Meta sessions, fall back to
    # "an inherited parent exists".
    legacy = _facts(is_initial_artifact=False)["artifact_contract"]["parent"]
    assert legacy["parent_control_available"] is True


def test_the_compact_history_carries_turnover_and_the_parent_delta(
    tmp_path: Path,
) -> None:
    """Both are named Meta evidence, so the Agent-visible manifest must keep them.

    ``turnover`` is the overfitting tell behind a healthy-looking development
    metric and ``vs_parent`` is the per-candidate selection evidence; the run
    manifest the Agent and the next Meta session read is the projected copy, so
    a key missing from that projection never reaches the compact history at all.
    """

    manifest = RunManifest.create(
        tmp_path / "artifacts" / "run_manifest.json",
        {"kind": "fold", "run_id": "run_x"},
        ref_store=AgentRefStore(tmp_path / "experiment"),
    )
    manifest.append_backtest_summary(
        {
            "result_name": "valid_001",
            "mode": "valid",
            "status": "ok",
            "complete_validation": True,
            "total_return": 0.02,
            "turnover": 8.5,
            "vs_parent": {"total_return": 0.01, "excess_return": 0.004},
        }
    )
    compact = compact_fold_history(
        {
            "record_type": "fold",
            "epoch_id": "epoch_001",
            "fold_id": "fold_2022",
            "run_manifest_ref": str(manifest.path),
        },
        ref_store=AgentRefStore(tmp_path / "experiment"),
    )
    published = compact["backtest_summaries"][0]
    assert published["turnover"] == 8.5
    assert published["vs_parent"]["total_return"] == 0.01


def test_the_neutralization_caliber_is_stated_once_not_once_per_result(
    tmp_path: Path,
) -> None:
    """One constant sentence belongs in the facts, not in every summary.

    ``development_history`` grows with every completed Fold, so repeating the
    ~130-char caliber in each of its backtest summaries spends prompt budget
    that the shared prefix never releases. The NL counters are different: a
    zero under a mounted text corpus is a real reading (the strategy never
    asked), and only a run with the corpus switched off drops them.
    """

    assert _facts()["neutralized_excess_method"] == NEUTRALIZATION_METHOD
    benchmark = {
        "label": "沪深300",
        "neutralized_excess_return": 0.1554,
        "neutralized_excess_method": NEUTRALIZATION_METHOD,
    }
    summary = {
        "result_name": "valid_001",
        "mode": "valid",
        "status": "ok",
        "complete_validation": True,
        "total_return": -0.0056,
        "benchmark": benchmark,
        "nl_calls": 0,
        "nl_llm_calls": 0,
        "nl_wall_seconds": 0.0,
    }

    def history(*, include_text: bool) -> dict[str, object]:
        manifest_ref = tmp_path / f"text_{include_text}" / "run_manifest.json"
        write_json_atomic(
            manifest_ref,
            {
                "snapshot_config": {"replay": {"include_text": include_text}},
                "backtest_summaries": [summary],
            },
        )
        return compact_fold_history(
            {
                "record_type": "fold",
                "epoch_id": "epoch_001",
                "fold_id": "fold_2022",
                "run_manifest_ref": str(manifest_ref),
            },
            ref_store=AgentRefStore(tmp_path / "experiment"),
        )

    with_text = history(include_text=True)["backtest_summaries"][0]
    assert "neutralized_excess_method" not in with_text["benchmark"]
    assert with_text["benchmark"]["neutralized_excess_return"] == 0.1554
    assert with_text["nl_calls"] == 0 and with_text["nl_wall_seconds"] == 0.0

    without_text = history(include_text=False)["backtest_summaries"][0]
    assert not [key for key in without_text if key.startswith("nl_")]
    assert without_text["total_return"] == -0.0056


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


def test_a_meta_run_manifest_publishes_both_strategy_wall_clocks(tmp_path: Path) -> None:
    """Meta may rewrite main.py, ``fit`` included, so its own run facts must
    carry the same strategy wall clocks an ordinary Fold is given."""

    baseline = tmp_path / "baseline" / "main.py"
    baseline.parent.mkdir()
    baseline.write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    learner = LLMMetaLearner(
        llm=ScriptedLLM(
            [ProviderResponse(tool_calls=(ToolCall("f", "finish_meta", {}),))]
        ),
        baseline_strategy=baseline,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        experiment_dir=tmp_path / "experiment",
        runtime_root=tmp_path / "runtime",
        max_llm_calls=2,
        deadline_seconds=30.0,
        decision_timeout_seconds=30.0,
        fit_timeout_seconds=1800.0,
        use_docker=False,
        rebuild_enabled=False,
    )
    learner(
        {
            "run_id": "run_budgets",
            "experiment_id": "exp",
            "epoch_id": "epoch_002",
            "meta_learning_id": "epoch_002",
            "previous_prior": "keep the current transferable direction",
        }
    )
    manifest = json.loads(
        (tmp_path / "run_budgets" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["budgets"]["strategy_inference_timeout_seconds"] == 30.0
    assert manifest["budgets"]["strategy_fit_timeout_seconds"] == 1800.0
    facts = build_experiment_facts(
        manifest=manifest, ref_store=AgentRefStore(tmp_path / "experiment")
    )
    assert facts["budgets"]["strategy_fit_timeout_seconds"] == 1800.0


def test_meta_facts_read_the_data_summary_the_session_is_given(tmp_path: Path) -> None:
    """The Meta system prompt must not call text/events unavailable while the
    data_summary.json installed for the same session lists their rows."""

    baseline = tmp_path / "baseline" / "main.py"
    baseline.parent.mkdir()
    baseline.write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    bundle_summary = tmp_path / "bundle" / "data_summary.json"
    write_json_atomic(
        bundle_summary,
        _data_summary({"events.parquet": 12_600_000, "text_index.parquet": 0}),
    )
    llm = ScriptedLLM(
        [ProviderResponse(tool_calls=(ToolCall("f", "finish_meta", {}),))]
    )
    LLMMetaLearner(
        llm=llm,
        baseline_strategy=baseline,
        artifact_store=FilesystemArtifactStore(tmp_path / "artifacts"),
        experiment_dir=tmp_path / "experiment",
        runtime_root=tmp_path / "runtime",
        max_llm_calls=2,
        deadline_seconds=30.0,
        use_docker=False,
        rebuild_enabled=False,
    )(
        {
            "run_id": "run_facts",
            "experiment_id": "exp",
            "epoch_id": "epoch_002",
            "meta_learning_id": "epoch_002",
            "previous_prior": "keep the current transferable direction",
            "data_summary_ref": str(bundle_summary),
        }
    )
    system_prompt = llm.calls[0]["messages"][0].content or ""
    assert '"events_available": true' in system_prompt
    assert '"text_available": false' in system_prompt


def _completed_fold_record(tmp_path: Path, *, candidates: int) -> dict[str, object]:
    """A ledger fold record shaped like the host writes it, with a run manifest
    carrying ``candidates`` per-candidate backtest summaries."""

    manifest_ref = tmp_path / f"run_{candidates}" / "run_manifest.json"
    write_json_atomic(
        manifest_ref,
        {
            "backtest_summaries": [
                {
                    "result_name": f"valid_{index:03d}",
                    "mode": "valid",
                    "status": "ok",
                    "complete_validation": True,
                    "total_return": 0.01 * index,
                    "sharpe": 0.1 * index,
                    "sub_windows": [{"label": "2023Q1", "return": 0.01}],
                }
                for index in range(candidates)
            ]
        },
    )
    return {
        "record_type": "fold",
        "epoch_id": "epoch_001",
        "fold_id": "fold_2023Q1",
        "run_id": "run_x",
        "run_manifest_ref": str(manifest_ref),
        "validation_period": "20220401..20230331",
        "fold_status": "frozen",
        "finish_reason": "llm_agent_finish_fold",
        "early_stop_reason": None,
        "accept_reasons": [],
        "accept_warnings": ["min_sharpe"],
        "validation_result": {
            "total_return": 0.1229,
            "sharpe": 0.8118,
            "per_stock": {"000001.SZ": [0.1] * 80},
            "benchmark": {
                "excess_return": 0.0836,
                "neutralized_excess_return": 0.1725,
                "neutralized_excess_method": NEUTRALIZATION_METHOD,
            },
        },
        "vs_parent": {
            "excess_return_delta": 0.1229,
            "neutralized_excess_return_delta": 0.2117,
            "max_drawdown_delta": 0.0647,
            "beats_parent": True,
        },
        "selection_statistics": {
            "candidates_evaluated": 17,
            "parent_included": False,
            "deflated_sharpe_probability": 0.3479,
            "trials": 17,
            "sharpe_star": 1.209,
            "unavailable_reason": None,
        },
        "null_control": {
            "dropped_trips_mean": 12.016,
            "excess_percentile": 0.966,
            "k": 500,
            "matched": "circ_mv_decile",
            "null_excess_mean": 0.0427,
            "null_excess_p05": -0.0459,
            "null_excess_p95": 0.1585,
            "observed_excess": 0.1775,
            "rejects_mean": 0.316,
            "seed": 1668853636,
        },
        "parent_control": {
            "status": "ok",
            "parent_strategy_artifact_id": "strategy_raw_id",
            "step_id": "node_raw",
            "validation_result": {"total_return": 0.0, "per_stock": {"000001.SZ": [0.1]}},
            "validation_result_ref": "/host/results/parent_control/result.json",
            "step_result": {
                "label": "2023Q1",
                "start": "20230103",
                "end": "20230331",
                "partial": False,
                "total_return": 0.0071,
                "benchmark": {"benchmark_return": 0.0463, "excess_return": -0.0392},
                "cost_sensitivity": {
                    "excess_at_2x_slippage": -0.0401,
                    "reason": "excess_not_positive",
                    "slippage_bps": 5.0,
                },
                "sharpe": 0.3109,
                "max_drawdown": 0.0639,
                "turnover": 1.819,
                "trade_count": 20,
            },
            "null_control": {
                "excess_percentile": 0.496,
                "observed_excess": 0.0547,
                "seed": 57050769,
                "step": {"start": "20230101", "end": "20230331", "excess_percentile": 0.11},
            },
        },
        "test_period": "20230401..20230630",
        "test_result": {"total_return": 0.5, "sharpe": 2.0},
    }


def test_fold_development_history_is_the_verdict_and_does_not_grow_with_candidates(
    tmp_path: Path,
) -> None:
    """A Fold session's system prompt carries every completed Fold and is never
    compacted, so the per-Fold projection must not scale with how many
    candidates that Fold ran; the trial log stays in the Meta projection."""

    store = AgentRefStore(tmp_path / "experiment")
    two = fold_development_summary(_completed_fold_record(tmp_path, candidates=2), ref_store=store)
    forty = fold_development_summary(_completed_fold_record(tmp_path, candidates=40), ref_store=store)
    assert two == forty
    assert "backtest_summaries" not in two
    # Meta still reads the whole trial log through its own projection.
    meta = compact_fold_history(_completed_fold_record(tmp_path, candidates=40), ref_store=store)
    assert len(meta["backtest_summaries"]) == 40

    assert two["fold_status"] == "frozen"
    assert two["early_stop_reason"] is None
    assert two["accept_warnings"] == ["min_sharpe"]
    assert two["validation_result"]["benchmark"]["neutralized_excess_return"] == 0.1725
    assert two["vs_parent"] == {
        "excess_return_delta": 0.1229,
        "neutralized_excess_return_delta": 0.2117,
        "max_drawdown_delta": 0.0647,
        "beats_parent": True,
    }
    assert two["selection_statistics"]["candidates_evaluated"] == 17
    assert two["selection_statistics"]["deflated_sharpe_probability"] == 0.3479
    assert two["null_control"] == {
        "observed_excess": 0.1775,
        "excess_percentile": 0.966,
        "null_excess_mean": 0.0427,
        "null_excess_p05": -0.0459,
        "null_excess_p95": 0.1585,
        "k": 500,
        "rejects_mean": 0.316,
        "dropped_trips_mean": 12.016,
    }
    control = two["parent_control"]
    assert control["status"] == "ok"
    assert control["step_result"] == {
        "label": "2023Q1",
        "start": "20230103",
        "end": "20230331",
        "partial": False,
        "total_return": 0.0071,
        "sharpe": 0.3109,
        "max_drawdown": 0.0639,
        "trade_count": 20,
        "turnover": 1.819,
        "benchmark": {"benchmark_return": 0.0463, "excess_return": -0.0392},
        "excess_at_2x_slippage": -0.0401,
    }
    assert control["null_control"]["step"]["excess_percentile"] == 0.11

    rendered = json.dumps(two, ensure_ascii=False)
    assert "test_result" not in two
    for leak in ("20230401", "fold_2023Q1", "per_stock", "000001.SZ", "/host/", "strategy_raw_id", "node_raw", "seed", "matched"):
        assert leak not in rendered, leak
    # The caliber is one constant sentence the facts state once, not once per Fold.
    assert "neutralized_excess_method" not in rendered
    assert two["fold_id"] == store.get_or_create("fold", "fold_2023Q1")


def test_fold_development_history_without_a_parent_or_null_is_honest(tmp_path: Path) -> None:
    """A first Fold has no parent control and a backend may run no null: those
    blocks are absent, never invented from a neighbour."""

    record = _completed_fold_record(tmp_path, candidates=1)
    record.update(parent_control=None, vs_parent=None, null_control=None)
    summary = fold_development_summary(record, ref_store=AgentRefStore(tmp_path / "experiment"))
    assert summary["parent_control"] is None
    assert summary["vs_parent"] is None
    assert summary["null_control"] is None
    assert summary["selection_statistics"]["candidates_evaluated"] == 17
