"""The run facts a session is told: development window, universe policy and
strategy call cadence — one sentence each, with Held-out invisible as ever —
plus the budgets and decision-input windows the same object publishes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm import ProviderResponse, ScriptedLLM, ToolCall
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
