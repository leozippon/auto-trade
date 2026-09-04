"""The console's pre-approval Prompt preview must be the session's own prompt.

The preview is only useful if approving what it shows approves what the session
receives, so these tests pin the properties that break silently: every stable
section of ``prompts.py`` reaches it, the numbers come from the pipeline
defaults rather than a console copy, the researcher's own parameters change it,
and an edit to a prompt section shows up without an edit to the preview.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from autotrade.agent import prompts
from autotrade.agent.experiment_facts import DEADLINE_SECONDS_NOTE
from autotrade.agent.prompts import (
    FOLD_DEFAULT_INSTRUCTION,
    FOLD_DYNAMIC_CONTEXT_HEADER,
    FOLD_STATIC_SECTIONS,
    META_STATIC_SECTIONS,
    STEP_TREE_SECTION,
)
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.sandbox import SandboxLimits
from autotrade.pipelines.config import (
    DEFAULT_DEADLINE_GRACE_MINUTES,
    fold_session_deadline_seconds,
    rolling_default,
)
from autotrade.webui.prompt_preview import RUNTIME_PLACEHOLDER, build_prompt_preview

FOLD_KEY = "epoch_001/fold_2022"
META_KEY = "epoch_001/meta_learning"

# Wording retired from the prompts and the calendar. A preview that still shows
# any of it is serving a stale copy rather than the current builder.
RETIRED_WORDING = ("fold_period=quarter", "单文件", "30 秒", "first_test_period")


def _trading_days() -> list[str]:
    """Two trading days a month across the fixture's whole calendar range."""
    days: list[str] = []
    for year in range(2019, 2027):
        for month in range(1, 13):
            for day in (5, 20):
                days.append(f"{year}{month:02d}{day:02d}")
    return days


def _repo(tmp_path: Path) -> Path:
    """A repository root holding just what the worker's resolution reads."""
    repo = tmp_path / "repo"
    template = repo / "configs" / "agent_output_template"
    template.mkdir(parents=True)
    (template / "main.py").write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    # The local gateway profile resolves its key from the repository env file.
    (repo / ".env").write_text("VLLM_API_KEY=test-key\n", encoding="utf-8")
    (repo / "data" / "pit" / "fundamental_events").mkdir(parents=True)
    raw = repo / "data" / "raw"
    days = _trading_days()
    calendar = raw / "trade_cal" / "exchange=SSE"
    calendar.mkdir(parents=True)
    pd.DataFrame({"cal_date": days, "is_open": ["1"] * len(days)}).to_parquet(
        calendar / "year=2019.parquet"
    )
    # The pinned release must carry every core dataset; only ``daily`` needs its
    # real per-day partitions, which are the pipeline's trading calendar.
    for dataset in ("daily", "daily_basic", "adj_factor", "stk_limit", "suspend_d"):
        directory = raw / dataset
        directory.mkdir(parents=True)
        partitions = days if dataset == "daily" else days[:1]
        for day in partitions:
            pd.DataFrame({"trade_date": [day]}).to_parquet(
                directory / f"trade_date={day}.parquet"
            )
    return repo


def _experiment(tmp_path: Path, **overrides: object) -> tuple[Path, Path]:
    """One console-shaped experiment ready for a preview, minus the worker."""
    repo = _repo(tmp_path)
    experiment_id = "preview_exp"
    directory = tmp_path / "experiments" / experiment_id
    # The console seeds the reference store at creation; without it any ledger
    # row naming a raw fold id would read as a legacy experiment.
    AgentRefStore(directory)
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    params: dict[str, object] = {
        "experiment_id": experiment_id,
        "strategy_path": "configs/agent_output_template/main.py",
        "data_backend": "pit",
        "raw_dir": "data/raw",
        "fundamental_events_root": "data/pit/fundamental_events",
        "fundamental_events_status": "results/data_quality/fundamental_events_status.json",
        "execution_mode": "sandbox",
        "developer_mode": "llm",
        "development_first_period": "2022",
        "development_last_period": "2025",
        "heldout_first_period": "20260101..20260630",
        "heldout_last_period": "20260101..20260630",
        "fold_period": "year",
        # Keeps the pinned release to the core datasets the fixture provides.
        "include_fundamentals": False,
        "include_macro": False,
        "include_events": False,
        "include_text": False,
        "include_intraday": False,
        "gpu_count": 0,
    }
    params.update(overrides)
    (hitl / "params.json").write_text(json.dumps(params), encoding="utf-8")
    (hitl / "schedule.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sessions": [
                    {
                        "session_key": META_KEY,
                        "kind": "meta",
                        "epoch_id": "epoch_001",
                        "fold_id": "fold_2022",
                        "fold_index": 0,
                    },
                    {
                        "session_key": FOLD_KEY,
                        "kind": "fold",
                        "epoch_id": "epoch_001",
                        "fold_id": "fold_2022",
                        "fold_index": 0,
                    },
                    {"key": "heldout", "kind": "heldout", "epoch_id": "epoch_001"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return directory, repo


def _preview(tmp_path: Path, session_key: str = FOLD_KEY, directive: str = "", **overrides):
    directory, repo = _experiment(tmp_path, **overrides)
    return build_prompt_preview(directory, session_key, directive, repo_root=repo)


def test_fold_preview_carries_every_current_prompt_section(tmp_path: Path):
    prompt = str(_preview(tmp_path)["prompt"])
    for section in FOLD_STATIC_SECTIONS:
        assert section.strip() in prompt
    # Enabled by default, so the lineage rules must be in the preview too.
    assert STEP_TREE_SECTION.strip() in prompt
    assert FOLD_DYNAMIC_CONTEXT_HEADER.strip() in prompt
    # The opening user message the session is actually started with.
    assert FOLD_DEFAULT_INSTRUCTION.strip() in prompt
    for tool in (
        "batch_validate",
        "memory_feedback",
        "report_issue",
        "modification_check",
        "smoke_backtest",
        "daily_backtest",
        "step_rollback",
        "finish_fold",
        "write_skill",
    ):
        assert f"`{tool}`" in prompt
    for retired in RETIRED_WORDING:
        assert retired not in prompt


def test_fold_preview_states_the_pipeline_budgets_and_scope(tmp_path: Path):
    prompt = str(_preview(tmp_path)["prompt"])
    facts = _facts(prompt)
    limits = SandboxLimits()
    assert facts["budgets"] == {
        "context_compaction": facts["budgets"]["context_compaction"],
        "deadline_seconds": fold_session_deadline_seconds(
            rolling_default("max_fold_minutes"), DEFAULT_DEADLINE_GRACE_MINUTES
        ),
        "deadline_seconds_note": DEADLINE_SECONDS_NOTE,
        "deadline_grace_seconds": DEFAULT_DEADLINE_GRACE_MINUTES * 60.0,
        "finalize_before_deadline_seconds": rolling_default(
            "finalize_before_deadline_seconds"
        ),
        "max_backtests_per_fold": rolling_default("max_backtests_per_fold"),
        "max_llm_calls": rolling_default("max_llm_calls"),
        "max_steps": rolling_default("max_steps_per_fold"),
        "strategy_fit_timeout_seconds": float(
            rolling_default("strategy_fit_timeout_seconds")
        ),
        "strategy_inference_timeout_seconds": limits.timeout_seconds,
    }
    # The calendar the console configures: one yearly Fold, no frozen Test.
    assert facts["visible_timeline"]["fold_period"] == "year"
    assert facts["visible_timeline"]["visible_validation_replay_period"] == "20220101..20221231"
    assert "one Fold per year" in facts["research_scope"]["development_window"]
    assert facts["identity"]["phase"] == "exploration"
    assert facts["artifact_contract"]["step_tree_enabled"] is True
    # Runtime-only facts are marked, never invented.
    assert facts["identity"]["run_id"] == RUNTIME_PLACEHOLDER
    assert facts["visible_timeline"]["execution_policy"]["text_available"] == RUNTIME_PLACEHOLDER
    # deadline_seconds already contains the grace, so the two must differ by
    # exactly the main deadline the directive and the wrap-up prompt name.
    assert (
        facts["budgets"]["deadline_seconds"] - facts["budgets"]["deadline_grace_seconds"]
        == rolling_default("max_fold_minutes") * 60.0
    )
    # A first Fold inherits the template, so there is no parent and no parent
    # control: the absence is stated, not left to be inferred from a missing
    # block (four first-Fold sessions read that silence as a pipeline fault).
    assert facts["artifact_contract"]["parent"]["kind"] == "initial_template"
    assert facts["artifact_contract"]["parent"]["parent_control_available"] is False
    assert "parent_control" not in facts


def test_meta_preview_is_the_meta_session_prompt(tmp_path: Path):
    preview = _preview(tmp_path, META_KEY)
    prompt = str(preview["prompt"])
    for section in META_STATIC_SECTIONS:
        assert section.strip() in prompt
    assert "`finish_meta`" in prompt
    assert "`memory_feedback`" in prompt
    assert "`report_issue`" in prompt
    # A Meta session runs no replay and is given no strategy schedule block.
    assert "## 本轮调度" not in prompt
    assert "## 当前实验事实" in prompt
    assert "开始本轮 Meta。" in prompt
    facts = _facts(prompt)
    assert facts["identity"]["session_kind"] == "meta_learning"
    assert facts["meta_learning"]["backtest_allowed"] is False
    assert facts["budgets"]["max_llm_calls"] == rolling_default("max_llm_calls")
    for retired in RETIRED_WORDING:
        assert retired not in prompt


@pytest.mark.parametrize(
    ("section", "session_key"),
    (
        ("PROTOCOL_INSTRUCTION", FOLD_KEY),
        ("STEP_TREE_SECTION", FOLD_KEY),
        ("FOLD_DYNAMIC_CONTEXT_HEADER", FOLD_KEY),
        ("EXPLORATION_PHASE_PROMPT", FOLD_KEY),
        ("META_STATIC_SECTIONS", META_KEY),
    ),
)
def test_prompt_section_edits_reach_the_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, section: str, session_key: str
):
    marker = "EDITED-PROMPT-SECTION-MARKER"
    before = str(_preview(tmp_path, session_key)["prompt"])
    assert marker not in before
    current = getattr(prompts, section)
    edited = (marker,) if isinstance(current, tuple) else marker
    monkeypatch.setattr(prompts, section, edited)
    after = str(_preview(tmp_path / "edited", session_key)["prompt"])
    assert marker in after
    assert after != before


def test_preview_follows_the_experiment_parameters(tmp_path: Path):
    directive = "只检验一条可证伪机制"
    prompt = str(
        _preview(
            tmp_path,
            FOLD_KEY,
            directive,
            max_fold_minutes=90,
            max_steps_per_fold=7,
            max_backtests_per_fold=5,
            max_llm_calls=123,
            disable_step_tree=True,
            convergence_start_epoch=1,
            fold_exploration_directive="以截面因子为主线",
        )["prompt"]
    )
    facts = _facts(prompt)
    assert facts["budgets"]["max_steps"] == 7
    assert facts["budgets"]["max_backtests_per_fold"] == 5
    assert facts["budgets"]["max_llm_calls"] == 123
    assert facts["budgets"]["deadline_seconds"] == fold_session_deadline_seconds(
        90, DEFAULT_DEADLINE_GRACE_MINUTES
    )
    assert facts["identity"]["phase"] == "convergence"
    assert STEP_TREE_SECTION.strip() not in prompt
    assert "## 实验级默认 Fold 探索方向（用户注入）" in prompt
    assert "以截面因子为主线" in prompt
    assert "## 研究者本 Fold 指令（用户注入）" in prompt
    assert directive in prompt


def test_inherited_parent_is_stated_without_inventing_the_artifact(tmp_path: Path):
    """A later Fold inherits a parent whose identity the preview cannot know:
    which artifact it is depends on the sessions that still run before it."""
    directory, repo = _experiment(tmp_path)
    ledger = directory / "ledgers" / "experiment_ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "fold",
                "experiment_id": "preview_exp",
                "epoch_id": "epoch_001",
                "fold_id": "fold_2022",
                "run_id": "run_001",
                "fold_status": "frozen",
                "frozen_strategy_artifact_id": "strategy_epoch_001_fold_2022",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompt = str(build_prompt_preview(directory, FOLD_KEY, "", repo_root=repo)["prompt"])
    facts = _facts(prompt)
    parent = facts["artifact_contract"]["parent"]
    assert parent["kind"] == "frozen_artifact"
    assert parent["parent_control_available"] is True
    assert parent["id"] == RUNTIME_PLACEHOLDER
    assert parent["model_artifacts_empty"] == RUNTIME_PLACEHOLDER
    # The host replays the parent on this window before the session starts.
    assert facts["parent_control"] == RUNTIME_PLACEHOLDER
    assert "strategy_epoch_001_fold_2022" not in prompt


def test_unknown_and_heldout_sessions_are_rejected(tmp_path: Path):
    directory, repo = _experiment(tmp_path)
    with pytest.raises(ValueError, match="held-out"):
        build_prompt_preview(directory, "heldout", "", repo_root=repo)
    with pytest.raises(KeyError):
        build_prompt_preview(directory, "epoch_009/fold_1999", "", repo_root=repo)


def _facts(prompt: str) -> dict:
    """The run-facts JSON block the preview embeds in the system prompt."""
    body = prompt.split("```json\n", 1)[1].split("\n```", 1)[0]
    return json.loads(body)
