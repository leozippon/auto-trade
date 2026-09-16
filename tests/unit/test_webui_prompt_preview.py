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
from autotrade.agent.experiment_facts import (
    BATCH_VALIDATE_FIT_TIMEOUT_NOTE,
    DEADLINE_SECONDS_NOTE,
    REPLAY_YEARS_NOTE,
    SMOKE_PROBE_NOTE,
)
from autotrade.agent.prompts import (
    SESSION_DEFAULT_INSTRUCTION,
    SESSION_DYNAMIC_CONTEXT_HEADER,
    SESSION_STATIC_SECTIONS,
    STEP_TREE_SECTION,
)
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.sandbox import SandboxLimits
from autotrade.pipelines.config import (
    DEFAULT_DEADLINE_GRACE_MINUTES,
    DEFAULT_RESEARCH_GEOMETRY,
    rolling_default,
    session_deadline_seconds,
)
from autotrade.pipelines.hitl_state import build_session_plan
from autotrade.pipelines.local_backend import BATCH_VALIDATE_MAX_CONCURRENCY
from autotrade.webui.prompt_preview import RUNTIME_PLACEHOLDER, build_prompt_preview

SESSION_KEY = "research"

# Wording retired from the prompts and the calendar. A preview that still shows
# any of it is serving a stale copy rather than the current builder.
RETIRED_WORDING = ("fold_period=quarter", "单文件", "30 秒", "first_test_period")


GEOMETRY = DEFAULT_RESEARCH_GEOMETRY


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
        **GEOMETRY.to_record(),
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
        json.dumps(build_session_plan(forward={})), encoding="utf-8"
    )
    return directory, repo


def _preview(tmp_path: Path, session_key: str = SESSION_KEY, directive: str = "", **overrides):
    directory, repo = _experiment(tmp_path, **overrides)
    return build_prompt_preview(directory, session_key, directive, repo_root=repo)


def _preview_of(directory: Path, repo: Path, session_key: str) -> str:
    return str(build_prompt_preview(directory, session_key, "", repo_root=repo)["prompt"])


def test_research_preview_carries_every_current_prompt_section(tmp_path: Path):
    prompt = str(_preview(tmp_path)["prompt"])
    for section in SESSION_STATIC_SECTIONS:
        assert section.strip() in prompt
    assert STEP_TREE_SECTION.strip() in prompt
    assert SESSION_DYNAMIC_CONTEXT_HEADER.strip() in prompt
    # The opening user message the session is actually started with.
    assert SESSION_DEFAULT_INSTRUCTION.strip() in prompt
    for tool in (
        "batch_validate",
        "report_issue",
        "skill_feedback",
        "modification_check",
        "smoke_backtest",
        "step_rollback",
        "finish_session",
        "write_skill",
    ):
        assert f"`{tool}`" in prompt
    for retired in RETIRED_WORDING:
        assert retired not in prompt


def test_research_preview_states_the_pipeline_budgets_and_window(tmp_path: Path):
    prompt = str(_preview(tmp_path)["prompt"])
    facts = _facts(prompt)
    limits = SandboxLimits()
    assert facts["budgets"] == {
        "context_compaction": facts["budgets"]["context_compaction"],
        "deadline_seconds": session_deadline_seconds(
            rolling_default("max_research_minutes"), DEFAULT_DEADLINE_GRACE_MINUTES
        ),
        "deadline_seconds_note": DEADLINE_SECONDS_NOTE,
        "deadline_grace_seconds": DEFAULT_DEADLINE_GRACE_MINUTES * 60.0,
        "finalize_before_deadline_seconds": rolling_default(
            "finalize_before_deadline_seconds"
        ),
        "max_replay_years": rolling_default("max_replay_years"),
        "max_replay_years_note": REPLAY_YEARS_NOTE,
        "max_null_controls": rolling_default("max_null_controls"),
        "max_llm_calls": rolling_default("max_llm_calls"),
        "used_before_this_attempt": {
            "inference_seconds": 0.0,
            "llm_calls": 0,
            "main_calls": 0,
            "subagent_calls": 0,
            "compact_calls": 0,
            "replay_years": 0,
            "null_controls": 0,
        },
        "strategy_fit_timeout_seconds": float(
            rolling_default("strategy_fit_timeout_seconds")
        ),
        "strategy_inference_timeout_seconds": limits.timeout_seconds,
        "strategy_gpu_count": limits.gpu_count,
        "strategy_cpus": limits.cpus,
        "strategy_memory_bytes": limits.memory_bytes,
        "batch_validate_max_concurrency": BATCH_VALIDATE_MAX_CONCURRENCY,
        "batch_validate_fit_timeout_note": BATCH_VALIDATE_FIT_TIMEOUT_NOTE,
        "smoke_backtest_probe_note": SMOKE_PROBE_NOTE,
    }
    # The whole research period, read at research end.
    geometry = facts["research_geometry"]
    assert geometry["research_period"] == f"{GEOMETRY.research_start}..{GEOMETRY.research_end}"
    assert geometry["decision_time"] == GEOMETRY.research_decision_time.isoformat()
    assert [year["label"] for year in geometry["years"]] == ["Y1", "Y2", "Y3", "Y4"]
    assert "session" not in facts["identity"]
    assert facts["budgets"]["used_before_this_attempt"]["replay_years"] == 0
    # Runtime-only facts are marked, never invented.
    assert facts["identity"]["run_id"] == RUNTIME_PLACEHOLDER
    assert facts["visible_timeline"]["execution_policy"]["text_available"] == RUNTIME_PLACEHOLDER
    assert (
        facts["budgets"]["deadline_seconds"] - facts["budgets"]["deadline_grace_seconds"]
        == rolling_default("max_research_minutes") * 60.0
    )
    # The session starts from the template.
    assert facts["artifact_contract"]["start"]["kind"] == "template"
    # No date after research end appears anywhere in the prompt.
    for day in (GEOMETRY.forward_start, GEOMETRY.forward_end, GEOMETRY.heldout_start, GEOMETRY.heldout_end):
        assert day not in prompt
        assert f"{day[:4]}-{day[4:6]}-{day[6:]}" not in prompt


@pytest.mark.parametrize(
    "section", ("PROTOCOL_INSTRUCTION", "SESSION_DYNAMIC_CONTEXT_HEADER")
)
def test_prompt_section_edits_reach_the_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, section: str
):
    marker = "EDITED-PROMPT-SECTION-MARKER"
    before = str(_preview(tmp_path)["prompt"])
    assert marker not in before
    current = getattr(prompts, section)
    edited = (marker,) if isinstance(current, tuple) else marker
    monkeypatch.setattr(prompts, section, edited)
    after = str(_preview(tmp_path / "edited")["prompt"])
    assert marker in after
    assert after != before


def test_preview_follows_the_experiment_parameters(tmp_path: Path):
    directive = "只检验一条可证伪机制"
    prompt = str(
        _preview(
            tmp_path,
            SESSION_KEY,
            directive,
            max_research_minutes=90,
            max_replay_years=7,
            max_llm_calls=123,
            research_directive="以截面因子为主线",
        )["prompt"]
    )
    facts = _facts(prompt)
    assert facts["budgets"]["max_replay_years"] == 7
    assert facts["budgets"]["max_llm_calls"] == 123
    assert facts["budgets"]["deadline_seconds"] == session_deadline_seconds(
        90, DEFAULT_DEADLINE_GRACE_MINUTES
    )
    assert "以截面因子为主线" in prompt
    assert directive in prompt


def test_unknown_and_forward_sessions_are_rejected(tmp_path: Path):
    directory, repo = _experiment(tmp_path)
    with pytest.raises(ValueError, match="forward replay"):
        build_prompt_preview(directory, "forward", "", repo_root=repo)
    with pytest.raises(KeyError):
        build_prompt_preview(directory, "s1", "", repo_root=repo)


def _facts(prompt: str) -> dict:
    """The run-facts JSON block the preview embeds in the system prompt."""
    body = prompt.split("```json\n", 1)[1].split("\n```", 1)[0]
    return json.loads(body)
