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
    build_experiment_facts,
)
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
from autotrade.pipelines.local_backend import BATCH_VALIDATE_MAX_CONCURRENCY
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


def _experiment(
    tmp_path: Path, *, fold_id: str = "fold_2022", **overrides: object
) -> tuple[Path, Path]:
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
                        "fold_id": fold_id,
                        "fold_index": 0,
                    },
                    {
                        "session_key": f"epoch_001/{fold_id}",
                        "kind": "fold",
                        "epoch_id": "epoch_001",
                        "fold_id": fold_id,
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


def _preview_of(directory: Path, repo: Path, session_key: str) -> str:
    return str(build_prompt_preview(directory, session_key, "", repo_root=repo)["prompt"])


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
        # The formal strategy container's GPU allocation; 0 is published as
        # "every formal replay runs on CPU", not omitted.
        "strategy_gpu_count": limits.gpu_count,
        # Its CPU quota, which is also the thread count its numeric libraries
        # are pinned to, and how many replays one batch runs at once — plus the
        # rule that the fit clock scales with that width.
        "strategy_cpus": limits.cpus,
        "batch_validate_max_concurrency": BATCH_VALIDATE_MAX_CONCURRENCY,
        "batch_validate_fit_timeout_note": BATCH_VALIDATE_FIT_TIMEOUT_NOTE,
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


def test_preview_manifests_carry_the_session_s_own_experiment_parameters(tmp_path: Path):
    """A preview that drops a parameter publishes a different fact, not a gap.

    A Meta manifest has no Fold of its own, so the geometry, the universe
    screen, the cadence and the broker profile ride in ``experiment_parameters``
    (``pipelines/experiment.py``). Building the preview's manifest without that
    block silently rendered the unscreened-universe sentence, an empty cadence
    and a broker profile missing its costs, which is worse than the
    ``RUNTIME_PLACEHOLDER`` the preview promises for facts it cannot know.
    """
    from autotrade.pipelines.meta_schedule import meta_learning_id
    from autotrade.pipelines.worker import load_worker_options

    screened_quarterly = {
        "fold_period": "quarter",
        "validation_periods": 4,
        "development_first_period": "2022Q1",
        "development_last_period": "2025Q4",
        "screen_exclude_st": True,
        "screen_exclude_new_listed_days": 60,
    }
    directory, repo = _experiment(tmp_path, fold_id="fold_2022Q4", **screened_quarterly)
    options = load_worker_options(directory, repo_root=repo)
    rolling = options.rolling
    # The Meta manifest a session writes, assembled as the pipeline does.
    session_facts = build_experiment_facts(
        manifest={
            "experiment_id": rolling.experiment_id,
            "epoch_id": "epoch_001",
            "run_id": "run_001",
            "meta_learning_id": meta_learning_id("epoch_001", 0),
            "trigger_after_folds": 0,
            "kind": "meta_learning",
            "experiment_parameters": {
                "fold_period": rolling.fold_period,
                "validation_periods": rolling.validation_periods,
                "schedule": rolling.schedule.to_record(),
                "broker_profile": rolling.broker_profile.to_record(),
                "snapshot_config": options.snapshot_config.to_record(),
            },
            "is_initial_artifact": True,
            "template_ref": "agent_output_template",
            "budgets": {"max_llm_calls": rolling.max_llm_calls},
        },
        ref_store=AgentRefStore(directory),
    )
    meta = _facts(str(_preview_of(directory, repo, META_KEY)))
    assert set(meta) == set(session_facts)
    assert set(meta["visible_timeline"]) == set(session_facts["visible_timeline"])
    assert set(meta["broker_replay"]) == set(session_facts["broker_replay"])
    assert meta["research_scope"] == session_facts["research_scope"]
    assert meta["broker_replay"]["profile_id"] == session_facts["broker_replay"]["profile_id"]
    # The two facts the missing block used to invert: a screened universe read
    # as unfiltered, and the geometry that separates this round from the
    # console default.
    assert "screened at the decision anchor" in meta["research_scope"]["universe"]
    assert "exclude_st=True" in meta["research_scope"]["universe"]
    assert meta["visible_timeline"]["fold_period"] == "quarter"
    assert meta["visible_timeline"]["validation_periods"] == 4
    # A Fold session carries the same geometry on the manifest itself.
    fold = _facts(str(_preview_of(directory, repo, "epoch_001/fold_2022Q4")))
    assert fold["visible_timeline"]["validation_periods"] == 4
    assert fold["research_scope"]["universe"] == meta["research_scope"]["universe"]
    assert fold["research_scope"]["strategy_cadence"] == meta["research_scope"]["strategy_cadence"]


def test_inherited_prior_reaches_the_preview_before_the_first_meta(tmp_path: Path):
    """An experiment created with ``inherit_memory_from`` starts from another
    experiment's PRIOR; until its own first Meta row exists the preview shows
    that PRIOR, exactly as the worker hands it to the first session."""
    from autotrade.pipelines.inherited_memory import import_inherited_memory
    from autotrade.pipelines.ledger import ExperimentLedger
    from autotrade.pipelines.prior import ExperimentPriorStore

    prior = "Closed: small-cap reversal (null percentile 0.5). Next: post-event drift."
    source = tmp_path / "experiments" / "src"
    ExperimentPriorStore(source).publish(prior, generation_id="gen_1")
    ExperimentLedger(source / "ledgers" / "experiment_ledger.jsonl").append(
        {
            "record_type": "meta_learning",
            "experiment_id": "src",
            "epoch_id": "epoch_001",
            "fold_id": "meta_001",
            "run_id": "run_m",
            "prior": prior,
            "prior_generation_id": "gen_1",
        }
    )
    directory, repo = _experiment(tmp_path)
    params_path = directory / "hitl" / "params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    params["_inherited_memory"] = import_inherited_memory(directory, source, source_id="src")
    params_path.write_text(json.dumps(params), encoding="utf-8")

    # The Fold session reads the PRIOR in its prompt; the Meta session reads
    # it from its workspace and is told only that a previous PRIOR exists.
    assert prior in _preview_of(directory, repo, FOLD_KEY)
    meta_facts = _facts(_preview_of(directory, repo, META_KEY))
    assert meta_facts["meta_learning"]["previous_prior_available"] is True


def test_deployment_adjustment_preview_is_the_deployment_prompt(tmp_path: Path):
    """The post-Held-out session previews through the same chain: its window
    runs from the configured start to the release's last trading day, the
    facts say the Held-out is visible, and the deployment contract replaces
    the research guardrails."""
    from autotrade.agent.prompts import (
        DEPLOYMENT_DEFAULT_INSTRUCTION,
        DEPLOYMENT_SECTION,
        FOLD_GUARDRAILS_SECTION,
    )

    directory, repo = _experiment(tmp_path, deployment_adjustment_start="20190301")
    schedule_path = directory / "hitl" / "schedule.json"
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    schedule["sessions"].append(
        {
            "key": "deployment_adjustment",
            "kind": "deployment_adjustment",
            "epoch_id": "epoch_001",
            "fold_id": "deployment_20190301..20191231",
        }
    )
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")
    preview = build_prompt_preview(directory, "deployment_adjustment", "", repo_root=repo)
    prompt = str(preview["prompt"])
    assert DEPLOYMENT_SECTION.strip() in prompt
    assert FOLD_GUARDRAILS_SECTION.strip() not in prompt
    assert DEPLOYMENT_DEFAULT_INSTRUCTION.strip() in prompt
    facts = _facts(prompt)
    assert facts["identity"]["session_kind"] == "deployment_adjustment"
    assert facts["visibility_policy"]["heldout_visible"] is True
    assert "heldout" not in facts["forbidden"]
    assert facts["budgets"]["max_backtests_per_fold"] == rolling_default("deployment_max_backtests")
    assert facts["budgets"]["max_steps"] == rolling_default("deployment_max_backtests")
    # From the configured start to the release's last trading day.
    assert facts["research_scope"]["development_window"].startswith(
        f"This Fold's validation period is 20190301..{_trading_days()[-1]}."
    )


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
