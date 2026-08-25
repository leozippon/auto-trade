"""AGENTS.md section injection and experiment-level PRIOR loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from autotrade.agent.agents_md import AgentsMdError, load_required_agents_md_sections
from autotrade.agent.prompts import build_system_prompt
from autotrade.agent.runner import (
    PRIOR_MAX_CHARS,
    TasteFinishTool,
    prior_policy_violation,
)
from autotrade.environment.tools import SafeWorkspace, ToolRegistry, WriteFileTool
from autotrade.pipelines.config import MetaSessionResult
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.meta_inputs import build_meta_fold_reviews, compact_explore_trace
from autotrade.pipelines.prior import ExperimentPriorStore, latest_prior_text
from autotrade.pipelines.worker import _restore_prior_store


def _agents_md(root: Path, *, missing: str | None = None) -> Path:
    sections = {
        "Rules for Multi-Agent Cooperation": "Coordinate; identify sub-agents.",
        "Development Principles": "Smallest complete solution.",
        "Operational Guardrails": "Read enough before writing.",
    }
    parts = ["# Global Guidelines"]
    for title, body in sections.items():
        if title == missing:
            continue
        parts.append(f"## {title}\n{body}")
    path = root / "AGENTS.md"
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return path


def test_required_agents_sections_are_injected_into_fold_and_meta(tmp_path: Path) -> None:
    path = _agents_md(tmp_path)
    extracted = load_required_agents_md_sections(path)
    fold = build_system_prompt(mode="fold", agents_md_path=path)
    meta = build_system_prompt(mode="meta", agents_md_path=path)
    for title in (
        "Rules for Multi-Agent Cooperation",
        "Development Principles",
        "Operational Guardrails",
    ):
        assert f"## {title}" in extracted.text
        assert f"## {title}" in fold
        assert f"## {title}" in meta
    assert "只能通过已注入的 `explore`" in fold
    assert "共享同一会话" in fold
    assert "Meta 子代理/阶段" in meta
    assert "不要再委托子代理" in meta
    assert extracted.sha256
    assert extracted.version == extracted.sha256[:12]


def test_missing_agents_section_fails_explicitly(tmp_path: Path) -> None:
    path = _agents_md(tmp_path, missing="Operational Guardrails")
    with pytest.raises(AgentsMdError, match="Operational Guardrails"):
        load_required_agents_md_sections(path)
    with pytest.raises(AgentsMdError, match="Operational Guardrails"):
        build_system_prompt(mode="fold", agents_md_path=path)
    with pytest.raises(AgentsMdError, match="missing"):
        load_required_agents_md_sections(tmp_path / "absent.md")


def test_fold_system_prompt_injects_prior_full_text(tmp_path: Path) -> None:
    path = _agents_md(tmp_path)
    prior = "先用 grep 定向，再抽样 parquet。\n不要并行委托。"
    prompt = build_system_prompt(
        mode="fold",
        agents_md_path=path,
        prior_prompt=prior,
        taste_prompt="prefer simple signals",
    )
    assert prior in prompt
    assert "只读且不得修改" in prompt
    assert "prefer simple signals" in prompt
    meta = build_system_prompt(
        mode="meta", agents_md_path=path, prior_prompt=prior
    )
    assert prior in meta
    assert "上一份已发布 PRIOR.md 的全文" in meta


def test_fold_write_tools_cannot_overwrite_authoritative_prior(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    store = ExperimentPriorStore(experiment)
    published = store.publish("sample then count", generation_id="meta_001_run_a")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    WriteFileTool(SafeWorkspace(workspace)).invoke(
        {"path": "PRIOR.md", "content": "tampered workspace copy"}
    )
    assert (workspace / "PRIOR.md").read_text(encoding="utf-8") == "tampered workspace copy"
    assert store.current_text().strip() == "sample then count"
    assert Path(published.prior_ref).read_text(encoding="utf-8").strip() == "sample then count"


def test_meta_publishes_keeps_and_rejects_overlong_prior(tmp_path: Path) -> None:
    from autotrade.pipelines.experiment import RollingExperimentPipeline

    experiment = tmp_path / "experiment"
    store = ExperimentPriorStore(experiment)
    first = store.publish("first workflow", generation_id="gen_1")
    pipeline = RollingExperimentPipeline.__new__(RollingExperimentPipeline)
    pipeline.config = type("Cfg", (), {"experiment_dir": experiment})()

    published = pipeline._publish_or_keep_prior(
        MetaSessionResult(taste="keep short", prior="updated workflow notes"),
        previous_prior=first.text,
        generation_id="gen_2",
        deadline_exceeded=False,
    )
    assert published[1] is True
    assert store.current_text().strip() == "updated workflow notes"
    assert Path(published[2]).is_file()

    kept = pipeline._publish_or_keep_prior(
        MetaSessionResult(taste="keep short", prior=""),
        previous_prior="updated workflow notes",
        generation_id="gen_3",
        deadline_exceeded=False,
    )
    assert kept[0] == "updated workflow notes"
    assert kept[1] is False
    assert store.current_text().strip() == "updated workflow notes"

    deadline = pipeline._publish_or_keep_prior(
        MetaSessionResult(taste="keep short", prior="should not publish"),
        previous_prior="updated workflow notes",
        generation_id="gen_4",
        deadline_exceeded=True,
    )
    assert deadline[1] is False
    assert store.current_text().strip() == "updated workflow notes"

    overlong = tmp_path / "PRIOR.md"
    overlong.write_text("x" * (PRIOR_MAX_CHARS + 1), encoding="utf-8")
    assert "characters" in prior_policy_violation(overlong)

    with pytest.raises(FileExistsError):
        pipeline._publish_or_keep_prior(
            MetaSessionResult(taste="keep short", prior="collision"),
            previous_prior="updated workflow notes",
            generation_id="gen_2",
            deadline_exceeded=False,
        )


def test_finish_meta_refuses_overlong_prior_but_taste_still_required(
    tmp_path: Path,
) -> None:
    (tmp_path / "taste.md").write_text("prefer simple signals\n", encoding="utf-8")
    (tmp_path / "PRIOR.md").write_text("y" * (PRIOR_MAX_CHARS + 8), encoding="utf-8")
    result = ToolRegistry([TasteFinishTool(SafeWorkspace(tmp_path))]).invoke(
        "finish_meta", {"taste_path": "taste.md"}
    )
    assert result.ok is False
    assert result.value["error_type"] == "prior_policy"
    (tmp_path / "PRIOR.md").write_text("keep grep first\n", encoding="utf-8")
    accepted = ToolRegistry([TasteFinishTool(SafeWorkspace(tmp_path))]).invoke(
        "finish_meta", {"taste_path": "taste.md"}
    )
    assert accepted.ok is True
    assert accepted.value["status"] == "meta_learning_done"


def test_latest_prior_resume_reads_last_meta_record() -> None:
    records = [
        {"record_type": "fold", "prior": "ignored"},
        {"record_type": "meta_learning", "prior": "first"},
        {"record_type": "meta_learning", "prior": "second", "prior_published": False},
    ]
    assert latest_prior_text(records) == "second"
    assert latest_prior_text([]) == ""
    assert latest_prior_text([{"record_type": "fold", "prior": "ignored"}]) == ""


def test_prior_store_restore_points_current_at_earlier_generation(
    tmp_path: Path,
) -> None:
    store = ExperimentPriorStore(tmp_path / "experiment")
    first = store.publish("first workflow", generation_id="gen_1")
    store.publish("second workflow", generation_id="gen_2")
    restored = store.restore("gen_1")
    assert store.current_text().strip() == "first workflow"
    assert store.current_generation_id() == "gen_1"
    assert restored.prior_ref == first.prior_ref
    assert Path(first.prior_ref).read_text(encoding="utf-8").strip() == "first workflow"
    assert (
        Path(store.root / "generations" / "gen_2" / "PRIOR.md")
        .read_text(encoding="utf-8")
        .strip()
        == "second workflow"
    )


def _append_meta(
    ledger: ExperimentLedger,
    *,
    run_id: str,
    generation_id: str = "",
    prior: str = "",
) -> None:
    ledger.append(
        {
            "record_type": "meta_learning",
            "experiment_id": "exp",
            "epoch_id": "epoch_001",
            "fold_id": run_id,
            "run_id": run_id,
            "prior": prior,
            "prior_generation_id": generation_id or None,
        }
    )


def test_restore_prior_store_rewinds_current_after_rollback(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    store = ExperimentPriorStore(experiment)
    store.publish("first workflow", generation_id="gen_1")
    store.publish("second workflow", generation_id="gen_2")
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    _append_meta(
        ledger, run_id="run_1", generation_id="gen_1", prior="first workflow"
    )
    _restore_prior_store(experiment, ledger)
    assert store.current_generation_id() == "gen_1"
    assert store.current_text().strip() == "first workflow"
    assert (
        Path(store.root / "generations" / "gen_2" / "PRIOR.md")
        .read_text(encoding="utf-8")
        .strip()
        == "second workflow"
    )


def test_restore_prior_store_clears_current_when_no_generation_remains(
    tmp_path: Path,
) -> None:
    experiment = tmp_path / "experiment"
    store = ExperimentPriorStore(experiment)
    store.publish("later workflow", generation_id="gen_2")
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    _restore_prior_store(experiment, ledger)
    assert store.current_generation_id() == ""
    assert store.current_text() == ""
    assert not store.current_path.exists()
    assert (
        Path(store.root / "generations" / "gen_2" / "PRIOR.md")
        .read_text(encoding="utf-8")
        .strip()
        == "later workflow"
    )


def test_restore_prior_store_fails_if_generation_is_missing(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    _append_meta(ledger, run_id="run_ghost", generation_id="ghost", prior="gone")
    with pytest.raises(FileNotFoundError, match="ghost"):
        _restore_prior_store(experiment, ledger)


def test_meta_fold_reviews_include_strategy_and_explore_not_heldout(tmp_path: Path) -> None:
    strategy = tmp_path / "frozen" / "output"
    strategy.mkdir(parents=True)
    (strategy / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    trace = tmp_path / "traces" / "run_fold.jsonl"
    trace.parent.mkdir()
    events = [
        {
            "event_type": "explore_task",
            "task_id": "explore_abc",
            "parent_call_id": "call_1",
            "task": "inspect daily schema",
            "status": "started",
        },
        {
            "event_type": "explore_llm",
            "task_id": "explore_abc",
            "parent_call_id": "call_1",
            "round": 1,
            "model": "test",
            "tool_names": ["grep"],
        },
        {
            "event_type": "explore_tool",
            "task_id": "explore_abc",
            "parent_call_id": "call_1",
            "tool": "grep",
            "result": {"ok": True},
        },
        {
            "event_type": "explore",
            "task_id": "explore_abc",
            "parent_call_id": "call_1",
            "status": "completed",
            "task": "inspect daily schema",
            "digest": "daily has trade_date",
        },
        {"event_type": "llm_call", "content": "should not appear as explore"},
    ]
    compact = compact_explore_trace(events)
    assert [item["event_type"] for item in compact] == [
        "explore_task",
        "explore_llm",
        "explore_tool",
        "explore",
    ]
    assert compact[0]["task"] == "inspect daily schema"
    assert compact[2]["ok"] is True
    trace.write_text(
        "\n".join(
            __import__("json").dumps(event, ensure_ascii=False) for event in events
        )
        + "\n",
        encoding="utf-8",
    )
    fold = {
        "record_type": "fold",
        "epoch_id": "epoch_001",
        "fold_id": "fold_2024Q1",
        "fold_status": "frozen",
        "frozen_strategy_artifact_id": "strategy_epoch_001_fold_2024Q1",
        "frozen_strategy_artifact_path": str(strategy),
        "validation_result": {"total_return": 0.02, "per_stock": {"000001.SZ": [0.1]}},
        "test_result": {"sharpe": 0.4, "weekly_returns": [0.01] * 20},
        "agent_trace_ref": str(trace),
    }
    heldout = {
        "record_type": "heldout",
        "fold_id": "heldout_2026Q1",
        "result": {"total_return": 0.99},
        "frozen_strategy_artifact_path": str(strategy),
    }
    reviews = build_meta_fold_reviews([fold, heldout])
    assert len(reviews) == 1
    review = reviews[0]
    assert review["strategy_files"][0]["path"] == "main.py"
    assert "generate_orders" in str(review["strategy_files"][0]["content"])
    assert review["explore_trace"][0]["task"] == "inspect daily schema"
    assert review["validation_result"]["total_return"] == 0.02
    assert "per_stock" not in review["validation_result"]
    assert review["test_result"]["sharpe"] == 0.4
    assert "weekly_returns" not in review["test_result"]
    rendered = str(reviews)
    assert "heldout_2026Q1" not in rendered
    assert 0.99 not in (
        review["test_result"].values() if review["test_result"] else []
    )


def test_meta_fold_reviews_resolve_trace_from_artifacts_root(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    trace = artifacts / "traces" / "run_fold.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        '{"event_type": "explore_task", "task_id": "explore_xyz", '
        '"parent_call_id": "call_9", "task": "count rows", "status": "started"}\n',
        encoding="utf-8",
    )
    reviews = build_meta_fold_reviews(
        [
            {
                "record_type": "fold",
                "epoch_id": "epoch_001",
                "fold_id": "fold_2024Q1",
                "run_id": "run_fold",
                "fold_status": "frozen",
            }
        ],
        artifacts_root=artifacts,
    )
    assert reviews[0]["explore_trace"][0]["task"] == "count rows"
    assert reviews[0]["explore_trace"][0]["parent_call_id"] == "call_9"


def test_compact_explore_trace_keeps_recent_complete_tasks() -> None:
    early = [
        {
            "event_type": "explore_task",
            "task_id": "explore_old",
            "task": "early schema",
            "status": "started",
        },
        *[
            {
                "event_type": "explore_llm",
                "task_id": "explore_old",
                "round": index,
                "model": "test",
            }
            for index in range(1, 90)
        ],
        {
            "event_type": "explore",
            "task_id": "explore_old",
            "task": "early schema",
            "status": "completed",
            "digest": "old digest",
        },
    ]
    late = [
        {
            "event_type": "explore_task",
            "task_id": "explore_new",
            "parent_call_id": "call_late",
            "task": "later rows",
            "status": "started",
        },
        {
            "event_type": "explore_tool",
            "task_id": "explore_new",
            "tool": "grep",
            "result": {"ok": True},
        },
        {
            "event_type": "explore",
            "task_id": "explore_new",
            "task": "later rows",
            "status": "completed",
            "digest": "new digest",
        },
    ]
    noise = [
        {"event_type": "llm_call", "content": "main dialogue"},
        {"event_type": "heldout", "task": "should not appear"},
    ]
    compact = compact_explore_trace(early + late + noise)
    assert [item["event_type"] for item in compact] == [
        "explore_task",
        "explore_tool",
        "explore",
    ]
    assert all(item["task_id"] == "explore_new" for item in compact)
    assert compact[0]["task"] == "later rows"
    assert compact[0]["parent_call_id"] == "call_late"
    assert compact[2]["digest"] == "new digest"


def test_compact_explore_trace_keeps_multiple_recent_tasks_in_order() -> None:
    events: list[dict[str, object]] = []
    for name in ("a", "b", "c"):
        events.extend(
            [
                {
                    "event_type": "explore_task",
                    "task_id": name,
                    "task": name,
                    "status": "started",
                },
                {
                    "event_type": "explore",
                    "task_id": name,
                    "task": name,
                    "status": "completed",
                },
            ]
        )
    compact = compact_explore_trace(events, max_events=4)
    assert [item["task_id"] for item in compact] == ["b", "b", "c", "c"]
    assert [item["event_type"] for item in compact] == [
        "explore_task",
        "explore",
        "explore_task",
        "explore",
    ]


def test_compact_explore_trace_oversized_latest_task_keeps_trailing_events() -> None:
    events = [
        {
            "event_type": "explore_llm",
            "task_id": "explore_big",
            "round": index,
            "model": "test",
        }
        for index in range(1, 100)
    ]
    compact = compact_explore_trace(events, max_events=80)
    assert len(compact) == 80
    assert compact[0]["round"] == 20
    assert compact[-1]["round"] == 99
