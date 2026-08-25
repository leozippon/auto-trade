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
from autotrade.environment.identity import agent_visible_ref
from autotrade.pipelines.experiment import _development_history
from autotrade.pipelines.meta_inputs import (
    build_meta_fold_reviews,
    compact_agent_trace,
    select_meta_review_folds,
)
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
    assert "可写 coding 子代理" in fold
    assert "Meta 子代理/阶段" in meta
    assert "不要再委托子代理" in meta
    assert "agent_trace" in meta
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


def test_meta_fold_reviews_include_strategy_and_agent_trace_not_heldout(tmp_path: Path) -> None:
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
        {"event_type": "llm_call", "content": "planning the next edit", "status": "ok"},
    ]
    compact = compact_agent_trace(events)
    assert [item["event_type"] for item in compact] == [
        "explore_task",
        "explore_llm",
        "explore_tool",
        "explore",
        "llm_call",
    ]
    assert compact[0]["task"] == "inspect daily schema"
    assert compact[2]["ok"] is True
    assert compact[-1]["content"] == "planning the next edit"
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
    strategy_files = review["strategy_files"]
    validation = review["validation_result"]
    test_result = review["test_result"]
    agent_trace = review["agent_trace"]
    assert isinstance(strategy_files, list)
    assert strategy_files[0]["path"] == "main.py"
    assert "generate_orders" in str(strategy_files[0]["content"])
    assert isinstance(agent_trace, list)
    assert agent_trace[0]["task"] == "inspect daily schema"
    assert isinstance(validation, dict)
    assert validation["total_return"] == 0.02
    assert "per_stock" not in validation
    assert isinstance(test_result, dict)
    assert test_result["sharpe"] == 0.4
    assert "weekly_returns" not in test_result
    rendered = str(reviews)
    assert "heldout_2026Q1" not in rendered
    assert 0.99 not in test_result.values()


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
    agent_trace = reviews[0]["agent_trace"]
    assert isinstance(agent_trace, list)
    assert agent_trace[0]["task"] == "count rows"
    assert agent_trace[0]["parent_call_id"] == "call_9"


def test_compact_agent_trace_keeps_recent_complete_tasks() -> None:
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
    compact = compact_agent_trace(early + late + noise)
    assert [item["event_type"] for item in compact] == [
        "explore_task",
        "explore_tool",
        "explore",
        "llm_call",
    ]
    assert compact[0]["task"] == "later rows"
    assert compact[0]["parent_call_id"] == "call_late"
    assert compact[2]["digest"] == "new digest"
    assert compact[3]["content"] == "main dialogue"
    assert all("heldout" not in str(item) for item in compact)


def test_compact_agent_trace_keeps_multiple_recent_tasks_in_order() -> None:
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
    compact = compact_agent_trace(events, max_events=4)
    assert [item["task_id"] for item in compact] == ["b", "b", "c", "c"]
    assert [item["event_type"] for item in compact] == [
        "explore_task",
        "explore",
        "explore_task",
        "explore",
    ]


def test_compact_agent_trace_oversized_latest_task_keeps_trailing_events() -> None:
    events = [
        {
            "event_type": "explore_llm",
            "task_id": "explore_big",
            "round": index,
            "model": "test",
        }
        for index in range(1, 100)
    ]
    compact = compact_agent_trace(events, max_events=80)
    assert len(compact) == 80
    assert compact[0]["round"] == 20
    assert compact[-1]["round"] == 99


def test_compact_agent_trace_keeps_main_and_subagent_without_forbidden_content() -> None:
    body = "def generate_orders(context):\n    return []\n" + ("x" * 200)
    events = [
        {
            "event_type": "session_start",
            "mode": "fold",
            "system_prompt": "FULL SYSTEM PROMPT SECRET",
            "instruction": "USER INSTRUCTION FULL TEXT",
        },
        {
            "event_type": "llm_call",
            "status": "ok",
            "model": "test",
            "tool_names": ["explore", "write_file"],
            "content": "delegate then edit",
        },
        {
            "event_type": "tool_call",
            "tool": "explore",
            "parent_call_id": "call_1",
            "arguments": {"task": "edit strategy"},
            "result": {"ok": True},
        },
        {
            "event_type": "explore_task",
            "task_id": "explore_abc",
            "parent_call_id": "call_1",
            "task": "edit strategy",
            "status": "started",
        },
        {
            "event_type": "explore_tool",
            "task_id": "explore_abc",
            "parent_call_id": "call_1",
            "tool": "write_file",
            "result": {"ok": True},
        },
        {
            "event_type": "explore",
            "task_id": "explore_abc",
            "parent_call_id": "call_1",
            "status": "completed",
            "digest": "wrote main.py",
        },
        {
            "event_type": "tool_call",
            "tool": "write_file",
            "arguments": {
                "path": "/Data2/lzp/ADMCubeQuant/experiments/x/output/main.py",
                "content": body,
            },
            "result": {"ok": True, "value": {"path": "output/main.py"}},
        },
        {"event_type": "wrap_up_started", "remaining_seconds": 12.0},
        {"event_type": "trace_limit_reached", "max_bytes": 32},
        {"event_type": "session_end", "status": "finished", "llm_calls": 4},
        {"event_type": "heldout", "result": {"total_return": 0.99}},
    ]
    compact = compact_agent_trace(events)
    types = [item["event_type"] for item in compact]
    assert types == [
        "session_start",
        "llm_call",
        "tool_call",
        "explore_task",
        "explore_tool",
        "explore",
        "tool_call",
        "wrap_up_started",
        "trace_limit_reached",
        "session_end",
    ]
    rendered = str(compact)
    assert "FULL SYSTEM PROMPT SECRET" not in rendered
    assert "USER INSTRUCTION FULL TEXT" not in rendered
    assert body not in rendered
    assert "/Data2/" not in rendered
    assert "heldout" not in rendered
    write_event = compact[6]
    args = write_event["args"]
    assert isinstance(args, dict)
    assert args["path"] == "[host_path]"
    assert args["content"] == {"omitted": True, "chars": len(body)}
    assert compact[3]["parent_call_id"] == "call_1"
    assert compact[5]["digest"] == "wrote main.py"


def test_compact_agent_trace_redacts_embedded_host_paths_not_sandbox() -> None:
    body = "def generate_orders(context):\n    return []\n" + ("x" * 200)
    events = [
        {
            "event_type": "llm_call",
            "content": (
                "failed reading /Data2/lzp/secret; keep /mnt/agent/workspace/main.py "
                + ("n" * 500)
            ),
        },
        {
            "event_type": "explore_task",
            "task": (
                "inspect (/home/lzp/hidden) and '/tmp/cache' "
                "then /mnt/agent/output/main.py"
            ),
        },
        {
            "event_type": "explore",
            "digest": "ratio 3/4 json 1.5 and a / b stay; host /var/tmp/x goes",
            "error": "boom at /tmp/foo",
        },
        {
            "event_type": "tool_call",
            "tool": "write_file",
            "arguments": {"path": "output/main.py", "content": body},
            "result": {"ok": False, "error": "cannot write (/Data2/lzp/out)"},
        },
    ]
    compact = compact_agent_trace(events)
    rendered = str(compact)
    assert "/Data2/" not in rendered
    assert "/home/" not in rendered
    assert "/tmp/" not in rendered
    assert "/var/" not in rendered
    content = compact[0]["content"]
    original = events[0]["content"]
    assert isinstance(content, str)
    assert isinstance(original, str)
    assert content.startswith(
        "failed reading [host_path]; keep /mnt/agent/workspace/main.py "
    )
    assert "/mnt/agent/workspace/main.py" in content
    assert len(content) < len(original)
    assert compact[1]["task"] == (
        "inspect ([host_path]) and '[host_path]' then /mnt/agent/output/main.py"
    )
    assert compact[2]["digest"] == (
        "ratio 3/4 json 1.5 and a / b stay; host [host_path] goes"
    )
    assert compact[2]["error"] == "boom at [host_path]"
    args = compact[3]["args"]
    assert isinstance(args, dict)
    assert args["content"] == {"omitted": True, "chars": len(body)}
    assert compact[3]["error"] == "cannot write ([host_path])"
    assert body not in rendered


def _fold_record(fold_id: str, run_id: str, *, status: str = "frozen") -> dict[str, object]:
    return {
        "record_type": "fold",
        "epoch_id": "epoch_001",
        "fold_id": fold_id,
        "run_id": run_id,
        "fold_status": status,
    }


def _meta_record(run_id: str, meta_id: str = "epoch_001") -> dict[str, object]:
    return {
        "record_type": "meta_learning",
        "epoch_id": "epoch_001",
        "fold_id": meta_id,
        "run_id": run_id,
        "meta_learning_id": meta_id,
    }


def test_first_meta_review_window_is_empty() -> None:
    folds, window = select_meta_review_folds([_fold_record("fold_2024Q1", "run_a")])
    assert folds == []
    assert window["fold_count"] == 0
    assert window["fold_run_refs"] == []
    assert window["previous_meta_ref"] is None


def test_meta_review_window_only_includes_folds_after_previous_meta() -> None:
    records = [
        _meta_record("run_m1", "epoch_001"),
        _fold_record("fold_old", "run_old"),
        _meta_record("run_m2", "epoch_001_after_fold_001"),
        _fold_record("fold_new", "run_new"),
    ]
    folds, window = select_meta_review_folds(records)
    assert [record["run_id"] for record in folds] == ["run_new"]
    assert window["fold_count"] == 1
    assert window["previous_meta_ref"] == agent_visible_ref(
        "epoch_001_after_fold_001", prefix="meta_ref"
    )
    assert window["fold_run_refs"] == [
        agent_visible_ref("run_new", prefix="run_ref")
    ]
    assert "fold_old" not in str(window)
    assert "fold_new" not in str(window)


def test_meta_review_window_dedupes_and_excludes_heldout_failed_in_progress() -> None:
    records = [
        _meta_record("run_m1"),
        _fold_record("fold_a", "run_a1"),
        _fold_record("fold_a", "run_a2"),
        {
            "record_type": "heldout",
            "epoch_id": "epoch_001",
            "fold_id": "heldout_2026Q1",
            "run_id": "run_h",
        },
        {
            "record_type": "attempt_failed",
            "epoch_id": "epoch_001",
            "fold_id": "fold_b",
            "run_id": "run_fail",
            "phase": "fold",
        },
        _fold_record("fold_c", "run_c", status="in_progress"),
        _fold_record("fold_d", "run_d"),
    ]
    folds, window = select_meta_review_folds(records)
    assert [record["run_id"] for record in folds] == ["run_a2", "run_d"]
    assert window["fold_count"] == 2
    rendered = str(window)
    assert "fold_a" not in rendered
    assert "heldout_2026Q1" not in rendered
    assert "run_fail" not in rendered


def test_meta_review_window_rollback_and_resume_recompute_deterministically() -> None:
    full = [
        _meta_record("run_m1"),
        _fold_record("fold_a", "run_a"),
        _fold_record("fold_b", "run_b"),
        _meta_record("run_m2", "epoch_001_after_fold_002"),
        _fold_record("fold_c", "run_c"),
    ]
    first = select_meta_review_folds(full)
    assert [record["run_id"] for record in first[0]] == ["run_c"]
    assert select_meta_review_folds(full) == first
    rewound = full[:3]
    second = select_meta_review_folds(rewound)
    assert [record["run_id"] for record in second[0]] == ["run_a", "run_b"]
    assert select_meta_review_folds(rewound) == second
    history = _development_history(rewound)
    summaries = history["fold_backtest_summaries"]
    reviews = history["fold_reviews"]
    assert history["review_window"] == second[1]
    assert isinstance(summaries, list)
    assert isinstance(reviews, list)
    assert [row["fold_id"] for row in summaries] == [
        agent_visible_ref("fold_a", prefix="fold_ref"),
        agent_visible_ref("fold_b", prefix="fold_ref"),
    ]
    assert len(reviews) == 2
