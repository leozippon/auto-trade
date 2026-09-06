"""Experiment-level PRIOR loop and the Meta-visible agent trace projection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from autotrade.agent.prompts import build_meta_learning_prompt, build_system_prompt
from autotrade.environment.tools import (
    FinishMetaTool,
    SafeWorkspace,
    ToolRegistry,
    WriteFileTool,
)
from autotrade.environment.tools.prior_policy import (
    PRIOR_MAX_CHARS,
    prior_content_violation,
    prior_policy_violation,
)
from autotrade.pipelines.config import MetaSessionResult
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.environment.identity import AgentRefStore
from autotrade.pipelines.experiment import _development_inputs
from autotrade.pipelines.meta_inputs import (
    build_agent_process_summary,
    build_meta_fold_review_bundle,
    compact_agent_trace,
    select_meta_review_folds,
)
from autotrade.pipelines.prior import ExperimentPriorStore, latest_prior_text
from autotrade.pipelines.worker import _restore_prior_store


def test_fold_system_prompt_injects_prior_full_text(tmp_path: Path) -> None:
    prior = "先用 grep 定向，再抽样 parquet。\n不要并行委托。"
    prompt = build_system_prompt(
        mode="fold",
        prior_prompt=prior,
    )
    assert prior in prompt
    assert "权威 PRIOR 不在本 Fold 可写树中" in prompt
    assert "策略方向" in prompt
    meta = build_system_prompt(mode="meta")
    assert prior not in meta
    assert "工作区根的 `PRIOR.md`" in meta


def test_prompts_define_no_edge_pre_registration_and_meta_fold_labels() -> None:
    """Round 20260910: two first Folds froze nodes with no demonstrated edge
    (neutralized excess +0.07%; a style overlay picked for min_return>0), one
    Fold burned backtests on the untouched template, and three Meta sessions
    misread the upcoming Fold's window as a data defect. The prompts now state
    each standard where the Agent reads it."""

    fold = build_system_prompt(mode="fold", experiment_facts={})
    guardrails = fold[fold.index("# 研究方向与守则") :]
    # What "no edge" looks like, in the host's own field names.
    for clause in (
        "「没有证明边际」的三项检验",
        "`vs_parent.beats_parent=false`",
        "没有父本对照时为 null，附 `vs_parent_note`",
        "验证窗口跨多个周期时",
        "`run_null_control(node_id)` 按需算出 `excess_percentile`",
        "`selection_statistics.deflated_sharpe_probability`",
        "以 `outcome=\"no_edge\"` 结束是诚实的结果",
        "不是选择标准",
    ):
        assert clause in guardrails, clause
    # The abstention route and its ledger outcome sit in the submit contract.
    contract = fold[fold.index("# 提交合同") : fold.index("# 禁止事项")]
    assert "过硬门的提名一律被冻结" in contract
    assert '`finish_fold(outcome="no_edge", reason=<证据>)`' in contract
    assert "有父产物记 `no_update`" in contract and "`baseline_missing`" in contract
    assert "才可显式提名 `parent_control` 保留父本" in contract
    assert "否则 Pipeline 不会冻结它" not in fold
    # The hypothesis argument is the binding pre-registration; notes are optional
    # and only count when written before the call.
    assert "`hypothesis` 参数就是有约束力的预登记记录" in guardrails
    assert "调用之后补写的笔记不算预登记" in guardrails
    # The template is not a comparator; audits do not gate a ready batch.
    assert "模板只是交付合同的可运行示例而不是研究基线" in contract
    assert "需要基线就自己跑一次" not in fold
    assert "只读审计不在 Validation 的关键路径上" in fold
    assert '`["python", "/mnt/tools/screen.py", "--help"]`' in guardrails
    # The objective is stated once, at the top of the role section, and the
    # procedure is framed as protecting that judgement, not replacing it.
    role = fold[: fold.index("# 工具")]
    assert "目标是找到真实、可部署的边际" in role and "不是替代它" in role
    # Mechanism family is defined once; a different estimator on the same
    # features no longer counts as structurally different.
    assert "机制家族指收益来源的经济解释" in guardrails
    assert "同一特征集换个估计器不算结构不同" in guardrails
    assert "另一类模型" not in fold
    # One worked pre-registration example, structurally unlike the template.
    assert "示例（只示形式）" in guardrails and "无预告的匹配组" in guardrails
    # Enforced preconditions sit where the tool is described.
    assert "每条不超过 500 字符" in guardrails
    assert "弃权同样要求本会话至少有一次完整 Validation" in contract
    assert "本实验自己的 skills 不是目标" in fold
    # Each shared rule has one home: the tie-break, the one-third threshold
    # and the warn-is-not-selection sentence each appear exactly once.
    assert fold.count("子区间一致性") == 1
    assert fold.count("三分之一") == 1
    assert fold.count("不是选择标准") == 1
    # Delegating a candidate carries its hypothesis and falsification rule.
    assert "再写进它的假设与证伪条件" in fold

    meta = build_system_prompt(mode="meta")
    assert "目标是真实、可部署的边际" in meta
    assert "子区间一致性" not in meta
    assert "描述的是本次 Meta 之后即将开始的 Fold" in meta
    assert "`development_history.fold_reviews[]`" in meta
    assert "两者窗口不同不是数据缺陷" in meta
    prior_rules = meta[meta.index("# PRIOR") : meta.index("# 守则")]
    # Every history entry carries the host statistics: earlier Folds are
    # verifiable and their figures are carried forward or corrected, never
    # dropped as outside the review window.
    assert "`fold_validation_history[]`" in prior_rules
    assert "不得以不在审查窗口或「不可核」为由丢弃" in prior_rules
    for clause in (
        # What each reviewed Fold froze is read from the ledger, never from the
        # Fold session's own narrative (a Meta once asserted "nothing frozen"
        # while the ledger said frozen).
        "`fold_reviews[]` 的 `fold_status`、`finish_mode`",
        "`agent_no_edge`",
        "`hard_reject_reasons`",
        "`null_control.excess_percentile`",
        "`selection_statistics.deflated_sharpe_probability`",
        "PRIOR 逐 Fold 引用这些数值",
        "只能写成待检验，不能写成主线",
        "`no_update` 或 `baseline_missing` 是正当结果",
    ):
        assert clause in prior_rules, clause
    assert "`fold_status` 与 `finish_mode`" in build_meta_learning_prompt()


def test_fold_write_tools_cannot_overwrite_authoritative_prior(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment"
    store = ExperimentPriorStore(experiment)
    store.root.mkdir(parents=True)
    legacy_current = store.root / "CURRENT.md"
    legacy_current.write_text("stale duplicate\n", encoding="utf-8")
    published = store.publish("sample then count", generation_id="meta_001_run_a")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    WriteFileTool(SafeWorkspace(workspace)).invoke(
        {"path": "PRIOR.md", "content": "tampered workspace copy"}
    )
    assert (workspace / "PRIOR.md").read_text(encoding="utf-8") == "tampered workspace copy"
    assert store.current_text().strip() == "sample then count"
    assert legacy_current.read_text(encoding="utf-8") == "stale duplicate\n"
    assert Path(published.prior_ref).read_text(encoding="utf-8").strip() == "sample then count"
    assert not hasattr(published, "sha256")


def test_meta_publishes_keeps_and_rejects_overlong_prior(tmp_path: Path) -> None:
    from autotrade.pipelines.experiment import RollingExperimentPipeline

    experiment = tmp_path / "experiment"
    store = ExperimentPriorStore(experiment)
    first = store.publish("first workflow", generation_id="gen_1")
    pipeline = RollingExperimentPipeline.__new__(RollingExperimentPipeline)
    pipeline.config = type("Cfg", (), {"experiment_dir": experiment})()

    published = pipeline._publish_or_keep_prior(
        MetaSessionResult(prior="updated workflow notes"),
        previous_prior=first.text,
        generation_id="gen_2",
        deadline_exceeded=False,
    )
    assert published[1] is True
    assert store.current_text().strip() == "updated workflow notes"
    assert Path(published[2]).is_file()

    kept = pipeline._publish_or_keep_prior(
        MetaSessionResult(prior=""),
        previous_prior="updated workflow notes",
        generation_id="gen_3",
        deadline_exceeded=False,
    )
    assert kept[0] == "updated workflow notes"
    assert kept[1] is False
    assert store.current_text().strip() == "updated workflow notes"

    unchanged = pipeline._publish_or_keep_prior(
        MetaSessionResult(prior="updated workflow notes\n"),
        previous_prior="updated workflow notes",
        generation_id="gen_same",
        deadline_exceeded=False,
    )
    assert unchanged[1] is False
    assert unchanged[3] == "gen_2"
    assert store.current_generation_id() == "gen_2"

    deadline = pipeline._publish_or_keep_prior(
        MetaSessionResult(prior="should not publish"),
        previous_prior="updated workflow notes",
        generation_id="gen_4",
        deadline_exceeded=True,
    )
    assert deadline[1] is False
    assert store.current_text().strip() == "updated workflow notes"

    overlong = tmp_path / "PRIOR.md"
    overlong.write_text("x" * (PRIOR_MAX_CHARS + 1), encoding="utf-8")
    assert "characters" in prior_policy_violation(overlong)
    with pytest.raises(ValueError, match="characters"):
        pipeline._publish_or_keep_prior(
            MetaSessionResult(prior="x" * (PRIOR_MAX_CHARS + 1)),
            previous_prior="updated workflow notes",
            generation_id="gen_overlong",
            deadline_exceeded=False,
        )

    empty_pipeline = RollingExperimentPipeline.__new__(RollingExperimentPipeline)
    empty_pipeline.config = type("Cfg", (), {"experiment_dir": tmp_path / "empty"})()
    with pytest.raises(ValueError, match="first Meta session"):
        empty_pipeline._publish_or_keep_prior(
            MetaSessionResult(prior=""),
            previous_prior="",
            generation_id="gen_first",
            deadline_exceeded=False,
        )

    with pytest.raises(FileExistsError):
        pipeline._publish_or_keep_prior(
            MetaSessionResult(prior="collision"),
            previous_prior="updated workflow notes",
            generation_id="gen_2",
            deadline_exceeded=False,
        )


def test_finish_meta_requires_a_bounded_nonempty_prior(tmp_path: Path) -> None:
    registry = ToolRegistry([FinishMetaTool(SafeWorkspace(tmp_path))])
    missing = registry.invoke("finish_meta", {})
    assert missing.ok is False
    assert missing.value["error_type"] == "prior_policy"
    (tmp_path / "PRIOR.md").write_text("y" * (PRIOR_MAX_CHARS + 8), encoding="utf-8")
    overlong = registry.invoke("finish_meta", {})
    assert overlong.ok is False
    assert overlong.value["error_type"] == "prior_policy"
    # A content redline names the line and says the check is a line pattern,
    # not a semantic review; the description lists the redlines up front.
    (tmp_path / "PRIOR.md").write_text("方向\nheld-out 表现 0.9\n", encoding="utf-8")
    leaked = registry.invoke("finish_meta", {})
    assert leaked.ok is False and "line 2 leaks Held-out" in leaked.error
    assert "line-level patterns" in leaked.value["retry_hint"]
    for redline in ("held-out/holdout/持有期外/隐藏区间", "YYYYMMDD", "Test/测试", "line-level"):
        assert redline in FinishMetaTool.spec.description
    (tmp_path / "PRIOR.md").write_text("keep grep first\n", encoding="utf-8")
    accepted = registry.invoke("finish_meta", {})
    assert accepted.ok is True
    assert accepted.value["status"] == "meta_learning_done"


def test_latest_prior_resume_reads_last_meta_record() -> None:
    records = [
        {"record_type": "fold", "prior": "ignored"},
        {"record_type": "meta_learning", "prior": "first"},
        {
            "record_type": "meta_learning",
            "prior": "second",
            "prior_published": False,
            "sha256": "ignored legacy field",
        },
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


def test_prior_store_rejects_malformed_or_dangling_current_ref(tmp_path: Path) -> None:
    store = ExperimentPriorStore(tmp_path / "experiment")
    store.root.mkdir(parents=True)
    store.current_pointer_path.write_text("../outside\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generation_id"):
        store.current_text()
    store.current_pointer_path.write_text("missing\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing"):
        store.current_text()


def test_prior_store_serializes_concurrent_publications(tmp_path: Path) -> None:
    store = ExperimentPriorStore(tmp_path / "experiment")
    publications = (("first", "gen_1"), ("second", "gen_2"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda item: store.publish(item[0], generation_id=item[1]), publications)
        )
    current_generation = store.current_generation_id()
    assert current_generation in {"gen_1", "gen_2"}
    expected = {result.generation_id: result.text for result in results}
    assert store.current_text().strip() == expected[current_generation]
    assert all(Path(result.prior_ref).is_file() for result in results)


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
    legacy_current = store.root / "CURRENT.md"
    legacy_current.write_text("ignored legacy copy\n", encoding="utf-8")
    ledger = ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl")
    _restore_prior_store(experiment, ledger)
    assert store.current_generation_id() == ""
    assert store.current_text() == ""
    assert legacy_current.read_text(encoding="utf-8") == "ignored legacy copy\n"
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
            "event_type": "subagent_task",
            "task_id": "agent_abc",
            "parent_call_id": "call_1",
            "role": "auditor",
            "task": "inspect daily schema",
            "status": "started",
        },
        {
            "event_type": "subagent_llm",
            "task_id": "agent_abc",
            "parent_call_id": "call_1",
            "round": 1,
            "model": "test",
            "tool_names": ["grep"],
        },
        {
            "event_type": "subagent_tool",
            "task_id": "agent_abc",
            "parent_call_id": "call_1",
            "tool": "grep",
            "result": {"ok": True},
        },
        {
            "event_type": "subagent",
            "task_id": "agent_abc",
            "parent_call_id": "call_1",
            "status": "completed",
            "role": "auditor",
            "task": "inspect daily schema",
            "summary": "daily has trade_date",
        },
        {"event_type": "llm_call", "content": "planning the next edit", "status": "ok"},
    ]
    compact = compact_agent_trace(events)
    assert [item["event_type"] for item in compact] == [
        "subagent_task",
        "subagent_llm",
        "subagent_tool",
        "subagent",
        "llm_call",
    ]
    assert compact[0]["role"] == "auditor"
    assert "task" not in compact[0]
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
    reviews, _sidecars = build_meta_fold_review_bundle(
        [fold, heldout], ref_store=AgentRefStore(tmp_path / "experiment")
    )
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
    assert agent_trace[0]["role"] == "auditor"
    assert "task" not in agent_trace[0]
    assert isinstance(validation, dict)
    assert validation["total_return"] == 0.02
    assert "per_stock" not in validation
    assert isinstance(test_result, dict)
    assert test_result["sharpe"] == 0.4
    assert "weekly_returns" not in test_result
    rendered = str(reviews)
    assert "heldout_2026Q1" not in rendered
    assert 0.99 not in test_result.values()
    summary = review["agent_process_summary"]
    assert isinstance(summary, dict)
    assert summary["llm_calls"] == 1
    assert summary["subagent"] == {"attempts": 1, "completed": 1, "failed": 0}
    assert summary["daily_backtest"] == 0
    assert "heldout" not in str(summary).lower()
    assert "0.4" not in str(summary)
    full = review["agent_trace_full"]
    assert isinstance(full, dict)
    assert full["available"] is True
    assert full["events"] == 5
    assert full["source_truncated"] is False
    assert str(full["path"]).startswith("inputs/agent_traces/")
    assert str(full["path"]).endswith(".jsonl")
    assert full["raw_jsonl"] is True
    assert full["byte_exact"] is True
    assert "sha256" not in full


def test_meta_fold_reviews_without_trace_ref_are_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    trace = artifacts / "traces" / "run_fold.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        '{"event_type": "subagent_task", "task_id": "agent_xyz", '
        '"parent_call_id": "call_9", "role": "auditor", "task": "count rows", "status": "started"}\n',
        encoding="utf-8",
    )
    reviews, _sidecars = build_meta_fold_review_bundle(
        [
            {
                "record_type": "fold",
                "epoch_id": "epoch_001",
                "fold_id": "fold_2024Q1",
                "run_id": "run_fold",
                "fold_status": "frozen",
            }
        ],
        ref_store=AgentRefStore(tmp_path / "experiment"),
        artifacts_root=artifacts,
    )
    assert reviews[0]["agent_trace"] == []
    full = reviews[0]["agent_trace_full"]
    assert isinstance(full, dict)
    assert full["available"] is False
    assert full["events"] == 0
    assert full["bytes"] == 0
    assert full["source_truncated"] is False
    assert full["path"] is None


def test_meta_fold_reviews_resolve_relative_trace_ref(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    trace = artifacts / "traces" / "run_fold.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text(
        '{"event_type": "subagent_task", "task_id": "agent_xyz", '
        '"parent_call_id": "call_9", "role": "auditor", "task": "count rows", "status": "started"}\n',
        encoding="utf-8",
    )
    reviews, _sidecars = build_meta_fold_review_bundle(
        [
            {
                "record_type": "fold",
                "epoch_id": "epoch_001",
                "fold_id": "fold_2024Q1",
                "run_id": "run_fold",
                "fold_status": "frozen",
                "agent_trace_ref": "traces/run_fold.jsonl",
            }
        ],
        ref_store=AgentRefStore(tmp_path / "experiment"),
        artifacts_root=artifacts,
    )
    agent_trace = reviews[0]["agent_trace"]
    assert isinstance(agent_trace, list)
    assert agent_trace[0]["role"] == "auditor"
    assert "task" not in agent_trace[0]
    assert agent_trace[0]["parent_call_id"] == "call_9"
    full = reviews[0]["agent_trace_full"]
    assert isinstance(full, dict)
    assert full["available"] is True
    assert full["events"] == 1


def test_compact_agent_trace_keeps_recent_complete_tasks() -> None:
    early = [
        {
            "event_type": "subagent_task",
            "task_id": "agent_old",
            "task": "early schema",
            "status": "started",
        },
        *[
            {
                "event_type": "subagent_llm",
                "task_id": "agent_old",
                "round": index,
                "model": "test",
            }
            for index in range(1, 90)
        ],
        {
            "event_type": "subagent",
            "task_id": "agent_old",
            "task": "early schema",
            "status": "completed",
            "summary": "old summary",
        },
    ]
    late = [
        {
            "event_type": "subagent_task",
            "task_id": "agent_new",
            "parent_call_id": "call_late",
            "task": "later rows",
            "status": "started",
        },
        {
            "event_type": "subagent_tool",
            "task_id": "agent_new",
            "tool": "grep",
            "result": {"ok": True},
        },
        {
            "event_type": "subagent",
            "task_id": "agent_new",
            "task": "later rows",
            "status": "completed",
            "summary": "new summary",
        },
    ]
    noise = [
        {"event_type": "llm_call", "content": "main dialogue"},
        {"event_type": "heldout", "task": "should not appear"},
    ]
    compact = compact_agent_trace(early + late + noise)
    assert [item["event_type"] for item in compact] == [
        "subagent_task",
        "subagent_tool",
        "subagent",
        "llm_call",
    ]
    assert "task" not in compact[0]
    assert compact[0]["parent_call_id"] == "call_late"
    assert compact[2]["summary"] == "new summary"
    assert compact[3]["content"] == "main dialogue"
    assert all("heldout" not in str(item) for item in compact)


def test_compact_agent_trace_ignores_legacy_digest() -> None:
    compact = compact_agent_trace(
        [{"event_type": "subagent", "task_id": "legacy", "digest": "do not migrate"}]
    )
    assert "digest" not in compact[0]
    assert "summary" not in compact[0]


def test_compact_agent_trace_keeps_multiple_recent_tasks_in_order() -> None:
    events: list[dict[str, object]] = []
    for name in ("a", "b", "c"):
        events.extend(
            [
                {
                    "event_type": "subagent_task",
                    "task_id": name,
                    "task": name,
                    "status": "started",
                },
                {
                    "event_type": "subagent",
                    "task_id": name,
                    "task": name,
                    "status": "completed",
                },
            ]
        )
    compact = compact_agent_trace(events, max_events=4)
    assert [item["task_id"] for item in compact] == ["b", "b", "c", "c"]
    assert [item["event_type"] for item in compact] == [
        "subagent_task",
        "subagent",
        "subagent_task",
        "subagent",
    ]


def test_compact_agent_trace_oversized_latest_task_keeps_trailing_events() -> None:
    events = [
        {
            "event_type": "subagent_llm",
            "task_id": "agent_big",
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
            "tool_names": ["agent", "write_file"],
            "content": "delegate then edit",
        },
        {
            "event_type": "tool_call",
            "tool": "agent",
            "parent_call_id": "call_1",
            "arguments": {"task": "edit strategy"},
            "result": {"ok": True},
        },
        {
            "event_type": "subagent_task",
            "task_id": "agent_abc",
            "parent_call_id": "call_1",
            "task": "edit strategy",
            "status": "started",
        },
        {
            "event_type": "subagent_tool",
            "task_id": "agent_abc",
            "parent_call_id": "call_1",
            "tool": "write_file",
            "result": {"ok": True},
        },
        {
            "event_type": "subagent",
            "task_id": "agent_abc",
            "parent_call_id": "call_1",
            "status": "completed",
            "summary": "wrote main.py",
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
        "subagent_task",
        "subagent_tool",
        "subagent",
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
    assert compact[5]["summary"] == "wrote main.py"


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
            "event_type": "subagent_task",
            "task": (
                "inspect (/home/lzp/hidden) and '/tmp/cache' "
                "then /mnt/agent/output/main.py"
            ),
        },
        {
            "event_type": "subagent",
            "summary": "ratio 3/4 json 1.5 and a / b stay; host /var/tmp/x goes",
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
    assert "task" not in compact[1]
    assert compact[2]["summary"] == (
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


def test_first_meta_review_window_is_empty(tmp_path: Path) -> None:
    ref_store = AgentRefStore(tmp_path / "experiment")
    folds, window = select_meta_review_folds(
        [_fold_record("fold_2024Q1", "run_a")], ref_store=ref_store
    )
    assert folds == []
    assert window["fold_count"] == 0
    assert window["fold_run_refs"] == []
    assert window["previous_meta_ref"] is None


def test_meta_review_window_only_includes_folds_after_previous_meta(
    tmp_path: Path,
) -> None:
    ref_store = AgentRefStore(tmp_path / "experiment")
    records = [
        _meta_record("run_m1", "epoch_001"),
        _fold_record("fold_old", "run_old"),
        _meta_record("run_m2", "epoch_001_after_fold_001"),
        _fold_record("fold_new", "run_new"),
    ]
    folds, window = select_meta_review_folds(records, ref_store=ref_store)
    assert [record["run_id"] for record in folds] == ["run_new"]
    assert window["fold_count"] == 1
    assert window["previous_meta_ref"] == ref_store.get_or_create(
        "meta", "epoch_001_after_fold_001"
    )
    assert window["fold_run_refs"] == [
        ref_store.get_or_create("run", "run_new")
    ]
    assert "fold_old" not in str(window)
    assert "fold_new" not in str(window)


def test_meta_review_window_dedupes_and_excludes_heldout_failed_in_progress(
    tmp_path: Path,
) -> None:
    ref_store = AgentRefStore(tmp_path / "experiment")
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
    folds, window = select_meta_review_folds(records, ref_store=ref_store)
    assert [record["run_id"] for record in folds] == ["run_a2", "run_d"]
    assert window["fold_count"] == 2
    rendered = str(window)
    assert "fold_a" not in rendered
    assert "heldout_2026Q1" not in rendered
    assert "run_fail" not in rendered


def test_meta_review_window_rollback_and_resume_recompute_deterministically(
    tmp_path: Path,
) -> None:
    ref_store = AgentRefStore(tmp_path / "experiment")
    full = [
        _meta_record("run_m1"),
        _fold_record("fold_a", "run_a"),
        _fold_record("fold_b", "run_b"),
        _meta_record("run_m2", "epoch_001_after_fold_002"),
        _fold_record("fold_c", "run_c"),
    ]
    first = select_meta_review_folds(full, ref_store=ref_store)
    assert [record["run_id"] for record in first[0]] == ["run_c"]
    assert select_meta_review_folds(full, ref_store=ref_store) == first
    rewound = full[:3]
    second = select_meta_review_folds(rewound, ref_store=ref_store)
    assert [record["run_id"] for record in second[0]] == ["run_a", "run_b"]
    assert select_meta_review_folds(rewound, ref_store=ref_store) == second
    history, _sidecars = _development_inputs(rewound, ref_store=ref_store)
    reviews = history["fold_reviews"]
    assert history["review_window"] == second[1]
    assert isinstance(reviews, list)
    assert [row["fold_id"] for row in reviews] == [
        ref_store.get_or_create("fold", "fold_a"),
        ref_store.get_or_create("fold", "fold_b"),
    ]


def test_development_history_lists_each_fold_once(tmp_path: Path) -> None:
    """One completed Fold must never occupy two rows of ``development_history``.

    The review window is a slice of the accumulated history, so a second
    window-scoped copy of the same ``compact_fold_history`` projection makes a
    single Fold look like two to a Meta that counts rows.
    """

    ref_store = AgentRefStore(tmp_path / "experiment")
    records = [
        _meta_record("run_m1"),
        _fold_record("fold_a", "run_a", status="baseline_missing"),
    ]
    history, _sidecars = _development_inputs(records, ref_store=ref_store)
    fold_ref = ref_store.get_or_create("fold", "fold_a")
    assert history["review_window"]["fold_count"] == 1
    assert [row["fold_id"] for row in history["fold_validation_history"]] == [fold_ref]
    assert [row["fold_id"] for row in history["fold_reviews"]] == [fold_ref]
    compact_rows = [
        row
        for key, value in history.items()
        if isinstance(value, list)
        for row in value
        if isinstance(row, dict) and "backtest_summaries" in row
    ]
    assert len(compact_rows) == 1


def test_history_keeps_each_fold_statistics_after_its_review_window_closes(
    tmp_path: Path,
) -> None:
    """An older Fold's statistics stay readable once its window has passed.

    While only ``fold_reviews`` carried ``null_control`` /
    ``selection_statistics`` / ``vs_parent`` / ``parent_control``, two
    consecutive Meta generations discarded their predecessor's (correct)
    figures for an earlier Fold as unverifiable, and the record of how an
    edge decays across Folds was lost every cycle.
    """

    ref_store = AgentRefStore(tmp_path / "experiment")
    older = _fold_record("fold_a", "run_a")
    older.update(
        {
            "frozen_strategy_artifact_id": "raw_strategy_id_a",
            "null_control": {"excess_percentile": 0.48, "observed_excess": 0.2},
            "selection_statistics": {
                "candidates_evaluated": 10,
                "deflated_sharpe_probability": 0.23,
                "trials": 10,
            },
            "vs_parent": {"beats_parent": False, "excess_return_delta": -0.01},
            "parent_control": {
                "status": "ok",
                "step_result": {
                    "label": "2023Q1",
                    "sharpe": 0.2,
                    "benchmark": {"excess_return": -0.0417},
                },
                "null_control": {
                    "excess_percentile": 0.596,
                    "step": {"excess_percentile": 0.592},
                },
            },
        }
    )
    records = [
        _meta_record("run_m1"),
        older,
        _meta_record("run_m2", "epoch_001_after_fold_001"),
        _fold_record("fold_b", "run_b"),
    ]

    history, _sidecars = _development_inputs(records, ref_store=ref_store)

    # The window holds only the newer Fold; the older one is history alone.
    assert [row["fold_id"] for row in history["fold_reviews"]] == [
        ref_store.get_or_create("fold", "fold_b")
    ]
    entry = next(
        row
        for row in history["fold_validation_history"]
        if row["fold_id"] == ref_store.get_or_create("fold", "fold_a")
    )
    assert entry["null_control"]["excess_percentile"] == 0.48
    assert entry["selection_statistics"]["deflated_sharpe_probability"] == 0.23
    assert entry["selection_statistics"]["trials"] == 10
    assert entry["vs_parent"]["beats_parent"] is False
    assert entry["frozen_strategy_artifact_id"] == ref_store.get_or_create(
        "strategy", "raw_strategy_id_a"
    )
    # The inherited parent's new-period row and the null percentile of that
    # same span: the one forward result a trailing window holds.
    parent_control = entry["parent_control"]
    assert parent_control["step_result"]["benchmark"]["excess_return"] == -0.0417
    assert parent_control["null_control"]["step"]["excess_percentile"] == 0.592
    # Host ids never cross the boundary, here as anywhere else.
    assert "raw_strategy_id_a" not in str(entry)
    assert "fold_a" not in str(entry)


def test_agent_process_summary_counts_are_bounded_and_redacted() -> None:
    events = [
        {"event_type": "session_end", "llm_calls": 7, "subagent_attempts": 2},
        {
            "event_type": "subagent",
            "status": "completed",
        },
        {
            "event_type": "subagent",
            "status": "error",
        },
        {
            "event_type": "tool_call",
            "tool": "daily_backtest",
            "ok": True,
        },
        {
            "event_type": "tool_call",
            "tool": "grep",
            "ok": False,
            "error": "boom at /Data2/lzp/secret",
        },
        {
            "event_type": "tool_call",
            "tool": "grep",
            "ok": False,
            "error": "boom at /Data2/lzp/secret",
        },
        {
            "event_type": "tool_call",
            "tool": "read_file",
            "ok": False,
            "error": "missing",
        },
    ]
    summary = build_agent_process_summary(events)
    assert summary["llm_calls"] == 7
    assert summary["subagent"] == {"attempts": 2, "completed": 1, "failed": 1}
    assert summary["daily_backtest"] == 1
    assert summary["tool_failures"] == 3
    # The by-tool tally is the complete breakdown and sums back to the total;
    # only the signature list drops the one-off read_file error.
    assert summary["tool_failures_by_tool"] == {"grep": 2, "read_file": 1}
    assert sum(summary["tool_failures_by_tool"].values()) == summary["tool_failures"]
    assert summary["repeated_failure_signatures"] == [
        {"tool": "grep", "count": 2, "error": "boom at [host_path]"}
    ]
    rendered = str(summary)
    assert "/Data2/" not in rendered
    assert "inspect daily schema" not in rendered


def test_agent_process_summary_caps_repeated_failure_signatures() -> None:
    events = [
        {
            "event_type": "tool_call",
            "tool": f"tool_{index}",
            "ok": False,
            "error": f"fail-{index}",
        }
        for index in range(12)
        for _repeat in range(2)
    ]
    summary = build_agent_process_summary(events)
    assert summary["tool_failures"] == 24
    # The signature list is capped; the by-tool tally stays complete, so the
    # two can never be read as the same breakdown.
    assert len(summary["repeated_failure_signatures"]) == 8
    assert len(summary["tool_failures_by_tool"]) == 12
    assert sum(summary["tool_failures_by_tool"].values()) == 24


def test_agent_process_summary_by_tool_covers_one_off_and_subagent_failures() -> None:
    """Every failed call reaches the by-tool tally, repeated or not.

    A one-off error and a sub-agent tool failure are exactly what the
    ``repeated_failure_signatures`` rule drops, and reading that list as the
    failure breakdown is what makes the block look self-contradictory.
    """

    events = [
        {"event_type": "tool_call", "tool": "shell", "ok": False, "error": "bad argv"},
        {"event_type": "tool_call", "tool": "shell", "ok": False, "error": "bad argv"},
        {"event_type": "tool_call", "tool": "shell", "ok": False, "error": "no such cwd"},
        {"event_type": "tool_call", "tool": "agent", "ok": False, "error": "resume is not an action"},
        {
            "event_type": "subagent_tool",
            "tool": "read_file",
            "result": {"ok": False, "error": "path must be relative"},
        },
    ]
    summary = build_agent_process_summary(events)
    assert summary["tool_failures"] == 5
    assert summary["tool_failures_by_tool"] == {"shell": 3, "agent": 1, "read_file": 1}
    assert sum(summary["tool_failures_by_tool"].values()) == summary["tool_failures"]
    assert summary["repeated_failure_signatures"] == [
        {"tool": "shell", "count": 2, "error": "bad argv"}
    ]


def test_prior_content_allows_boundary_sentences_and_rejects_leaks() -> None:
    assert prior_content_violation("不得使用 Test/Held-out。\n") == ""
    assert prior_content_violation("Test 与 Held-out 不可见。\n") == ""
    assert prior_content_violation("不要用 Test 水平做选择。\n") == ""
    assert prior_content_violation("审查窗口排除 Held-out。\n") == ""
    assert "Held-out" in prior_content_violation("Held-out sharpe 1.2\n")
    assert "Test figure" in prior_content_violation("Fold1 Test 收益 0.31\n")
    assert "choose a strategy" in prior_content_violation("根据Test选择动量因子\n")
    assert "choose a strategy" in prior_content_violation(
        "Test上动量更稳所以保留该方向\n"
    )
