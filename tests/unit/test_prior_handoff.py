"""The experiment-level PRIOR handoff between research sessions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from autotrade.agent.prompts import build_system_prompt
from autotrade.environment.tools import SafeWorkspace, WriteFileTool
from autotrade.environment.tools.prior_policy import PRIOR_MAX_CHARS
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.prior import (
    ExperimentPriorStore,
    latest_prior_text,
    restore_current_from_records,
)


def test_fold_system_prompt_injects_prior_full_text(tmp_path: Path) -> None:
    prior = "先用 grep 定向，再抽样 parquet。\n不要并行委托。"
    prompt = build_system_prompt(
        mode="fold",
        prior_prompt=prior,
    )
    assert prior in prompt
    assert "权威 PRIOR 不在本 Fold 可写树中" in prompt
    assert "策略方向" in prompt


def test_prompts_define_no_edge_pre_registration_and_meta_fold_labels() -> None:
    """Round 20260910: two first Folds froze nodes with no demonstrated edge
    (neutralized excess +0.07%; a style overlay picked for min_return>0), one
    Fold burned backtests on the untouched template, and three Meta sessions
    misread the upcoming Fold's window as a data defect. The prompts now state
    each standard where the Agent reads it."""

    fold = build_system_prompt(mode="fold", experiment_facts={})
    guardrails = fold[fold.index("# 研究协议") :]
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
    contract = fold[fold.index("# 提交合同") : fold.index("# 证据标准")]
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
    # The screen script is named through the fact that carries its usage;
    # the mount path itself belongs to the fact and the shell description.
    assert "`source_refs.signal_screen_ref`" in guardrails
    assert "/mnt/tools/screen.py" not in fold
    # The objective is stated once, at the top of the role section, and the
    # procedure is framed as protecting that judgement, not replacing it.
    role = fold[: fold.index("# 研究协议")]
    assert "目标是找到真实、可部署的边际" in role and "不是替代它" in role
    # Mechanism family is defined once; a different estimator on the same
    # features no longer counts as structurally different.
    assert "机制家族指收益来源的经济解释" in guardrails
    assert "同一特征集换个估计器不算结构不同" in guardrails
    assert "另一类模型" not in fold
    # One worked pre-registration example, structurally unlike the template.
    assert "示例（只示形式）" in guardrails and "无预告的匹配组" in guardrails
    # Enforced limits sit in the tool schema, not in the prompt.
    assert "500 字符" not in fold
    assert "弃权同样要求本会话至少有一次完整 Validation" in contract
    assert "`report_issue(category=\"docs\")`" in fold
    # Each shared rule has one home: the tie-break, the one-third threshold
    # and the warn-is-not-selection sentence each appear exactly once.
    assert fold.count("子区间一致性") == 1
    assert fold.count("三分之一") == 1
    assert fold.count("不是选择标准") == 1
    # Delegating a candidate carries its hypothesis and falsification rule.
    assert "再写进它的假设与证伪条件" in fold


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


def test_a_session_publishes_keeps_and_rejects_an_overlong_prior(tmp_path: Path) -> None:
    from autotrade.pipelines.experiment import RollingExperimentPipeline

    experiment = tmp_path / "experiment"
    store = ExperimentPriorStore(experiment)
    first = store.publish("first workflow", generation_id="gen_1")
    pipeline = RollingExperimentPipeline.__new__(RollingExperimentPipeline)
    pipeline.config = type("Cfg", (), {"experiment_dir": experiment})()

    published = pipeline._publish_or_keep_prior(
        "updated workflow notes", previous=first.text, generation_id="gen_2"
    )
    assert published["prior_published"] is True
    assert store.current_text().strip() == "updated workflow notes"
    assert Path(published["prior_ref"]).is_file()

    # An empty or unchanged PRIOR.md keeps the generation in force.
    for candidate, generation in (("", "gen_3"), ("updated workflow notes\n", "gen_same")):
        kept = pipeline._publish_or_keep_prior(
            candidate, previous="updated workflow notes", generation_id=generation
        )
        assert kept["prior"] == "updated workflow notes"
        assert kept["prior_published"] is False
        assert kept["prior_generation_id"] == "gen_2"
    assert store.current_generation_id() == "gen_2"

    # Before any PRIOR exists, an empty handoff is simply empty.
    empty = RollingExperimentPipeline.__new__(RollingExperimentPipeline)
    empty.config = type("Cfg", (), {"experiment_dir": tmp_path / "empty"})()
    assert empty._publish_or_keep_prior("", previous="", generation_id="gen_first") == {
        "prior": "",
        "prior_published": False,
        "prior_ref": None,
        "prior_generation_id": None,
        "prior_chars": 0,
    }

    with pytest.raises(ValueError, match="characters"):
        pipeline._publish_or_keep_prior(
            "x" * (PRIOR_MAX_CHARS + 1),
            previous="updated workflow notes",
            generation_id="gen_overlong",
        )
    with pytest.raises(FileExistsError):
        pipeline._publish_or_keep_prior(
            "collision", previous="updated workflow notes", generation_id="gen_2"
        )


def test_latest_prior_reads_the_last_research_session_record() -> None:
    records = [
        {"record_type": "meta_learning", "prior": "ignored"},
        {"record_type": "research_session", "prior": "first"},
        {"record_type": "research_session", "prior": "second", "prior_published": False},
        {"record_type": "forward", "prior": "ignored"},
    ]
    assert latest_prior_text(records) == "second"
    assert latest_prior_text([]) == ""
    assert latest_prior_text([{"record_type": "meta_learning", "prior": "ignored"}]) == ""


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


def _append_session(
    ledger: ExperimentLedger,
    *,
    run_id: str,
    generation_id: str = "",
    prior: str = "",
) -> None:
    ledger.append(
        {
            "record_type": "research_session",
            "experiment_id": "exp",
            "epoch_id": "research",
            "fold_id": "s1",
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
    _append_session(
        ledger, run_id="run_1", generation_id="gen_1", prior="first workflow"
    )
    restore_current_from_records(experiment, ledger.read())
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
    restore_current_from_records(experiment, ledger.read())
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
    _append_session(ledger, run_id="run_ghost", generation_id="ghost", prior="gone")
    with pytest.raises(FileNotFoundError, match="ghost"):
        restore_current_from_records(experiment, ledger.read())
