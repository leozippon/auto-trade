"""The Agent must never see a raw fold id.

Folds are named after their **test** period on the host side
(``pipelines/folds.py``: ``fold_id = f"fold_{test_label}"``), so the raw label
is itself hidden-schedule evidence: an Agent that learns its Fold is
``fold_2026Q1`` learns exactly which quarter it will be graded on, and can
shape the strategy around it. The boundary is defended by one projection —
``environment.identity.agent_visible_ref`` — applied at every agent-readable
surface.

This test drives a real fold session and sweeps every surface the Agent can
actually read for the raw label: the Fold system prompt and user messages, the
sandbox input tree (``inputs/fold_context.json``, ``artifacts/data_summary.json``,
``artifacts/run_manifest.json``, ``steps/tree.json``/``tree.txt``), the step ids
that come back on the ledger record, and the Meta-visible development history.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from autotrade.agent.experiment_facts import build_experiment_facts
from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.identity import agent_visible_ref
from autotrade.environment.llm import ScriptedLLM, ToolCall
from autotrade.pipelines.agent_views import (
    agent_visible_ledger_record,
    compact_fold_history,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.local_backend import LLMMetaLearner
from autotrade.pipelines.worker import load_worker_options, run_local_interactive_worker

from autotrade.agent.explore import FOLD_REQUIRED_EXPLORE_ROLES
from .test_interactive_worker_local import _NoShellRunner, _explore_then, _experiment

TEST_LABEL = "2026Q1"
RAW_FOLD_ID = f"fold_{TEST_LABEL}"
FOLD_REF = agent_visible_ref(RAW_FOLD_ID, prefix="fold_ref")
SOURCE = "def generate_orders(context):\n    return []\n"


class _SandboxCapturingLLM:
    """A ScriptedLLM that snapshots the sandbox tree on every model call.

    The per-run sandbox tree is removed when the session ends, so the only
    honest place to read what the Agent could see is while it is being asked
    to act.
    """

    provider = "scripted"
    model = "scripted"

    def __init__(self, responses, *, work_root: Path) -> None:
        self._inner = ScriptedLLM(responses)
        self._work_root = Path(work_root)
        self.sandbox_files: dict[str, str] = {}

    @property
    def calls(self):
        return self._inner.calls

    # Exactly the trees `DockerSandbox.start()` binds into the container
    # (sandbox.py: snapshots/train, snapshots/valid, current_snapshot,
    # artifacts read-only, agent read-write). `runtime/` is host-only and is
    # deliberately excluded: it is where the host audit copy lives.
    MOUNTED = ("agent", "artifacts", "snapshot", "snapshots")

    def _capture(self) -> None:
        if not self._work_root.is_dir():
            return
        for run_root in sorted(self._work_root.iterdir()):
            if not run_root.is_dir():
                continue
            for mount in self.MOUNTED:
                base = run_root / mount
                if not base.is_dir():
                    continue
                for path in sorted(base.rglob("*")):
                    if not path.is_file():
                        continue
                    try:
                        self.sandbox_files[str(path.relative_to(run_root))] = (
                            path.read_text(encoding="utf-8", errors="replace")
                        )
                    except (
                        OSError
                    ):  # pragma: no cover - binary mounts are not prompt input
                        continue

    def complete(self, messages, **kwargs):
        self._capture()
        return self._inner.complete(messages, **kwargs)


def _prompt_text(call) -> str:
    return "\n".join(
        message.content or ""
        for message in call["messages"]
        if message.role in {"system", "user"}
    )


@pytest.fixture(scope="module")
def provider_key():
    previous = {
        name: os.environ.get(name) for name in ("DEEPSEEK_API_KEY", "VLLM_API_KEY")
    }
    os.environ["DEEPSEEK_API_KEY"] = "test-only-key"
    os.environ["VLLM_API_KEY"] = "local-test-key"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(scope="module")
def fold_session(tmp_path_factory, provider_key):
    tmp_path = tmp_path_factory.mktemp("fold_ref_isolation")
    repo, experiment = _experiment(tmp_path, developer_mode="llm")
    options = load_worker_options(experiment, repo_root=repo)
    llm = _SandboxCapturingLLM(
        [
            *_explore_then(
                ToolCall(
                    "taste", "write_taste", {"taste": "prefer simple signals"}
                ),
                ToolCall("finish_meta", "finish_meta", {"taste_path": "taste.md"}),
            ),
            *_explore_then(
                ToolCall("check", "modification_check", {}),
                ToolCall("valid", "daily_backtest", {}),
                ToolCall("finish", "finish_fold", {}),
                roles=FOLD_REQUIRED_EXPLORE_ROLES,
                implement={"path": "output/main.py", "content": SOURCE},
            ),
        ],
        work_root=options.work_root / options.experiment_id,
    )
    result = run_local_interactive_worker(
        options, llm=llm, command_runner_factory=lambda _workspace: _NoShellRunner()
    )
    assert result["state"] == "completed"
    records = ExperimentLedger(options.rolling.ledger_path).read()
    return {
        "llm": llm,
        "records": records,
        "fold": next(record for record in records if record["record_type"] == "fold"),
        "experiment": experiment,
    }


def test_fold_id_is_named_after_the_hidden_test_period(fold_session):
    """The premise: the host-side fold id IS the hidden test period label."""
    fold = fold_session["fold"]
    assert fold["fold_id"] == RAW_FOLD_ID
    # 2026Q1 == 20260101..20260331: the label IS the hidden test window.
    assert fold["test_period"] == "20260101..20260331"
    assert FOLD_REF.startswith("fold_ref_")
    assert RAW_FOLD_ID not in FOLD_REF


def _call_has_tool(call, name: str) -> bool:
    return any(item["function"]["name"] == name for item in call["tools"])


def test_fold_system_prompt_and_user_messages_carry_only_the_opaque_ref(fold_session):
    fold_call = next(
        call
        for call in reversed(fold_session["llm"].calls)
        if _call_has_tool(call, "finish_fold")
    )
    prompt = _prompt_text(fold_call)
    assert RAW_FOLD_ID not in prompt
    assert TEST_LABEL not in prompt
    assert FOLD_REF in prompt


def test_no_message_of_any_session_carries_the_raw_fold_id(fold_session):
    for index, call in enumerate(fold_session["llm"].calls):
        for message in call["messages"]:
            content = message.content or ""
            assert RAW_FOLD_ID not in content, f"call {index} role {message.role}"
            assert TEST_LABEL not in content, f"call {index} role {message.role}"


def test_sandbox_input_tree_never_materializes_the_raw_fold_id(fold_session):
    captured = fold_session["llm"].sandbox_files
    # The capture must be real: a session that mounted nothing proves nothing.
    assert captured, "no sandbox files were captured during the fold session"
    # And it must cover the surfaces the Agent actually reads.
    assert any(name.startswith("agent/") for name in captured)
    assert any(name.startswith("artifacts/") for name in captured)
    offenders = {
        name: text
        for name, text in captured.items()
        if RAW_FOLD_ID in text or TEST_LABEL in text
    }
    assert offenders == {}, f"raw fold id reached the sandbox: {sorted(offenders)}"
    # And the surfaces that name the fold at all use the opaque ref.
    named = [name for name, text in captured.items() if FOLD_REF in text]
    assert named, f"no captured sandbox file carried {FOLD_REF}: {sorted(captured)}"


def test_agent_readable_file_names_never_encode_the_raw_fold_id(fold_session):
    for name in fold_session["llm"].sandbox_files:
        assert RAW_FOLD_ID not in name
        assert TEST_LABEL not in name


def test_step_tree_node_ids_the_agent_reads_back_are_opaque(fold_session):
    fold = fold_session["fold"]
    # The Agent reads node ids out of steps/tree.txt and passes them back to
    # finish_fold / step_rollback, so the id itself must not be evidence.
    step_ids = [str(step["step_id"]) for step in fold["steps"]]
    assert step_ids
    for step_id in [*step_ids, str(fold["selected_step_id"])]:
        assert RAW_FOLD_ID not in step_id
        assert TEST_LABEL not in step_id
        assert FOLD_REF in step_id


def test_compact_fold_history_keeps_metrics_and_drops_per_stock_series():
    compact = compact_fold_history(
        {
            "epoch_id": "epoch_001",
            "fold_id": "fold_2024Q2",
            "fold_status": "frozen",
            "finish_reason": "finish_fold",
            "validation_result": {
                "total_return": 0.01,
                "sharpe": 0.4,
                "per_stock": {"000001.SZ": [0.1] * 80},
                "weekly_returns": [0.01] * 40,
            },
        }
    )
    metrics = compact["validation_result"]
    assert isinstance(metrics, dict)
    assert metrics["total_return"] == 0.01
    assert "per_stock" not in metrics
    assert "weekly_returns" not in metrics
    rendered = json.dumps(compact, ensure_ascii=False)
    assert "000001.SZ" not in rendered


def test_meta_learning_prompt_does_not_inline_development_history():
    from autotrade.agent.prompts import build_meta_learning_prompt

    prompt = build_meta_learning_prompt(
        {
            "fold_backtest_summaries": [
                {
                    "fold_id": "fold_ref_deadbeef",
                    "validation_result": {"per_stock": {"000001.SZ": [0.1] * 80}},
                }
            ]
        },
        previous_taste="keep the daily floor",
    )
    assert "inputs/meta_context.json" in prompt
    assert "keep the daily floor" in prompt
    assert "per_stock" not in prompt
    assert "fold_backtest_summaries" not in prompt


def test_meta_visible_projections_opaque_the_fold_id(fold_session):
    fold = fold_session["fold"]
    compact = compact_fold_history(fold, include_frozen_test_metrics=True)
    projected = agent_visible_ledger_record(fold, include_frozen_test_metrics=True)
    for view in (compact, projected):
        rendered = json.dumps(view, ensure_ascii=False, sort_keys=True, default=str)
        assert RAW_FOLD_ID not in rendered
        assert TEST_LABEL not in rendered
        assert FOLD_REF in rendered


def test_experiment_facts_opaque_the_fold_id_for_every_session_kind(fold_session):
    for kind in ("fold", "meta_learning"):
        facts = build_experiment_facts(
            manifest={
                "experiment_id": "smoke",
                "run_id": "run_x",
                "epoch_id": "epoch_001",
                "fold_id": RAW_FOLD_ID,
                "kind": kind,
                "fold": {"test_period": "20260101..20260331"},
            }
        )
        rendered = json.dumps(facts, ensure_ascii=False, sort_keys=True, default=str)
        assert RAW_FOLD_ID not in rendered, kind
        assert TEST_LABEL not in rendered, kind


def test_agent_visible_ref_is_stable_and_not_reversible_by_prefix_collision():
    assert agent_visible_ref(RAW_FOLD_ID, prefix="fold_ref") == FOLD_REF
    assert agent_visible_ref("fold_2026Q2", prefix="fold_ref") != FOLD_REF
    # A different namespace never collides with the fold namespace.
    assert agent_visible_ref(RAW_FOLD_ID, prefix="strategy_ref") != FOLD_REF
    # An absent id still projects to a ref, never to the raw value: a
    # passthrough would be exactly the leak this helper exists to prevent.
    assert agent_visible_ref("", prefix="fold_ref").startswith("fold_ref_")
    assert agent_visible_ref(None, prefix="fold_ref") == agent_visible_ref(
        "", prefix="fold_ref"
    )


def test_meta_context_parent_artifact_id_is_an_opaque_strategy_ref(tmp_path: Path):
    parent_id = "strategy_epoch_002_fold_2025Q2_59852cdf4fb8"
    baseline = tmp_path / "baseline" / "main.py"
    baseline.parent.mkdir()
    baseline.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    parent_output = store.frozen_root / parent_id / "output"
    parent_output.mkdir(parents=True)
    (parent_output / "main.py").write_text(baseline.read_text(encoding="utf-8"), encoding="utf-8")
    llm = ScriptedLLM(
        [
            *_explore_then(
                ToolCall(
                    "taste", "write_taste", {"taste": "prefer simple signals"}
                ),
                ToolCall("finish_meta", "finish_meta", {"taste_path": "taste.md"}),
            )
        ]
    )
    learner = LLMMetaLearner(
        llm=llm,
        baseline_strategy=baseline,
        artifact_store=store,
        experiment_dir=tmp_path / "experiment",
        runtime_root=tmp_path / "runtime",
        max_llm_calls=2,
        deadline_seconds=30.0,
        use_docker=False,
        rebuild_enabled=False,
    )
    learner(
        {
            "run_id": "run_meta",
            "experiment_id": "exp",
            "epoch_id": "epoch_002",
            "meta_learning_id": "epoch_002_after_fold",
            "parent_artifact_id": parent_id,
        }
    )
    expected = agent_visible_ref(parent_id, prefix="strategy_ref")
    public = json.loads(
        (tmp_path / "run_meta" / "workspace" / "inputs" / "meta_context.json").read_text(
            encoding="utf-8"
        )
    )
    assert public["parent_artifact_id"] == expected
    assert public["parent_artifact_id"].startswith("strategy_ref_")
    assert "strategy_epoch_" not in public["parent_artifact_id"]
    assert "fold_2025Q2" not in public["parent_artifact_id"]
    host = json.loads(
        (tmp_path / "run_meta" / "host_run_manifest.json").read_text(encoding="utf-8")
    )
    assert host["parent_strategy_artifact_id"] == parent_id
