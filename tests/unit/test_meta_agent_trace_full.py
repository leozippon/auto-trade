"""Byte-exact raw Fold Agent Trace sidecars for Meta learning."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from autotrade.agent.prompts import META_SYSTEM_PROMPT, build_meta_learning_prompt
from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm import ScriptedLLM, ToolCall
from autotrade.pipelines.experiment import _development_inputs
from autotrade.pipelines.local_backend import LLMMetaLearner
from autotrade.pipelines.meta_inputs import (
    AGENT_TRACE_FULL_MAX_FILE_BYTES,
    AGENT_TRACE_FULL_MAX_WINDOW_BYTES,
    AGENT_TRACE_FULL_RELATIVE_DIR,
    AgentTraceSourceError,
    build_meta_fold_review_bundle,
    select_meta_review_folds,
    write_meta_agent_trace_sidecars,
)
from tests.unit.test_interactive_worker_local import _explore_then


def _as_map(value: object) -> dict[str, object]:
    assert isinstance(value, Mapping)
    return dict(value)


def _raw_trace(marker: str = "A") -> bytes:
    return (
        '{ "event_type" : "session_start", "mode":"fold", '
        '"system_prompt":"SYSTEM PROMPT BODY", '
        '"instruction":"USER INSTRUCTION BODY" }\r\n'
        "  \n"
        ' {"event_type":"user_message","content":"USER MESSAGE BODY",'
        '"host":"/Data2/lzp/ADMCubeQuant/runs/fold"}\n'
        '{"event_type":"tool_call","tool":"write_file","arguments":'
        '{"path":"/Data2/lzp/ADMCubeQuant/output/main.py",'
        f'"content":"TOOL ARGUMENT BODY {marker}"}},"result":'
        '{"ok":true,"body":"TOOL RESULT BODY",'
        '"sandbox_path":"/mnt/agent/workspace/output/main.py"}}   \n'
        '{"event_type":"trace_limit_reached","max_bytes":32}\n'
    ).encode("utf-8")


def _fold(
    tmp_path: Path,
    payload: bytes | None = None,
    *,
    epoch_id: str = "epoch_001",
    fold_id: str = "fold_2024Q1",
    run_id: str = "run_fold",
    status: str = "frozen",
    with_ref: bool = True,
) -> tuple[dict[str, object], Path | None]:
    record: dict[str, object] = {
        "record_type": "fold",
        "epoch_id": epoch_id,
        "fold_id": fold_id,
        "run_id": run_id,
        "fold_status": status,
    }
    if not with_ref:
        return record, None
    source = tmp_path / "traces" / f"{run_id}.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(payload if payload is not None else _raw_trace())
    record["agent_trace_ref"] = str(source)
    return record, source


def test_raw_sidecar_is_byte_exact_and_retains_every_recorded_field(
    tmp_path: Path,
) -> None:
    record, source = _fold(tmp_path)
    assert source is not None

    ref_store = AgentRefStore(tmp_path / "experiment")
    reviews, sidecars = build_meta_fold_review_bundle(
        [record], ref_store=ref_store
    )
    sidecar = sidecars[0]
    metadata = _as_map(reviews[0]["agent_trace_full"])

    assert sidecar.payload == source.read_bytes()
    assert sidecar.bytes == len(source.read_bytes())
    assert sidecar.events == 4
    assert sidecar.source_truncated is True
    assert metadata == sidecar.metadata()
    assert metadata["raw_jsonl"] is True
    assert metadata["byte_exact"] is True
    assert "sha256" not in metadata
    assert not hasattr(sidecar, "sha256")

    raw = cast(bytes, sidecar.payload)
    for expected in (
        b"SYSTEM PROMPT BODY",
        b"USER INSTRUCTION BODY",
        b"USER MESSAGE BODY",
        b"TOOL ARGUMENT BODY A",
        b"TOOL RESULT BODY",
        b"/Data2/lzp/ADMCubeQuant/output/main.py",
        b"/mnt/agent/workspace/output/main.py",
    ):
        assert expected in raw
    assert b"\r\n  \n" in raw
    assert b"}}   \n" in raw

    public = json.dumps(reviews[0], ensure_ascii=False)
    assert "SYSTEM PROMPT BODY" not in public
    assert "TOOL RESULT BODY" not in public
    assert "payload" not in metadata

    fold_ref = ref_store.get_or_create("fold", "fold_2024Q1")
    trace_ref = ref_store.get_or_create("trace", "epoch_001:fold_2024Q1")
    assert reviews[0]["fold_id"] == fold_ref
    assert sidecar.relative_path == f"{AGENT_TRACE_FULL_RELATIVE_DIR}/{trace_ref}.jsonl"

    workspace = tmp_path / "workspace"
    write_meta_agent_trace_sidecars(workspace, sidecars)
    destination = workspace / sidecar.relative_path
    assert destination.read_bytes() == source.read_bytes()
    assert _stat_mode(destination) == 0o444
    assert not list(destination.parent.glob(".*.tmp"))


def test_changing_one_source_byte_changes_the_sidecar(tmp_path: Path) -> None:
    record, source = _fold(tmp_path)
    assert source is not None
    ref_store = AgentRefStore(tmp_path / "experiment")
    _reviews, first = build_meta_fold_review_bundle([record], ref_store=ref_store)
    first_payload = cast(bytes, first[0].payload)

    changed = bytearray(source.read_bytes())
    offset = changed.index(b"TOOL ARGUMENT BODY A") + len(b"TOOL ARGUMENT BODY ")
    changed[offset] = ord("B")
    source.write_bytes(changed)
    _reviews, second = build_meta_fold_review_bundle([record], ref_store=ref_store)

    assert first_payload != second[0].payload
    assert second[0].payload == source.read_bytes()
    assert len(first_payload) == len(cast(bytes, second[0].payload))


def test_missing_ref_is_unavailable_but_referenced_bad_sources_fail(
    tmp_path: Path,
) -> None:
    no_ref, _source = _fold(tmp_path, with_ref=False)
    reviews, sidecars = build_meta_fold_review_bundle(
        [no_ref], ref_store=AgentRefStore(tmp_path / "experiment")
    )
    metadata = _as_map(reviews[0]["agent_trace_full"])
    assert metadata == {
        "path": None,
        "events": 0,
        "bytes": 0,
        "source_truncated": False,
        "available": False,
        "raw_jsonl": True,
        "byte_exact": True,
    }
    assert sidecars[0].payload is None

    missing = dict(no_ref, agent_trace_ref=str(tmp_path / "absent.jsonl"))
    with pytest.raises(AgentTraceSourceError, match="missing"):
        build_meta_fold_review_bundle(
            [missing], ref_store=AgentRefStore(tmp_path / "experiment")
        )

    invalid_ref = dict(no_ref, agent_trace_ref=123)
    with pytest.raises(AgentTraceSourceError, match="string path"):
        build_meta_fold_review_bundle(
            [invalid_ref], ref_store=AgentRefStore(tmp_path / "experiment")
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"{not-json\n", "invalid JSON"),
        (b"[1, 2]\n", "not a JSON object"),
        (b"\xff\xfe\n", "UTF-8"),
    ],
)
def test_corrupt_trace_source_fails_fast(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    record, _source = _fold(tmp_path, payload)
    with pytest.raises(AgentTraceSourceError, match=message):
        build_meta_fold_review_bundle(
            [record], ref_store=AgentRefStore(tmp_path / "experiment")
        )


def test_event_count_and_trace_limit_are_validated_without_reserializing(
    tmp_path: Path,
) -> None:
    payload = (
        b' {"event_type":"session_start","x":1}\r\n'
        b"\n"
        b'{"event_type":"trace_limit_reached","max_bytes":10}  \n'
        b'{"event_type":"session_end","status":"finished"}'
    )
    record, source = _fold(tmp_path, payload)
    assert source is not None
    reviews, sidecars = build_meta_fold_review_bundle(
        [record], ref_store=AgentRefStore(tmp_path / "experiment")
    )
    metadata = _as_map(reviews[0]["agent_trace_full"])

    assert metadata["events"] == 3
    assert metadata["source_truncated"] is True
    assert metadata["bytes"] == len(payload)
    assert sidecars[0].payload == payload == source.read_bytes()
    assert not payload.endswith(b"\n")


def test_raw_sidecar_budgets_fail_instead_of_truncating(tmp_path: Path) -> None:
    assert AGENT_TRACE_FULL_MAX_FILE_BYTES == 8 * 1024 * 1024
    assert AGENT_TRACE_FULL_MAX_WINDOW_BYTES == 16 * 1024 * 1024

    first, first_source = _fold(tmp_path, _raw_trace("A"), run_id="run_a")
    assert first_source is not None
    with pytest.raises(ValueError, match="sidecar exceeds"):
        build_meta_fold_review_bundle(
            [first],
            ref_store=AgentRefStore(tmp_path / "experiment"),
            max_file_bytes=len(first_source.read_bytes()) - 1
        )

    second, second_source = _fold(
        tmp_path,
        _raw_trace("B"),
        fold_id="fold_2024Q2",
        run_id="run_b",
    )
    assert second_source is not None
    total = len(first_source.read_bytes()) + len(second_source.read_bytes())
    with pytest.raises(ValueError, match="window exceeds"):
        build_meta_fold_review_bundle(
            [first, second],
            ref_store=AgentRefStore(tmp_path / "experiment"),
            max_window_bytes=total - 1
        )


def test_review_window_keeps_regular_completed_folds_isolated(tmp_path: Path) -> None:
    old, _ = _fold(tmp_path, _raw_trace("O"), run_id="run_old")
    current, current_source = _fold(
        tmp_path,
        _raw_trace("C"),
        epoch_id="epoch_002",
        fold_id="fold_current",
        run_id="run_current",
    )
    assert current_source is not None
    current["validation_result"] = {
        "total_return": 0.02,
        "per_stock": {"000001.SZ": [0.1]},
    }
    current["test_result"] = {"sharpe": 0.4, "weekly_returns": [0.01]}
    in_progress, _ = _fold(
        tmp_path,
        _raw_trace("I"),
        fold_id="fold_in_progress",
        run_id="run_in_progress",
        status="running",
    )
    failed, _ = _fold(
        tmp_path,
        _raw_trace("F"),
        fold_id="fold_failed",
        run_id="run_failed",
        status="attempt_failed",
    )
    heldout_source = tmp_path / "traces" / "heldout.jsonl"
    heldout_source.write_bytes(_raw_trace("H"))
    heldout = {
        "record_type": "heldout",
        "epoch_id": "epoch_002",
        "fold_id": "heldout_secret",
        "agent_trace_ref": str(heldout_source),
        "result": {"total_return": 9.9},
    }
    records = [
        old,
        {
            "record_type": "meta_learning",
            "meta_learning_id": "meta_previous",
            "run_id": "run_meta_previous",
        },
        current,
        in_progress,
        failed,
        heldout,
    ]

    ref_store = AgentRefStore(tmp_path / "experiment")
    selected, window = select_meta_review_folds(records, ref_store=ref_store)
    assert [record["run_id"] for record in selected] == ["run_current"]
    assert window["fold_count"] == 1
    history, sidecars = _development_inputs(records, ref_store=ref_store)
    reviews = cast(list[object], history["fold_reviews"])
    review = _as_map(reviews[0])
    assert len(sidecars) == 1
    assert sidecars[0].payload == current_source.read_bytes()
    assert b"TOOL ARGUMENT BODY H" not in cast(bytes, sidecars[0].payload)
    assert _as_map(review["validation_result"]) == {"total_return": 0.02}
    assert _as_map(review["test_result"]) == {"sharpe": 0.4}
    assert _as_map(history["review_window"])["fold_count"] == 1

    no_previous_meta, empty_window = select_meta_review_folds(
        [current], ref_store=ref_store
    )
    assert no_previous_meta == []
    assert empty_window["fold_count"] == 0


def test_atomic_tmp_fsync_replace_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autotrade.pipelines.meta_inputs as meta_inputs

    record, source = _fold(tmp_path)
    assert source is not None
    _reviews, sidecars = build_meta_fold_review_bundle(
        [record], ref_store=AgentRefStore(tmp_path / "experiment")
    )
    calls: list[str] = []
    real_fsync = os.fsync
    real_replace = Path.replace

    def recording_fsync(fd: int) -> None:
        calls.append("fsync")
        real_fsync(fd)

    def recording_replace(path: Path, target: Path) -> Path:
        calls.append("replace")
        return real_replace(path, target)

    monkeypatch.setattr(meta_inputs.os, "fsync", recording_fsync)
    monkeypatch.setattr(Path, "replace", recording_replace)
    workspace = tmp_path / "success"
    write_meta_agent_trace_sidecars(workspace, sidecars)
    destination = workspace / sidecars[0].relative_path
    assert calls == ["fsync", "replace"]
    assert destination.read_bytes() == source.read_bytes()
    assert not list(destination.parent.glob(".*.tmp"))

    def failing_replace(_path: Path, _target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", failing_replace)
    failed_workspace = tmp_path / "failed"
    with pytest.raises(OSError, match="replace failed"):
        write_meta_agent_trace_sidecars(failed_workspace, sidecars)
    failed_destination = failed_workspace / sidecars[0].relative_path
    assert not failed_destination.exists()
    assert not list(failed_destination.parent.glob(".*.tmp"))


def test_llm_meta_publishes_raw_sidecar_before_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autotrade.pipelines.meta_inputs as meta_inputs

    record, source = _fold(
        tmp_path,
        _raw_trace("E"),
        epoch_id="epoch_002",
        fold_id="fold_2025Q2",
    )
    assert source is not None
    reviews, sidecars = build_meta_fold_review_bundle(
        [record], ref_store=AgentRefStore(tmp_path / "experiment")
    )
    baseline = tmp_path / "baseline" / "main.py"
    baseline.parent.mkdir()
    baseline.write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    store = FilesystemArtifactStore(tmp_path / "artifacts")
    learner = LLMMetaLearner(
        llm=ScriptedLLM(
            [
                *_explore_then(
                    ToolCall(
                        "prior",
                        "write_file",
                        {"path": "PRIOR.md", "content": "prefer simple signals"},
                    ),
                    ToolCall("finish_meta", "finish_meta", {}),
                )
            ]
        ),
        baseline_strategy=baseline,
        artifact_store=store,
        experiment_dir=tmp_path / "experiment",
        runtime_root=tmp_path / "runtime",
        max_llm_calls=2,
        deadline_seconds=30.0,
        use_docker=False,
        rebuild_enabled=False,
    )
    real_writer = meta_inputs.write_meta_agent_trace_sidecars
    publication_order: list[str] = []

    def checking_writer(workspace: str | Path, values: Sequence[object]) -> None:
        context = Path(workspace) / "inputs" / "meta_context.json"
        assert not context.exists()
        publication_order.append("sidecar")
        real_writer(workspace, cast(Sequence[meta_inputs.AgentTraceFullSidecar], values))

    monkeypatch.setattr(meta_inputs, "write_meta_agent_trace_sidecars", checking_writer)
    learner(
        {
            "run_id": "run_meta",
            "experiment_id": "exp",
            "epoch_id": "epoch_002",
            "meta_learning_id": "epoch_002_after_fold",
            "development_history": {"fold_reviews": reviews},
            "agent_trace_sidecars": sidecars,
        }
    )

    collected = tmp_path / "run_meta"
    context_path = collected / "workspace" / "inputs" / "meta_context.json"
    public = json.loads(context_path.read_text(encoding="utf-8"))
    assert publication_order == ["sidecar"]
    public_reviews = cast(
        list[object], _as_map(public["development_history"])["fold_reviews"]
    )
    metadata = _as_map(_as_map(public_reviews[0])["agent_trace_full"])
    assert "sha256" not in metadata
    assert metadata["raw_jsonl"] is True
    assert metadata["byte_exact"] is True

    live = (
        tmp_path
        / "runtime"
        / "run_meta"
        / "agent"
        / "workspace"
        / str(metadata["path"])
    )
    copied = collected / "workspace" / str(metadata["path"])
    assert live.read_bytes() == source.read_bytes()
    assert copied.read_bytes() == source.read_bytes()
    assert b"SYSTEM PROMPT BODY" in live.read_bytes()
    assert b"/Data2/lzp/ADMCubeQuant/output/main.py" in live.read_bytes()
    assert _stat_mode(live) == 0o444
    assert _stat_mode(live.parent) == 0o555
    assert "2025Q2" not in live.name
    assert "SYSTEM PROMPT BODY" not in json.dumps(public, ensure_ascii=False)

    host_manifest = json.loads(
        (collected / "host_run_manifest.json").read_text(encoding="utf-8")
    )
    overview = _as_map(_as_map(host_manifest["development_inputs"])["agent_trace_full"])
    assert overview["directory"] == "/mnt/agent/workspace/inputs/agent_traces"
    assert overview["available"] == 1
    assert overview["fold_count"] == 1
    assert "/Data2/" not in json.dumps(overview)
    assert not (collected / "output" / "agent_traces").exists()


def test_prompt_contract_leaves_sidecar_exploration_to_meta() -> None:
    prompt = build_meta_learning_prompt({"fold_reviews": [{"agent_trace": ["raw"]}]})
    assert "inputs/meta_context.json" in prompt
    assert "自主选择" in prompt
    assert "fold_reviews" not in prompt
    assert "逐个读取" not in prompt
    assert "process summary" not in prompt
    assert "raw traces" in prompt
    assert "sidecar 引用" in META_SYSTEM_PROMPT
    assert "原始 sidecar 不改变 PIT/Test/Held-out 边界" in META_SYSTEM_PROMPT
    assert "逐个读取" not in META_SYSTEM_PROMPT


def _stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
