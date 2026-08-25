"""Session-local todo tool: CRUD, bounds, isolation, registration, collection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from autotrade.agent.explore import ExploreSubAgentEngine
from autotrade.agent.prompts import HARD_FINALIZATION_SYSTEM_PROMPT, build_system_prompt
from autotrade.agent.runner import (
    AgentSessionConfig,
    AgentSessionRunner,
    _FOLD_TOOLS,
    _META_TOOLS,
)
from autotrade.environment.llm import ScriptedLLM
from autotrade.environment.sandbox import LocalSandbox
from autotrade.environment.tools import (
    ModificationCheckTool,
    SafeWorkspace,
    SearchRoots,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from autotrade.environment.tools import TodoTool
from autotrade.environment.tools.todo import (
    MAX_DESCRIPTION_CHARS,
    MAX_SUBJECT_CHARS,
    MAX_TODO_ITEMS,
    TODO_FILENAME,
)
from autotrade.pipelines.local_backend import build_fold_explore_tools
from autotrade.pipelines.meta_inputs import compact_agent_trace


def _registry(root: Path) -> ToolRegistry:
    return ToolRegistry([TodoTool(SafeWorkspace(root))])


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, Mapping)
    return dict(value)


def test_todo_crud_and_list_hides_deleted(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    created = registry.invoke(
        "todo",
        {
            "action": "create",
            "subject": "inspect daily schema",
            "description": "confirm columns first",
        },
    )
    assert created.ok
    item = _mapping(created.value["item"])
    assert item == {
        "id": 1,
        "subject": "inspect daily schema",
        "status": "pending",
        "description": "confirm columns first",
    }
    updated = registry.invoke(
        "todo",
        {"action": "update", "id": 1, "status": "completed", "subject": "schema done"},
    )
    assert updated.ok
    updated_item = _mapping(updated.value["item"])
    assert updated_item["status"] == "completed"
    assert updated_item["subject"] == "schema done"
    deleted = registry.invoke("todo", {"action": "delete", "id": 1})
    assert deleted.ok
    listed = registry.invoke("todo", {"action": "list"})
    assert listed.ok
    assert listed.value["items"] == []
    assert listed.value["count"] == 0
    with_deleted = registry.invoke("todo", {"action": "list", "include_deleted": True})
    deleted_items = with_deleted.value["items"]
    assert isinstance(deleted_items, list)
    assert with_deleted.value["count"] == 1
    assert _mapping(deleted_items[0])["status"] == "deleted"
    store = json.loads((tmp_path / TODO_FILENAME).read_text(encoding="utf-8"))
    assert store["next_id"] == 2
    assert store["items"][0]["id"] == 1


def test_todo_single_in_progress_and_monotonic_ids(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    first = registry.invoke(
        "todo", {"action": "create", "subject": "one", "status": "in_progress"}
    )
    second = registry.invoke(
        "todo", {"action": "create", "subject": "two", "status": "in_progress"}
    )
    assert _mapping(first.value["item"])["id"] == 1
    assert _mapping(second.value["item"])["id"] == 2
    listed = registry.invoke("todo", {"action": "list"}).value["items"]
    assert isinstance(listed, list)
    by_id = {_mapping(item)["id"]: _mapping(item) for item in listed}
    assert by_id[1]["status"] == "pending"
    assert by_id[2]["status"] == "in_progress"
    registry.invoke("todo", {"action": "update", "id": 1, "status": "in_progress"})
    listed = registry.invoke("todo", {"action": "list"}).value["items"]
    assert isinstance(listed, list)
    by_id = {_mapping(item)["id"]: _mapping(item) for item in listed}
    assert by_id[1]["status"] == "in_progress"
    assert by_id[2]["status"] == "pending"


def test_todo_fail_fast_on_bad_action_id_status_and_limits(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    bad_action = registry.invoke("todo", {"action": "schedule"})
    assert not bad_action.ok
    assert "action" in bad_action.error
    missing_subject = registry.invoke("todo", {"action": "create"})
    assert not missing_subject.ok
    create_with_id = registry.invoke(
        "todo", {"action": "create", "id": 9, "subject": "nope"}
    )
    assert not create_with_id.ok
    bad_status = registry.invoke(
        "todo", {"action": "create", "subject": "x", "status": "blocked"}
    )
    assert not bad_status.ok
    too_long = registry.invoke(
        "todo", {"action": "create", "subject": "s" * (MAX_SUBJECT_CHARS + 1)}
    )
    assert not too_long.ok
    long_desc = registry.invoke(
        "todo",
        {
            "action": "create",
            "subject": "ok",
            "description": "d" * (MAX_DESCRIPTION_CHARS + 1),
        },
    )
    assert not long_desc.ok
    missing_id = registry.invoke("todo", {"action": "update", "status": "completed"})
    assert not missing_id.ok
    unknown = registry.invoke(
        "todo", {"action": "update", "id": 1, "status": "completed"}
    )
    assert not unknown.ok
    assert "unknown todo id" in unknown.error
    for index in range(MAX_TODO_ITEMS):
        result = registry.invoke("todo", {"action": "create", "subject": f"item {index}"})
        assert result.ok
    overflow = registry.invoke("todo", {"action": "create", "subject": "one more"})
    assert not overflow.ok
    assert "cannot exceed" in overflow.error
    store = json.loads((tmp_path / TODO_FILENAME).read_text(encoding="utf-8"))
    assert len(store["items"]) == MAX_TODO_ITEMS


def test_todo_atomic_write_leaves_complete_json(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    assert registry.invoke("todo", {"action": "create", "subject": "keep"}).ok
    leftovers = list(tmp_path.glob(".TODO.json*.tmp"))
    assert leftovers == []
    payload = json.loads((tmp_path / TODO_FILENAME).read_text(encoding="utf-8"))
    assert payload["items"][0]["subject"] == "keep"
    before = (tmp_path / TODO_FILENAME).read_text(encoding="utf-8")
    failed = registry.invoke(
        "todo", {"action": "update", "id": 99, "status": "completed"}
    )
    assert not failed.ok
    assert (tmp_path / TODO_FILENAME).read_text(encoding="utf-8") == before


def test_todo_corrupt_store_fails_fast(tmp_path: Path) -> None:
    (tmp_path / TODO_FILENAME).write_text("{not-json", encoding="utf-8")
    result = _registry(tmp_path).invoke("todo", {"action": "list"})
    assert not result.ok
    assert "not valid JSON" in result.error


def test_todo_workspaces_are_isolated(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    left_reg = _registry(left)
    right_reg = _registry(right)
    assert left_reg.invoke("todo", {"action": "create", "subject": "only left"}).ok
    assert right_reg.invoke("todo", {"action": "list"}).value["items"] == []
    assert (left / TODO_FILENAME).is_file()
    assert not (right / TODO_FILENAME).exists()
    assert not (tmp_path / "output" / TODO_FILENAME).exists()
    assert not (tmp_path / "models" / TODO_FILENAME).exists()


def test_todo_is_registered_for_fold_meta_and_explore(tmp_path: Path) -> None:
    assert "todo" in _FOLD_TOOLS
    assert "todo" in _META_TOOLS
    fold_prompt = build_system_prompt(mode="fold", experiment_facts={})
    meta_prompt = build_system_prompt(mode="meta", experiment_facts={})
    assert "`todo`" in fold_prompt
    assert "`todo`" in meta_prompt
    assert "`todo`" not in HARD_FINALIZATION_SYSTEM_PROMPT
    workspace = tmp_path / "agent"
    workspace.mkdir()
    (workspace / "output").mkdir()
    safe = SafeWorkspace(workspace)

    class UnusedRunner:
        def run(self, argv, *, cwd, timeout_seconds, max_output_chars, input_text=None):
            raise AssertionError("runner is unused")

    names = [
        tool.spec.name
        for tool in build_fold_explore_tools(
            SearchRoots(safe),
            safe,
            UnusedRunner(),
            ModificationCheckTool(workspace / "output"),
        )
    ]
    assert "todo" in names
    ExploreSubAgentEngine(
        llm=ScriptedLLM([]),
        tools=ToolRegistry([TodoTool(safe)]),
    )


def test_hard_finalization_does_not_expose_todo(tmp_path: Path) -> None:
    class NamedTool:
        def __init__(self, name: str) -> None:
            self.spec = ToolSpec(
                name,
                "finalization stub",
                {
                    "type": "object",
                    "properties": {"node_id": {"type": "string"}},
                    "required": [],
                },
            )

        def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
            del arguments
            return ToolResult(True, value={})

    runner = AgentSessionRunner(
        llm=ScriptedLLM([]),
        tools=ToolRegistry(
            [
                TodoTool(SafeWorkspace(tmp_path)),
                NamedTool("step_rollback"),
                NamedTool("finish_fold"),
            ]
        ),
        system_prompt="test",
        config=AgentSessionConfig(mode="fold"),
    )
    runner._hard_finalization = True
    runner._complete_validation_nodes = [
        {"node_id": "node_a", "revision_id": "revision_a"}
    ]
    assert runner._finalization_tool_names() == frozenset(
        {"finish_fold", "step_rollback"}
    )
    names: set[str] = set()
    for record in runner._provider_tools():
        function = record["function"]
        assert isinstance(function, dict)
        names.add(str(function["name"]))
    assert names == {"finish_fold", "step_rollback"}
    assert "todo" not in names


def test_collect_artifacts_excludes_todo_json(tmp_path: Path) -> None:
    local = LocalSandbox(tmp_path / "session")
    paths = local.prepare_layout()
    work_output = paths.workspace / "output"
    work_output.mkdir(parents=True, exist_ok=True)
    (work_output / "main.py").write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    assert _registry(paths.workspace).invoke(
        "todo", {"action": "create", "subject": "session plan"}
    ).ok
    assert (paths.workspace / TODO_FILENAME).is_file()
    dest = local.collect_artifacts(tmp_path / "collected")
    assert (dest / "output" / "main.py").is_file()
    assert not (dest / TODO_FILENAME).exists()
    assert not (dest / "workspace" / TODO_FILENAME).exists()
    assert not (dest / "output" / TODO_FILENAME).exists()
    assert not (dest / "models" / TODO_FILENAME).exists()


def test_meta_agent_trace_keeps_todo_name_not_body() -> None:
    events = [
        {
            "event_type": "tool_call",
            "tool": "todo",
            "arguments": {
                "action": "create",
                "subject": "secret research plan",
                "description": "do not leak this body",
                "status": "in_progress",
            },
            "result": {
                "ok": True,
                "value": {
                    "item": {
                        "id": 1,
                        "subject": "secret research plan",
                        "status": "in_progress",
                    }
                },
            },
        }
    ]
    compact = compact_agent_trace(events)
    assert compact[0]["tool"] == "todo"
    assert compact[0]["ok"] is True
    args = compact[0]["args"]
    assert isinstance(args, dict)
    assert args["action"] == "create"
    assert args["status"] == "in_progress"
    assert args["subject"] == {"omitted": True, "chars": len("secret research plan")}
    assert args["description"] == {
        "omitted": True,
        "chars": len("do not leak this body"),
    }
    rendered = str(compact)
    assert "secret research plan" not in rendered
    assert "do not leak this body" not in rendered
