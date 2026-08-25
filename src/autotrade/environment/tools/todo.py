"""Session-local research plan for Fold, Explore, and Meta.

The store is the current Agent SafeWorkspace file ``TODO.json``. That is the
sandbox path ``workspace/TODO.json``. The tool never writes ``output/``,
``models/``, artifacts, or the host tree. Official Fold collection ignores the
file, and it is not handed to the next Fold or PRIOR unless an Agent summarizes
it into an existing artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import NotRequired, TypedDict

from autotrade.environment.runtime import write_json_atomic

from .base import ToolError, ToolResult, ToolSpec
from .workspace import SafeWorkspace

TODO_FILENAME = "TODO.json"
MAX_TODO_ITEMS = 64
MAX_SUBJECT_CHARS = 200
MAX_DESCRIPTION_CHARS = 2000
TODO_ACTIONS = ("create", "list", "update", "delete")
TODO_STATUSES = ("pending", "in_progress", "completed", "deleted")


class TodoItem(TypedDict):
    id: int
    subject: str
    status: str
    description: NotRequired[str]


class TodoStore(TypedDict):
    next_id: int
    items: list[TodoItem]


class TodoTool:
    spec = ToolSpec(
        "todo",
        "Maintain a session-local research plan in workspace/TODO.json. "
        "Actions are create, list, update, and delete. At most one item may be "
        "in_progress. This file is not a formal Fold artifact.",
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(TODO_ACTIONS),
                },
                "id": {"type": "integer", "minimum": 1},
                "subject": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SUBJECT_CHARS,
                },
                "status": {
                    "type": "string",
                    "enum": list(TODO_STATUSES),
                },
                "description": {
                    "type": "string",
                    "maxLength": MAX_DESCRIPTION_CHARS,
                },
                "include_deleted": {"type": "boolean"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        mutating=True,
    )

    def __init__(self, workspace: SafeWorkspace) -> None:
        self.workspace = workspace

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        action = str(arguments["action"])
        store = _load_store(self._path())
        if action == "list":
            include_deleted = bool(arguments.get("include_deleted"))
            items = [
                dict(item)
                for item in store["items"]
                if include_deleted or item["status"] != "deleted"
            ]
            return ToolResult(True, value={"items": items, "count": len(items)})
        if action == "create":
            return ToolResult(True, value=_create_item(store, arguments, self._path()))
        if action == "update":
            return ToolResult(True, value=_update_item(store, arguments, self._path()))
        if action == "delete":
            return ToolResult(True, value=_delete_item(store, arguments, self._path()))
        raise ToolError(
            f"unsupported todo action: {action}",
            error_type="schema_error",
        )

    def _path(self) -> Path:
        return self.workspace.resolve(TODO_FILENAME)


def _create_item(
    store: TodoStore, arguments: Mapping[str, object], path: Path
) -> dict[str, object]:
    if "id" in arguments:
        raise ToolError("create does not accept id", error_type="schema_error")
    subject = arguments.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        raise ToolError("create requires subject", error_type="schema_error")
    status = str(arguments.get("status") or "pending")
    if status == "deleted":
        raise ToolError("create cannot use deleted status", error_type="schema_error")
    if len(store["items"]) >= MAX_TODO_ITEMS:
        raise ToolError(
            f"todo list cannot exceed {MAX_TODO_ITEMS} items",
            error_type="limit_error",
        )
    item: TodoItem = {
        "id": store["next_id"],
        "subject": subject,
        "status": status,
    }
    description = arguments.get("description")
    if isinstance(description, str) and description:
        item["description"] = description
    if status == "in_progress":
        _clear_in_progress(store["items"])
    store["items"].append(item)
    store["next_id"] = store["next_id"] + 1
    _save_store(path, store)
    return {"item": dict(item)}


def _update_item(
    store: TodoStore, arguments: Mapping[str, object], path: Path
) -> dict[str, object]:
    item = _require_item(store, arguments)
    if (
        "subject" not in arguments
        and "status" not in arguments
        and "description" not in arguments
    ):
        raise ToolError(
            "update requires subject, status, or description",
            error_type="schema_error",
        )
    if "subject" in arguments:
        subject = arguments["subject"]
        if not isinstance(subject, str) or not subject.strip():
            raise ToolError(
                "subject must be a non-empty string", error_type="schema_error"
            )
        item["subject"] = subject
    if "description" in arguments:
        description = arguments["description"]
        if not isinstance(description, str):
            raise ToolError("description must be a string", error_type="schema_error")
        if description:
            item["description"] = description
        else:
            item.pop("description", None)
    if "status" in arguments:
        status = str(arguments["status"])
        if status == "in_progress":
            _clear_in_progress(store["items"], keep_id=item["id"])
        item["status"] = status
    _save_store(path, store)
    return {"item": dict(item)}


def _delete_item(
    store: TodoStore, arguments: Mapping[str, object], path: Path
) -> dict[str, object]:
    item = _require_item(store, arguments)
    item["status"] = "deleted"
    _save_store(path, store)
    return {"item": dict(item)}


def _require_item(
    store: TodoStore, arguments: Mapping[str, object]
) -> TodoItem:
    raw_id = arguments.get("id")
    if not isinstance(raw_id, int) or isinstance(raw_id, bool):
        raise ToolError("id is required", error_type="schema_error")
    for item in store["items"]:
        if item["id"] == raw_id:
            return item
    raise ToolError(
        f"unknown todo id: {raw_id}",
        error_type="not_found",
        blocked_target=str(raw_id),
    )


def _clear_in_progress(
    items: list[TodoItem], *, keep_id: int | None = None
) -> None:
    for item in items:
        if item["status"] == "in_progress" and item["id"] != keep_id:
            item["status"] = "pending"


def _load_store(path: Path) -> TodoStore:
    if not path.exists():
        return {"next_id": 1, "items": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ToolError(
            "TODO.json is not valid JSON",
            error_type="store_error",
            blocked_target=TODO_FILENAME,
        ) from exc
    if not isinstance(data, dict):
        raise ToolError("TODO.json must be an object", error_type="store_error")
    raw_items = data.get("items")
    if not isinstance(raw_items, list):
        raise ToolError("TODO.json items must be a list", error_type="store_error")
    items: list[TodoItem] = []
    seen: set[int] = set()
    for raw in raw_items:
        item = _parse_item(raw)
        item_id = item["id"]
        if item_id in seen:
            raise ToolError("TODO.json contains duplicate ids", error_type="store_error")
        seen.add(item_id)
        items.append(item)
    if len(items) > MAX_TODO_ITEMS:
        raise ToolError(
            f"TODO.json exceeds {MAX_TODO_ITEMS} items",
            error_type="store_error",
        )
    next_id = data.get("next_id")
    if not isinstance(next_id, int) or isinstance(next_id, bool) or next_id < 1:
        next_id = max(seen, default=0) + 1
    elif seen and next_id <= max(seen):
        next_id = max(seen) + 1
    return {"next_id": next_id, "items": items}


def _parse_item(raw: object) -> TodoItem:
    if not isinstance(raw, dict):
        raise ToolError("TODO.json item must be an object", error_type="store_error")
    raw_id = raw.get("id")
    subject = raw.get("subject")
    status = raw.get("status")
    if not isinstance(raw_id, int) or isinstance(raw_id, bool) or raw_id < 1:
        raise ToolError("TODO.json item id is invalid", error_type="store_error")
    if not isinstance(subject, str) or not subject or len(subject) > MAX_SUBJECT_CHARS:
        raise ToolError("TODO.json item subject is invalid", error_type="store_error")
    if status not in TODO_STATUSES:
        raise ToolError("TODO.json item status is invalid", error_type="store_error")
    item: TodoItem = {"id": raw_id, "subject": subject, "status": status}
    description = raw.get("description")
    if description is None:
        return item
    if not isinstance(description, str) or len(description) > MAX_DESCRIPTION_CHARS:
        raise ToolError(
            "TODO.json item description is invalid", error_type="store_error"
        )
    if description:
        item["description"] = description
    return item


def _save_store(path: Path, store: TodoStore) -> None:
    write_json_atomic(
        path,
        {
            "next_id": store["next_id"],
            "items": [dict(item) for item in store["items"]],
        },
    )


__all__ = [
    "MAX_DESCRIPTION_CHARS",
    "MAX_SUBJECT_CHARS",
    "MAX_TODO_ITEMS",
    "TODO_ACTIONS",
    "TODO_FILENAME",
    "TODO_STATUSES",
    "TodoTool",
]
