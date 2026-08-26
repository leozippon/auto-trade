"""Typed artifact write/edit tools for a strategy workspace.

Promoting artifact writes from opaque ``shell`` strings to typed ``write_file`` /
``edit_file`` tools gives the harness a typed, audited record of every formal
change and a deterministic write boundary, plus a staleness check:
``edit_file`` rejects an ``old_string`` that does not match the current file.
Binary model weights are still written with shell/python.

Bounded reading and searching live in ``tools.search``, which reaches the
read-only PIT and prior-fold roots as well as this workspace.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.artifacts import READONLY_FILES

from .base import ToolError, ToolResult, ToolSpec
from .workspace import SafeWorkspace

MAX_WRITE_CHARS = 200_000
_PATH = {"type": "string", "minLength": 1, "maxLength": 500}


class _WorkspaceWriteTool:
    """Shared write-boundary resolution for the typed artifact writers."""

    def __init__(self, workspace: SafeWorkspace) -> None:
        self.workspace = workspace

    def _resolve(self, path: str, *, must_exist: bool = False) -> Path:
        target = self.workspace.resolve(path, must_exist=must_exist, directory=False if must_exist else None)
        relative = self.workspace.relative(target)
        if relative == "skills" or relative.startswith("skills/"):
            raise ToolError(
                "skills/ may only be changed with write_skill or delete_skill",
                error_type="readonly",
                blocked_target=str(path),
            )
        if relative in READONLY_FILES or relative.removeprefix("output/") in READONLY_FILES:
            # Compare the RESOLVED relative path: './output/README.md' and
            # friends must get the documented typed error, not a raw
            # PermissionError later.
            raise ToolError(
                f"{relative} is read-only",
                error_type="readonly",
                blocked_target=str(path),
            )
        return target


class WriteFileTool(_WorkspaceWriteTool):
    spec = ToolSpec(
        "write_file",
        "Create or overwrite a UTF-8 text file in the strategy workspace "
        "(formal code under output/, drafts elsewhere, text metadata under models/). "
        "Binary model weights are still written with shell/python.",
        {
            "type": "object",
            # No schema-level maxLength: the size refusal is raised in invoke()
            # as a typed `too_large` the Agent can act on, and a schema rejection
            # here would mask it with a generic schema error.
            "properties": {"path": _PATH, "content": {"type": "string"}},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        mutating=True,
    )

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        content = str(arguments["content"])
        if len(content) > MAX_WRITE_CHARS:
            raise ToolError(f"content exceeds {MAX_WRITE_CHARS} chars", error_type="too_large")
        target = self._resolve(str(arguments["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        target.write_text(content, encoding="utf-8")
        return ToolResult(True, value={
            "path": self.workspace.relative(target),
            "created": not existed,
            "chars": len(content),
            "bytes_written": len(content.encode("utf-8")),
        })


class EditFileTool(_WorkspaceWriteTool):
    spec = ToolSpec(
        "edit_file",
        "Replace exact UTF-8 text in an existing strategy workspace file; "
        "old_text must match the current content uniquely unless replace_all is set.",
        {
            "type": "object",
            "properties": {
                "path": _PATH,
                "old_text": {"type": "string", "minLength": 1, "maxLength": 100_000},
                "new_text": {"type": "string", "maxLength": 100_000},
                "replace_all": {
                    "type": "boolean",
                    "description": "Set true only when every occurrence of old_text should be replaced.",
                },
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
        mutating=True,
    )

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        target = self._resolve(str(arguments["path"]), must_exist=True)
        current = target.read_text(encoding="utf-8", errors="replace")
        old = str(arguments["old_text"])
        replace_all = bool(arguments.get("replace_all"))
        count = current.count(old)
        if count == 0:
            raise ToolError(
                "old_text not found in file",
                error_type="stale",
                retry_hint="re-read the file; old_text must match the current content exactly",
            )
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_text matched {count} times; pass replace_all=true or make it unique",
                error_type="ambiguous",
                details={"matches": count},
            )
        new = str(arguments["new_text"])
        updated = current.replace(old, new) if replace_all else current.replace(old, new, 1)
        if len(updated) > MAX_WRITE_CHARS:
            raise ToolError(f"resulting file exceeds {MAX_WRITE_CHARS} chars", error_type="too_large")
        target.write_text(updated, encoding="utf-8")
        return ToolResult(True, value={
            "path": self.workspace.relative(target),
            "changed": True,
            "replacements": count if replace_all else 1,
            "bytes_written": len(updated.encode("utf-8")),
        })


__all__ = ["EditFileTool", "WriteFileTool"]
