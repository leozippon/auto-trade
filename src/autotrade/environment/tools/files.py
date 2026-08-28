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
# Everything the host-side writers create must stay writable by the sandbox
# user that runs ``shell``: the workspace root is world-writable, but a
# default-umask mkdir or write from the host user would lock the sandbox out
# of its own scratch tree (files could then only travel through the model).
_SANDBOX_DIR_MODE = 0o777
_SANDBOX_FILE_MODE = 0o666
_PATH = {"type": "string", "minLength": 1, "maxLength": 500}
# The same root convention as the search tools, restricted to the writable
# tree: `workspace` is the workspace root, `output`/`models` its formal
# subtrees. Any other search root is read-only and refused.
WRITABLE_ROOTS = ("workspace", "output", "models")
_ROOT = {
    "type": "string",
    "enum": list(WRITABLE_ROOTS),
    "description": (
        "Optional, same convention as read_file: `output` = output/, `models` = models/, "
        "`workspace` (default) = the workspace root; read-only roots are refused."
    ),
}


class _WorkspaceWriteTool:
    """Shared write-boundary resolution for the typed artifact writers."""

    def __init__(self, workspace: SafeWorkspace) -> None:
        self.workspace = workspace

    def _resolve(self, path: str, *, root: object = None, must_exist: bool = False) -> Path:
        if root is not None:
            root = str(root)
            if root not in WRITABLE_ROOTS:
                raise ToolError(
                    f"root {root!r} is read-only; writable roots: {', '.join(WRITABLE_ROOTS)}",
                    error_type="path_error",
                    blocked_target=f"{root}:{path}",
                )
            if root != "workspace" and not (path == root or path.startswith(f"{root}/")):
                path = f"{root}/{path}"
        # ``workspace/x`` under the workspace root means ``x`` (the root is
        # not a subdirectory of itself); only a real ``workspace/`` directory
        # keeps the literal form. The accepted path is echoed in the result.
        if (root is None or root == "workspace") and path.startswith("workspace/"):
            if not (self.workspace.root / "workspace").is_dir():
                path = path[len("workspace/"):]
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

    def _write(self, target: Path, content: str) -> None:
        """Write ``content`` so the sandbox user can read, change and delete it."""

        created: list[Path] = []
        parent = target.parent
        while not parent.exists() and parent != parent.parent:
            created.append(parent)
            parent = parent.parent
        target.parent.mkdir(parents=True, exist_ok=True)
        for directory in created:
            directory.chmod(_SANDBOX_DIR_MODE)
        target.write_text(content, encoding="utf-8")
        try:
            target.chmod(_SANDBOX_FILE_MODE)
        except PermissionError:
            # A file the sandbox user created is not ours to chmod; it already
            # carries that user's own mode, which is what this write preserves.
            pass


class WriteFileTool(_WorkspaceWriteTool):
    spec = ToolSpec(
        "write_file",
        "Create or overwrite a UTF-8 text file in the strategy workspace "
        "(formal code under output/, drafts elsewhere, text metadata under models/). "
        "`path` is relative to the workspace root, or to the optional writable `root`. "
        "Binary model weights are still written with shell/python.",
        {
            "type": "object",
            # No schema-level maxLength: the size refusal is raised in invoke()
            # as a typed `too_large` the Agent can act on, and a schema rejection
            # here would mask it with a generic schema error.
            "properties": {"path": _PATH, "content": {"type": "string"}, "root": _ROOT},
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        mutating=True,
        example={"path": "output/main.py", "content": "def generate_orders(context):\n    return []\n"},
    )

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        content = str(arguments["content"])
        if len(content) > MAX_WRITE_CHARS:
            raise ToolError(f"content exceeds {MAX_WRITE_CHARS} chars", error_type="too_large")
        target = self._resolve(str(arguments["path"]), root=arguments.get("root"))
        existed = target.exists()
        self._write(target, content)
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
        "old_text must match the current content uniquely unless replace_all is set. "
        "`path` is relative to the workspace root, or to the optional writable `root`; "
        "there is no offset/limit (edits match text, not lines).",
        {
            "type": "object",
            "properties": {
                "path": _PATH,
                "root": _ROOT,
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
        example={"path": "output/main.py", "old_text": "return []", "new_text": "return orders"},
    )

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        target = self._resolve(
            str(arguments["path"]), root=arguments.get("root"), must_exist=True
        )
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
        self._write(target, updated)
        return ToolResult(True, value={
            "path": self.workspace.relative(target),
            "changed": True,
            "replacements": count if replace_all else 1,
            "bytes_written": len(updated.encode("utf-8")),
        })


__all__ = ["EditFileTool", "WriteFileTool"]
