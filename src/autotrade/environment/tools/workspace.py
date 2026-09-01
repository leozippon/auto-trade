"""One path boundary shared by every workspace tool."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from .base import ToolError

# Stated in every file tool's description because the two path conventions are
# easy to mix up: `shell` runs on the real sandbox filesystem, while these tools
# address files as a root name plus a path relative to it. Trace audits show
# fresh sub-agents carrying the absolute path out of a shell call straight into
# their first read_file/edit_file call.
ROOT_RELATIVE_PATH_RULE = (
    "`root` plus `path` is a virtual address, not a sandbox filesystem path: the "
    "`workspace` root is the directory shell runs in as `.` (/mnt/agent/workspace), "
    "so pass path='inputs/x.json', never path='/mnt/agent/workspace/inputs/x.json'. "
    "A leading 'workspace/' is read as that root name and dropped when it does not "
    "resolve as a real directory, while shell always takes it literally, so leave it "
    "out: path='notes/probe.py' is what shell runs as ['python', 'notes/probe.py']."
)


class SafeWorkspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("workspace root must be a directory")

    def resolve(
        self,
        relative: str,
        *,
        must_exist: bool = False,
        directory: bool | None = None,
    ) -> Path:
        if not isinstance(relative, str) or not relative.strip():
            raise ToolError("path must be a non-empty relative path", error_type="schema_error")
        if relative == ".":
            if directory is False:
                raise ToolError(
                    "workspace root is not a file", error_type="path_error", blocked_target=relative
                )
            return self.root
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ToolError(
                "path must stay within the workspace", error_type="path_error", blocked_target=relative
            )
        if any(part.startswith(".") for part in pure.parts):
            raise ToolError(
                "hidden workspace paths are not available", error_type="path_error", blocked_target=relative
            )
        candidate = self.root.joinpath(*pure.parts)
        resolved = candidate.resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise ToolError("path escapes the workspace", error_type="path_error", blocked_target=relative)
        if must_exist and not candidate.exists():
            raise ToolError(
                f"path does not exist: {relative}",
                error_type="not_found",
                blocked_target=relative,
                retry_hint="write_file first, or fix the path",
            )
        if candidate.exists():
            actual = candidate.resolve(strict=True)
            if not actual.is_relative_to(self.root):
                raise ToolError(
                    "path escapes the workspace through a symlink",
                    error_type="path_error",
                    blocked_target=relative,
                )
            if directory is True and not actual.is_dir():
                raise ToolError(
                    f"path is not a directory: {relative}", error_type="path_error", blocked_target=relative
                )
            if directory is False and not actual.is_file():
                raise ToolError(
                    f"path is not a file: {relative}", error_type="path_error", blocked_target=relative
                )
            return actual
        parent = candidate.parent.resolve(strict=False)
        if not parent.is_relative_to(self.root):
            raise ToolError(
                "path parent escapes the workspace", error_type="path_error", blocked_target=relative
            )
        if directory is True:
            raise ToolError(
                f"directory does not exist: {relative}", error_type="not_found", blocked_target=relative
            )
        return candidate

    def relative(self, path: Path) -> str:
        return path.resolve(strict=False).relative_to(self.root).as_posix()


__all__ = ["ROOT_RELATIVE_PATH_RULE", "SafeWorkspace"]
