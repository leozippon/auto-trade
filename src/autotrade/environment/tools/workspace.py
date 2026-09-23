"""One path boundary shared by every workspace tool."""

from __future__ import annotations

import shutil
from pathlib import Path, PurePosixPath

from .base import ToolError

# Where the workspace is mounted inside the sandbox. The rule text below and the
# refusal hint for an absolute path quote the same constant.
WORKSPACE_MOUNT = "/mnt/agent/workspace"
# Stated in every file tool's description because the two path conventions are
# easy to mix up: `shell` runs on the real sandbox filesystem, while these tools
# address files as a root name plus a path relative to it. Trace audits show
# fresh sub-agents carrying the absolute path out of a shell call straight into
# their first read_file/edit_file call.
ROOT_RELATIVE_PATH_RULE = (
    "`root` plus `path` is a virtual address, not a sandbox filesystem path: the "
    f"`workspace` root is the directory shell runs in as `.` ({WORKSPACE_MOUNT}), "
    f"so pass path='inputs/x.json', never path='{WORKSPACE_MOUNT}/inputs/x.json'. "
    "A leading 'workspace/' is read as that root name and dropped when it does not "
    "resolve as a real directory, while shell always takes it literally, so leave it "
    "out: path='notes/probe.py' is what shell runs as ['python', 'notes/probe.py']."
)

# The workspace bind mount has no quota and shares one host filesystem with the
# data lake, the experiment directories and the logs: one arm wrote 6.4 GB of
# intermediates, and a naive full-market minute concatenation would need about
# 490 GB. A full disk would fail the nightly data update, other arms' ledger
# appends and the host's result writes at once, so the two calls that write in
# bulk -- ``shell`` and ``batch_validate`` -- are refused below this floor.
WORKSPACE_MIN_FREE_BYTES = 50 * 1024**3
LOW_DISK_SPACE = "low_disk_space"
# Below the floor ``shell`` still runs these, sent as an argv array (not as a
# command string, which runs under bash): shell is the only tool that can
# delete a file, so without them the refusal's own remedy could not be applied.
FREE_SPACE_RECOVERY_COMMANDS = frozenset({"du", "ls", "rm"})


def _relative_form_hint(pure: PurePosixPath) -> str:
    """Separate "you spelled the mount out" from "you tried to leave the tree".

    Both refusals share one message, but only the first has a mechanical
    repair, so the hint names it instead of restating the rule."""

    if pure.is_relative_to(WORKSPACE_MOUNT) and ".." not in pure.parts:
        return (
            f"the workspace is addressed relatively: drop the {WORKSPACE_MOUNT} prefix "
            f"and pass '{pure.relative_to(WORKSPACE_MOUNT).as_posix()}'"
        )
    return (
        "paths are relative to the workspace root: pass '.' for the root itself or a "
        "workspace-relative path such as 'candidates/x'; '..' and paths outside the "
        "workspace are refused"
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
                "path must stay within the workspace",
                error_type="path_error",
                blocked_target=relative,
                retry_hint=_relative_form_hint(pure),
            )
        if any(part.startswith(".") for part in pure.parts):
            raise ToolError(
                "hidden workspace paths are not available",
                error_type="path_error",
                blocked_target=relative,
                retry_hint=(
                    "no dot-prefixed name is readable or writable through these tools; "
                    "keep scratch files under a plain name such as 'notes/<topic>/' or "
                    "'candidates/<name>/scratch/'"
                ),
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

    def require_free_space(self, tool: str) -> None:
        """Refuse ``tool`` while the filesystem holding the workspace is below
        ``WORKSPACE_MIN_FREE_BYTES`` free."""

        free = shutil.disk_usage(self.root).free
        if free >= WORKSPACE_MIN_FREE_BYTES:
            return
        raise ToolError(
            f"{tool} refused: the disk holding the workspace has "
            f"{free / 1024**3:.1f} GiB free, below the "
            f"{WORKSPACE_MIN_FREE_BYTES / 1024**3:g} GiB floor that keeps the data "
            "lake and the other runs on it writable; delete intermediate files you "
            "no longer need, then call again",
            error_type=LOW_DISK_SPACE,
            retry_hint=(
                'shell ["du", "-sh", "notes"] finds large intermediates and '
                '["rm", "-r", "notes/<dir>"] removes them: du, ls and rm sent as an '
                "argv array (not a command string) still run below the floor"
            ),
            details={"free_bytes": free, "floor_bytes": WORKSPACE_MIN_FREE_BYTES},
        )


__all__ = [
    "FREE_SPACE_RECOVERY_COMMANDS",
    "LOW_DISK_SPACE",
    "ROOT_RELATIVE_PATH_RULE",
    "WORKSPACE_MIN_FREE_BYTES",
    "WORKSPACE_MOUNT",
    "SafeWorkspace",
]
