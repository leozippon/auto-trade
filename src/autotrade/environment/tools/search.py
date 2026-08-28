"""Structured read-only search tools for Agent exploration.

These tools borrow the useful parts of Claude Code's Grep/Glob design without
opening a general host filesystem search surface. They only read allowlisted
sandbox roots and return paginated, budgeted observations.

Deliberately retained although upstream agent CLIs moved to shell-only search
(Codex never shipped model-facing file-search tools; Claude Code has since
deprecated its Glob/Grep tools while keeping Read): under this framework's
~128k-token main-conversation budget they are load-bearing — bounded paginated
observations, locator summaries that survive tool-result clearing, and the only
concurrency-safe calls the runner may parallelize within a turn (shell is
serialized by design). Shell `rg` stays available for pipelines; the overlap
is a budget/parallelism boundary, not redundancy.

The read-only roots reach beyond the writable workspace on purpose: the Agent
authors against PIT decision inputs and prior-fold artifacts, so it must be
able to locate and read them (``snapshot``/``train``/``valid`` PIT views,
``parent_output``/``parent_models`` inherited artifacts, ``results`` backtest
outputs and the ``steps`` lineage) without shelling out.
"""

from __future__ import annotations

import fnmatch
import os
import selectors
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from autotrade.environment.executor import close_process_pipes

from .base import ToolError, ToolResult, ToolSpec
from .workspace import SafeWorkspace

DEFAULT_GREP_LIMIT = 250
DEFAULT_GLOB_LIMIT = 100
DEFAULT_READ_LIMIT = 2000
MAX_HEAD_LIMIT = 1000
MAX_READ_LIMIT = 5000
MAX_RESULT_CHARS = 20_000
# `read_file` decodes the whole file host-side before line pagination, so it
# must refuse unbounded inputs (multi-GB parquet files live under readable
# roots); large/binary files belong to shell head/tail or DuckDB/pyarrow reads.
MAX_READ_BYTES = 10 * 1024 * 1024
RG_TIMEOUT_SECONDS = 20.0
VCS_DIRS = (".git", ".hg", ".svn", ".bzr", ".jj", ".sl")
SEARCH_ROOTS = (
    "agent",
    "workspace",
    "output",
    "models",
    "snapshot",
    "train",
    "valid",
    "artifacts",
    "parent_output",
    "parent_models",
    "results",
    "steps",
)
GREP_OUTPUT_MODES = ("content", "files", "count")
# Roots that live outside the writable workspace tree; resolved from the
# sandbox layout when one is available.
_LAYOUT_ROOTS = {
    "agent": "agent",
    # The decision view bound into the container as /mnt/snapshot.
    "snapshot": "current_snapshot",
    "train": "train",
    "valid": "valid",
    "artifacts": "artifacts",
    "parent_output": "parent_output",
    "parent_models": "parent_model_artifacts",
    "results": "results",
    "steps": "steps",
}


class SearchRoots:
    """Allowlisted read roots for the structured search tools.

    ``workspace``/``output``/``models`` always resolve inside the caller's own
    session tree, because that is what the session actually writes. The
    remaining roots come from the sandbox layout and are simply absent when a
    caller has no layout, so a root is only ever offered when it exists.
    """

    def __init__(self, workspace: SafeWorkspace, *, paths: object | None = None) -> None:
        self.workspace = workspace
        roots: dict[str, Path] = {
            "workspace": workspace.root,
            "output": workspace.root / "output",
            "models": workspace.root / "models",
        }
        if paths is not None:
            for name, attribute in _LAYOUT_ROOTS.items():
                base = getattr(paths, attribute, None)
                if base is not None:
                    roots[name] = Path(base)
        self._roots = {name: roots[name] for name in SEARCH_ROOTS if name in roots}
        self.log_root = Path(getattr(paths, "logs", workspace.root / "logs"))

    @property
    def names(self) -> tuple[str, ...]:
        """Root names offered to the Agent: allowlisted AND present on disk."""
        available = tuple(name for name, base in self._roots.items() if base.is_dir())
        return available or ("workspace",)

    def base(self, root: str) -> Path:
        if root not in self._roots:
            raise ToolError(
                f"unsupported search root: {root}",
                error_type="path_error",
                blocked_target=root,
                retry_hint=f"available roots: {', '.join(self.names)}",
            )
        return self._roots[root]

    def resolve(self, root: str, path: str) -> tuple[Path, str]:
        base = self.base(root)
        if not base.is_dir():
            raise ToolError(
                f"search root is not available in this session: {root}",
                error_type="not_found",
                blocked_target=root,
                retry_hint=f"available roots: {', '.join(self.names)}",
            )
        target = _safe_subpath(base, path)
        if not target.exists():
            raise ToolError(f"search path does not exist: {root}:{path}", error_type="not_found")
        return target, str(base)

    def store_tool_result(self, *, tool: str, kind: str, content: str) -> dict[str, object]:
        """Persist an oversized tool result outside the model context budget."""
        result_dir = self.log_root / "tool_results" / f"{tool}_{kind}_{os.getpid()}_{time.monotonic_ns()}"
        try:
            result_dir.mkdir(parents=True, exist_ok=True)
            path = result_dir / f"{kind}.txt"
            path.write_text(content, encoding="utf-8", errors="replace")
        except OSError:
            return {}
        return {"result_path": str(path)}


def _root_field(roots: SearchRoots, description: str) -> dict[str, object]:
    return {"type": "string", "enum": list(roots.names), "description": description}


class _SearchToolBase:
    """Shared root resolution, ripgrep runner and result budgeting."""

    def __init__(self, roots: SearchRoots, *, timeout_seconds: float = RG_TIMEOUT_SECONDS) -> None:
        self.roots = roots
        self.timeout_seconds = timeout_seconds

    def _run_rg(self, args: list[str], cwd: Path, *, max_lines: int) -> dict[str, object]:
        try:
            proc = subprocess.Popen(args, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise ToolError(
                "ripgrep executable 'rg' is not available",
                error_type="unavailable",
                retry_hint="use shell grep for this search",
            ) from exc
        return _collect_rg_lines(proc, timeout_seconds=self.timeout_seconds, max_lines=max_lines)

    def _apply_result_budget(self, content: str, *, tool_kind: str) -> tuple[str, dict[str, object]]:
        if len(content) <= MAX_RESULT_CHARS:
            return content, {"truncated_by_chars": False}
        stored = self.roots.store_tool_result(tool=self.spec.name, kind=tool_kind, content=content)
        return content[:MAX_RESULT_CHARS], {"truncated_by_chars": True, **stored}

    def _budget_page_lines(self, lines: list[str], *, tool_kind: str) -> tuple[list[str], dict[str, object]]:
        """Apply the inline char budget to a filename page as a whole.

        The full page is persisted on overflow (same as text content); the
        returned list drops the char-cut partial last name instead of leaking
        a fabricated path into the observation."""
        truncated, budget = self._apply_result_budget("\n".join(lines), tool_kind=tool_kind)
        if not budget.get("truncated_by_chars"):
            return lines, budget
        return truncated.split("\n")[:-1], budget


class GrepTool(_SearchToolBase):
    """Ripgrep over an allowlisted root with structured pagination."""

    def __init__(self, roots: SearchRoots, *, timeout_seconds: float = RG_TIMEOUT_SECONDS) -> None:
        super().__init__(roots, timeout_seconds=timeout_seconds)
        self.spec = ToolSpec(
            "grep",
            "Search allowlisted sandbox roots with ripgrep and structured pagination. "
            "Use for targeted text/code/log search; choose files/count modes before content when possible.",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "minLength": 1, "maxLength": 1000,
                                "description": "Ripgrep pattern to search for."},
                    "root": _root_field(roots, "Allowlisted sandbox root to search."),
                    "path": {"type": "string", "maxLength": 500,
                             "description": "Optional relative subpath under root; empty searches the whole root."},
                    "glob": {"type": "string", "maxLength": 500,
                             "description": "Optional ripgrep glob filter such as '*.py' or '**/*.md'."},
                    "output_mode": {"type": "string", "enum": list(GREP_OUTPUT_MODES),
                                    "description": "Return matching files, counts, or content lines."},
                    "head_limit": {"type": "integer", "minimum": 1, "maximum": MAX_HEAD_LIMIT,
                                   "description": "Maximum number of paginated matches to return."},
                    "offset": {"type": "integer", "minimum": 0, "description": "Pagination offset."},
                    "context": {"type": "integer", "minimum": 0, "maximum": 20,
                                "description": "Context lines around content matches (output_mode='content')."},
                    "case_insensitive": {"type": "boolean", "description": "Enable case-insensitive search."},
                    "multiline": {"type": "boolean", "description": "Enable ripgrep multiline search."},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        )

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        pattern = str(arguments["pattern"])
        root = str(arguments.get("root") or "workspace")
        path = str(arguments.get("path") or "")
        glob = str(arguments.get("glob") or "")
        output_mode = str(arguments.get("output_mode") or "files")
        head_limit = int(arguments.get("head_limit") or DEFAULT_GREP_LIMIT)
        offset = int(arguments.get("offset") or 0)
        context = int(arguments.get("context") or 0)
        if output_mode not in GREP_OUTPUT_MODES:
            raise ToolError(f"unsupported grep output_mode: {output_mode}", error_type="schema_error")
        target, display_root = self.roots.resolve(root, path)
        cwd, target_arg = (target.parent, target.name) if target.is_file() else (target, ".")
        args = [
            "rg", "--no-heading", "--color", "never", "--max-columns", "500",
            # Search roots are data/artifact trees, not repositories: a stray
            # .ignore/.rgignore file must not silently filter results out of
            # sync with the glob walker (VCS dirs are excluded explicitly).
            "--no-ignore",
        ]
        for vcs_dir in VCS_DIRS:
            args.extend(["--glob", f"!{vcs_dir}/**"])
        if glob:
            _validate_relative_pattern(glob, label="glob")
            args.extend(["--glob", glob])
        if output_mode == "files":
            args.append("--files-with-matches")
        elif output_mode == "count":
            args.append("--count-matches")
        else:
            args.append("--line-number")
            if context:
                args.extend(["-C", str(context)])
        if bool(arguments.get("case_insensitive")):
            args.append("-i")
        if bool(arguments.get("multiline")):
            args.extend(["-U", "--multiline-dotall"])
        args.extend(["-e", pattern, target_arg])

        completed = self._run_rg(args, cwd, max_lines=offset + head_limit + 1)
        raw_lines = _clean_rg_lines(completed["stdout_lines"])
        stderr = str(completed["stderr"])
        if completed["exit_code"] not in (0, 1) and not completed["line_limited"]:
            raise ToolError(
                stderr.strip() or f"ripgrep failed with exit code {completed['exit_code']}",
                error_type="timeout" if completed["timeout"] else "tool_error",
                details={"exit_code": completed["exit_code"]},
            )

        common = {
            "mode": output_mode, "root": root, "root_path": display_root, "path": path,
            "pattern": pattern, "glob": glob, "stderr": stderr, "timeout": completed["timeout"],
        }
        if output_mode == "count":
            visible_lines, paging = _apply_paging(
                raw_lines, offset=offset, head_limit=head_limit, source_truncated=completed["line_limited"]
            )
            content, budget = self._apply_result_budget("\n".join(visible_lines), tool_kind="grep_count")
            value = {
                **common,
                "num_lines": len(raw_lines),
                "page_matches": _sum_count_lines(visible_lines),
                "num_matches_lower_bound": _sum_count_lines(raw_lines),
                "num_matches_known": not completed["line_limited"],
                "content": content,
                **paging, **budget,
            }
        elif output_mode == "files":
            visible_lines, paging = _apply_paging(
                raw_lines, offset=offset, head_limit=head_limit, source_truncated=completed["line_limited"]
            )
            # The page rides the observation once, as `filenames`: duplicating
            # it into a joined `content` copy doubled the bytes and the copy
            # escaped the char budget entirely.
            filenames, budget = self._budget_page_lines(visible_lines, tool_kind="grep_files")
            if len(filenames) != len(visible_lines):
                paging["returned"] = len(filenames)
                paging["truncated"] = True
            value = {**common, "num_files": paging["total"], "filenames": filenames, **paging, **budget}
        else:
            visible_lines, paging = _apply_paging(
                raw_lines, offset=offset, head_limit=head_limit, source_truncated=completed["line_limited"]
            )
            content, budget = self._apply_result_budget("\n".join(visible_lines), tool_kind="grep_content")
            # Derive the page's file list from the budgeted visible content
            # (dropping a char-cut partial last line), not from the raw lines:
            # every observation field must respect the same inline budget.
            content_lines = content.split("\n")
            if budget.get("truncated_by_chars"):
                content_lines = content_lines[:-1]
            value = {
                **common,
                "num_lines": len(raw_lines),
                "filenames": sorted(_filenames_from_content(content_lines)),
                "content": content,
                **paging, **budget,
            }
        return ToolResult(True, value=value)


class GlobTool(_SearchToolBase):
    """List files under an allowlisted root with structured pagination."""

    def __init__(self, roots: SearchRoots, *, timeout_seconds: float = RG_TIMEOUT_SECONDS) -> None:
        super().__init__(roots, timeout_seconds=timeout_seconds)
        self.spec = ToolSpec(
            "glob",
            "List files under an allowlisted sandbox root with structured pagination. "
            "Use to discover files by name/pattern before reading or grepping them.",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "minLength": 1, "maxLength": 500,
                                "description": "File glob such as '*.py' for one directory or '**/*.py' recursively."},
                    "root": _root_field(roots, "Allowlisted sandbox root to list."),
                    "path": {"type": "string", "maxLength": 500,
                             "description": "Optional relative subpath under root; empty lists from the root."},
                    "head_limit": {"type": "integer", "minimum": 1, "maximum": MAX_HEAD_LIMIT,
                                   "description": "Maximum number of paginated paths to return."},
                    "offset": {"type": "integer", "minimum": 0, "description": "Pagination offset."},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        )

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        pattern = str(arguments["pattern"])
        root = str(arguments.get("root") or "workspace")
        path = str(arguments.get("path") or "")
        head_limit = int(arguments.get("head_limit") or DEFAULT_GLOB_LIMIT)
        offset = int(arguments.get("offset") or 0)
        _validate_relative_pattern(pattern, label="pattern")
        target, display_root = self.roots.resolve(root, path)
        if not target.is_dir():
            raise ToolError(f"glob path must be a directory: {root}:{path}", error_type="path_error")
        files: list[str] = []
        seen_matches = 0
        source_truncated = False
        for candidate in _iter_glob_matches(target, pattern):
            seen_matches += 1
            if seen_matches <= offset:
                continue
            if len(files) >= head_limit:
                source_truncated = True
                break
            files.append(str(candidate.relative_to(target)))
        visible, paging = _apply_paging(
            files, offset=0, head_limit=head_limit, source_truncated=source_truncated,
            total_prefix=max(seen_matches - len(files), 0),
        )
        paging["offset"] = offset
        # Same single-field page contract as files-mode grep.
        filenames, budget = self._budget_page_lines(visible, tool_kind="glob")
        if len(filenames) != len(visible):
            paging["returned"] = len(filenames)
            paging["truncated"] = True
        return ToolResult(True, value={
            "root": root, "root_path": display_root, "path": path, "pattern": pattern,
            "num_files": paging["total"], "filenames": filenames, **paging, **budget,
        })


class ReadFileTool(_SearchToolBase):
    """Read a file under an allowlisted root with line numbers and pagination."""

    def __init__(self, roots: SearchRoots, *, timeout_seconds: float = RG_TIMEOUT_SECONDS) -> None:
        super().__init__(roots, timeout_seconds=timeout_seconds)
        self.spec = ToolSpec(
            "read_file",
            "Read a file under an allowlisted sandbox root with line numbers and pagination. "
            "Prefer this over `shell cat`/`head` for code you will edit (line-numbered, bounded output); "
            "`cat`/`head` stay available for pipelines. Rejects files over the size cap "
            "(use shell or DuckDB/pyarrow for large data files).",
            {
                "type": "object",
                "properties": {
                    "root": _root_field(roots, "Allowlisted sandbox root the file lives under."),
                    "path": {"type": "string", "minLength": 1, "maxLength": 500,
                             "description": "Relative file path under root."},
                    "offset": {"type": "integer", "minimum": 0,
                               "description": "Starting line offset (0-based)."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_READ_LIMIT,
                              "description": "Maximum number of lines to return from offset."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        root = str(arguments.get("root") or "workspace")
        path = str(arguments["path"])
        offset = int(arguments.get("offset") or 0)
        limit = int(arguments.get("limit") or DEFAULT_READ_LIMIT)
        target, display_root = self.roots.resolve(root, path)
        if target.is_dir():
            raise ToolError(f"read path is a directory, not a file: {root}:{path}", error_type="path_error")
        try:
            size = target.stat().st_size
        except OSError as exc:
            raise ToolError(f"read failed for {root}:{path}: {exc}", error_type="tool_error") from exc
        if size > MAX_READ_BYTES:
            # The whole file is decoded host-side before pagination, so an
            # unbounded input (e.g. a multi-GB parquet) must be refused, not
            # silently absorbed outside the sandbox resource limits.
            raise ToolError(
                f"file is {size} bytes, over the {MAX_READ_BYTES}-byte read cap: {root}:{path}",
                error_type="too_large",
                blocked_target=f"{root}:{path}",
                retry_hint=(
                    "read_file is for bounded text files; use shell head/tail/wc for large text, "
                    "or Parquet metadata and DuckDB/pyarrow column reads for data files"
                ),
            )
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolError(f"read failed for {root}:{path}: {exc}", error_type="tool_error") from exc
        # cat -n style line numbering, paginated by line so large files stay bounded.
        numbered = [f"{index}\t{line}" for index, line in enumerate(text.splitlines(), start=1)]
        visible, paging = _apply_paging(numbered, offset=offset, head_limit=limit, source_truncated=False)
        content, budget = self._apply_result_budget("\n".join(visible), tool_kind="read")
        return ToolResult(True, value={
            "root": root, "root_path": display_root, "path": path,
            "line_count": len(numbered), "content": content, **paging, **budget,
        })


def _safe_subpath(base: Path, path: str) -> Path:
    base_resolved = base.resolve()
    if not path:
        return base_resolved
    candidate = Path(path)
    if candidate.is_absolute():
        raise ToolError(
            "search path must be relative to the selected root",
            error_type="path_error", blocked_target=path,
        )
    parts = PurePosixPath(path).parts
    if ".." in parts:
        raise ToolError("search path must not contain '..'", error_type="path_error", blocked_target=path)
    if any(part.startswith(".") for part in parts):
        raise ToolError(
            "search path must not contain hidden path components",
            error_type="path_error", blocked_target=path,
        )
    target = (base_resolved / candidate).resolve()
    _assert_inside(target, base_resolved)
    return target


def _iter_glob_matches(root: Path, pattern: str):
    pattern_parts = PurePosixPath(pattern).parts
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise ToolError(f"glob failed to list directory: {directory}", error_type="tool_error") from exc
        subdirs: list[Path] = []
        for candidate in entries:
            if candidate.is_symlink():
                continue
            if _is_vcs_path(candidate, root) or _has_hidden_part(candidate, root):
                continue
            _assert_inside(candidate, root)
            if candidate.is_dir():
                subdirs.append(candidate)
                continue
            if candidate.is_file() and _glob_match(candidate.relative_to(root), pattern_parts):
                yield candidate
        stack.extend(reversed(subdirs))


def _glob_match(relative_path: Path, pattern_parts: tuple[str, ...]) -> bool:
    return _glob_match_parts(tuple(PurePosixPath(relative_path.as_posix()).parts), pattern_parts)


def _glob_match_parts(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    if not pattern_parts:
        return not path_parts
    first = pattern_parts[0]
    if first == "**":
        return _glob_match_parts(path_parts, pattern_parts[1:]) or (
            bool(path_parts) and _glob_match_parts(path_parts[1:], pattern_parts)
        )
    if not path_parts:
        return False
    return fnmatch.fnmatchcase(path_parts[0], first) and _glob_match_parts(path_parts[1:], pattern_parts[1:])


def _assert_inside(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ToolError(
            f"path escapes the selected search root: {path}",
            error_type="path_error", blocked_target=str(path),
        ) from exc


def _validate_relative_pattern(pattern: str, *, label: str) -> None:
    if not pattern:
        raise ToolError(f"{label} must not be empty", error_type="schema_error")
    pure = PurePosixPath(pattern)
    if pure.is_absolute() or ".." in pure.parts:
        raise ToolError(
            f"{label} must be relative and must not contain '..'",
            error_type="path_error", blocked_target=pattern,
        )
    if any(part.startswith(".") for part in pure.parts):
        raise ToolError(
            f"{label} must not contain hidden path components",
            error_type="path_error", blocked_target=pattern,
        )


def _apply_paging(
    lines: list[str],
    *,
    offset: int,
    head_limit: int,
    source_truncated: bool = False,
    total_prefix: int = 0,
) -> tuple[list[str], dict[str, object]]:
    available_len = len(lines)
    total_lower_bound = total_prefix + len(lines)
    if offset > available_len:
        visible: list[str] = []
    else:
        visible = lines[offset : offset + head_limit]
    truncated = source_truncated or offset + len(visible) < available_len
    total = None if source_truncated else total_lower_bound
    return visible, {
        "offset": offset,
        "head_limit": head_limit,
        "returned": len(visible),
        "truncated": truncated,
        "total": total,
        "total_lower_bound": total_lower_bound,
        "total_known": not source_truncated,
    }


def _clean_rg_lines(lines: list[str]) -> list[str]:
    return [line[2:] if line.startswith("./") else line for line in lines]


def _sum_count_lines(lines: list[str]) -> int:
    total = 0
    for line in lines:
        _, _, count_text = line.rpartition(":")
        try:
            total += int(count_text)
        except ValueError:
            pass
    return total


def _filenames_from_content(lines: list[str]) -> set[str]:
    filenames: set[str] = set()
    for line in lines:
        if not line or line == "--":
            continue
        separator = line.find(":")
        if separator <= 0:
            continue
        filenames.add(line[:separator])
    return filenames


def _collect_rg_lines(
    proc: subprocess.Popen,
    *,
    timeout_seconds: float,
    max_lines: int,
) -> dict[str, object]:
    selector = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
    selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
    stdout_lines: list[str] = []
    pending = bytearray()
    stderr = bytearray()
    timeout = False
    line_limited = False
    deadline = time.monotonic() + timeout_seconds

    while selector.get_map():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timeout = True
            proc.kill()
            break
        events = selector.select(timeout=min(0.1, remaining))
        if not events and proc.poll() is not None:
            events = selector.select(timeout=0)
        for key, _ in events:
            chunk = os.read(key.fileobj.fileno(), 8192)
            if not chunk:
                if key.data == "stdout" and pending:
                    line_limited = _append_rg_line(stdout_lines, pending, max_lines=max_lines) or line_limited
                    pending = bytearray()
                selector.unregister(key.fileobj)
                continue
            if key.data == "stderr":
                if len(stderr) < MAX_RESULT_CHARS:
                    stderr.extend(chunk[: MAX_RESULT_CHARS - len(stderr)])
                continue
            pending.extend(chunk)
            while b"\n" in pending:
                line, _, rest = pending.partition(b"\n")
                pending = bytearray(rest)
                if _append_rg_line(stdout_lines, line, max_lines=max_lines):
                    line_limited = True
                    proc.terminate()
                    break
            if line_limited:
                break
        if line_limited:
            break

    for key in list(selector.get_map().values()):
        try:
            selector.unregister(key.fileobj)
        except (KeyError, ValueError):
            pass
    selector.close()
    if timeout:
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        close_process_pipes(proc)
        return {
            "exit_code": 124,
            "stdout_lines": stdout_lines,
            "stderr": (stderr.decode("utf-8", errors="replace") or f"timeout after {timeout_seconds}s"),
            "timeout": True,
            "line_limited": line_limited,
        }
    if line_limited:
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        close_process_pipes(proc)
        return {
            "exit_code": 0,
            "stdout_lines": stdout_lines,
            "stderr": stderr.decode("utf-8", errors="replace"),
            "timeout": False,
            "line_limited": True,
        }
    return_code = proc.wait()
    close_process_pipes(proc)
    return {
        "exit_code": return_code,
        "stdout_lines": stdout_lines,
        "stderr": stderr.decode("utf-8", errors="replace"),
        "timeout": False,
        "line_limited": False,
    }


def _append_rg_line(lines: list[str], line: bytes | bytearray, *, max_lines: int) -> bool:
    if len(lines) >= max_lines:
        return True
    lines.append(bytes(line).decode("utf-8", errors="replace").rstrip("\r"))
    return False


def _is_vcs_path(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part in VCS_DIRS for part in parts)


def _has_hidden_part(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part.startswith(".") for part in parts)


__all__ = [
    "GlobTool",
    "GrepTool",
    "ReadFileTool",
    "SEARCH_ROOTS",
    "SearchRoots",
]
