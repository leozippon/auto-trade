"""Isolated-shell tool; it cannot execute a host subprocess."""

from __future__ import annotations

import json
import re
import shlex
import threading
from collections.abc import Mapping, Sequence

from .base import (
    CommandRunner,
    ToolError,
    ToolResult,
    ToolResultStore,
    ToolSchemaError,
    ToolSpec,
)
from .workspace import SafeWorkspace

# Advisory (not enforced): nudge the Agent away from hiding stderr, which breaks audit.
STDERR_SUPPRESSION_RE = re.compile(r"2\s*>\s*/dev/null|&>\s*/dev/null|/dev/null\s+2\s*>\s*&\s*1")
STDERR_SUPPRESSION_REMINDER = (
    "stderr 被重定向到 /dev/null：错误输出对审计与调试很重要，请保留 stderr（去掉 2>/dev/null 等）。"
)
# Default per-call timeout when the Agent omits ``timeout_seconds``, and the
# hard cap it may request. Data checks over PIT parquet (IC tables, coverage
# scans) regularly need more than 30 s, and a full-market pass in the 4-CPU
# sandbox did not fit 300 s; 600 s keeps every call a bounded foreground
# command while no longer starving a child of its numbers.
DEFAULT_SHELL_TIMEOUT_SECONDS = 60.0
MAX_SHELL_TIMEOUT_SECONDS = 600.0
SHELL_ARGV_MAX_CHARS = 1000
# Per-stream inline budget for one observation, and the host-side capture cap
# behind it: a stream over the inline budget keeps a head and a tail inline
# and spills the whole capture to the result store; a command that produces
# more than the capture cap loses the rest, explicitly.
DEFAULT_SHELL_OUTPUT_CHARS = 40_000
SHELL_CAPTURE_MAX_CHARS = 1_000_000
# Longest command string still echoed back as its argv form in a shape error:
# past this the suggestion stops being readable and the rule alone is clearer.
_ARGV_SUGGESTION_MAX_CHARS = 300
# Trace audits show two argv shapes recurring in every Fold, mostly on a fresh
# sub-agent's first shell call: a JSON-encoded array (repaired below, with the
# repair named in the result so the next call is a real array) and a long
# ``python -c`` script inlined as one element (refused with the file recipe).
ARGV_STRING_NOTE = (
    "argv arrived as a JSON-encoded string and was parsed into an array; "
    "send argv as a real JSON array of strings, not a string containing one"
)
ARGV_TOO_LONG_HINT = (
    f"each argv element is at most {SHELL_ARGV_MAX_CHARS} chars: write the "
    "script to a file with write_file (e.g. notes/probe.py) and run "
    '["python", "notes/probe.py"] instead of inlining it after -c'
)
FORBIDDEN_WAIT = "forbidden_wait"
_WAIT_COMMANDS = frozenset({"sleep", "usleep"})
_WAIT_WRAPPERS = frozenset({"env", "timeout", "nice", "stdbuf", "nohup", "time"})
_WAIT_SHELLS = frozenset({"sh", "bash", "dash"})
_DURATION_RE = re.compile(r"\d+(?:\.\d+)?[smhd]?")
_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_SHELL_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "|&"})
READ_ONLY_COMMANDS = {
    "awk",
    "cat",
    "cut",
    "du",
    "file",
    "find",
    "grep",
    "head",
    "jq",
    "less",
    "ls",
    "nl",
    "pwd",
    "rg",
    "sort",
    "stat",
    "tail",
    "wc",
}
SEARCH_COMMANDS = {"ag", "ack", "find", "grep", "locate", "rg", "which", "whereis"}
LIST_COMMANDS = {"du", "ls", "tree"}
SHELL_NEUTRAL_COMMANDS = {"echo", "printf", "true", "false", ":"}
WRITE_COMMANDS = {
    "apply_patch",
    "chmod",
    "chown",
    "cp",
    "dd",
    "install",
    "ln",
    "mkdir",
    "mv",
    "rm",
    "rmdir",
    "tee",
    "touch",
    "truncate",
}


def _shell_input_schema(timeout_seconds: float) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "argv": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": SHELL_ARGV_MAX_CHARS},
            },
            "cwd": {"type": "string", "minLength": 1, "maxLength": 500},
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.1,
                "maximum": timeout_seconds,
            },
            "input": {"type": "string", "maxLength": 100_000},
        },
        "required": ["argv"],
        "additionalProperties": False,
    }


def _shell_description(
    timeout_seconds: float, max_timeout_seconds: float, max_output_chars: int
) -> str:
    return (
        "Run one bounded foreground argv command in the injected network-disabled "
        "Agent sandbox. `argv` is a JSON array of strings, e.g. "
        '["python", "-c", "print(1)"] or ["bash", "-lc", "ls output"]; a single '
        "command-line string is rejected. Each argv element is at most "
        f"{SHELL_ARGV_MAX_CHARS} chars: put longer code in a file with write_file "
        '(e.g. notes/probe.py) and run ["python", "notes/probe.py"]. '
        "`cwd` and every path must stay inside the workspace (relative, no `..`); "
        "`skills/` is read-only for shell too (write_skill/delete_skill are its only writers). "
        "These are real sandbox filesystem paths under the workspace root, which the "
        "sandbox mounts at /mnt/agent/workspace and this tool enters as `.`; the file "
        "tools (read_file/write_file/edit_file/grep/glob) address the same files as a "
        "`root` name plus a path relative to it, reject that absolute form, and read a "
        "leading `workspace/` as that root name (dropping it unless a real `workspace/` "
        "directory exists) while shell always takes it literally — so keep script paths "
        "free of that prefix. "
        f"`timeout_seconds` defaults to {timeout_seconds:g} and is at most {max_timeout_seconds:g}; "
        "the command runs in the foreground and is killed at the timeout, so bound the work: "
        "validate a script on a sample of dates/stocks first, split a full-market or "
        "full-history pass into chunks that each finish within the cap, and checkpoint "
        "intermediate results to files under the workspace (they persist) rather than "
        "starting anything in the background. stdout and stderr over "
        f"{max_output_chars} chars come back as an inline head plus `<stream>_tail`, with the "
        "full stream spilled to a file: `<stream>_spill.result_hint` gives the read_file call "
        "for the omitted lines. Write large outputs to a workspace file and read them selectively."
    )


def _shell_example(timeout_seconds: float) -> dict[str, object]:
    return {"argv": ["python", "-c", "print(1)"], "cwd": ".", "timeout_seconds": timeout_seconds}


class SandboxShellTool:
    spec = ToolSpec(
        "shell",
        _shell_description(
            DEFAULT_SHELL_TIMEOUT_SECONDS, MAX_SHELL_TIMEOUT_SECONDS, DEFAULT_SHELL_OUTPUT_CHARS
        ),
        _shell_input_schema(MAX_SHELL_TIMEOUT_SECONDS),
        mutating=True,
        example=_shell_example(DEFAULT_SHELL_TIMEOUT_SECONDS),
    )

    def __init__(
        self,
        workspace: SafeWorkspace,
        runner: CommandRunner,
        *,
        timeout_seconds: float = DEFAULT_SHELL_TIMEOUT_SECONDS,
        max_timeout_seconds: float = MAX_SHELL_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_SHELL_OUTPUT_CHARS,
        capture_output_chars: int = SHELL_CAPTURE_MAX_CHARS,
        result_store: ToolResultStore | None = None,
    ) -> None:
        if timeout_seconds <= 0 or max_timeout_seconds <= 0 or max_output_chars <= 0:
            raise ValueError("shell limits must be positive")
        # A capture the runner cut is then always over the inline budget too.
        if capture_output_chars <= max_output_chars:
            raise ValueError("shell capture cap must exceed the inline output budget")
        # The default never exceeds the cap; a lower cap pulls it down.
        timeout_seconds = min(timeout_seconds, max_timeout_seconds)
        self.spec = ToolSpec(
            "shell",
            _shell_description(timeout_seconds, max_timeout_seconds, max_output_chars),
            _shell_input_schema(max_timeout_seconds),
            mutating=True,
            example=_shell_example(timeout_seconds),
        )
        self.workspace = workspace
        self.runner = runner
        self.timeout_seconds = timeout_seconds
        self.max_timeout_seconds = max_timeout_seconds
        self.max_output_chars = max_output_chars
        self.capture_output_chars = capture_output_chars
        self.result_store = result_store
        # One registry instance serves every concurrent sub-agent, so the note
        # ``normalize_arguments`` leaves for ``invoke`` (they always run back to
        # back on the same thread) must not be shared between calls.
        self._repair = threading.local()

    def normalize_arguments(self, arguments: Mapping[str, object]) -> Mapping[str, object]:
        """Repair or refuse the call shape before the schema sees it.

        A JSON-encoded array is the same command with one layer of quoting too
        many, so it is parsed and the repair is reported back in the result. A
        plain command line is a different call and stays refused, now with that
        very command written as an array; an element over the per-element cap is
        refused with the write_file recipe instead of a bare length error.
        """

        self._repair.note = None
        argv = arguments.get("argv")
        if isinstance(argv, str):
            parsed = _json_string_argv(argv)
            if parsed is None:
                raise _argv_shape_error(argv)
            self._repair.note = ARGV_STRING_NOTE
            arguments = {**arguments, "argv": parsed}
            argv = parsed
        if isinstance(argv, list):
            _reject_long_argv_elements(argv)
        return arguments

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        raw_argv = arguments["argv"]
        if not isinstance(raw_argv, list) or not raw_argv:
            raise ToolError("argv must contain at least one argument")
        argv: Sequence[str] = tuple(str(item) for item in raw_argv)
        reject_forbidden_wait(argv)
        requested_cwd = str(arguments.get("cwd", "."))
        cwd = self.workspace.resolve(requested_cwd, must_exist=True, directory=True)
        requested_timeout = float(arguments.get("timeout_seconds", self.timeout_seconds))
        timeout = min(requested_timeout, self.max_timeout_seconds)
        result = self.runner.run(
            argv,
            cwd=self.workspace.relative(cwd) if cwd != self.workspace.root else ".",
            timeout_seconds=timeout,
            max_output_chars=self.capture_output_chars,
            input_text=str(arguments["input"]) if "input" in arguments else None,
        )
        record = result.to_record()
        self._bound_stream(record, "stdout", result.stdout, capture_cut=result.stdout_truncated)
        self._bound_stream(record, "stderr", result.stderr, capture_cut=result.stderr_truncated)
        # Audit statistics only; permissions stay with the sandbox, the
        # filesystem and the tool registry's post-finish write lock.
        record["command_kind"] = _classify_command(argv)
        reminder = _stderr_suppression_reminder(argv)
        if reminder:
            record["stderr_suppression_reminder"] = reminder
        # Consumed once: the note belongs to the call that was repaired.
        note = getattr(self._repair, "note", None)
        self._repair.note = None
        if note:
            record["argv_normalized"] = note
        return ToolResult(True, value=record)

    def _bound_stream(
        self, record: dict[str, object], name: str, text: str, *, capture_cut: bool
    ) -> None:
        """Keep one stream within the inline budget without losing it.

        Over budget, the observation carries a head and a tail (errors sit at
        the end) plus the line geometry of the omitted middle, and the whole
        capture goes to the result store; ``read_file`` pages it by line, so
        the hint names the offset to resume from. A capture the runner itself
        cut has no true tail, so none is shown and the hint says so.
        """

        if len(text) <= self.max_output_chars and not capture_cut:
            return
        tail_chars = self.max_output_chars // 4
        head = text[: self.max_output_chars - tail_chars]
        if "\n" in head:
            head = head[: head.rfind("\n") + 1]
        total_lines = len(text.splitlines())
        head_lines = head.count("\n")
        tail = text[-tail_chars:] if tail_chars and not capture_cut else ""
        if "\n" in tail and text[-tail_chars - 1] != "\n":
            # Drop the partial first line unless the window starts on a line.
            tail = tail[tail.find("\n") + 1 :]
        tail_lines = len(tail.splitlines())
        omitted = total_lines - head_lines - tail_lines
        record[name] = head
        record[f"{name}_truncated"] = True
        record[f"{name}_lines"] = total_lines
        if tail:
            record[f"{name}_tail"] = tail
        if capture_cut:
            record[f"{name}_capture_truncated"] = True
        stored = (
            self.result_store.store_tool_result(tool="shell", kind=name, content=text)
            if self.result_store is not None
            else {}
        )
        if "result_ref" in stored:
            hint = (
                f"{name} exceeded the {self.max_output_chars}-char inline budget: "
                f"{head_lines} head lines inline, {omitted} lines omitted, {tail_lines} tail "
                f"lines in {name}_tail; full {name} spilled ({total_lines} lines), read the "
                f"omitted lines with: read_file root='{stored['result_root']}' "
                f"path='{stored['result_ref']}' offset={head_lines}"
            )
        else:
            hint = (
                f"{name} exceeded the {self.max_output_chars}-char inline budget and was not "
                f"persisted: {omitted} lines are lost; rerun with head/tail/grep or redirect "
                "the output to a workspace file"
            )
        if capture_cut:
            hint += (
                f"; the command produced more than the {self.capture_output_chars}-char "
                "capture cap, so the capture ends there and its true tail is lost"
            )
        record[f"{name}_spill"] = {**stored, "result_hint": hint}


def _json_string_argv(value: str) -> list[str] | None:
    """The argv array a JSON-encoded array string holds, else ``None``."""

    text = value.strip()
    if not text.startswith("["):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    if (
        isinstance(parsed, list)
        and parsed
        and all(isinstance(item, str) and item for item in parsed)
    ):
        return parsed
    return None


def _argv_shape_error(value: str) -> ToolSchemaError:
    """Refuse a command line, showing that same command as an argv array.

    A value that already looks like a JSON array failed to parse as one of
    non-empty strings, so splitting it as a command line would suggest
    nonsense (``"[1, 2]"`` -> ``["[1,", "2]"]``): it is told what the array
    must hold instead.
    """

    text = value.strip()
    if text.startswith(("[", "{")):
        return ToolSchemaError(
            "argv must be an array of separate strings, not one command string; "
            "this value parses as neither, so send a real JSON array whose "
            "elements are all non-empty strings"
        )
    message = "argv must be an array of separate strings, not one command string"
    try:
        tokens = shlex.split(value, posix=True)
    except ValueError:
        tokens = []
    if tokens:
        suggestion = json.dumps(tokens, ensure_ascii=False)
        if len(suggestion) <= _ARGV_SUGGESTION_MAX_CHARS:
            message = f"{message}; send argv: {suggestion}"
    return ToolSchemaError(message)


def _reject_long_argv_elements(argv: Sequence[object]) -> None:
    """Name the over-long element and the file-based way to run it instead.

    The schema enforces the same cap, but only as a length error; the recipe
    the Agent needs (write the script, then run the file) lives here.
    """

    for index, item in enumerate(argv):
        if isinstance(item, str) and len(item) > SHELL_ARGV_MAX_CHARS:
            # Same message as the schema check, so the shape stays familiar;
            # the retry hint is what this earlier check adds.
            raise ToolSchemaError(
                f"argv[{index}] is too long", retry_hint=ARGV_TOO_LONG_HINT
            )


def reject_forbidden_wait(argv: Sequence[str]) -> None:
    """Refuse sleep/usleep and a small set of wait wrappers before execution."""

    if not argv_is_forbidden_wait(argv):
        return
    raise ToolError(
        "shell wait is forbidden",
        error_type=FORBIDDEN_WAIT,
        reason=FORBIDDEN_WAIT,
        retry_hint="Run one bounded foreground command; do not sleep or poll.",
    )


def argv_is_forbidden_wait(argv: Sequence[str]) -> bool:
    """True when the effective first command is sleep/usleep.

    Unwraps env/timeout/nice/stdbuf/nohup/time and sh/bash/dash -c/-lc first
    commands. Unparseable scripts and unrelated binaries (pyright, python,
    grep, echo) are not waits.
    """

    tokens = [str(item) for item in argv if str(item)]
    return _argv_waits(tokens)


def _argv_waits(tokens: list[str]) -> bool:
    if not tokens:
        return False
    name = _basename(tokens[0]).lower()
    if name in _WAIT_COMMANDS:
        return True
    if name in _WAIT_SHELLS:
        script = _shell_c_script(tokens)
        if script is None:
            return False
        return _argv_waits(_first_command_tokens(script))
    if name in _WAIT_WRAPPERS:
        return _argv_waits(_unwrap_wrapper(name, tokens[1:]))
    return False


def _shell_c_script(argv: Sequence[str]) -> str | None:
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            return None
        if arg == "-c":
            return argv[index + 1] if index + 1 < len(argv) else None
        if arg.startswith("-") and not arg.startswith("--") and "c" in arg[1:]:
            return argv[index + 1] if index + 1 < len(argv) else None
        if arg.startswith("-"):
            index += 1
            continue
        return None
    return None


def _first_command_tokens(script: str) -> list[str]:
    try:
        tokens = shlex.split(script, posix=True)
    except ValueError:
        return []
    first: list[str] = []
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            break
        if (
            not first
            and _ASSIGNMENT_RE.match(token)
            and not token.startswith("-")
        ):
            continue
        first.append(token)
    return first


_WRAPPER_VALUE_FLAGS = {
    "timeout": frozenset({"-k", "-s", "--kill-after", "--signal"}),
    "env": frozenset(
        {
            "-u",
            "-C",
            "-S",
            "--unset",
            "--chdir",
            "--split-string",
            "--block-signal",
            "--default-signal",
        }
    ),
    "nice": frozenset({"-n", "--adjustment"}),
    "stdbuf": frozenset({"-i", "-o", "-e", "--input", "--output", "--error"}),
    "time": frozenset({"-f", "-o", "--format", "--output"}),
}
_WRAPPER_GLUED_PREFIXES = {
    "nice": ("-n",),
    "stdbuf": ("-i", "-o", "-e"),
}


def _unwrap_wrapper(name: str, args: list[str]) -> list[str]:
    if name == "nohup":
        return args[1:] if args[:1] == ["--"] else args
    value_flags = _WRAPPER_VALUE_FLAGS.get(name, frozenset())
    glued = _WRAPPER_GLUED_PREFIXES.get(name, ())
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            rest = args[index + 1 :]
            return _skip_duration(rest) if name == "timeout" else rest
        if name == "env" and (
            arg == "-" or (_ASSIGNMENT_RE.match(arg) and not arg.startswith("-"))
        ):
            index += 1
            continue
        if arg.startswith("-") and arg not in {"-", "--"}:
            option, eq, _tail = arg.partition("=")
            if eq or any(arg.startswith(prefix) and arg != prefix for prefix in glued):
                index += 1
                continue
            index += 2 if option in value_flags else 1
            continue
        rest = args[index:]
        return _skip_duration(rest) if name == "timeout" else rest
    return []


def _skip_duration(args: list[str]) -> list[str]:
    if args and _DURATION_RE.fullmatch(args[0]):
        return args[1:]
    return args


def _classify_command(tokens: Sequence[str]) -> str:
    """Best-effort audit label only; permissions are enforced by Docker/filesystem."""
    words = [_basename(token) for token in tokens if token and not token.startswith("-")]
    if not words:
        return "unknown"
    meaningful = [word for word in words if word not in SHELL_NEUTRAL_COMMANDS]
    if not meaningful:
        return "neutral"
    first = meaningful[0]
    if first in WRITE_COMMANDS:
        return "write"
    if first in SEARCH_COMMANDS:
        return "search"
    if first in LIST_COMMANDS:
        return "list"
    if first in READ_ONLY_COMMANDS:
        return "read"
    return "unknown"


def _stderr_suppression_reminder(tokens: Sequence[str]) -> str | None:
    """Advisory when the command redirects stderr away; never blocks the call.

    The argv contract keeps redirections out of the shell's own parsing, but an
    Agent can still reach one through ``bash -lc "... 2>/dev/null"``, so the
    whole command line is scanned.
    """
    return STDERR_SUPPRESSION_REMINDER if STDERR_SUPPRESSION_RE.search(" ".join(tokens)) else None


def _basename(token: str) -> str:
    return token.rstrip("/").rsplit("/", 1)[-1]


__all__ = [
    "ARGV_STRING_NOTE",
    "ARGV_TOO_LONG_HINT",
    "DEFAULT_SHELL_OUTPUT_CHARS",
    "DEFAULT_SHELL_TIMEOUT_SECONDS",
    "FORBIDDEN_WAIT",
    "MAX_SHELL_TIMEOUT_SECONDS",
    "SHELL_ARGV_MAX_CHARS",
    "SHELL_CAPTURE_MAX_CHARS",
    "SandboxShellTool",
    "argv_is_forbidden_wait",
    "reject_forbidden_wait",
]
