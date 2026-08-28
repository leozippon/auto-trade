"""Isolated-shell tool; it cannot execute a host subprocess."""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping, Sequence

from .base import CommandRunner, ToolError, ToolResult, ToolSpec
from .workspace import SafeWorkspace

# Advisory (not enforced): nudge the Agent away from hiding stderr, which breaks audit.
STDERR_SUPPRESSION_RE = re.compile(r"2\s*>\s*/dev/null|&>\s*/dev/null|/dev/null\s+2\s*>\s*&\s*1")
STDERR_SUPPRESSION_REMINDER = (
    "stderr 被重定向到 /dev/null：错误输出对审计与调试很重要，请保留 stderr（去掉 2>/dev/null 等）。"
)
DEFAULT_SHELL_TIMEOUT_SECONDS = 30.0
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
                "items": {"type": "string", "minLength": 1, "maxLength": 1000},
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


def _shell_description(timeout_seconds: float) -> str:
    return (
        "Run one bounded foreground argv command in the injected network-disabled "
        "Agent sandbox. `argv` is a JSON array of strings, e.g. "
        '["python", "-c", "print(1)"] or ["bash", "-lc", "ls output"]; a single '
        "command-line string is rejected. `cwd` and every path must stay inside "
        f"the workspace (relative, no `..`). `timeout_seconds` is at most {timeout_seconds:g}."
    )


class SandboxShellTool:
    spec = ToolSpec(
        "shell",
        _shell_description(DEFAULT_SHELL_TIMEOUT_SECONDS),
        _shell_input_schema(DEFAULT_SHELL_TIMEOUT_SECONDS),
        mutating=True,
    )

    def __init__(
        self,
        workspace: SafeWorkspace,
        runner: CommandRunner,
        *,
        timeout_seconds: float = DEFAULT_SHELL_TIMEOUT_SECONDS,
        max_output_chars: int = 40_000,
    ) -> None:
        if timeout_seconds <= 0 or max_output_chars <= 0:
            raise ValueError("shell limits must be positive")
        self.spec = ToolSpec(
            "shell",
            _shell_description(timeout_seconds),
            _shell_input_schema(timeout_seconds),
            mutating=True,
        )
        self.workspace = workspace
        self.runner = runner
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        raw_argv = arguments["argv"]
        if not isinstance(raw_argv, list) or not raw_argv:
            raise ToolError("argv must contain at least one argument")
        argv: Sequence[str] = tuple(str(item) for item in raw_argv)
        reject_forbidden_wait(argv)
        requested_cwd = str(arguments.get("cwd", "."))
        cwd = self.workspace.resolve(requested_cwd, must_exist=True, directory=True)
        requested_timeout = float(arguments.get("timeout_seconds", self.timeout_seconds))
        timeout = min(requested_timeout, self.timeout_seconds)
        result = self.runner.run(
            argv,
            cwd=self.workspace.relative(cwd) if cwd != self.workspace.root else ".",
            timeout_seconds=timeout,
            max_output_chars=self.max_output_chars,
            input_text=str(arguments["input"]) if "input" in arguments else None,
        )
        record = result.to_record()
        # Audit statistics only; permissions stay with the sandbox, the
        # filesystem and the tool registry's post-finish write lock.
        record["command_kind"] = _classify_command(argv)
        reminder = _stderr_suppression_reminder(argv)
        if reminder:
            record["stderr_suppression_reminder"] = reminder
        return ToolResult(True, value=record)


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
    "DEFAULT_SHELL_TIMEOUT_SECONDS",
    "FORBIDDEN_WAIT",
    "SandboxShellTool",
    "argv_is_forbidden_wait",
    "reject_forbidden_wait",
]
