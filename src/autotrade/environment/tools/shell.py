"""Isolated-shell tool; it cannot execute a host subprocess."""

from __future__ import annotations

import json
import re
import shlex
import threading
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath

from ..sandbox import SCREENING_TOOL_MOUNT
from .base import (
    CommandRunner,
    ToolError,
    ToolResult,
    ToolResultStore,
    ToolSchemaError,
    ToolSpec,
)
from .workspace import FREE_SPACE_RECOVERY_COMMANDS, SafeWorkspace

# Advisory (not enforced): nudge the Agent away from hiding stderr, which breaks audit.
STDERR_SUPPRESSION_RE = re.compile(r"2\s*>\s*/dev/null|&>\s*/dev/null|/dev/null\s+2\s*>\s*&\s*1")
STDERR_SUPPRESSION_REMINDER = (
    "stderr 被重定向到 /dev/null：错误输出对审计与调试很重要，请保留 stderr（去掉 2>/dev/null 等）。"
)
# Default per-call timeout when the Agent omits ``timeout_seconds``, and the
# hard cap it may request. Data checks over PIT parquet (IC tables, coverage
# scans) regularly need more than 30 s, and a full-market pass in the session
# sandbox (``SandboxSpec.cpus``, 8) did not fit 300 s; 600 s keeps every call a
# bounded foreground command while no longer starving a child of its numbers.
DEFAULT_SHELL_TIMEOUT_SECONDS = 60.0
MAX_SHELL_TIMEOUT_SECONDS = 600.0
# Per-element cap. Trace audits show legitimate `bash -lc` heredocs and
# `python -c` probes running 1-2k chars, so a 1000-char cap refused work that
# was already written correctly; 4000 admits those while still pushing a real
# script into a file.
SHELL_ARGV_MAX_CHARS = 4000
# Per-stream inline budget for one observation, and the host-side capture cap
# behind it: a stream over the inline budget keeps a head and a tail inline
# and spills the whole capture to the result store; a command that produces
# more than the capture cap loses the rest, explicitly.
DEFAULT_SHELL_OUTPUT_CHARS = 40_000
SHELL_CAPTURE_MAX_CHARS = 1_000_000
# Trace audits show the same argv shapes recurring in every session, mostly on a
# fresh sub-agent's first shell call: the whole command line as one string
# (bare, or as the single element of a one-element array), a JSON-encoded
# array, and a long ``python -c`` script inlined as one element. All but the
# last are the same command in a different wrapper, so they are repaired here
# and the repair is named in the result; only the over-long element is refused,
# with the recipe that replaces it. A command sent under another key is not a
# wrapper of the command but a wrong field name: the schema names ``argv`` and
# answers with a correct call example, which costs one round and teaches the
# field, so this tool guesses no key.
ARGV_STRING_NOTE = (
    "argv arrived as a JSON-encoded string and was parsed into an array; "
    "send argv as a real JSON array of strings, not a string containing one"
)
ARGV_SHELL_NOTE = (
    "argv arrived as one command string, so it was run as "
    '["bash", "-lc", "<that command line>"] -- pipes, redirections, globs, '
    "`&&` and $VAR are the shell's, and an argv array is what runs without one"
)
ARGV_ONE_ELEMENT_NOTE = (
    "argv held a single element that was itself a whole command line, so the "
    "one-element array was unwrapped"
)
ARGV_TOO_LONG_HINT = (
    f"each argv element is at most {SHELL_ARGV_MAX_CHARS} chars: write the "
    "script to a file with write_file (e.g. notes/probe.py) and run "
    '["python", "notes/probe.py"] instead of inlining it after -c'
)
# The shell a command string is handed to: the same form this tool's own
# description names, so a repaired call and an explicit one run identically.
COMMAND_STRING_SHELL = ("bash", "-lc")
FORBIDDEN_WAIT = "forbidden_wait"
_WAIT_COMMANDS = frozenset({"sleep", "usleep"})
_WAIT_WRAPPERS = frozenset({"env", "timeout", "nice", "stdbuf", "nohup", "time"})
_SHELLS = frozenset({"sh", "bash", "dash"})
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
        "Run one bounded foreground command in the injected network-disabled "
        "Agent sandbox. Send `argv` either as a JSON array of strings, which runs "
        'directly with no shell (["python", "-c", "print(1)"]), or as one command '
        'string, which runs as ["bash", "-lc", "<that command line>"] so pipes, '
        "redirections, globs, `&&` and $VAR work as written "
        '("ls output | wc -l"). A one-element array holding a whole command line, '
        "and the array sent as one JSON-encoded string, are read the same way; the "
        "result names whichever repair was applied. "
        "Each argv element is at most "
        f"{SHELL_ARGV_MAX_CHARS} chars: put longer code in a file with write_file "
        '(e.g. notes/probe.py) and run ["python", "notes/probe.py"]. '
        "`cwd` and every path must stay inside the workspace (relative, no `..`); "
        "`skills/` is read-only for shell too (write_skill/delete_skill are its only writers). "
        "The one path outside the workspace this tool may run is the read-only signal screen "
        f"`{SCREENING_TOOL_MOUNT}` (e.g. [\"python\", \"{SCREENING_TOOL_MOUNT}\", \"--help\"]); "
        "it is a bind mount outside every file-tool root, so read_file/grep/glob cannot open it. "
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
        """Repair or refuse the shape of ``argv`` before the schema sees it.

        A JSON-encoded array, a plain command line and that command line
        wrapped in a one-element array are the same command in a different
        wrapper: each is rewritten into the argv array and the repair is
        reported back in the result, so the next call can be the canonical
        shape. A command line is handed to ``bash -lc`` rather than word-split,
        because that is what the Agent meant by writing one; what stays refused
        is what cannot be rewritten faithfully: an element over the per-element
        cap (refused with the write_file recipe instead of a bare length error)
        and a malformed JSON array, which nothing can reconstruct.
        """

        self._repair.note = None
        notes: list[str] = []
        argv = arguments.get("argv")
        if isinstance(argv, str):
            argv, note = _argv_from_string(argv)
            notes.append(note)
            arguments = {**arguments, "argv": argv}
        elif (lone := _lone_command_line(argv)) is not None:
            argv, note = _argv_from_string(lone)
            notes.extend((ARGV_ONE_ELEMENT_NOTE, note))
            arguments = {**arguments, "argv": argv}
        if isinstance(argv, list):
            _reject_long_argv_elements(argv)
        self._repair.note = "; ".join(notes) or None
        return arguments

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        raw_argv = arguments["argv"]
        if not isinstance(raw_argv, list) or not raw_argv:
            raise ToolError("argv must contain at least one argument")
        argv: Sequence[str] = tuple(str(item) for item in raw_argv)
        reject_forbidden_wait(argv)
        if PurePosixPath(argv[0]).name not in FREE_SPACE_RECOVERY_COMMANDS:
            self.workspace.require_free_space("shell")
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
    """The argv array a JSON-encoded array string holds, else ``None``.

    Parsed with ``strict=False`` because the recurring shape carries a
    multi-line ``python -c`` body whose newlines were never escaped: strict
    JSON rejects those raw control characters, yet the array around them is
    unambiguous and rewrites faithfully, element for element.
    """

    text = value.strip()
    if not text.startswith("["):
        return None
    try:
        parsed = json.loads(text, strict=False)
    except ValueError:
        return None
    if (
        isinstance(parsed, list)
        and parsed
        and all(isinstance(item, str) and item for item in parsed)
    ):
        return parsed
    return None


def _lone_command_line(argv: object) -> str | None:
    """The whole command line a one-element argv array holds, else ``None``.

    ``["python -c 'print(1)'"]`` names no executable -- nothing is called that
    -- so the element can only be a command line the model failed to split.
    Internal whitespace is the whole test: a genuine one-element argv is a bare
    program name, and a program whose own path carries a space is the accepted
    cost of repairing the shape that actually recurs.
    """

    if not (isinstance(argv, list) and len(argv) == 1 and isinstance(argv[0], str)):
        return None
    element: str = argv[0]
    return element if any(char.isspace() for char in element.strip()) else None


def _argv_from_string(value: str) -> tuple[list[str], str]:
    """The argv array one string form holds, and the note naming the repair.

    A JSON-encoded array is the same command with one layer of quoting too
    many. Anything else is a command line, and a command line is what a shell
    reads: word-splitting it here instead would hand `|`, `&&`, `2>` or `$VAR`
    to the program as literal words, so it goes to ``bash -lc`` whole -- the
    very form this tool's description names. What is still refused is a value
    that looks like JSON but is not an array of non-empty strings: it is a
    mis-encoded array, not a command line, and running it would report a
    shell syntax error instead of what actually went wrong.
    """

    parsed = _json_string_argv(value)
    if parsed is not None:
        return parsed, ARGV_STRING_NOTE
    text = value.strip()
    if not text:
        raise ToolSchemaError("argv is empty")
    if text.startswith(("[", "{")):
        raise ToolSchemaError(
            "argv looks like a JSON array or object but does not parse as an "
            "array of non-empty strings, so it cannot be reconstructed: send a "
            "real JSON array of strings, or send the command line itself as one "
            "plain string (no JSON), which this tool runs with bash -lc"
        )
    return [*COMMAND_STRING_SHELL, value], ARGV_SHELL_NOTE


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
    if name in _SHELLS:
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
    """Best-effort audit label only; permissions are enforced by Docker/filesystem.

    A command line runs as ``bash -lc <script>``, whether the Agent wrote that
    array or this tool built it from a command string, so the label comes from
    the script's first command; labelling every such call by its shell would
    say nothing about what ran.
    """
    if tokens and _basename(str(tokens[0])).lower() in _SHELLS:
        script = _shell_c_script(tokens)
        if script is not None:
            tokens = _first_command_tokens(script)
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
    "ARGV_ONE_ELEMENT_NOTE",
    "ARGV_SHELL_NOTE",
    "ARGV_STRING_NOTE",
    "ARGV_TOO_LONG_HINT",
    "COMMAND_STRING_SHELL",
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
