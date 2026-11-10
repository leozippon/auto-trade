"""Isolated-shell tool; it cannot execute a host subprocess."""

from __future__ import annotations

import json
import re
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
from .search import file_lines
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
# and the repair is named in the result in one short sentence (a command
# string is more than half of all calls, so the note rides on most results);
# only the over-long element is refused, with the recipe that replaces it. A
# command sent under another key is not a wrapper of the command but a wrong
# field name: the schema names ``argv`` and answers with a correct call
# example, which costs one round and teaches the field, so this tool guesses
# no key.
ARGV_STRING_NOTE = "argv was a JSON-encoded string; it ran as the array it holds"
ARGV_SHELL_NOTE = "argv was a command string; it ran under bash -lc"
ARGV_ONE_ELEMENT_NOTE = "argv's one element was a whole command line; it ran under bash -lc"
ARGV_TOO_LONG_HINT = (
    f"each argv element is at most {SHELL_ARGV_MAX_CHARS} chars: write the "
    "script to a file with write_file (e.g. notes/probe.py) and run "
    '["python", "notes/probe.py"] instead of inlining it after -c'
)
# The shell a command string is handed to: the same form this tool's own
# description names, so a repaired call and an explicit one run identically.
COMMAND_STRING_SHELL = ("bash", "-lc")


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
        "Run one bounded foreground command in the network-disabled sandbox, from the "
        "workspace directory. `argv` is a JSON array of strings, run directly with no "
        'shell (["python", "-c", "print(1)"]), or one command string, run under '
        "`bash -lc` so pipes, redirections, globs, `&&` and $VAR work. Each element is "
        f"at most {SHELL_ARGV_MAX_CHARS} chars: write longer code to a file and run "
        '["python", "notes/probe.py"]. `cwd` and every path stay inside the workspace '
        "(relative, no `..`); `skills/` is read-only here too. The one outside path it "
        f"may run is the read-only signal screen `{SCREENING_TOOL_MOUNT}`, which no file "
        "tool can open. "
        f"`timeout_seconds` defaults to {timeout_seconds:g} and is at most "
        f"{max_timeout_seconds:g}; the command is killed at the timeout, so try a script "
        "on a sample of dates/stocks first. Longer work runs as chunks that each finish in time, or as a "
        "detached job: start it with setsid, stdin, stdout and stderr redirected to "
        "workspace files, checkpoint there and read its log in later calls (a bounded "
        "`sleep` can wait for it); it runs past "
        "the timeout until the session container ends (a `vllm serve` on 127.0.0.1 "
        f"starts the same way). stdout and stderr over {max_output_chars} chars come back "
        "as an inline head plus `<stream>_tail`; the whole stream is spilled and "
        "`<stream>_spill.result_hint` gives the read_file call for the omitted lines."
    )


def _shell_example(timeout_seconds: float) -> dict[str, object]:
    return {"argv": ["python", "-c", "print(1)"], "cwd": ".", "timeout_seconds": timeout_seconds}


class SandboxShellTool:
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
        argv = arguments.get("argv")
        note = None
        if isinstance(argv, str):
            argv, note = _argv_from_string(argv)
            arguments = {**arguments, "argv": argv}
        elif (lone := _lone_command_line(argv)) is not None:
            argv, note = _argv_from_string(lone)
            if note == ARGV_SHELL_NOTE:
                note = ARGV_ONE_ELEMENT_NOTE
            arguments = {**arguments, "argv": argv}
        if isinstance(argv, list):
            _reject_long_argv_elements(argv)
        self._repair.note = note
        return arguments

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        raw_argv = arguments["argv"]
        if not isinstance(raw_argv, list) or not raw_argv:
            raise ToolError("argv must contain at least one argument")
        argv: Sequence[str] = tuple(str(item) for item in raw_argv)
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
        total_lines = len(file_lines(text))
        head_lines = head.count("\n")
        tail = text[-tail_chars:] if tail_chars and not capture_cut else ""
        if "\n" in tail and text[-tail_chars - 1] != "\n":
            # Drop the partial first line unless the window starts on a line.
            tail = tail[tail.find("\n") + 1 :]
        tail_lines = len(file_lines(tail))
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


def _stderr_suppression_reminder(tokens: Sequence[str]) -> str | None:
    """Advisory when the command redirects stderr away; never blocks the call.

    The argv contract keeps redirections out of the shell's own parsing, but an
    Agent can still reach one through ``bash -lc "... 2>/dev/null"``, so the
    whole command line is scanned.
    """
    return STDERR_SUPPRESSION_REMINDER if STDERR_SUPPRESSION_RE.search(" ".join(tokens)) else None


__all__ = [
    "ARGV_ONE_ELEMENT_NOTE",
    "ARGV_SHELL_NOTE",
    "ARGV_STRING_NOTE",
    "ARGV_TOO_LONG_HINT",
    "COMMAND_STRING_SHELL",
    "DEFAULT_SHELL_OUTPUT_CHARS",
    "DEFAULT_SHELL_TIMEOUT_SECONDS",
    "MAX_SHELL_TIMEOUT_SECONDS",
    "SHELL_ARGV_MAX_CHARS",
    "SHELL_CAPTURE_MAX_CHARS",
    "SandboxShellTool",
]
