"""Small provider-neutral tool protocol and strict dispatcher."""

from __future__ import annotations

import json
import traceback
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from autotrade.environment.time_budget import (
    SessionTimeBudgetAware,
    TimeBudgetBinding,
)
from autotrade.environment.runtime import redact_host_paths


class SessionInterrupt(Exception):
    """Control-flow signal (researcher stop at a gate): the session must abort.

    ``ToolRegistry.invoke`` converts every other exception a tool raises into
    an error observation so an action can never kill the fold — this class is
    the deliberate exception: it re-raises through the dispatch so the
    worker's session loop can honor the stop immediately."""


class ToolError(RuntimeError):
    """Explicit, agent-visible tool failure with a fixable reason.

    The structured fields ride back to the Agent through the dispatcher, so a
    failure says what kind of failure it was and what to do next instead of
    being a bare sentence the model has to parse."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "tool_error",
        reason: str | None = None,
        retry_hint: str | None = None,
        blocked_target: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.reason = reason
        self.retry_hint = retry_hint
        self.blocked_target = blocked_target
        self.details = details or {}

    def to_record(self) -> dict[str, object]:
        return {
            key: value
            for key, value in {
                "error_type": self.error_type,
                "reason": self.reason,
                "retry_hint": self.retry_hint,
                "blocked_target": self.blocked_target,
                "details": self.details or None,
            }.items()
            if value is not None
        }


class ToolSchemaError(ToolError):
    """Action payload failed the Runner-side tool schema."""

    def __init__(self, message: str, **kwargs: object) -> None:
        kwargs.setdefault("error_type", "schema_error")
        super().__init__(message, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    # The runner discarded output beyond its capture cap: the stream held
    # here is a prefix of what the command produced, not the whole of it.
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise TypeError("exit_code must be an integer")

    def to_record(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
        }


class ToolResultStore(Protocol):
    """Where an oversized result goes when it must leave the conversation
    (the search tools' spill store)."""

    def store_tool_result(
        self, *, tool: str, kind: str, content: str
    ) -> dict[str, object]: ...


class CommandRunner(Protocol):
    """Injected isolated command runner; implementations need not inherit it."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        timeout_seconds: float,
        max_output_chars: int,
        input_text: str | None = None,
    ) -> CommandResult: ...


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, object]
    mutating: bool = False
    # One correct call, echoed in every schema error for this tool so the
    # model can fix the shape in its next call instead of guessing.
    example: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.description:
            raise ValueError("tool name and description must be non-empty")
        schema = _json_object(self.input_schema, name="input_schema")
        if schema.get("type") != "object":
            raise ValueError("tool input_schema must describe an object")
        object.__setattr__(self, "input_schema", schema)
        if self.example is not None:
            object.__setattr__(self, "example", _json_object(self.example, name="example"))

    def provider_record(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.input_schema),
            },
        }


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    value: Mapping[str, object] = field(default_factory=dict)
    error: str = ""
    finish: bool = False

    def __post_init__(self) -> None:
        value = _json_object(self.value, name="tool result")
        object.__setattr__(self, "value", value)
        if self.ok and self.error:
            raise ValueError("successful tool result cannot contain an error")
        if not self.ok and not self.error:
            raise ValueError("failed tool result requires an error")
        if self.finish and not self.ok:
            raise ValueError("failed result cannot finish the session")

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {"ok": self.ok}
        if self.value:
            record["value"] = dict(self.value)
        if self.error:
            record["error"] = self.error
        return record


class Tool(Protocol):
    spec: ToolSpec

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult: ...


# Tools whose calls must run in order even though their spec is not mutating:
# they finish the session or wait for a human.
SEQUENTIAL_TOOL_NAMES = frozenset({"ask_user", "finish_fold", "finish_meta"})


def is_sequential_tool(spec: ToolSpec | None) -> bool:
    """Whether a call must run in order rather than concurrently with its batch.

    Every tool call in one assistant turn is dispatched concurrently unless
    the batch contains a sequential tool: a mutating tool (writes, edits,
    shell, skills, both backtests, rollback), a finish gate, a human wait, or
    an unregistered name (its rejection keeps the batch order).
    Then the whole batch runs in order.
    """

    return spec is None or spec.mutating or spec.name in SEQUENTIAL_TOOL_NAMES


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        self._finished = False
        self._finish_value: dict[str, object] | None = None
        for tool in tools:
            self.register(tool)

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def finish_value(self) -> Mapping[str, object] | None:
        return self._finish_value

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def spec(self, name: str) -> ToolSpec | None:
        tool = self._tools.get(name)
        return tool.spec if tool is not None else None

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def provider_tools(
        self, allowed_names: Collection[str] | None = None
    ) -> tuple[dict[str, object], ...]:
        if allowed_names is None:
            selected = self._tools.values()
        else:
            allowed = frozenset(allowed_names)
            missing = sorted(allowed.difference(self._tools))
            if missing:
                raise ValueError(f"tool view references unregistered tools: {missing}")
            selected = (
                tool for name, tool in self._tools.items() if name in allowed
            )
        return tuple(tool.spec.provider_record() for tool in selected)

    def time_budget_bindings(self) -> tuple[TimeBudgetBinding, ...]:
        """Return only tools that explicitly opt into the session budget contract."""

        return tuple(
            TimeBudgetBinding(f"tool:{tool.spec.name}", tool.session_time_budget)
            for tool in self._tools.values()
            if isinstance(tool, SessionTimeBudgetAware)
        )

    def result_store(self) -> ToolResultStore | None:
        """The registered search tools' spill store for oversized results, if
        any: the one place a result too large for the conversation goes, so
        other components (shell output, sub-agent reports) reuse the same
        read-back path."""

        for tool in self._tools.values():
            store = getattr(tool, "result_store", None)
            if store is not None:
                return store
        return None

    def invoke(
        self,
        name: str,
        arguments: Mapping[str, object],
        *,
        allowed_names: Collection[str] | None = None,
    ) -> ToolResult:
        if allowed_names is not None and name not in allowed_names:
            available = ", ".join(sorted(str(item) for item in allowed_names))
            return ToolResult(
                False,
                error=(
                    f"tool is unavailable in the current session phase: {name}; "
                    f"available now: {available}"
                ),
            )
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools))
            return ToolResult(
                False,
                error=f"unknown tool: {name}; tools in this session: {available}",
            )
        if self._finished and tool.spec.mutating:
            return ToolResult(False, error="the strategy workspace is locked after finish")
        try:
            # A tool may repair or reject its own call shape before the schema
            # runs (duck-typed like ``result_store``): a documented argument
            # alias has to be mapped to its canonical value before the enum the
            # model is shown rejects it, and a wrong shape the tool can name
            # precisely says so instead of listing the enum.
            normalize = getattr(tool, "normalize_arguments", None)
            if normalize is not None and isinstance(arguments, Mapping):
                arguments = normalize(arguments)
            validated = validate_arguments(tool.spec.input_schema, arguments)
            result = tool.invoke(validated)
        except ToolSchemaError as exc:
            # A shape error is self-correcting: the message carries one
            # correct call for this tool.
            message = str(exc)
            if tool.spec.example is not None:
                example = json.dumps(tool.spec.example, ensure_ascii=False)
                message = f"{message}; correct call example: {example}"
                if exc.retry_hint is None:
                    exc.retry_hint = f"correct call example: {example}"
            return ToolResult(False, value=exc.to_record(), error=message)
        except ToolError as exc:
            # Structured failure detail rides back with the message so the Agent
            # can act on the kind of failure, not just read a sentence.
            return ToolResult(False, value=exc.to_record(), error=str(exc))
        except (OSError, ValueError) as exc:
            return ToolResult(False, error=str(exc))
        except SessionInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - a tool bug fails the call, not the fold
            return ToolResult(
                False,
                value={
                    "error_type": "tool_exception",
                    "traceback": _traceback_tail(exc),
                },
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        if result.finish:
            self._finished = True
            self._finish_value = dict(result.value)
        return result


def _traceback_tail(exc: BaseException, limit: int = 4) -> str:
    """The innermost frames of a tool failure, bounded for an observation."""

    frames = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return redact_host_paths("".join(frames[-limit:])[-1_500:])


def validate_arguments(
    schema: Mapping[str, object], arguments: Mapping[str, object]
) -> dict[str, object]:
    if not isinstance(arguments, Mapping):
        raise ToolSchemaError("tool arguments must be an object")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise ToolSchemaError("invalid tool schema")
    unknown = sorted(set(arguments).difference(properties))
    if unknown and schema.get("additionalProperties", False) is False:
        raise ToolSchemaError(f"unknown argument(s): {unknown}")
    missing = sorted(str(name) for name in required if name not in arguments)
    if missing:
        raise ToolSchemaError(f"missing required argument(s): {missing}")
    normalized = _json_object(arguments, name="tool arguments")
    for name, value in normalized.items():
        field_schema = properties.get(name)
        if isinstance(field_schema, Mapping):
            normalized[name] = _validate_value(str(name), value, field_schema)
    return normalized


def _validate_value(name: str, value: object, schema: Mapping[str, object]) -> object:
    expected = schema.get("type")
    if (
        expected == "integer"
        and isinstance(value, float)
        and value.is_integer()
    ):
        # JSON has one number type; an integral float (5.0) is an integer.
        value = int(value)
    valid = {
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }.get(str(expected), False)
    if not valid:
        raise ToolSchemaError(f"{name} must be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ToolSchemaError(f"{name} must be one of {schema['enum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            raise ToolSchemaError(f"{name} is too short")
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            raise ToolSchemaError(f"{name} is too long")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < float(schema["minimum"]):
            raise ToolSchemaError(f"{name} is below its minimum")
        if "maximum" in schema and value > float(schema["maximum"]):
            raise ToolSchemaError(f"{name} is above its maximum")
    if isinstance(value, list) and isinstance(schema.get("items"), Mapping):
        for index, item in enumerate(value):
            value[index] = _validate_value(f"{name}[{index}]", item, schema["items"])  # type: ignore[arg-type]
    return value


def _json_object(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    try:
        normalized = json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-compatible") from exc
    if not isinstance(normalized, dict):
        raise TypeError(f"{name} must be a JSON object")
    return normalized


__all__ = [
    "SEQUENTIAL_TOOL_NAMES",
    "CommandResult",
    "CommandRunner",
    "SessionInterrupt",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolResultStore",
    "ToolSchemaError",
    "ToolSpec",
    "is_sequential_tool",
    "validate_arguments",
]
