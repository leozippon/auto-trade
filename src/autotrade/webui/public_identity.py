"""Single host boundary for public experiment identities.

The pipeline ledger and HITL files retain raw identities (run ids, artifact
ids, host paths). Every Web/API projection passes through
:class:`PublicIdentity`, which exposes durable experiment-scoped UUID4
references for runs, traces, strategies and session refs, and strips host
paths. Research sessions (``s1``..``sN``) and the ``forward`` replay name no
hidden period, so their plan keys are public as they are. Host operations
resolve the opaque references back; they never send the mapping table to the
Agent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.identity import (
    LEGACY_EXPERIMENT_MESSAGE,
    AgentRefStore,
    LegacyExperimentError,
)
from autotrade.pipelines.hitl_state import HITL_DIR_NAME, SCHEDULE_NAME, read_json

_PATH_KEYS = frozenset(
    {
        "agent_trace_ref",
        "result_ref",
        "run_manifest_ref",
        "validation_result_ref",
    }
)
_STRATEGY_ID_KEYS = frozenset(
    {
        "artifact_id",
        "revision_id",
        "final_strategy_artifact",
    }
)
# Opaque references a projection may carry through unchanged.
_PUBLIC_REF_PREFIXES = ("session_ref_", "run_ref_", "strategy_ref_", "trace_ref_")
_ALLOWED_PUBLIC_PATH_PREFIXES = (
    "/api",
    "/mnt/agent",
    "/mnt/artifacts",
    "/mnt/snapshot",
    "/mnt/snapshots",
    "/mnt/tools",
    # Container FHS roots: fixed by the sandbox image, identical on every
    # machine, so they carry no host identity. Keeping them readable preserves
    # traceback frames (`/usr/local/lib/python3.11/...`) and `2>/dev/null`
    # in the console. The host interpreter lives under a home directory and is
    # still redacted.
    "/dev",
    "/usr",
)
_FILE_URI = re.compile(r"file://(?:[^\s\"'`<>])+", re.IGNORECASE)
# A host path has at least two segments and an ASCII path body. Requiring the
# second segment keeps division and prose out (`(C-O)/C`, `(x-mean)/std`,
# `asof_dir + "/daily"`); restricting the body to ASCII path characters stops a
# trailing CJK run from being absorbed into the token, which would otherwise
# make an allow-listed root (`/mnt/snapshot。pandas`) fail the prefix test.
_POSIX_PATH = re.compile(
    r"(?<![A-Za-z0-9_/])/[A-Za-z._][A-Za-z0-9._+@%~-]*"
    r"(?:/[A-Za-z0-9._+@%~-]+)+"
)
_WINDOWS_PATH = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/]|\\\\)"
    r"[^\s\"'`<>|()\[\]{},;]+"
)

# The planned session kinds of a research arm (pipelines/hitl_state.py).
SESSION_KINDS = frozenset({"research", "forward"})


class PublicIdentity:
    """Validated experiment public projection and host resolver."""

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir).resolve()
        self._host_roots = tuple(
            sorted(
                {
                    str(self.experiment_dir),
                    str(self.experiment_dir.parent),
                    str(self.experiment_dir.parent.parent),
                },
                key=len,
                reverse=True,
            )
        )
        store = AgentRefStore.existing(self.experiment_dir)
        if store is None:
            raise LegacyExperimentError(LEGACY_EXPERIMENT_MESSAGE)
        self.store = store
        self._text_replacements: dict[str, str] = {}
        schedule = read_json(self.experiment_dir / HITL_DIR_NAME / SCHEDULE_NAME)
        raw_sessions = schedule.get("sessions")
        # The plan of record, in order; a Fold-era plan is refused here, so such
        # an experiment lists as unreadable instead of half-rendering.
        self.sessions: list[dict[str, object]] = []
        if raw_sessions is None:
            return
        if not isinstance(raw_sessions, list):
            raise ValueError("experiment session plan is invalid")
        for raw in raw_sessions:
            if not isinstance(raw, dict):
                raise ValueError("experiment session plan is invalid")
            key = str(raw.get("session_key") or "")
            if not key or str(raw.get("kind") or "") not in SESSION_KINDS:
                raise ValueError("experiment session plan contains an invalid session")
            if any(entry["session_key"] == key for entry in self.sessions):
                raise ValueError("experiment session plan contains duplicate identities")
            self.sessions.append(dict(raw))

    def run_ref(self, raw_run_id: object) -> str:
        return self._public_ref("run", raw_run_id, "run id")

    def strategy_ref(self, raw_strategy_id: object) -> str:
        return self._public_ref("strategy", raw_strategy_id, "strategy id")

    def trace_ref(self, raw_run_id: object) -> str:
        return self._public_ref("trace", raw_run_id, "trace id")

    def _public_ref(self, namespace: str, raw_value: object, label: str) -> str:
        raw = _required_text(raw_value, label)
        public = self.store.get_or_create(namespace, raw)
        self._text_replacements[raw] = public
        return public

    def raw_strategy_id(self, public_ref: str) -> str:
        raw = self.store.resolve("strategy", public_ref)
        self._text_replacements[raw] = public_ref
        return raw

    def raw_run_id(self, public_ref: str) -> str:
        if public_ref.startswith("run_ref_"):
            raw = self.store.resolve("run", public_ref)
        elif public_ref.startswith("trace_ref_"):
            raw = self.store.resolve("trace", public_ref)
        else:
            raise KeyError("unknown run or trace reference")
        self._text_replacements[raw] = public_ref
        return raw

    def session(self, key: object) -> dict[str, object]:
        """The planned session named ``key`` (an inbox suffix ``#…`` allowed)."""

        base, _suffix = _split_suffix(_required_text(key, "session key"))
        entry = next(
            (item for item in self.sessions if item["session_key"] == base), None
        )
        if entry is None:
            raise KeyError("unknown session key")
        return entry

    def public_session_key(self, raw_key: object) -> str:
        self.session(raw_key)
        return str(raw_key)

    def raw_session_key(self, public_key: object) -> str:
        self.session(public_key)
        return str(public_key)

    def public_status(self, status: Mapping[str, object]) -> dict[str, object]:
        out: dict[str, object] = {}
        raw_run = status.get("run_id")
        for key, value in status.items():
            if key in {"run_id", "fold_id"}:
                continue
            if key in _STRATEGY_ID_KEYS and isinstance(value, str) and value:
                out[_strategy_ref_key(key)] = self.strategy_ref(value)
                continue
            out[key] = self._safe_value(value)
        if isinstance(raw_run, str) and raw_run:
            out["run_ref"] = self.run_ref(raw_run)
            out["trace_ref"] = self.trace_ref(raw_run)
        return out

    def public_control(self, control: Mapping[str, object]) -> dict[str, object]:
        return {key: self._safe_value(value) for key, value in control.items()}

    def public_record(self, record: Mapping[str, object]) -> dict[str, object]:
        """One ledger row, trace event or Step node past the host boundary."""

        out = self._safe_value(record)
        assert isinstance(out, dict)
        raw_run = record.get("run_id")
        if isinstance(raw_run, str) and raw_run and not raw_run.startswith("run_ref_"):
            out["trace_ref"] = self.trace_ref(raw_run)
        return out

    def public_text(self, text: str) -> str:
        """Remove known host identities and roots from one public text field."""

        safe = self._safe_value(text)
        return safe if isinstance(safe, str) else ""

    def _safe_value(self, value: object) -> object | None:
        if isinstance(value, Mapping):
            out: dict[str, object] = {}
            for key, item in value.items():
                name = str(key)
                if (
                    name in _PATH_KEYS
                    or name.endswith(("_path", "_generation_id"))
                    or (
                        name.endswith("_ref")
                        and isinstance(item, str)
                        and not item.startswith(_PUBLIC_REF_PREFIXES)
                    )
                ):
                    continue
                if name == "run_id" and isinstance(item, str) and item:
                    out["run_ref"] = item if item.startswith("run_ref_") else self.run_ref(item)
                    continue
                if name == "fold_id" and isinstance(item, str) and item:
                    # A raw ledger ``fold_id`` is the public session key; the
                    # session refs the Agent sees ride under ``session_ref``.
                    out["session_key"] = item
                    continue
                if name in _STRATEGY_ID_KEYS and isinstance(item, str) and item:
                    out[_strategy_ref_key(name)] = (
                        item
                        if item.startswith("strategy_ref_")
                        else self.strategy_ref(item)
                    )
                    continue
                out[name] = self._safe_value(item)
            return out
        if isinstance(value, (list, tuple)):
            return [self._safe_value(item) for item in value]
        if isinstance(value, str):
            if _is_host_absolute_path(value):
                return "[host path omitted]"
            safe = value
            for raw, public in sorted(
                self._text_replacements.items(), key=lambda item: len(item[0]), reverse=True
            ):
                safe = safe.replace(raw, public)
            for root in self._host_roots:
                if root and root != "/":
                    safe = safe.replace(root, "[host]")
            return redact_host_paths(safe)
        return value


def redact_host_paths(text: str) -> str:
    """Scrub host paths out of text that belongs to no single experiment.

    :class:`PublicIdentity` adds the experiment's own roots and reference
    mappings on top; repository-level content (the curated memory library) has
    neither, so this is the whole rule for it — and the same rule, not a second
    one, is what every experiment-scoped string ends with."""

    safe = _FILE_URI.sub("[host path omitted]", text)
    safe = _WINDOWS_PATH.sub("[host path omitted]", safe)
    return _POSIX_PATH.sub(_redact_posix_path, safe)


def _strategy_ref_key(name: str) -> str:
    if name in {"artifact_id", "revision_id"}:
        return "strategy_ref"
    return "final_strategy_ref"


def _allowed_public_path(value: str) -> bool:
    return any(
        value == prefix or value.startswith(f"{prefix}/")
        for prefix in _ALLOWED_PUBLIC_PATH_PREFIXES
    )


def _is_host_absolute_path(value: str) -> bool:
    if _allowed_public_path(value):
        return False
    return Path(value).is_absolute() or bool(
        re.match(r"^(?:[A-Za-z]:[\\/]|\\\\)", value)
    )


def _redact_posix_path(match: re.Match[str]) -> str:
    value = match.group(0)
    return value if _allowed_public_path(value) else "[host path omitted]"


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _split_suffix(session_key: str) -> tuple[str, str]:
    base, separator, suffix = session_key.partition("#")
    return base, f"#{suffix}" if separator else ""
