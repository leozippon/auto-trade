"""Single host boundary for public experiment identities.

The pipeline ledger and HITL files retain raw schedule identities.  Every Web/API
projection passes through :class:`PublicIdentity`, which exposes durable
experiment-scoped UUID4 references for control and Agent-facing fields, plus
operator display labels that name a Fold by its schedule period (``2022Q1``)
without echoing the raw ``fold_2022Q1`` token.  Host operations resolve the
opaque references back; they never send the mapping table to the Agent.
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
from autotrade.pipelines.meta_schedule import meta_record_session_key

_PATH_KEYS = frozenset(
    {
        "agent_trace_ref",
        "analysis_path",
        "combined_artifact_ref",
        "frozen_model_artifact_path",
        "frozen_strategy_artifact_path",
        "model_artifact_ref",
        "result_ref",
        "run_manifest_ref",
        "strategy_artifact_ref",
        "strategy_dir",
        "test_result_ref",
        "validation_result_ref",
    }
)
_STRATEGY_ID_KEYS = frozenset(
    {
        "artifact_id",
        "frozen_strategy_artifact_id",
        "parent_artifact_id",
        "parent_strategy_artifact_id",
        "revision_id",
        "source_artifact_id",
        "strategy_artifact_id",
        "final_strategy_artifact",
    }
)
_ALLOWED_PUBLIC_PATH_PREFIXES = (
    "/api",
    "/mnt/agent",
    "/mnt/artifacts",
    "/mnt/snapshot",
    "/mnt/snapshots",
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


class PublicIdentity:
    """Validated modern-experiment public projection and host resolver."""

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
        self.sessions: list[dict[str, object]] = []
        self._raw_to_public: dict[str, str] = {}
        self._public_to_raw: dict[str, str] = {}
        if raw_sessions is None:
            return
        if not isinstance(raw_sessions, list):
            raise ValueError("experiment session plan is invalid")
        for raw in raw_sessions:
            if not isinstance(raw, dict):
                raise ValueError("experiment session plan is invalid")
            entry = dict(raw)
            raw_key = str(entry.get("session_key") or entry.get("key") or "")
            kind = str(entry.get("kind") or "")
            if kind == "meta_learning":
                kind = "meta"
            if not raw_key or kind not in {"fold", "meta", "heldout"}:
                raise ValueError("experiment session plan contains an invalid session")
            public_key = self._project_session_key(entry, raw_key, kind)
            if raw_key in self._raw_to_public or public_key in self._public_to_raw:
                raise ValueError("experiment session plan contains duplicate identities")
            self._raw_to_public[raw_key] = public_key
            self._public_to_raw[public_key] = raw_key
            entry["kind"] = kind
            entry["_raw_key"] = raw_key
            entry["_public_key"] = public_key
            self.sessions.append(entry)

    def fold_ref(self, raw_fold_id: object) -> str:
        return self._public_ref("fold", raw_fold_id, "fold id")

    def run_ref(self, raw_run_id: object) -> str:
        return self._public_ref("run", raw_run_id, "run id")

    def strategy_ref(self, raw_strategy_id: object) -> str:
        return self._public_ref("strategy", raw_strategy_id, "strategy id")

    def trace_ref(self, raw_run_id: object) -> str:
        return self._public_ref("trace", raw_run_id, "trace id")

    def meta_ref(self, raw_meta_id: object) -> str:
        return self._public_ref("meta", raw_meta_id, "meta id")

    def _public_ref(self, namespace: str, raw_value: object, label: str) -> str:
        raw = _required_text(raw_value, label)
        public = self.store.get_or_create(namespace, raw)
        self._text_replacements[raw] = public
        return public

    def raw_fold_id(self, epoch_id: str, fold_ref: str) -> str:
        raw = self.store.resolve("fold", fold_ref)
        expected = f"{epoch_id}/{raw}"
        if self._raw_to_public.get(expected) != f"{epoch_id}/{fold_ref}":
            raise KeyError("unknown fold reference for epoch")
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

    def public_session_key(self, raw_key: object) -> str:
        text = _required_text(raw_key, "session key")
        base, suffix = _split_suffix(text)
        public = self._raw_to_public.get(base)
        if public is None:
            raise KeyError("unknown session key")
        return public + suffix

    def session_display_key(self, raw_key: object) -> str:
        text = _required_text(raw_key, "session key")
        base, suffix = _split_suffix(text)
        raw = self._public_to_raw.get(base, base)
        if raw not in self._raw_to_public:
            raise KeyError("unknown session key")
        entry = next(item for item in self.sessions if item.get("_raw_key") == raw)
        projected = self.public_session(entry, heldout_revealed=True)
        display = str(projected.get("display_key") or projected.get("label") or "")
        if not display:
            raise KeyError("session has no display key")
        return display + suffix

    def raw_session_key(self, public_key: object) -> str:
        text = _required_text(public_key, "public session key")
        base, suffix = _split_suffix(text)
        raw = self._public_to_raw.get(base)
        if raw is None:
            raise KeyError("unknown public session key")
        return raw + suffix

    def public_session(
        self, entry: Mapping[str, object], *, heldout_revealed: bool
    ) -> dict[str, object]:
        kind = str(entry.get("kind") or "")
        if kind == "meta_learning":
            kind = "meta"
        raw_key = str(entry.get("_raw_key") or entry.get("session_key") or entry.get("key") or "")
        public_key = str(entry.get("_public_key") or self.public_session_key(raw_key))
        out: dict[str, object] = {
            "kind": "meta_learning" if kind == "meta" else kind,
            "key": public_key,
            "session_key": public_key,
        }
        epoch_id = entry.get("epoch_id")
        if epoch_id is not None:
            out["epoch_id"] = epoch_id
        if kind == "fold":
            raw_fold = str(entry.get("fold_id") or raw_key.partition("/")[2])
            period = schedule_period_label(raw_fold)
            out["fold_ref"] = self.fold_ref(raw_fold)
            out["label"] = period
            out["display_key"] = f"{epoch_id}/{period}" if epoch_id else period
        elif kind == "meta":
            out["meta_ref"] = self.meta_ref(raw_key)
            out["display_key"] = raw_key
        elif kind == "heldout":
            out["display_key"] = "heldout"
            if not heldout_revealed:
                out["hidden"] = True
        for key, value in entry.items():
            if key in {
                "_raw_key",
                "_public_key",
                "key",
                "session_key",
                "kind",
                "fold_id",
                "meta_learning_id",
                "periods",
                "test_period",
            }:
                continue
            out.setdefault(key, self._safe_value(value))
        if kind == "heldout" and heldout_revealed:
            periods = entry.get("periods")
            if isinstance(periods, list):
                out["periods"] = [self._safe_value(period) for period in periods]
        return out

    def public_status(self, status: Mapping[str, object]) -> dict[str, object]:
        out: dict[str, object] = {}
        raw_run = status.get("run_id")
        raw_fold = status.get("fold_id")
        if isinstance(raw_run, str) and raw_run:
            self.run_ref(raw_run)
        if isinstance(raw_fold, str) and raw_fold:
            if str(status.get("session_kind") or "") == "meta_learning":
                self.meta_ref(raw_fold)
            else:
                self.fold_ref(raw_fold)
        for key, value in status.items():
            if key in {"run_id", "fold_id", "session_key", "question_key"}:
                continue
            if key in _STRATEGY_ID_KEYS and isinstance(value, str) and value:
                out[_strategy_ref_key(key)] = self.strategy_ref(value)
                continue
            out[key] = self._safe_value(value)
        if isinstance(raw_run, str) and raw_run:
            out["run_ref"] = self.run_ref(raw_run)
            out["trace_ref"] = self.trace_ref(raw_run)
        if isinstance(raw_fold, str) and raw_fold:
            namespace = "meta" if str(status.get("session_kind") or "") == "meta_learning" else "fold"
            out[f"{namespace}_ref"] = (
                self.meta_ref(raw_fold) if namespace == "meta" else self.fold_ref(raw_fold)
            )
        session_key = status.get("session_key")
        if isinstance(session_key, str) and session_key:
            out["session_key"] = self.public_session_key(session_key)
            try:
                out["session_label"] = self.session_display_key(session_key)
            except (KeyError, ValueError):
                pass
        question_key = status.get("question_key")
        if isinstance(question_key, str) and question_key:
            out["question_key"] = self.public_session_key(question_key)
        return out

    def public_control(self, control: Mapping[str, object]) -> dict[str, object]:
        session_lists = {"approved_sessions"}
        session_maps = {
            "directives",
            "gpu_counts",
            "parent_overrides",
            "prompt_overrides",
            "rerun_sessions",
            "resource_overrides",
            "step_directives",
            "step_gate",
            "step_go",
            "user_replies",
        }
        out: dict[str, object] = {}
        for key, value in control.items():
            if key in session_lists and isinstance(value, (list, tuple)):
                out[key] = [self.public_session_key(item) for item in value]
            elif key in session_maps and isinstance(value, Mapping):
                out[key] = {
                    self.public_session_key(raw_key): self._safe_value(item)
                    for raw_key, item in value.items()
                }
            else:
                out[key] = self._safe_value(value)
        return out

    def public_record(
        self, record: Mapping[str, object], *, heldout_revealed: bool
    ) -> dict[str, object]:
        record_type = str(record.get("record_type") or "")
        if record_type == "heldout" and not heldout_revealed:
            return {
                "record_type": "heldout",
                "epoch_id": record.get("epoch_id"),
                "hidden": True,
            }
        out: dict[str, object] = {}
        raw_run = record.get("run_id")
        raw_fold = record.get("fold_id")
        if isinstance(raw_run, str) and raw_run and not raw_run.startswith("run_ref_"):
            self.run_ref(raw_run)
        if isinstance(raw_fold, str) and raw_fold:
            if record_type == "meta_learning":
                self.meta_ref(meta_record_session_key(record))
            elif not raw_fold.startswith(("fold_ref_", "meta_ref_")):
                self.fold_ref(raw_fold)
        for key, value in record.items():
            if (
                key in _PATH_KEYS
                or key.endswith("_path")
                or key.endswith("_generation_id")
                or (
                    key.endswith("_ref")
                    and isinstance(value, str)
                    and not value.startswith(("fold_ref_", "run_ref_", "strategy_ref_", "trace_ref_", "meta_ref_"))
                )
            ):
                continue
            if key in {"run_id", "fold_id", "session_key", "meta_learning_id"}:
                continue
            if key in _STRATEGY_ID_KEYS and isinstance(value, str) and value:
                out[_strategy_ref_key(key)] = self.strategy_ref(value)
                continue
            out[key] = self._safe_value(value)
        if isinstance(raw_fold, str) and raw_fold:
            if raw_fold.startswith("fold_ref_"):
                out["fold_ref"] = raw_fold
            elif raw_fold.startswith("meta_ref_"):
                out["meta_ref"] = raw_fold
            elif record_type == "meta_learning":
                raw_meta = meta_record_session_key(record)
                out["meta_ref"] = self.meta_ref(raw_meta)
            else:
                out["fold_ref"] = self.fold_ref(raw_fold)
        elif record_type == "meta_learning":
            raw_meta = meta_record_session_key(record)
            if raw_meta:
                out["meta_ref"] = self.meta_ref(raw_meta)
        if isinstance(raw_run, str) and raw_run:
            if raw_run.startswith("run_ref_"):
                out["run_ref"] = raw_run
            else:
                out["run_ref"] = self.run_ref(raw_run)
                out["trace_ref"] = self.trace_ref(raw_run)
        session_key = record.get("session_key")
        if isinstance(session_key, str) and session_key:
            out["session_key"] = self.public_session_key(session_key)
        return out

    def public_text(self, text: str) -> str:
        """Remove known host identities and roots from one public text field."""

        safe = self._safe_value(text)
        return safe if isinstance(safe, str) else ""

    def public_value(self, value: object) -> object | None:
        """Project one nested block that carries no record-level identities.

        The record/status/control projections above key off field names they
        know. A block the console republishes verbatim (a run manifest's
        mounted-memory record) has none of them and still must not carry host
        paths out, so it goes through the same generic projector."""

        return self._safe_value(value)

    def public_analysis_meta(self, payload: Mapping[str, object]) -> dict[str, object]:
        return {
            key: self._safe_value(value)
            for key, value in payload.items()
            if key not in {"fold_id", "run_id", "analysis_path"}
        }

    def _project_session_key(
        self, entry: Mapping[str, object], raw_key: str, kind: str
    ) -> str:
        if kind == "heldout":
            return "heldout"
        epoch_id = str(entry.get("epoch_id") or raw_key.partition("/")[0])
        if not epoch_id:
            raise ValueError("planned session has no epoch")
        if kind == "fold":
            raw_fold = str(entry.get("fold_id") or raw_key.partition("/")[2])
            return f"{epoch_id}/{self.fold_ref(raw_fold)}"
        return f"{epoch_id}/{self.meta_ref(raw_key)}"

    def _safe_value(self, value: object) -> object | None:
        if isinstance(value, Mapping):
            out: dict[str, object] = {}
            for key, item in value.items():
                name = str(key)
                if (
                    name in _PATH_KEYS
                    or name.endswith("_path")
                    or name.endswith("_generation_id")
                    or (
                        name.endswith("_ref")
                        and isinstance(item, str)
                        and not item.startswith(("fold_ref_", "run_ref_", "strategy_ref_", "trace_ref_", "meta_ref_"))
                    )
                ):
                    continue
                if name == "run_id" and isinstance(item, str) and item:
                    out["run_ref"] = item if item.startswith("run_ref_") else self.run_ref(item)
                    continue
                if name == "fold_id" and isinstance(item, str) and item:
                    out["fold_ref"] = item if item.startswith("fold_ref_") else self.fold_ref(item)
                    continue
                if name == "session_key" and isinstance(item, str) and item:
                    out["session_key"] = self.public_session_key(item)
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
        if isinstance(value, list):
            return [self._safe_value(item) for item in value]
        if isinstance(value, tuple):
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
    if name == "final_strategy_artifact":
        return "final_strategy_ref"
    return name.replace("_id", "_ref")


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


def schedule_period_label(raw_fold_id: object) -> str:
    """Operator-facing Fold period (``2022Q1``) from a raw ``fold_2022Q1`` id."""

    text = str(raw_fold_id or "").strip()
    if text.startswith("fold_"):
        return text[5:] or text
    return text


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _split_suffix(session_key: str) -> tuple[str, str]:
    base, separator, suffix = session_key.partition("#")
    return base, f"#{suffix}" if separator else ""
