"""Sandbox runtime files: paths, run_manifest.json, and agent_trace.jsonl.

Trusted logs are produced only by Runner / Execution Gateway / LLM Proxy /
simulated Broker code paths (docs/environment-design.md §4.1). Agent text
never replaces these records.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from autotrade.environment.identity import AgentRefStore

SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [redacted]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*)[^\s,;]+"), r"\1[redacted]"),
    (re.compile(r"sk-[A-Za-z0-9_-]{8,}"), "sk-[redacted]"),
    (re.compile(r"hf_[A-Za-z0-9]{8,}"), "hf_[redacted]"),
    (re.compile(r"vless:" + r"//[^\s'\"<>]+"), "vless:" + "//[redacted]"),
    (
        re.compile(r"\b((?:https?|socks5h?|socks4)://)[^/\s'\"<>:@]+:[^@\s'\"<>]+@"),
        r"\1[redacted]@",
    ),
)
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "token",
    "secret",
    "password",
    "hf_token",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
    "proxy",
    "proxy_url",
}

# Everything the Agent session actually writes under ``/mnt/artifacts``. The
# AgentTrace is not here: it is written straight to the experiment's
# ``artifacts/traces/<run_id>.jsonl`` (see :func:`agent_trace_path`), never
# into the sandbox.
ARTIFACT_TOP_LEVEL = (
    "run_manifest.json",
    "runtime_env.json",
    "data_summary.json",
    "unit_reference.json",
    "parent_output",
    "parent_models",
    "steps",
    "logs",
)
# Absolute host paths must never reach a model: the Agent sees only sandbox
# mounts under /mnt. Shared by the data summary, error summaries and tool
# tracebacks.
HOST_PATH_RE = re.compile(r"(?<![\w.:\-/])/(?!mnt/)(?:[^\s'\";|,)]*)")


def redact_host_paths(text: str) -> str:
    return HOST_PATH_RE.sub("[host_path]", text)
# Python bytecode-cache dirs/suffixes that are never experiment artifacts. Single
# source for both the artifact-collection ignore list (sandbox._COLLECT_IGNORE, which
# adds VCS/venv/tooling dirs on top) and the formal-file runtime-cache predicate
# (artifacts._is_runtime_cache).
RUNTIME_CACHE_DIR_NAMES = ("__pycache__",)
RUNTIME_CACHE_SUFFIXES = (".pyc", ".pyo")
# The Agent-visible projection of one ``backtest_summaries`` entry. Every key
# here is produced end to end: the tool layer writes the record identity and
# outcome, ``compute_return_stats`` writes the return/order/timing block, and
# ``NLService.counters`` writes the ``nl_*`` block. A key that nothing populates
# does not belong in this tuple — it would advertise telemetry that never
# arrives. The Fold tool layer also uses it to decide which structured (dict)
# summary values are worth carrying into the run manifest at all.
AGENT_VISIBLE_BACKTEST_SUMMARY_KEYS = (
    "result_name",
    "mode",
    "status",
    "complete_validation",
    "error",
    "total_return",
    "long_return",
    "sharpe",
    "max_drawdown",
    "order_count",
    "order_lifecycle",
    "reject_counts",
    "benchmark",
    "strategy_exit_fill_count",
    "trade_count",
    "decision_calls",
    "started_at",
    "finished_at",
    "replay_wall_seconds",
    "replayed_trade_days",
    "phase_seconds",
    "nl_calls",
    "nl_executed_calls",
    "nl_search_calls",
    "nl_llm_calls",
    "nl_event_filter_calls",
    "nl_evidence_gated_calls",
    "nl_budget_rejected_calls",
    "nl_wall_seconds",
    "nl_max_total_calls",
)
TRACE_MAX_BYTES = 32 * 1024 * 1024
TRACE_MAX_EVENT_BYTES = 256 * 1024
# What an over-sized event keeps. Identity plus these named fields survive
# verbatim; ``content`` (a model reply) keeps its own preview because the trace
# views read it back, and everything else — a ``tool_call``'s ``arguments`` and
# ``result`` above all — survives as one clipped JSON head, so an event with no
# ``content`` field still says what it did.
_TRACE_STUB_KEYS = ("status", "call_index", "tool", "tool_names")
TRACE_CONTENT_PREVIEW_CHARS = 32_000
TRACE_PAYLOAD_HEAD_CHARS = 8_000
_TRACE_LOCKS: dict[str, threading.Lock] = {}
_TRACE_LOCKS_GUARD = threading.Lock()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class SandboxPaths:
    """Resolved sandbox mount points.

    In Docker these are the fixed /mnt/... paths; the local driver maps them
    under a host directory with the same relative layout.
    """

    root: Path

    @property
    def snapshots(self) -> Path:
        return self.root / "snapshots"

    @property
    def train(self) -> Path:
        return self.snapshots / "train"

    @property
    def valid(self) -> Path:
        return self.snapshots / "valid"

    @property
    def test(self) -> Path:
        return self.snapshots / "test"

    @property
    def snapshot(self) -> Path:
        """Development-visible decision-input mirror exposed as /mnt/snapshot."""
        return self.root / "snapshot"

    @property
    def runtime(self) -> Path:
        """Host-only runtime scratch root; never mounted to the Agent."""
        return self.root / "runtime"

    @property
    def snapshot_views(self) -> Path:
        return self.runtime / "snapshot_views"

    @property
    def current_snapshot(self) -> Path:
        """Host-side current decision-input mirror mounted as /mnt/snapshot."""
        return self.runtime / "current_snapshot"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def agent(self) -> Path:
        """Agent-writable mount root."""
        return self.root / "agent"

    @property
    def run_manifest(self) -> Path:
        return self.artifacts / "run_manifest.json"

    @property
    def host_run_manifest(self) -> Path:
        """Host-only full manifest used for audit; never mounted to Agent."""
        return self.runtime / "host_run_manifest.json"

    @property
    def runtime_env(self) -> Path:
        return self.artifacts / "runtime_env.json"

    @property
    def data_summary(self) -> Path:
        return self.artifacts / "data_summary.json"

    @property
    def parent_output(self) -> Path:
        return self.artifacts / "parent_output"

    @property
    def parent_model_artifacts(self) -> Path:
        return self.artifacts / "parent_models"

    # There is no sandbox ``results`` directory: a full Validation record is
    # written host-side to the experiment's ``artifacts/results/`` and reaches
    # the Agent as the ``validation/result.json`` attachment inside the Step
    # node it belongs to (search root ``steps``).

    @property
    def steps(self) -> Path:
        """Step artifact tree (lineage of validated Step artifacts)."""
        return self.artifacts / "steps"

    @property
    def logs(self) -> Path:
        return self.artifacts / "logs"

    @property
    def workspace(self) -> Path:
        return self.agent / "workspace"

    # The formal working copies live INSIDE the workspace: ``workspace/output``
    # and ``workspace/models`` (the session backend creates them and the search
    # roots ``output``/``models`` resolve there). There are no sibling
    # ``agent/output`` or ``agent/models`` directories.


def sanitize_for_log(value: object) -> object:
    """Drop sensitive keys and redact secret-looking strings recursively."""
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if str(key).lower() in SENSITIVE_KEYS else sanitize_for_log(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_log(item) for item in value]
    if isinstance(value, str):
        text = value
        for pattern, replacement in SECRET_PATTERNS:
            text = pattern.sub(replacement, text)
        return text
    return value


def chmod_tree(root: Path, *, file_mode: int, dir_mode: int) -> None:
    """Recursive chmod with the tolerant per-path policy every lock/unlock
    site needs: under rootless Docker the agent's subuid may own files the
    host cannot chmod (EPERM) — skipping them beats crashing a freeze or a
    parent-restore mid-flight. Single source: the sandbox lock/unlock pair,
    the formal-replay readonly bracket, and the pipeline restore all share it."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            path.chmod(dir_mode if path.is_dir() else file_mode)
        except OSError:
            pass
    try:
        root.chmod(dir_mode if root.is_dir() else file_mode)
    except OSError:
        pass


@dataclass
class RunManifest:
    """Per-run manifest with an Agent-visible public view and host audit view."""

    path: Path
    data: dict[str, object] = field(default_factory=dict)
    host_path: Path | None = None
    ref_store: AgentRefStore | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def create(
        cls,
        path: str | Path,
        initial: dict[str, object],
        *,
        ref_store: AgentRefStore,
    ) -> "RunManifest":
        path = Path(path)
        manifest = cls(
            path=path,
            host_path=_default_host_manifest_path(path),
            data=dict(initial),
            ref_store=ref_store,
        )
        manifest.data.setdefault("created_at", utc_now_iso())
        manifest.data.setdefault("backtest_summaries", [])
        manifest.save()
        return manifest

    def save(self) -> None:
        if self.host_path is not None:
            write_json_atomic(self.host_path, sanitize_for_log(self.data))
        if self.ref_store is None:
            raise RuntimeError("RunManifest requires an experiment AgentRefStore")
        write_json_atomic(self.path, _agent_visible_manifest(self.data, self.ref_store))

    def update(self, **fields: object) -> None:
        with self._lock:
            self.data.update(fields)
            self.save()

    def append_backtest_summary(self, summary: dict[str, object]) -> None:
        with self._lock:
            raw_summaries = self.data.get("backtest_summaries")
            summaries = list(raw_summaries) if isinstance(raw_summaries, list) else []
            summaries.append(summary)
            self.data["backtest_summaries"] = summaries
            self.save()

    def get(self, key: str, default: object = None) -> object:
        return self.data.get(key, default)

    def require(self, key: str) -> object:
        if key not in self.data:
            raise KeyError(f"run manifest missing required key: {key}")
        return self.data[key]


def _default_host_manifest_path(public_path: Path) -> Path:
    if public_path.parent.name == "artifacts":
        return public_path.parent.parent / "runtime" / "host_run_manifest.json"
    return public_path.with_name("host_run_manifest.json")


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name: concurrent writers must never share a temp file, or
    # interleaved chunks get os.replace'd into place.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        # allow_nan=False: a NaN in a run manifest is an upstream bug — fail here.
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str, allow_nan=False),
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _agent_visible_manifest(
    data: dict[str, object], ref_store: AgentRefStore
) -> dict[str, object]:
    """Return the public manifest view mounted at /mnt/artifacts.

    The in-memory and host audit manifest keep the full schedule and frozen
    Test details for orchestration. Agent-visible manifests carry no raw Test
    schedule or result; Meta receives separately whitelisted historical metrics
    from completed Folds through its workspace projection.
    """

    record = json.loads(json.dumps(sanitize_for_log(data), ensure_ascii=False, default=str))
    if not isinstance(record, dict):
        return {}
    public: dict[str, object] = {
        key: record[key]
        for key in (
            "experiment_id",
            "epoch_id",
            "meta_learning_id",
            "trigger_after_folds",
            "run_id",
            "kind",
            "runtime_env_ref",
            "data_summary_ref",
            "fold_period",
            "schedule",
            "snapshot_config",
            "valid_decision_time",
            "is_initial_artifact",
            "parent_strategy_artifact_id",
            "template_ref",
            "modification_constraints",
            "acceptance_rules",
            "broker_profile",
            "nl_failure_policy",
            "step_tree_enabled",
            "record_failed_attempts",
            "epoch_index",
            "phase",
            "budgets",
            "max_steps",
            "max_backtests_per_fold",
            "deadline_seconds",
            "fold_deadline_at",
            "finalize_before_deadline_seconds",
            "per_call_timeout_seconds",
            "sandbox_spec",
            "sandbox_runtime",
            "prior_prompt",
            "operating_memory",
            "skills",
            "development_inputs",
            "prior_output",
            "meta_learning_directive",
            "fold_exploration_directive",
            "review_window",
            "created_at",
        )
        if key in record
    }
    if isinstance(record.get("sandbox_image_update"), dict):
        public["sandbox_image_update"] = _agent_visible_sandbox_image_update(
            record["sandbox_image_update"]
        )
    kind = str(record.get("kind") or "fold")
    if record.get("run_id"):
        public["run_id"] = ref_store.get_or_create("run", str(record["run_id"]))
    if record.get("meta_learning_id"):
        public["meta_learning_id"] = ref_store.get_or_create(
            "meta", str(record["meta_learning_id"])
        )
    if record.get("fold_id"):
        namespace = "meta" if kind == "meta_learning" else "fold"
        public["fold_id"] = ref_store.get_or_create(namespace, str(record["fold_id"]))
    # Artifact ids embed the raw fold label (strategy_<epoch>_fold_<period>), so they
    # must be projected exactly like the ledger view does.
    if public.get("parent_strategy_artifact_id"):
        public["parent_strategy_artifact_id"] = ref_store.get_or_create(
            "strategy", str(public["parent_strategy_artifact_id"])
        )
    if isinstance(record.get("fold"), dict):
        public["fold"] = _agent_visible_fold_record(record["fold"], ref_store)
    if isinstance(record.get("meta_learning_visible_fold"), dict):
        public["meta_learning_visible_fold"] = _agent_visible_fold_record(
            record["meta_learning_visible_fold"], ref_store
        )
    if isinstance(record.get("snapshots"), dict):
        public["snapshots"] = _agent_visible_snapshots(record["snapshots"])
    if isinstance(record.get("experiment_parameters"), dict):
        public["experiment_parameters"] = _agent_visible_experiment_parameters(
            record["experiment_parameters"]
        )
    if isinstance(record.get("backtest_summaries"), list):
        public["backtest_summaries"] = [
            _agent_visible_backtest_summary(item)
            for item in record["backtest_summaries"]
            if isinstance(item, dict) and item.get("mode") == "valid"
        ]
    return public


def _agent_visible_sandbox_image_update(record: dict[str, object]) -> dict[str, object]:
    """Keep rebuild outcome facts while withholding host build coordinates."""
    return {
        key: record[key]
        for key in (
            "status",
            "reason",
            "request_ref",
            "base_image_ref",
            "image_ref",
            "build_generation_id",
            "runtime",
            "pruned_image_refs",
            "started_at",
            "finished_at",
            "timeout_seconds",
            "returncode",
        )
        if key in record
    }


def _agent_visible_fold_record(
    record: dict[str, object], ref_store: AgentRefStore
) -> dict[str, object]:
    public = {
        key: record[key]
        for key in ("input_window", "validation_period", "valid_decision_time")
        if key in record
    }
    if record.get("fold_id"):
        public["fold_id"] = ref_store.get_or_create(
            "fold", str(record["fold_id"])
        )
    return public


def _agent_visible_snapshots(record: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"test_decision_input", "test_replay", "heldout_decision_input", "heldout_replay"}
        and not str(key).startswith("test_")
        and not str(key).startswith("heldout_")
    }


def _agent_visible_experiment_parameters(record: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if key != "periods"
        and not str(key).startswith("test_")
        and not str(key).startswith("heldout_")
    }


def _agent_visible_backtest_summary(record: dict[str, object]) -> dict[str, object]:
    return {
        key: record[key]
        for key in AGENT_VISIBLE_BACKTEST_SUMMARY_KEYS
        if key in record
    }


def agent_trace_path(artifacts_root: str | Path, run_id: str) -> Path:
    """Host-side per-run Agent trace file under an experiment's artifacts root.

    Single source for the writer and for the ledger's ``agent_trace_ref``, so a
    recorded reference always names the file the session actually wrote.
    """
    return Path(artifacts_root) / "traces" / f"{run_id}.jsonl"


def _trace_payload_head(payload: dict[str, object]) -> str | None:
    """Clipped JSON of the payload fields the truncation stub does not keep."""

    body = {
        key: value
        for key, value in payload.items()
        if key not in _TRACE_STUB_KEYS and key not in ("content", "error")
    }
    if not body:
        return None
    head = json.dumps(body, ensure_ascii=False, sort_keys=True, default=str)
    return head[:TRACE_PAYLOAD_HEAD_CHARS]


class AgentTraceWriter:
    """Bounded, redacted JSONL event stream for one Agent session."""

    def __init__(
        self,
        path: str | Path,
        *,
        ids: dict[str, str],
        max_bytes: int = TRACE_MAX_BYTES,
        max_event_bytes: int = TRACE_MAX_EVENT_BYTES,
    ) -> None:
        if max_bytes <= 0 or max_event_bytes <= 0 or max_event_bytes > max_bytes:
            raise ValueError("trace size limits are invalid")
        self.path = Path(path)
        self.ids = dict(ids)
        self.max_bytes = max_bytes
        self.max_event_bytes = max_event_bytes
        key = str(self.path.resolve())
        with _TRACE_LOCKS_GUARD:
            self._lock = _TRACE_LOCKS.setdefault(key, threading.Lock())
        self._full = False

    def emit(self, event_type: str, payload: dict[str, object]) -> dict[str, object]:
        sanitized = sanitize_for_log(payload)
        if not isinstance(sanitized, dict):
            raise TypeError("trace payload must sanitize to an object")
        record = {
            **sanitized,
            "event_type": str(event_type),
            "ts": utc_now_iso(),
            "event_id": new_id("event"),
            **self.ids,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)
        encoded = (line + "\n").encode("utf-8")
        if len(encoded) > self.max_event_bytes:
            record = {
                "event_type": str(event_type),
                "ts": record["ts"],
                "event_id": record["event_id"],
                **self.ids,
                "truncated": True,
                "original_bytes": len(encoded),
                "status": record.get("status"),
                "call_index": record.get("call_index"),
                "tool": record.get("tool"),
                "tool_names": record.get("tool_names"),
                "content_preview": str(record.get("content") or "")[:TRACE_CONTENT_PREVIEW_CHARS],
                "payload_head": _trace_payload_head(sanitized),
                "error": str(record.get("error") or "")[:4_000] or None,
            }
            encoded = (
                json.dumps(record, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)
                + "\n"
            ).encode("utf-8")
            if len(encoded) > self.max_event_bytes:
                record.pop("content_preview", None)
                record.pop("payload_head", None)
                record.pop("error", None)
                encoded = (
                    json.dumps(record, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)
                    + "\n"
                ).encode("utf-8")
            if len(encoded) > self.max_event_bytes:
                raise ValueError("trace identifiers exceed the per-event size limit")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.path.parent.chmod(0o700)
            current = self.path.stat().st_size if self.path.exists() else 0
            if self._full or current + len(encoded) > self.max_bytes:
                if not self._full:
                    marker = {
                        "event_type": "trace_limit_reached",
                        "ts": utc_now_iso(),
                        "event_id": new_id("event"),
                        **self.ids,
                        "max_bytes": self.max_bytes,
                    }
                    raw = (json.dumps(marker, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                    if current + len(raw) <= self.max_bytes:
                        with self.path.open("ab") as handle:
                            handle.write(raw)
                            handle.flush()
                            os.fsync(handle.fileno())
                self._full = True
                return record
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self.path.chmod(0o600)
        return record

