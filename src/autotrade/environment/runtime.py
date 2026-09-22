"""Sandbox runtime files: paths, run_manifest.json, and agent_trace.jsonl.

Trusted logs are produced only by Runner / Execution Gateway / LLM Proxy /
simulated Broker code paths (docs/environment-design.md §4.1). Agent text
never replaces these records. The durable versioned-JSONL log primitive every
such record stream is written with also lives here, so the experiment ledger
and the issue-report log share one implementation of the format.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import threading
import uuid
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from autotrade.environment.identity import AgentRefStore

SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [redacted]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*)[^\s,;]+"), r"\1[redacted]"),
    # A left boundary and a realistic key length keep ordinary words out:
    # without them "risk-controls", "disk-usage" and a "…-task-…" skill name
    # came back as "risk-[redacted]" from every sanitized tool result. A
    # hyphen still counts as a boundary, so "resp-sk-<key>" stays redacted.
    (re.compile(r"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{16,}"), "sk-[redacted]"),
    (re.compile(r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{8,}"), "hf_[redacted]"),
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
# mounts under /mnt. Shared by the data summary, error summaries, tool
# tracebacks and the Agent-readable transcript.
#
# A match has to look like a path — a top-level directory this host keeps its
# files under, then at least one more segment of ordinary path characters.
# Redacting every ``/`` at a token boundary instead rewrote the transcript the
# Agent reads its own work back from: 464 substitutions in one audited arm,
# nearly all of them division, backquoted names, ``2>/dev/null`` and
# ``<domain>/part_0000.parquet``.


def _host_path_prefixes() -> tuple[str, ...]:
    """The top-level directories an absolute host path can start with.

    The home directories plus the top of this checkout: the data, snapshot,
    experiment and sandbox work roots all live inside the repository, so one
    prefix covers every host tree a session could name. ``/Data2`` is named
    outright because that is where the worker runs from, and the derived one
    keeps this correct for a checkout elsewhere. ``/mnt``, ``/tmp``, ``/opt``
    and ``/usr`` are the Agent's view of its own sandbox, never host paths, so
    they can never become a prefix.
    """

    tops = {"/Data2", "/home", "/root"}
    repo = Path(__file__).resolve().parents[3]
    if len(repo.parts) > 1:
        tops.add("/" + repo.parts[1])
    return tuple(sorted(tops - {"/mnt", "/tmp", "/opt", "/usr"}))


HOST_PATH_PREFIXES = _host_path_prefixes()
HOST_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.\-/])(?:"
    + "|".join(re.escape(prefix) for prefix in HOST_PATH_PREFIXES)
    + r")(?:/[A-Za-z0-9._\-+@~%]+)+"
)


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
# arrives. The run manifest also uses it to decide which structured (dict)
# summary values of a completed Validation are worth carrying at all.
AGENT_VISIBLE_BACKTEST_SUMMARY_KEYS = (
    "result_name",
    "mode",
    "span",
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
    # Overfitting tell: turnover cost can drive a forward loss while the
    # research metrics still look healthy, so every Validation carries it.
    "turnover",
    # Net-of-cost robustness and how few trades/names carried the gains: an
    # excess that dies at twice the modelled slippage, or a return one name
    # produced, must be as visible as the return itself.
    "cost_sensitivity",
    "pnl_concentration",
    "strategy_exit_fill_count",
    "trade_count",
    "decision_calls",
    "started_at",
    "finished_at",
    "replay_wall_seconds",
    "replayed_trade_days",
    "phase_seconds",
    # Container telemetry of the replay: peak memory against the limit in
    # force, per-fit seconds against the fit timeout, GPU headroom.
    "resources",
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
# The Agent-readable transcript beside the JSONL trace: one text block per
# event, every field clipped to the content preview size, rotated into parts
# under ``read_file``'s 10 MiB cap so every part stays readable.
TRANSCRIPT_PART_MAX_BYTES = 8 * 1024 * 1024
_TRANSCRIPT_TEXT_FIELDS = ("instruction", "content", "message", "summary", "task", "text", "error")
_TRANSCRIPT_JSON_FIELDS = ("arguments", "result", "value", "report", "compaction", "context_edit", "usage")
_TRANSCRIPT_HEADER_FIELDS = (
    ("call", "call_index"),
    ("tool", "tool"),
    ("id", "tool_call_id"),
    ("status", "status"),
    ("task", "task_id"),
    ("round", "round"),
)
# Fields the transcript leaves out: the system prompt is the Agent's own
# context already, and the event ids are host-side bookkeeping.
_TRANSCRIPT_OMITTED = frozenset({"system_prompt", "event_type", "ts", "event_id"})
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


def rmtree_keeping_file_modes(root: Path) -> None:
    """Remove a tree whose files may be hardlinks this caller does not own.

    Only the directories are made writable. Unlinking a child needs write
    permission on the directory holding it and never on the child itself,
    while a file's mode belongs to its inode: chmod'ing a payload file on the
    way out would also unfreeze every other link to it. PIT staging trees are
    hardlink copies of published views, so one such chmod turns every
    experiment's decision snapshot writable at once. Single source for the
    PIT cache's staging and slot removals; :func:`chmod_tree`'s tolerant
    per-path policy applies to the directories for the same reason it does
    there."""

    root = Path(root)
    if not root.exists():
        return
    for directory in _tree_directories(root):
        try:
            directory.chmod(0o755)
        except OSError:
            pass
    shutil.rmtree(root)


def _tree_directories(root: Path) -> Iterator[Path]:
    """``root`` and every real directory under it, parents before children."""

    yield root
    for directory, dirnames, _filenames in os.walk(root):
        base = Path(directory)
        for name in dirnames:
            child = base / name
            if not child.is_symlink():
                yield child


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


def write_json_atomic(path: Path, payload: object, *, sort_keys: bool = True) -> None:
    """Publish one JSON file atomically.

    ``sort_keys`` defaults to True so manifests stay diffable. Pass False for a
    file whose builder orders keys deliberately -- a long Agent-visible index
    read in chunks needs its identifying keys at the top of each entry, and
    alphabetical order buries them under whatever bulky block sorts first.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique temp name: concurrent writers must never share a temp file, or
    # interleaved chunks get os.replace'd into place.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        # allow_nan=False: a NaN in a run manifest is an upstream bug — fail here.
        tmp.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=sort_keys,
                default=str,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def append_versioned_jsonl(
    path: str | Path, record: Mapping[str, object], *, schema_version: int
) -> dict[str, object]:
    """Append one sanitized, version-stamped record line and return it.

    The stamps come after the spread so a caller-supplied ``schema_version`` or
    ``recorded_at`` can never override the log's own. flock + fsync: records
    regularly exceed the 8 KiB text buffer, so an unlocked write reaches the
    file in multiple chunks and a concurrent reader can see a torn final line;
    and a finished run's record must survive power loss (the writer treats the
    record as durable once this returns).
    """
    payload = {
        **sanitize_for_log(record),
        "schema_version": schema_version,
        "recorded_at": utc_now_iso(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return payload


def read_versioned_jsonl(
    path: str | Path, *, schema_version: int, label: str
) -> list[dict[str, object]]:
    """All records in append order; a foreign or newer format fails fast.

    The shared lock pairs with :func:`append_versioned_jsonl`'s exclusive lock
    so a live reader never observes a half-written line. External corruption
    (truncation, foreign writers) still fails fast in ``json.loads`` below.
    There is no legacy tolerance: a missing or unknown version means a foreign
    or newer format that older code must not silently misinterpret — migrate
    the file, don't guess. ``label`` names the record kind in those errors.
    """
    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        try:
            text = handle.read()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    for record in records:
        if not isinstance(record, dict):
            raise TypeError(f"{label} line is not a JSON object in {target}")
        version = record.get("schema_version")
        # type() check: JSON true/1.0/"1" must not pass as 1 (bool subclasses
        # int and floats compare equal), and "1" must not pass either.
        if type(version) is not int or version != schema_version:
            raise ValueError(
                f"{label} schema_version {version!r} != {schema_version} in "
                f"{target}; migrate the log before reading"
            )
    return records


def _agent_visible_manifest(
    data: dict[str, object], ref_store: AgentRefStore
) -> dict[str, object]:
    """Return the public manifest view mounted at /mnt/artifacts.

    The in-memory and host audit manifest keep host paths and raw identities
    for orchestration; the Agent-visible manifest carries allowlisted keys with
    every identity projected through the experiment reference store.
    """

    record = json.loads(json.dumps(sanitize_for_log(data), ensure_ascii=False, default=str))
    if not isinstance(record, dict):
        return {}
    public: dict[str, object] = {
        key: record[key]
        for key in (
            "experiment_id",
            "epoch_id",
            "run_id",
            "kind",
            "session",
            "runtime_env_ref",
            "data_summary_ref",
            "research",
            "schedule",
            "benchmark_index",
            "snapshot_config",
            "start",
            "arm",
            "modification_constraints",
            "acceptance_rules",
            "broker_profile",
            "nl_failure_policy",
            "attempt",
            "record_failed_attempts",
            "budgets",
            "finalize_before_deadline_seconds",
            "per_call_timeout_seconds",
            "sandbox_spec",
            "sandbox_runtime",
            "operating_memory",
            "skills",
            "exploration_directive",
            "created_at",
        )
        if key in record
    }
    if record.get("run_id"):
        public["run_id"] = ref_store.get_or_create("run", str(record["run_id"]))
    if record.get("fold_id"):
        # The host records link on the raw session id; the Agent reads its ref.
        public["session_ref"] = ref_store.get_or_create("session", str(record["fold_id"]))
    if isinstance(record.get("snapshots"), dict):
        public["snapshots"] = _agent_visible_snapshots(record["snapshots"])
    if isinstance(record.get("backtest_summaries"), list):
        public["backtest_summaries"] = [
            _agent_visible_backtest_summary(item)
            for item in record["backtest_summaries"]
            if isinstance(item, dict) and item.get("mode") == "valid"
        ]
    return public


def _agent_visible_snapshots(record: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"test_decision_input", "test_replay", "heldout_decision_input", "heldout_replay"}
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


def agent_transcript_dir(artifacts_root: str | Path) -> Path:
    """The Agent-readable transcripts of an experiment's research attempts.

    One text file per attempt, named by the run's opaque ref, appended by the
    same writer as the JSONL trace; the session's ``trace`` read root.
    """
    return Path(artifacts_root) / "transcripts"


def _transcript_target(directory: Path, run_ref: str) -> Path:
    """The transcript part to append to: a new part once the last one is full."""

    part = 1
    while True:
        name = f"{run_ref}.txt" if part == 1 else f"{run_ref}.part{part}.txt"
        candidate = directory / name
        try:
            size = candidate.stat().st_size
        except FileNotFoundError:
            return candidate
        if size < TRANSCRIPT_PART_MAX_BYTES:
            return candidate
        part += 1


def _clip_transcript_text(text: str) -> str:
    if len(text) <= TRACE_CONTENT_PREVIEW_CHARS:
        return text
    omitted = len(text) - TRACE_CONTENT_PREVIEW_CHARS
    return f"{text[:TRACE_CONTENT_PREVIEW_CHARS]}\n[... {omitted} more characters in the host trace]"


def render_transcript_block(record: Mapping[str, object], *, omit: Collection[str] = ()) -> str:
    """One event as the Agent reads it: a header line, then its fields.

    Text fields are printed verbatim, structured fields as indented JSON so
    ``read_file``'s line paging and ``grep -C`` land on individual values, and
    whatever is left travels on one ``meta`` line. Every field is clipped at
    the trace's content preview size, and the block is the Agent-visible
    boundary: absolute host paths (an exception text, a host-side result
    reference) are redacted here, while the host JSONL keeps them.
    """

    header = f"=== {record.get('ts', '')} {record.get('event_type', '')}"
    for label, key in _TRANSCRIPT_HEADER_FIELDS:
        value = record.get(key)
        if value not in (None, ""):
            header += f" {label}={value}"
    lines = [header]
    skipped = set(_TRANSCRIPT_OMITTED) | set(omit) | {key for _label, key in _TRANSCRIPT_HEADER_FIELDS}
    meta: dict[str, object] = {}
    for key, value in record.items():
        if key in skipped:
            continue
        if key in _TRANSCRIPT_TEXT_FIELDS and isinstance(value, str):
            text = _clip_transcript_text(value)
            lines.append(f"{key}:\n{text}" if "\n" in text else f"{key}: {text}")
        elif key in _TRANSCRIPT_JSON_FIELDS and value is not None:
            rendered = json.dumps(value, ensure_ascii=False, indent=1, default=str)
            lines.append(f"{key}:\n{_clip_transcript_text(rendered)}")
        else:
            meta[key] = value
    if meta:
        lines.append(
            "meta: "
            + _clip_transcript_text(
                json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str)
            )
        )
    return redact_host_paths("\n".join(lines)) + "\n\n"


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
    """Bounded, redacted JSONL event stream for one Agent session.

    The session cap bounds one record, not two: once the JSONL is full it
    writes the ``trace_limit_reached`` marker and stops, and the Agent-readable
    transcript ends on the same marker. A trace that stopped there no longer
    carries the attempt's own spend, so a resume refuses to seed from it
    (:func:`autotrade.pipelines.session_resume.resume_state`).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        ids: dict[str, str],
        transcript_dir: str | Path | None = None,
        max_bytes: int = TRACE_MAX_BYTES,
        max_event_bytes: int = TRACE_MAX_EVENT_BYTES,
    ) -> None:
        if max_bytes <= 0 or max_event_bytes <= 0 or max_event_bytes > max_bytes:
            raise ValueError("trace size limits are invalid")
        self.path = Path(path)
        self.ids = dict(ids)
        # Where the Agent-readable transcript of this run goes, if anywhere;
        # its file is named by the run's opaque ref carried in ``ids``.
        self.transcript_dir = Path(transcript_dir) if transcript_dir is not None else None
        if self.transcript_dir is not None and not self.ids.get("run_id"):
            raise ValueError("a transcript needs the run's opaque ref in ids['run_id']")
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
                    # The transcript ends on the same marker: one bounded
                    # record of the attempt, not a host half and an Agent half
                    # that stop at different events.
                    self._append_transcript(marker)
                self._full = True
                return record
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self.path.chmod(0o600)
            self._append_transcript(record)
        return record

    def _append_transcript(self, record: dict[str, object]) -> None:
        if self.transcript_dir is None:
            return
        # Private like the JSONL beside it: the session reads it host-side
        # through the ``trace`` root, nothing else needs to.
        self.transcript_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.transcript_dir.chmod(0o700)
        target = _transcript_target(self.transcript_dir, str(self.ids["run_id"]))
        block = render_transcript_block(record, omit=self.ids)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(block)
        target.chmod(0o600)

