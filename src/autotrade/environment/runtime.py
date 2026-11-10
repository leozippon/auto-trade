"""Sandbox runtime files: paths and run_manifest.json (the AgentTrace is
``environment.trace``).

Trusted logs are produced only by Runner / Execution Gateway / LLM Proxy /
simulated Broker code paths (docs/ENVIRONMENT_DESIGN.md §4.1). Agent text
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
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from autotrade.common.records import write_json_atomic
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
# Credential keys ``is_sensitive_key`` does not already match by shape (an
# ``api``+``key``, ``secret`` or ``password`` part, a ``token`` ending),
# normalized as it reads them; a proxy URL can embed its login. The one list
# for traces, manifests, ledgers and provider conversation logs alike.
SENSITIVE_KEYS = frozenset(
    {
        "apikey",
        "authorization",
        "access_key",
        "private_key",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "proxy",
        "proxy_url",
    }
)

# Everything the Agent session actually writes under ``/mnt/artifacts``. The
# AgentTrace is not here: it is written straight to the experiment's
# ``artifacts/traces/<run_id>.jsonl`` (see ``trace.agent_trace_path``), never
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
# Python bytecode-cache dirs/suffixes that are never experiment artifacts. Single
# source for both the artifact-collection ignore list (sandbox._COLLECT_IGNORE, which
# adds VCS/venv/tooling dirs on top) and the formal-file runtime-cache predicate
# (artifacts._is_runtime_cache).
RUNTIME_CACHE_DIR_NAMES = ("__pycache__",)
RUNTIME_CACHE_SUFFIXES = (".pyc", ".pyo")
# The Agent-visible projection of one ``backtest_summaries`` entry. Every key
# here is produced end to end: the tool layer writes the record identity and
# outcome and ``compute_return_stats`` writes the return/order/timing block. A
# key that nothing populates does not belong in this tuple — it would advertise
# telemetry that never arrives. The run manifest also uses it to decide which
# structured (dict) summary values of a completed Validation are worth carrying
# at all.
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
    # force, per-fit seconds against the fit timeout, the seconds of each
    # generate_orders call, GPU headroom.
    "resources",
)


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


def is_sensitive_key(key: str) -> bool:
    """Whether a mapping key names a credential.

    The key is lower-cased with every run of other characters folded to one
    underscore, so ``X-Api-Key`` reads as ``x_api_key``. Besides the listed
    names, a key with both ``api`` and ``key`` parts, a ``secret`` or
    ``password`` part, or ending in ``token`` (``refresh_token``,
    ``x-auth-token``) is one. A token count or setting is not: traces carry
    ``token_threshold`` and ``token_usage`` for the console, and responses
    ``total_tokens``.
    """

    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    parts = set(normalized.split("_"))
    return (
        normalized in SENSITIVE_KEYS
        or {"api", "key"} <= parts
        or bool(parts & {"secret", "password"})
        or normalized.endswith("token")
    )


def sanitize_for_log(value: object) -> object:
    """Drop sensitive keys and redact secret-looking strings recursively."""
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if is_sensitive_key(str(key)) else sanitize_for_log(item)
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


def link_or_copy(source: str | Path, dest: str | Path) -> None:
    """Hard-link ``source`` at ``dest``, replacing a file already there; copy it
    with its metadata when no link can be made (another filesystem, the link
    limit). Also ``shutil.copytree``'s ``copy_function`` for a linked tree."""

    Path(dest).unlink(missing_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


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
    host_path: Path
    ref_store: AgentRefStore = field(repr=False)
    data: dict[str, object] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def create(
        cls,
        path: str | Path,
        initial: dict[str, object],
        *,
        ref_store: AgentRefStore,
    ) -> RunManifest:
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
        write_json_atomic(self.host_path, sanitize_for_log(self.data))
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


def _default_host_manifest_path(public_path: Path) -> Path:
    if public_path.parent.name == "artifacts":
        return public_path.parent.parent / "runtime" / "host_run_manifest.json"
    return public_path.with_name("host_run_manifest.json")


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
            "experiment",
            "modification_constraints",
            "acceptance_rules",
            "broker_profile",
            "attempt",
            "record_failed_attempts",
            "budgets",
            "per_call_timeout_seconds",
            "sandbox_spec",
            "sandbox_runtime",
            "skill_library",
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
        if not str(key).startswith(("test_", "heldout_"))
    }


def _agent_visible_backtest_summary(record: dict[str, object]) -> dict[str, object]:
    return {
        key: record[key]
        for key in AGENT_VISIBLE_BACKTEST_SUMMARY_KEYS
        if key in record
    }
