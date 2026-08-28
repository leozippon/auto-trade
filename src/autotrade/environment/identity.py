"""Host-only persistent Agent-visible opaque identifiers.

Agent references are random UUID4 values scoped to one experiment and one fixed
namespace.  The raw-to-reference table lives only under the experiment's
private ``.host`` directory; Agent-facing projections receive references, never
the table or its sources.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping

AGENT_REF_SCHEMA_VERSION = 1
AGENT_REF_PREFIXES: Mapping[str, str] = {
    "fold": "fold_ref",
    "strategy": "strategy_ref",
    "run": "run_ref",
    "trace": "trace_ref",
    "meta": "meta_ref",
}
LEGACY_EXPERIMENT_MESSAGE = "legacy experiment is read-only; start a new experiment"


class AgentRefStoreError(ValueError):
    """The host-only mapping is missing, corrupt, or inconsistent."""


class LegacyExperimentError(AgentRefStoreError):
    """A pre-random-ref experiment may be audited but never resumed or changed."""


class AgentRefStore:
    """Persistent per-experiment UUID4 mapping for Agent-visible references."""

    def __init__(self, experiment_dir: str | Path, *, initialize: bool = True) -> None:
        self.experiment_dir = Path(experiment_dir).resolve()
        self.host_dir = self.experiment_dir / ".host"
        self.path = self.host_dir / "agent-refs.json"
        self.lock_path = self.host_dir / "agent-refs.lock"
        self.experiment = self.experiment_dir.name
        if not self.experiment:
            raise AgentRefStoreError("experiment directory must have a name")
        if initialize:
            self.initialize()

    @classmethod
    def existing(cls, experiment_dir: str | Path) -> "AgentRefStore | None":
        """Return an existing store without creating one (trusted read paths)."""
        store = cls(experiment_dir, initialize=False)
        if not store.path.is_file():
            return None
        # Validate immediately so even read-only control-plane callers fail on
        # a corrupt store instead of treating it as absent.
        with store._locked(shared=True, create=False):
            store._read_unlocked()
        return store

    def initialize(self) -> None:
        """Create an empty mapping for a new experiment or validate an existing one."""
        if not self.path.exists() and _has_legacy_identity_artifacts(
            self.experiment_dir
        ):
            raise LegacyExperimentError(LEGACY_EXPERIMENT_MESSAGE)
        with self._locked():
            if self.path.exists():
                self._read_unlocked()
                return
            if _has_legacy_identity_artifacts(self.experiment_dir):
                raise LegacyExperimentError(LEGACY_EXPERIMENT_MESSAGE)
            self._write_unlocked(self._empty_payload())

    def get_or_create(self, namespace: str, raw_source: object) -> str:
        """Return a durable UUID4 ref, creating exactly one mapping under flock."""
        namespace = _validate_namespace(namespace)
        source = _source_text(raw_source)
        if not self.path.exists() and _has_legacy_identity_artifacts(
            self.experiment_dir
        ):
            raise LegacyExperimentError(LEGACY_EXPERIMENT_MESSAGE)
        with self._locked():
            if self.path.exists():
                payload = self._read_unlocked()
            else:
                if _has_legacy_identity_artifacts(self.experiment_dir):
                    raise LegacyExperimentError(LEGACY_EXPERIMENT_MESSAGE)
                payload = self._empty_payload()
            entries = payload["entries"]
            assert isinstance(entries, list)
            for entry in entries:
                assert isinstance(entry, dict)
                if entry["namespace"] == namespace and entry["source"] == source:
                    return str(entry["ref"])
            existing_refs = {str(entry["ref"]) for entry in entries}
            while True:
                ref = f"{AGENT_REF_PREFIXES[namespace]}_{uuid.uuid4()}"
                if ref not in existing_refs:
                    break
            entries.append({"namespace": namespace, "source": source, "ref": ref})
            # A reference is not usable until the complete updated table is on disk.
            self._write_unlocked(payload)
            return ref

    def resolve(self, namespace: str, ref: str) -> str:
        """Resolve a reference for trusted host control-plane use only."""
        namespace = _validate_namespace(namespace)
        _validate_ref(namespace, ref)
        with self._locked(shared=True, create=False):
            payload = self._read_unlocked()
        entries = payload["entries"]
        assert isinstance(entries, list)
        for entry in entries:
            assert isinstance(entry, dict)
            if entry["namespace"] == namespace and entry["ref"] == ref:
                return str(entry["source"])
        raise KeyError(f"unknown {namespace} reference")

    def _empty_payload(self) -> dict[str, object]:
        return {
            "schema_version": AGENT_REF_SCHEMA_VERSION,
            "experiment": self.experiment,
            "entries": [],
        }

    @contextmanager
    def _locked(self, *, shared: bool = False, create: bool = True) -> Iterator[None]:
        if create:
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            if self.host_dir.exists() and (self.host_dir.is_symlink() or not self.host_dir.is_dir()):
                raise AgentRefStoreError(f"invalid host directory: {self.host_dir}")
            self.host_dir.mkdir(mode=0o700, exist_ok=True)
            os.chmod(self.host_dir, 0o700)
        elif not self.lock_path.is_file():
            raise AgentRefStoreError("agent reference store is unavailable")
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.lock_path, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _read_unlocked(self) -> dict[str, object]:
        try:
            if self.path.is_symlink() or not self.path.is_file():
                raise AgentRefStoreError("agent reference store must be a regular file")
            if stat.S_IMODE(self.host_dir.stat().st_mode) != 0o700:
                raise AgentRefStoreError("agent reference host directory mode must be 0700")
            if stat.S_IMODE(self.path.stat().st_mode) != 0o600:
                raise AgentRefStoreError("agent reference store mode must be 0600")
            raw = self.path.read_bytes()
            text = raw.decode("utf-8", errors="strict")
            payload = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AgentRefStoreError(f"invalid agent reference store: {self.path}") from exc
        if not isinstance(payload, dict):
            raise AgentRefStoreError("agent reference store root must be an object")
        if set(payload) != {"schema_version", "experiment", "entries"}:
            raise AgentRefStoreError("invalid agent reference store fields")
        if type(payload.get("schema_version")) is not int or payload.get("schema_version") != AGENT_REF_SCHEMA_VERSION:
            raise AgentRefStoreError("unsupported agent reference store schema_version")
        if payload.get("experiment") != self.experiment:
            raise AgentRefStoreError("agent reference store experiment mismatch")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise AgentRefStoreError("agent reference store entries must be a list")
        sources: set[tuple[str, str]] = set()
        refs: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"namespace", "source", "ref"}:
                raise AgentRefStoreError("invalid agent reference entry")
            namespace = _validate_namespace(entry.get("namespace"))
            source = _source_text(entry.get("source"))
            ref = entry.get("ref")
            if not isinstance(ref, str):
                raise AgentRefStoreError("agent reference must be a string")
            _validate_ref(namespace, ref)
            source_key = (namespace, source)
            if source_key in sources or ref in refs:
                raise AgentRefStoreError("duplicate agent reference source or ref")
            sources.add(source_key)
            refs.add(ref)
        return payload

    def _write_unlocked(self, payload: dict[str, object]) -> None:
        # Validate the exact object before committing it, including newly added refs.
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        temp = self.host_dir / f".agent-refs.{uuid.uuid4()}.tmp"
        fd = -1
        try:
            fd = os.open(
                temp,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as handle:
                fd = -1
                os.fchmod(handle.fileno(), 0o600)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
            os.chmod(self.path, 0o600)
            directory_fd = os.open(self.host_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            if fd >= 0:
                os.close(fd)
            try:
                temp.unlink()
            except FileNotFoundError:
                pass
            raise


def _validate_namespace(namespace: object) -> str:
    if not isinstance(namespace, str) or namespace not in AGENT_REF_PREFIXES:
        raise AgentRefStoreError(f"unsupported agent reference namespace: {namespace!r}")
    return namespace


def _source_text(raw_source: object) -> str:
    if not isinstance(raw_source, str) or not raw_source:
        raise AgentRefStoreError("agent reference source must be a non-empty string")
    return raw_source


def _validate_ref(namespace: str, ref: str) -> None:
    prefix = f"{AGENT_REF_PREFIXES[namespace]}_"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        raise AgentRefStoreError(f"invalid {namespace} reference prefix")
    token = ref[len(prefix) :]
    try:
        parsed = uuid.UUID(token)
    except (ValueError, AttributeError) as exc:
        raise AgentRefStoreError(f"invalid {namespace} UUID reference") from exc
    if parsed.version != 4 or str(parsed) != token:
        raise AgentRefStoreError(f"non-canonical UUID4 {namespace} reference")


def _has_legacy_identity_artifacts(experiment_dir: Path) -> bool:
    ledger = experiment_dir / "ledgers" / "experiment_ledger.jsonl"
    if ledger.is_file() and ledger.stat().st_size:
        return True
    steps = experiment_dir / "steps"
    if steps.is_dir() and any(steps.iterdir()):
        return True
    traces = experiment_dir / "artifacts" / "traces"
    if traces.is_dir() and any(path.is_file() and path.stat().st_size for path in traces.iterdir()):
        return True
    return False
