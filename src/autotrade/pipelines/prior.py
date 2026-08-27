"""Experiment-level immutable PRIOR generations selected only by CURRENT.ref."""

from __future__ import annotations

import fcntl
import os
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from autotrade.environment.tools.prior_policy import PRIOR_MAX_CHARS

PRIOR_FILENAME = "PRIOR.md"
_LEGACY_DIRECTION_FIELDS = frozenset({"taste", "taste_path"})


@dataclass(frozen=True)
class PriorPublication:
    generation_id: str
    prior_ref: str
    chars: int
    text: str


class ExperimentPriorStore:
    """One immutable PRIOR file per Meta generation and one atomic current pointer."""

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir).resolve()
        self.root = self.experiment_dir / "artifacts" / "prior"

    @property
    def current_pointer_path(self) -> Path:
        return self.root / "CURRENT.ref"

    @property
    def lock_path(self) -> Path:
        return self.experiment_dir / ".prior.lock"

    def current_text(self) -> str:
        generation = self.current_generation_id()
        return self.load_generation(generation) if generation else ""

    def current_generation_id(self) -> str:
        if not self.current_pointer_path.is_file():
            return ""
        raw = self.current_pointer_path.read_text(encoding="utf-8")
        generation = raw.strip()
        if not generation or raw != generation + "\n":
            raise ValueError("PRIOR CURRENT.ref is malformed")
        self._generation_path(generation)
        return generation

    def current_ref(self) -> str:
        generation = self.current_generation_id()
        if not generation:
            return ""
        path = self._generation_path(generation)
        if not path.is_file():
            raise FileNotFoundError(f"PRIOR generation is missing: {generation}")
        return str(path)

    def load_generation(self, generation_id: str) -> str:
        path = self._generation_path(generation_id)
        if not path.is_file():
            raise FileNotFoundError(f"PRIOR generation is missing: {generation_id}")
        return path.read_text(encoding="utf-8")

    def publish(self, text: str, *, generation_id: str) -> PriorPublication:
        body = text.strip()
        if not body:
            raise ValueError("PRIOR.md must be non-empty to publish")
        nchars = len(body)
        if nchars > PRIOR_MAX_CHARS:
            raise ValueError(
                f"PRIOR.md is {nchars} characters; keep it to {PRIOR_MAX_CHARS}"
            )
        if not generation_id.strip():
            raise ValueError("PRIOR generation_id is required")
        generation = generation_id.strip()
        dest = self._generation_path(generation)
        payload = body + "\n"
        with _exclusive_flock(self.lock_path):
            if dest.exists():
                raise FileExistsError(f"PRIOR generation already exists: {generation_id}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            _write_text_atomic(dest, payload)
            dest.chmod(0o444)
            _write_text_atomic(self.current_pointer_path, generation + "\n")
        return PriorPublication(
            generation_id=generation,
            prior_ref=str(dest),
            chars=nchars,
            text=body,
        )

    def restore(self, generation_id: str) -> PriorPublication:
        """Point CURRENT at an existing immutable generation without rewriting it."""
        if not generation_id.strip():
            raise ValueError("PRIOR generation_id is required")
        generation = generation_id.strip()
        dest = self._generation_path(generation)
        with _exclusive_flock(self.lock_path):
            if not dest.is_file():
                raise FileNotFoundError(f"PRIOR generation is missing: {generation_id}")
            body = dest.read_text(encoding="utf-8").strip()
            if not body:
                raise ValueError(f"PRIOR generation is empty: {generation_id}")
            _write_text_atomic(self.current_pointer_path, generation + "\n")
        return PriorPublication(
            generation_id=generation,
            prior_ref=str(dest),
            chars=len(body),
            text=body,
        )

    def clear_current(self) -> None:
        """Drop the current pointer after rollback past every published generation."""
        with _exclusive_flock(self.lock_path):
            self.current_pointer_path.unlink(missing_ok=True)

    def _generation_path(self, generation_id: str) -> Path:
        if Path(generation_id).name != generation_id or generation_id.startswith("."):
            raise ValueError("PRIOR generation_id must be one path component")
        return self.root / "generations" / generation_id / PRIOR_FILENAME


def latest_prior_text(
    records: Sequence[Mapping[str, object]],
    *,
    experiment_dir: str | Path | None = None,
) -> str:
    """Return the current unified PRIOR, including one safe legacy migration.

    Old Meta rows could carry a separate direction document inline or by path.
    Only the latest legacy row is migrated, and path recovery is confined to the
    experiment directory. A later unified row has no legacy keys and wins as-is.
    """
    metas = [record for record in records if record.get("record_type") == "meta_learning"]
    if not metas:
        return ""
    latest = metas[-1]
    prior = str(latest.get("prior") or "").strip()
    # Legacy compatibility: these are the only old fields read by new code.
    has_legacy_direction = any(key in latest for key in _LEGACY_DIRECTION_FIELDS)
    if not has_legacy_direction:
        return prior
    if not prior:
        prior = next(
            (
                str(record.get("prior") or "").strip()
                for record in reversed(metas[:-1])
                if str(record.get("prior") or "").strip()
            ),
            "",
        )
    direction = _legacy_taste_text(latest, experiment_dir=experiment_dir)
    return _merge_legacy_direction(prior, direction)


def _legacy_taste_text(
    record: Mapping[str, object], *, experiment_dir: str | Path | None
) -> str:
    """Read an inline or experiment-local legacy Taste body."""
    inline = str(record.get("taste") or "").strip()
    if inline:
        return inline
    raw_path = str(record.get("taste_path") or "").strip()
    if not raw_path or experiment_dir is None:
        return ""
    root = Path(experiment_dir).resolve()
    candidate = Path(raw_path)
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    if path != root and root not in path.parents:
        return ""
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def _merge_legacy_direction(prior: str, direction: str) -> str:
    prior = prior.strip()
    direction = direction.strip()
    if not direction:
        return prior
    if direction in prior:
        return prior
    heading = re.search(r"(?m)^(#{1,6}[ \t]+策略探索方向[ \t]*)$", prior)
    if heading:
        insert_at = heading.end()
        return (
            prior[:insert_at]
            + "\n\n"
            + direction
            + prior[insert_at:]
        ).strip()
    section = f"## 策略探索方向\n\n{direction}"
    return f"{section}\n\n{prior}".strip() if prior else section


def unified_meta_record(
    record: Mapping[str, object], *, current_prior: str | None = None
) -> dict[str, object]:
    """Return a public Meta row with legacy direction fields removed.

    ``current_prior`` is supplied only for the latest row when callers need the
    one-time in-memory migration rendered through the sole current channel.
    The persisted ledger remains byte-for-byte unchanged.
    """
    public = {
        str(key): value
        for key, value in record.items()
        if str(key) not in _LEGACY_DIRECTION_FIELDS
    }
    if current_prior is not None:
        public["prior"] = current_prior
    return public


def restore_current_from_records(
    experiment_dir: str | Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    """Align CURRENT with the last remaining Meta generation after resume/rollback."""
    metas = [
        record for record in records if record.get("record_type") == "meta_learning"
    ]
    store = ExperimentPriorStore(experiment_dir)
    generation = (
        str(metas[-1].get("prior_generation_id") or "").strip() if metas else ""
    )
    if not generation:
        store.clear_current()
        return
    try:
        if store.current_generation_id() == generation:
            store.current_ref()
            return
    except (OSError, ValueError):
        pass
    store.restore(generation)


@contextmanager
def _exclusive_flock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
