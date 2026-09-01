"""Experiment-level immutable PRIOR generations selected only by CURRENT.ref."""

from __future__ import annotations

import fcntl
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from autotrade.environment.tools.prior_policy import PRIOR_MAX_CHARS

PRIOR_FILENAME = "PRIOR.md"


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


def latest_prior_text(records: Sequence[Mapping[str, object]]) -> str:
    """The PRIOR the last Meta row published; empty before the first Meta."""
    metas = [record for record in records if record.get("record_type") == "meta_learning"]
    if not metas:
        return ""
    return str(metas[-1].get("prior") or "").strip()


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
