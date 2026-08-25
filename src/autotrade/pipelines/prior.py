"""Experiment-level versioned PRIOR.md (process memory, not a repo-root file)."""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from autotrade.agent.runner import PRIOR_MAX_CHARS

PRIOR_FILENAME = "PRIOR.md"


@dataclass(frozen=True)
class PriorPublication:
    generation_id: str
    prior_ref: str
    sha256: str
    chars: int
    text: str


class ExperimentPriorStore:
    """Authoritative current PRIOR plus one immutable file per published Meta generation."""

    def __init__(self, experiment_dir: str | Path) -> None:
        self.root = Path(experiment_dir).resolve() / "artifacts" / "prior"

    @property
    def current_path(self) -> Path:
        return self.root / "CURRENT.md"

    @property
    def current_pointer_path(self) -> Path:
        return self.root / "CURRENT.ref"

    def current_text(self) -> str:
        if not self.current_path.is_file():
            return ""
        return self.current_path.read_text(encoding="utf-8")

    def current_generation_id(self) -> str:
        if not self.current_pointer_path.is_file():
            return ""
        return self.current_pointer_path.read_text(encoding="utf-8").strip()

    def current_ref(self) -> str:
        generation = self.current_generation_id()
        if not generation:
            return ""
        path = self._generation_path(generation)
        return str(path) if path.is_file() else ""

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
        dest = self._generation_path(generation_id.strip())
        if dest.exists():
            raise FileExistsError(f"PRIOR generation already exists: {generation_id}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = body + "\n"
        _write_text_atomic(dest, payload)
        dest.chmod(0o444)
        _write_text_atomic(self.current_path, payload)
        _write_text_atomic(self.current_pointer_path, generation_id.strip() + "\n")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return PriorPublication(
            generation_id=generation_id.strip(),
            prior_ref=str(dest),
            sha256=digest,
            chars=nchars,
            text=body,
        )

    def restore(self, generation_id: str) -> PriorPublication:
        """Point CURRENT at an existing immutable generation without rewriting it."""
        if not generation_id.strip():
            raise ValueError("PRIOR generation_id is required")
        dest = self._generation_path(generation_id.strip())
        if not dest.is_file():
            raise FileNotFoundError(f"PRIOR generation is missing: {generation_id}")
        body = dest.read_text(encoding="utf-8").strip()
        if not body:
            raise ValueError(f"PRIOR generation is empty: {generation_id}")
        payload = body + "\n"
        _write_text_atomic(self.current_path, payload)
        _write_text_atomic(self.current_pointer_path, generation_id.strip() + "\n")
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return PriorPublication(
            generation_id=generation_id.strip(),
            prior_ref=str(dest),
            sha256=digest,
            chars=len(body),
            text=body,
        )

    def clear_current(self) -> None:
        """Drop CURRENT after rollback past every published generation."""
        self.current_path.unlink(missing_ok=True)
        self.current_pointer_path.unlink(missing_ok=True)

    def _generation_path(self, generation_id: str) -> Path:
        if Path(generation_id).name != generation_id or generation_id.startswith("."):
            raise ValueError("PRIOR generation_id must be one path component")
        return self.root / "generations" / generation_id / PRIOR_FILENAME


def latest_prior_text(records: list[Mapping[str, object]]) -> str:
    metas = [record for record in records if record.get("record_type") == "meta_learning"]
    return str(metas[-1].get("prior") or "") if metas else ""


def restore_current_from_records(
    experiment_dir: str | Path,
    records: list[Mapping[str, object]],
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
    if store.current_generation_id() == generation:
        return
    store.restore(generation)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
