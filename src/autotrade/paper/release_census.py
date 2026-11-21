"""Which research releases are still read, and by whom: the keep rules of
``scripts/data/tushare_lake.py prune-releases``.

A release stays while the raw generation record names it, a seed (or a seed
build's workspace pin) one or two levels under ``data/`` was built from it,
an experiment under ``experiments/`` that has not ended pinned it or names a
seed built from it, or a Paper account that is not killed has it pinned. The
census lives here, the lowest package that sees every consumer (the
experiments' HITL state and the Paper accounts), and the lake command takes
it as an argument, so the ingest package imports neither.
"""

from __future__ import annotations

from pathlib import Path

from autotrade.common.records import read_json_object
from autotrade.data_sources.tushare.generation import read_raw_generation_file
from autotrade.environment.data.research_release import pinned_generation
from autotrade.pipelines.hitl_state import (
    HITL_DIR_NAME,
    PARAMS_NAME,
    STATUS_NAME,
    ExperimentState,
    experiment_is_archived,
    read_status,
    status_pid_alive,
)

from .book import list_books, open_capped_books
from .books import PAPER_STATE_DIR
from .pit import PIT_CACHE_NAME


def release_consumers(repo_root: Path, raw_dir: Path) -> dict[str, list[str]]:
    """Why each generation is still read: the keep rules of ``prune-releases``."""

    kept: dict[str, list[str]] = {}

    def keep(generation: str, reason: str) -> None:
        if generation:
            kept.setdefault(generation, []).append(reason)

    keep(str(read_raw_generation_file(raw_dir).get("generation_id") or ""), "raw generation record")
    data = repo_root / "data"
    for directory in sorted(path for pattern in ("*", "*/*") for path in data.glob(pattern) if path.is_dir()):
        name = directory.relative_to(repo_root)
        if (directory / "provider.json").is_file():
            keep(_seed_generation(directory), f"seed {name}")
        keep(pinned_generation(directory), f"pin {name}")
    experiments = repo_root / "experiments"
    for directory in sorted(experiments.iterdir()) if experiments.is_dir() else ():
        hitl = directory / HITL_DIR_NAME
        if directory.name.startswith(".") or not hitl.is_dir():
            continue
        try:
            status = read_status(hitl / STATUS_NAME)
            state = str(status.get("state") or ExperimentState.CREATED)
        except (OSError, TypeError, ValueError):
            status, state = {}, ExperimentState.UNREADABLE
        # An experiment has ended once completed (the console neither runs nor
        # resumes it) or archived while no worker still runs it; any other
        # experiment, one whose state is unknown or unreadable included, keeps its
        # release.
        if state == ExperimentState.COMPLETED or (experiment_is_archived(directory) and not status_pid_alive(status)):
            continue
        reason = f"experiment {directory.name} ({state})"
        keep(pinned_generation(directory), reason)
        seed = read_json_object(hitl / PARAMS_NAME, missing_ok=True).get("pit_views_seed")
        if isinstance(seed, str) and seed.strip() and (repo_root / seed.strip() / "provider.json").is_file():
            keep(_seed_generation(repo_root / seed.strip()), reason)
    paper = repo_root / PAPER_STATE_DIR
    try:
        books = open_capped_books(paper)
    except (OSError, ValueError):
        # A verdict log that cannot be read leaves every book's release kept.
        books = list_books(paper)
    for book_id in books:
        for cache in sorted((paper / book_id / PIT_CACHE_NAME).glob("*")):
            keep(pinned_generation(cache), f"Paper account {book_id}")
    return kept


def _seed_generation(seed: Path) -> str:
    """The generation a seed's views were built from, as its ``provider.json`` records it."""
    path = seed / "provider.json"
    try:
        return str(read_json_object(path).get("generation_id") or "")
    except (OSError, ValueError, AttributeError) as exc:
        raise RuntimeError(f"cannot read the release seed record {path} names: {exc}") from exc


__all__ = ["release_consumers"]
