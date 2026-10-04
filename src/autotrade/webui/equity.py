"""Daily equity series of one ledger-named replay result.

The curve itself is ``replay.curve``'s projection of the result file and its
style sidecar — the same one a Paper book copies at creation — so this module
only answers which results exist at all: ``registry.ledger_result``'s rule, by
which the forward replay has no curve until its verdict is recorded.

A result file is written once and never changes, and projecting a multi-MB
file costs most of a request, so the projection is kept per file identity
(path, size, mtime) for the last few dozen results read. A card's miniature
asks for ``points`` evenly spaced days instead of every day.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from autotrade.environment.replay.curve import result_curve

from . import registry


@lru_cache(maxsize=64)
def _curve(path: str, _size: int, _mtime_ns: int) -> dict[str, object]:
    # Read-only to every caller: _thin builds new lists, never edits these.
    return result_curve(Path(path))


def _thin(line: dict[str, object], points: int | None) -> dict[str, object]:
    """``line`` on at most ``points`` of its days, the first and last kept;
    every per-day list is cut on the same days, scalars stay exact."""

    dates = line.get("dates")
    if not points or not isinstance(dates, list) or len(dates) <= points:
        return line
    step = (len(dates) - 1) / (points - 1)
    keep = sorted({round(index * step) for index in range(points)})
    return {
        key: [value[index] for index in keep]
        if isinstance(value, list) and len(value) == len(dates)
        else value
        for key, value in line.items()
    }


def result_equity_payload(
    root: Path, experiment_id: str, name: str, *, points: int | None = None
) -> dict[str, object]:
    path = registry.ledger_result(root, experiment_id, name)
    info = path.stat()
    curve = _curve(str(path), info.st_size, info.st_mtime_ns)
    benchmark = curve["benchmark"]
    return {
        "experiment_id": experiment_id,
        "result": name,
        "series": [_thin(line, points) for line in curve["series"]],  # type: ignore[union-attr]
        "benchmark": _thin(benchmark, points) if isinstance(benchmark, dict) else benchmark,
        "exposure": {
            key: _thin(line, points)
            for key, line in curve["exposure"].items()  # type: ignore[union-attr]
        },
    }
