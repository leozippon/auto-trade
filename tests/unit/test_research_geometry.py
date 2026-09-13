"""The research geometry: whole research years, one forward year, a clipped Held-out.

The geometry decides which data every stage may read, so these assert the dates
and anchors themselves on the real SSE calendar, the refusals that keep the
stages whole, ordered and apart, the Held-out clip, and that the calendar
helpers kept their behaviour when they moved out of the Fold module.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.data.contracts import CN_TZ
from autotrade.pipelines import calendar
from autotrade.pipelines.calendar import ResearchGeometry, load_sse_trading_days
from autotrade.pipelines.config import DEFAULT_RESEARCH_GEOMETRY

REPO_ROOT = Path(__file__).resolve().parents[2]


def _close(day: str) -> datetime:
    return datetime.strptime(day, "%Y%m%d").replace(hour=23, minute=59, second=59, tzinfo=CN_TZ)


def _weekdays(end: str) -> list[str]:
    return [day.strftime("%Y%m%d") for day in pd.bdate_range("2019-01-02", end)]


def test_the_default_geometry_on_the_sse_calendar() -> None:
    """968 research, 242 forward and 53 Held-out trading days for a release
    ending 2026-09-11, with the Held-out clip stated on its slot."""

    raw = REPO_ROOT / "data" / "raw"
    if not (raw / "trade_cal" / "exchange=SSE").is_dir():
        pytest.skip("the SSE trade calendar is not in this checkout's data lake")
    release = [day for day in load_sse_trading_days(raw) if day <= "20260911"]
    geometry = DEFAULT_RESEARCH_GEOMETRY

    def count(slot: calendar.Slot) -> int:
        return sum(1 for day in release if slot.start <= day <= slot.end)

    years = geometry.research_years
    heldout = geometry.heldout(release)
    assert [(slot.label, slot.start, slot.end) for slot in years] == [
        ("Y1", "20210701", "20220630"),
        ("Y2", "20220701", "20230630"),
        ("Y3", "20230701", "20240630"),
        ("Y4", "20240701", "20250630"),
    ]
    assert [count(slot) for slot in years] == [242, 243, 241, 242]
    assert sum(count(slot) for slot in years) == 968
    assert count(geometry.forward) == 242
    assert (heldout.start, heldout.end, heldout.requested_end) == ("20260701", "20260911", "20260930")
    assert heldout.truncation_reason == "release_ends_20260911"
    assert count(heldout) == 53
    # Y4 is anchored on a Sunday: the calendar day before it, not the last
    # trading day, so the weekend's rows land in exactly one slot.
    assert "20240630" not in release
    assert years[3].anchor == _close("20240630")
    assert geometry.research_decision_time == geometry.forward.anchor == _close("20250630")


def test_each_slot_starts_the_instant_the_previous_one_ends() -> None:
    """Stages are contiguous and ordered: every slot's anchor is the close of
    the previous slot's last day, which is what makes a chain of slots read
    the same rows as one long slot."""

    for geometry, release_end in (
        (DEFAULT_RESEARCH_GEOMETRY, "20260911"),
        (ResearchGeometry("20190701", "20220630", "20230630", "20241231"), "20260911"),
    ):
        slots = (
            *geometry.research_years,
            geometry.forward,
            geometry.heldout(_weekdays(release_end)),
        )
        assert slots[0].start == geometry.research_start
        assert slots[len(geometry.research_years) - 1].end == geometry.research_end
        for previous, current in zip(slots, slots[1:]):
            assert current.start > previous.end
            assert current.anchor == _close(previous.end)
            assert _close(previous.end) + timedelta(seconds=1) == datetime.strptime(
                current.start, "%Y%m%d"
            ).replace(tzinfo=CN_TZ)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"research_start": "20210101"}, "whole July-June years"),
        ({"research_end": "20241231"}, "whole July-June years"),
        ({"research_end": "20200630", "forward_end": "20210630"}, "whole July-June years"),
        ({"forward_end": "20261231"}, "twelve months after research"),
        ({"forward_end": "20250630"}, "twelve months after research"),
        ({"heldout_end": "20260630"}, "Held-out follows the forward period"),
        ({"heldout_end": "20260601"}, "Held-out follows the forward period"),
        ({"research_start": "2021-07-01"}, "YYYYMMDD string"),
        ({"research_start": 20210701}, "YYYYMMDD string"),
        ({"heldout_end": "20260931"}, "not a calendar date"),
    ],
)
def test_a_misaligned_or_disordered_geometry_is_refused(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(DEFAULT_RESEARCH_GEOMETRY, **change)


def test_heldout_is_clipped_to_the_release_and_refused_without_two_days() -> None:
    geometry = DEFAULT_RESEARCH_GEOMETRY
    whole = geometry.heldout(_weekdays("20261231"))
    assert (whole.end, whole.requested_end, whole.truncation_reason) == ("20260930", "20260930", None)
    clipped = geometry.heldout(_weekdays("20260911"))
    assert (clipped.end, clipped.truncation_reason) == ("20260911", "release_ends_20260911")
    # One Held-out trading day, a release that stops inside the forward
    # period, and no release at all cannot replay Held-out.
    for release in (_weekdays("20260701"), _weekdays("20260615"), []):
        with pytest.raises(ValueError, match="Held-out|trading days"):
            geometry.heldout(release)


def test_the_moved_calendar_helpers_behave_as_before(tmp_path: Path) -> None:
    calendar_dir = tmp_path / "trade_cal" / "exchange=SSE"
    with pytest.raises(FileNotFoundError, match="missing SSE trade calendar"):
        load_sse_trading_days(tmp_path)
    calendar_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no trade calendar partitions"):
        load_sse_trading_days(tmp_path)
    pd.DataFrame(
        {"cal_date": ["20251231", "20251230", "20251227"], "is_open": [1, 1, 0]}
    ).to_parquet(calendar_dir / "year=2025.parquet")
    pd.DataFrame(
        {"cal_date": ["20260102", "20260101", "20251231"], "is_open": ["1", "0", "1"]}
    ).to_parquet(calendar_dir / "year=2026.parquet")
    assert load_sse_trading_days(tmp_path) == ["20251230", "20251231", "20260102"]
    assert calendar.yyyymmdd(" 2026-01-02 ") == "20260102"
    assert calendar.yyyymmdd(20260102) == "20260102"


def test_paper_reads_the_trading_calendar_from_its_new_home() -> None:
    from autotrade.paper import pit as paper_pit

    assert paper_pit.load_sse_trading_days is calendar.load_sse_trading_days
    spec = importlib.util.spec_from_file_location(
        "run_paper", REPO_ROOT / "scripts" / "paper" / "run_paper.py"
    )
    assert spec is not None and spec.loader is not None
    run_paper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_paper)
    assert run_paper.load_sse_trading_days is calendar.load_sse_trading_days
