"""The standard controls a pack starter copies (``configs/starter_lib/controls.py``).

What must hold. Each control changes one thing about the candidate: late entry
moves only the day, by trading days, and never acts before the calendar
reaches it; shuffled assignment keeps every event day and count and draws only
eligible names; the held shuffle moves every score onto an eligible stand-in
that stays the same from review to review, so a book with a keep band holds
the stand-ins exactly as long as the candidate holds its names; random skip
removes exactly as many names as the rule did and keeps the book's order; the
static version ignores the state it is given. The draws reproduce across calls
and do not move when the pool gains a name, so a later review recomputes an
old event to the same names. Turnover per year is the host's span figure on a
yearly scale, counting only fills.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from autotrade.environment.strategy_loader import validate_strategy_source

PATH = Path(__file__).resolve().parents[2] / "configs" / "starter_lib" / "controls.py"
# Executed from its source, so no bytecode cache lands in configs/.
controls = ModuleType("starter_controls")
exec(compile(PATH.read_text(encoding="utf-8"), str(PATH), "exec"), controls.__dict__)

# Mon 03-04 .. Fri 03-08, then Mon 03-11 and Tue 03-12.
CALENDAR = ["20240304", "20240305", "20240306", "20240307", "20240308", "20240311", "20240312"]
EVENTS = pd.DataFrame(
    {
        "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
        # A Saturday title, a trading day, the calendar's last day, and one
        # dated before the calendar starts.
        "day": ["20240309", "20240305", "20240312", "20240301"],
        "tag": ["a", "b", "c", "d"],
    }
)
POOL = [f"{i:06d}.SZ" for i in range(1, 41)]


def test_the_module_is_valid_strategy_code() -> None:
    validate_strategy_source(PATH.read_text(encoding="utf-8"), filename="controls.py", entrypoints=False)


def test_late_entry_moves_the_day_by_trading_days_and_drops_what_is_not_due() -> None:
    late = controls.late_entry(EVENTS, CALENDAR, 2)
    # Saturday 03-09 -> first trading day 03-11 -> +2 is past the calendar: dropped.
    # 03-05 -> 03-07; 03-12 + 2 is not due; 03-01 -> 03-04 -> 03-06.
    assert late[["ts_code", "day", "tag"]].values.tolist() == [
        ["000002.SZ", "20240307", "b"],
        ["000004.SZ", "20240306", "d"],
    ]
    with pytest.raises(ValueError, match="k >= 1"):
        controls.late_entry(EVENTS, CALENDAR, 0)
    with pytest.raises(ValueError, match="sorted"):
        controls.late_entry(EVENTS, CALENDAR[::-1], 1)


def test_shuffled_assignment_keeps_days_and_counts_and_draws_eligible_names() -> None:
    events = pd.DataFrame(
        {"ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"], "day": ["20240305", "20240305", "20240306"]}
    )
    drawn = controls.shuffled_assignment(events, POOL)
    assert drawn["day"].tolist() == events["day"].tolist()
    assert set(drawn["ts_code"]) <= set(POOL)
    assert drawn[drawn["day"] == "20240305"]["ts_code"].nunique() == 2
    # Reproducible whatever the pool's order, salted, and unmoved when the
    # pool loses names that were not drawn.
    assert drawn.equals(controls.shuffled_assignment(events, list(reversed(POOL))))
    assert not drawn.equals(controls.shuffled_assignment(events, POOL, salt=1))
    shrunk = [code for code in POOL if code in set(drawn["ts_code"]) or code > "000030.SZ"]
    assert drawn.equals(controls.shuffled_assignment(events, shrunk))
    with pytest.raises(ValueError, match="eligible names"):
        controls.shuffled_assignment(events, POOL[:1])


def _keep_band_book(values: pd.Series, held: set[str], seats: int, band: float) -> set[str]:
    """The review of a keep-band book: keep holdings ranked inside seats *
    band, fill the free seats from the top."""
    ranked = list(values.sort_values(ascending=False, kind="stable").index)
    keep = {code for code in held if code in ranked[: int(seats * band)]}
    return keep | set([code for code in ranked if code not in keep][: seats - len(keep)])


def test_the_held_shuffle_keeps_a_screen_s_holding_period() -> None:
    """Twelve monthly reviews of a drifting score: the candidate book and the
    held-shuffle control trade exactly as often; the stand-ins are eligible
    names, fixed across days, and unmoved when the pool gains a name."""
    # Distinct values on every day (i < 40), reshuffled in blocks month to month.
    scores = [
        pd.Series({code: float(i) + 40.0 * ((month * 7 + i * 13) % 5) for i, code in enumerate(POOL)})
        for month in range(12)
    ]
    held_candidate: set[str] = set()
    held_control: set[str] = set()
    trades = {"candidate": 0, "control": 0}
    for values in scores:
        new = _keep_band_book(values, held_candidate, 6, 2.0)
        trades["candidate"] += len(new ^ held_candidate)
        held_candidate = new
        new = _keep_band_book(controls.held_shuffle(values, POOL), held_control, 6, 2.0)
        trades["control"] += len(new ^ held_control)
        held_control = new
    assert trades["control"] == trades["candidate"]

    moved = controls.held_shuffle(scores[0], POOL)
    assert sorted(moved.to_numpy()) == sorted(scores[0].to_numpy())
    assert set(moved.index) == set(POOL)
    assert (moved != scores[0].reindex(moved.index)).all()
    # The same stand-ins on another day's scores and in another pool order.
    partner = {code: scores[0].index[scores[0].to_numpy() == value][0] for code, value in moved.items()}
    later = controls.held_shuffle(scores[5], list(reversed(POOL)))
    assert all(later[code] == scores[5][partner[code]] for code in later.index)
    assert not moved.equals(controls.held_shuffle(scores[0], POOL, salt=1))
    # A new eligible name re-partners one neighbour at most; scores on fewer
    # names leave the unscored stand-ins out.
    grown = controls.held_shuffle(scores[0], [*POOL, "000041.SZ"])
    assert sum(grown.get(code) != value for code, value in moved.items()) <= 1
    assert len(controls.held_shuffle(scores[0].iloc[:10], POOL)) == 10
    with pytest.raises(ValueError, match="not eligible"):
        controls.held_shuffle(scores[0], POOL[:5])


def test_turnover_per_year_counts_filled_notional_over_the_cash() -> None:
    fills = [
        {"price": 10.0, "quantity": 1_000, "status": "filled"},
        {"price": 10.0, "quantity": 1_000, "status": "filled"},
        {"price": 50.0, "quantity": 9_900, "status": "rejected"},
    ]
    # 20,000 traded on 100,000 over half a year: 0.2 a half-year, 0.4 a year.
    assert controls.turnover_per_year(fills, 100_000, 122) == pytest.approx(0.4)
    assert controls.turnover_per_year(pd.DataFrame(fills[:1]).drop(columns="status"), 100_000, 244) == pytest.approx(0.1)
    assert controls.turnover_per_year([], 100_000, 244) == 0.0
    with pytest.raises(ValueError, match="positive"):
        controls.turnover_per_year(fills, 100_000, 0)


def test_random_skip_removes_as_many_names_and_keeps_the_order() -> None:
    kept = controls.random_skip(POOL, 5, "20240305")
    assert len(kept) == len(POOL) - 5
    assert kept == [code for code in POOL if code in set(kept)]
    assert kept == controls.random_skip(POOL, 5, "20240305")
    assert kept != controls.random_skip(POOL, 5, "20240306")
    assert controls.random_skip(POOL, 0, "20240305") == POOL
    with pytest.raises(ValueError, match="cannot skip"):
        controls.random_skip(POOL[:3], 4, "20240305")


def test_the_static_version_ignores_the_state_and_the_parameter_is_declared() -> None:
    def seats_by_state(context, state):
        return 30 if state == "hot" else 12

    static = controls.static_version(seats_by_state, 20)
    assert static(None, "hot") == static(None, "cold") == 20
    assert static.__name__ == "static_seats_by_state"
    assert controls.most_common([12, 30, 30, 12, 20]) == 12
    with pytest.raises(ValueError):
        controls.most_common([])
