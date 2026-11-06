"""What the rounds send alike, stated once.

A round file is its arms and their directives. The values several rounds share
-- the geometry and dataset selection a seed was prebuilt for, the acceptance
rules and the round-level parameters -- live here, and a round imports them from
here and from `_round`, never from another round file, so a round can close
without its values moving.

Arms have been created under every value below. A round that needs another
value adds a new name rather than editing one in place. The one exception is a
key the console retires: it leaves every profile, so that a round built from
one is accepted (the deflated-Sharpe bars and the replay-year budget left on
2026-11-06; git history keeps what the closed rounds sent). No profile states
`research_years`: how many of a geometry's years a session researches on is
each arm's choice, which every round states for its arms.
"""

from __future__ import annotations

CSI1000 = "000852.SH"

# The research geometry and dataset selection the eight-year seeds were planned
# over (round 20260927, logs/notes/round_20260927/E8_eight_year_build.md); part
# of a seed's contract, so no arm may change one of them alone.
EIGHT_YEAR: dict[str, object] = {
    "research_start": "20170701",
    "research_end": "20250630",
    "forward_end": "20260630",
    "heldout_end": "20260930",
    "window_months": 108,
    "include_fundamentals": False,
    "include_events": False,
    "include_text": False,
    "macro_datasets": ["index_daily", "index_dailybasic", "sw_daily", "index_weight"],
}

# The eight-year geometry with the fundamentals domain switched on (round
# 20261005); leaving `fundamental_datasets` empty selects the default ten, the
# selection the fundamentals seeds were prebuilt for from release
# 55fc7fd82c3c4c77b752920ec4507609 (the dividend seed's own, its fundamental
# events audited from 2016-01). Daily, macro, universe and the Broker's
# corporate actions are byte-identical to the dividend seed's
# (logs/notes/round_20261005/SEED_fund_8y.md §1).
FUND_EIGHT_YEAR: dict[str, object] = {**EIGHT_YEAR, "include_fundamentals": True}

# The round-level parameters of every round on the fundamentals surface since
# round 20261005: the fundamentals geometry and the bars on a 100k account
# graded against CSI 1000, which drew down 48 % over the eight years, hence
# the 0.55 equity drawdown limit. The forward reading's confidence is named so
# the request records a choice.
FUND_EIGHT_YEAR_100K: dict[str, object] = {
    **FUND_EIGHT_YEAR,
    "forward_confidence": 0.80,
    "initial_cash": 100_000,
    "benchmark_index": CSI1000,
    "max_drawdown": 0.55,
    "active_max_drawdown": 0.30,
}

# The eight-year geometry with both the fundamentals domain (the default ten
# datasets) and the vendor's announcement titles (round 20261014): the first
# selection to mount statements and titles on one arm. Its seed is prebuilt
# from a research release whose fundamentals and titles both reach back to
# 2016-01 and whose Broker corporate actions carry the bonus-share column the
# dividend tax reads.
FUND_TITLE_EIGHT_YEAR: dict[str, object] = {**FUND_EIGHT_YEAR, "include_text": True, "text_datasets": ["anns_d"]}

# Round 20261005's 100k bars on that geometry.
FUND_TITLE_EIGHT_YEAR_100K: dict[str, object] = {**FUND_EIGHT_YEAR_100K, **FUND_TITLE_EIGHT_YEAR}

# Its seed at snapshot cache format 15 (fina_indicator rows no earlier than
# their period's latest statement) on release 56b289cf...
# (logs/data/seed_build_20261104_8y.log).
FUND_TITLE_PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_fundanns_20261104"

# The same selection on ten August-July research years from 2015-08 (rounds
# 20261031 and 20261102): the twelve forward months after them, a Held-out
# quarter the replay clips to the release end, and decision views reaching 132
# months back from the research end, the ten years plus the one their start
# reads, a window every domain inherits. Part of the seed's contract, like
# EIGHT_YEAR; the seed is prebuilt at snapshot cache format 15 from release
# 56b289cf02db41e68a33abf66de49016, the eight-year seed's.
FUND_TITLE_TEN_YEAR: dict[str, object] = {
    **FUND_TITLE_EIGHT_YEAR,
    "research_start": "20150801",
    "research_end": "20250731",
    "forward_end": "20260731",
    "heldout_end": "20261031",
    "window_months": 132,
}
FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED = "data/pit_views_seed_research_10y_fundanns_20261104"

# That selection graded against CSI 1000: the drawdown limits round 20261005
# set for the index (FUND_EIGHT_YEAR_100K). The account and the budgets are
# the round's.
FUND_TITLE_TEN_YEAR_CSI1000: dict[str, object] = {
    **FUND_TITLE_TEN_YEAR,
    "benchmark_index": CSI1000,
    "max_drawdown": 0.55,
    "active_max_drawdown": 0.30,
}
