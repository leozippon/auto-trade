"""What the rounds send alike, stated once.

A round file is its experiments and their directives. The values several rounds share
-- the geometry and dataset selection a seed was prebuilt for, the acceptance
rules and the round-level parameters -- live here, and a round imports them from
here and from `_round`, never from another round file, so a round can close
without its values moving.

Experiments have been created under every value below. A round that needs another
value adds a new name rather than editing one in place. The one exception is a
key the console retires: it leaves every profile, so that a round built from
one is accepted (the deflated-Sharpe bars and the replay-year budget left on
2026-11-06; git history keeps what the closed rounds sent). No profile states
`research_years`: how many of a geometry's years a session researches on is
each experiment's choice, which every round states for its experiments.
"""

from __future__ import annotations

CSI1000 = "000852.SH"

# The console's default selection is the ten-year one every current round runs:
# ten August-July research years from 2015-08 (``config.DEFAULT_RESEARCH_GEOMETRY``),
# decision views reaching 132 months back, the ten default fundamentals, the
# index and Shenwan macro datasets and the announcement titles, with events off
# (``experiment_params.DataParams``). A round relies on it through
# ``_round.BASE_EXPECTED_DEFAULTS``, which stops the launcher if the console's
# defaults move. Its seed, the console's default too, is prebuilt at snapshot
# cache format 17 from release 0ae8461dc6e84294b9f794a19e2d4c1a for research
# years 5, 6 and 7 (2026-11-10, after the disclosure_date fix).
FUND_TITLE_TEN_YEAR_PIT_VIEWS_SEED = "data/pit_views_seed_research_10y_fundanns_20261110"

# That selection graded against CSI 1000, which drew down 48 % over the
# eight years from 2017-07, hence the 0.55 equity drawdown limit round 20261005
# set for the index. The account and the budgets are the round's.
FUND_TITLE_TEN_YEAR_CSI1000: dict[str, object] = {
    "benchmark_index": CSI1000,
    "max_drawdown": 0.55,
    "active_max_drawdown": 0.30,
}
