"""What the rounds send alike, stated once.

A round file is its arms and their directives. The values several rounds share
-- the eight-year geometry and dataset selection a seed was prebuilt for, the
graduation bars, the round-level parameters, the lineage that grows as the
arms searching one baseline close -- live here, and a round imports them from
here and from `_round`, never from another round file, so a round can close
without its values moving.

Arms have been created under every value below. A round that needs another
value adds a new name rather than editing one in place.
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

# Create-time graduation bars, named on every arm so the request records a
# choice. They were the console's creation defaults when round 20260927 first
# named them.
GATES: dict[str, object] = {
    "min_active_ir": 0.75,
    "min_dsr_probability": 0.975,
    "min_positive_year_share": 0.75,
    "min_full_span_validations": 2,
    "forward_confidence": 0.80,
    "recency_months": 6,
    "min_mean_gross": 0.50,
    "min_round_trips_per_month": 1,
    "heldout_tolerance_z": 1.28,
}

# The round-level parameters of every round on the fundamentals surface since
# round 20261005: the fundamentals geometry and the bars on a 100k account
# graded against CSI 1000, which drew down 48 % over the eight years, hence
# the 0.55 equity drawdown limit.
FUND_EIGHT_YEAR_100K: dict[str, object] = {
    **FUND_EIGHT_YEAR,
    **GATES,
    "max_replay_years": 96,
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

# Its seed at snapshot cache format 14 (observed-era visibility) on release
# b50c220f... (logs/data/seed_8y_fundanns_20261024/).
FUND_TITLE_PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_fundanns_20261024"

# The four arms of rounds 20261014c and 20261014d that tested hierarchical
# continual memory on a numeric model (one pack, nine full-span legs replayed
# on two card models): the family a later HCM arm on another carrier
# continues (round 20261019).
HCM_LINEAGE = [
    "hcm_100k_8y_sol_qwen_20261014d",
    "hcm_100k_8y_sol_dsf_20261014d",
    "hcm_100k_8y_dsf_zgca_20261014c",
    "hcm_100k_8y_sol_dsf_zgca_20261014c",
]
