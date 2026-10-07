"""What the rounds send alike, stated once.

A round file is its arms and their directives. The values several rounds share
-- the eight-year geometry and dataset selection a seed was prebuilt for, the
graduation bars, the round-level parameters, the hosted roles of a model pair,
the lineage that grows as the arms searching one baseline close -- live here,
and a round imports them from here and from `_round`, never from another round
file, so a round can close without its values moving.

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

# The fundamentals seed's taxed-era twin: every market and fundamentals table is
# a hard link to round 20261006's seed, and only the Broker's corporate-action
# tables are rebuilt to carry the bonus-share column the dividend tax reads
# (logs/data/seed_8y_fund_20261008/). Arms created from round 20261008 on are
# taxed by the creation default, so a row on this seed differs from one on its
# twin by the dividend tax alone.
TAXED_FUND_PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_fund_20261008"

# The eight-year geometry with the vendor's announcement titles as the one text
# dataset (round 20261012): daily, universe, the four macro tables and
# `anns_d`, no fundamentals or events -- the selection the title seed below was
# prebuilt for. Its titles are the vendor's `anns_d` throughout: 2016-01..2019-12
# fetched from the official service, 2020-01 on the relay copy the lake held
# (logs/data/seed_8y_anns_20261006/).
TITLE_EIGHT_YEAR: dict[str, object] = {**EIGHT_YEAR, "include_text": True, "text_datasets": ["anns_d"]}

# Round 20261005's 100k bars on the title geometry.
TITLE_EIGHT_YEAR_100K: dict[str, object] = {**FUND_EIGHT_YEAR_100K, **TITLE_EIGHT_YEAR}

# The title seed: daily, macro and universe from release
# fa6a99174bcb4c43bcc957f0a91a0cce, the Broker's taxed corporate actions, and
# the text domain with `anns_d` alone.
TITLE_PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_anns_20261006"

# The eight-year geometry with both the fundamentals domain (the default ten
# datasets) and the vendor's announcement titles (round 20261014): the first
# selection to mount statements and titles on one arm. Its seed is prebuilt
# from the newest research release, whose fundamentals and titles both reach
# back to 2016-01 and whose Broker corporate actions carry the bonus-share
# column the dividend tax reads.
FUND_TITLE_EIGHT_YEAR: dict[str, object] = {**FUND_EIGHT_YEAR, "include_text": True, "text_datasets": ["anns_d"]}

# Round 20261005's 100k bars on that geometry.
FUND_TITLE_EIGHT_YEAR_100K: dict[str, object] = {**FUND_EIGHT_YEAR_100K, **FUND_TITLE_EIGHT_YEAR}

FUND_TITLE_PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_fundanns_20261014"

# The five-year geometry (round 20261015): research on the five July-June years
# from 2020-07, when the events domain begins, with a 72-month window (the
# research years plus one warm-up year, as the four- and eight-year windows
# have), the fundamentals domain (default ten), the announcement titles, the
# eight-year surface's four macro tables, and the events domain: its default
# fourteen datasets plus the two top-ten holder tables, which are complete
# only as a union (docs/data-documentation.md §4), and the famous-trader seats.
# The gates are GATES unchanged: a positive-year share of 0.75 already asks for
# four of five years.
FIVE_YEAR: dict[str, object] = {
    **EIGHT_YEAR,
    "research_start": "20200701",
    "window_months": 72,
    "include_fundamentals": True,
    "include_events": True,
    "events_datasets": [
        "margin",
        "margin_detail",
        "moneyflow",
        "cyq_perf",
        "bak_daily",
        "block_trade",
        "stk_holdernumber",
        "stk_holdertrade",
        "new_share",
        "share_float_complete",
        "top_list",
        "top_inst",
        "limit_list_d",
        "kpl_list",
        "top10_holders",
        "top10_floatholders",
        "hm_detail",
    ],
    "include_text": True,
    "text_datasets": ["anns_d"],
}

# Round 20261005's 100k bars on the five-year geometry.
FIVE_YEAR_100K: dict[str, object] = {**FUND_EIGHT_YEAR_100K, **FIVE_YEAR}

# Its seed, prebuilt from release b5dc339b708d4a0ba962805828e973af
# (logs/data/seed_5y_full_20261015/).
FIVE_YEAR_PIT_VIEWS_SEED = "data/pit_views_seed_research_5y_full_20261015"

# The hosted arm of a pair: main session, sub-agents and compaction on MiMo, the
# compaction threshold pinned to the local arms' value so both compact alike.
# `nl_model` is left on the local default: text-evidence scoring also runs in
# the forward and Held-out replays, whose data must not leave the machine.
MIMO: dict[str, object] = {
    "model": "mimo-v2.6-flash",
    "subagent_model": "mimo-v2.6-flash",
    "compact_model": "mimo-v2.6-flash",
    "compact_token_threshold": 221_184,
}

# The sequence bag's lineage, one list per stage of its search; an arm inherits
# the stage that had closed when it was created.
#
# Every arm that recorded a non-control trial on the whole-pool sequence recipe
# over the eight years (round 20261001; its pack README §8). The 100k MLP and
# the CSI 1000 GRU recorded controls only and cannot be lineage.
RECIPE_ARMS: tuple[str, ...] = (
    "mlp_alla_8y_20260925",
    "tcn_alla_8y_20260925",
    "lstm_alla_8y_20260926",
    "tattn_alla_8y_20260926",
    "shortmlp_alla_8y_20260926",
    "lagpool_alla_8y_20260926",
    "decay_alla_1m_20260926",
    "patchmlp_alla_100k_20260926",
    "spec_alla_100k_20260926",
    "lstm_alla_100k_20260926",
)
# The recipe arms and the 1m bag (round 20261003).
SEQBAG_LINEAGE = [*RECIPE_ARMS, "seqbag_alla_8y_20261001"]
# Plus every arm that rebuilt the bag as a 100k book (round 20261006): their
# trials are the search's own history on these eight years.
SEQAXES_LINEAGE = [
    *SEQBAG_LINEAGE,
    "seqbag_alla_100k_8y_qwen_20261003",
    "seqbag_alla_100k_8y_mimo_20261003",
    "seqbag_alla_100k_8y_deepseek_20261004",
]
# Plus every arm that has searched the frozen 100k bag as a baseline and closed:
# round 20261006's label and fundamentals-input pairs and round 20261009's
# clock arm (round 20261008's cost and pool pairs and round 20261009's beta
# pair, which were withdrawn before they ran).
SEQBOOK_LINEAGE = [
    *SEQAXES_LINEAGE,
    "seqlabel_alla_100k_8y_qwen_20261006",
    "seqlabel_alla_100k_8y_mimo_20261006",
    "seqfund_alla_100k_8y_qwen_20261006",
    "seqfund_alla_100k_8y_mimo_20261006",
    "seqhold_clock_100k_8y_qwen_20261009",
]
# Plus round 20261008's bag pair, which ran the bag with four and six seeds
# per head; the four-seed bag is the baseline of the lanes after it (round
# 20261011). The two arms' rows are byte-identical, and the gate counts each
# arm's trials, so the pair adds its four trials twice, as the label and
# fundamentals-input pairs above already do.
SEQBAG4_LINEAGE = [
    *SEQBOOK_LINEAGE,
    "seqbag2_bag_100k_8y_qwen_20261008",
    "seqbag2_bag_100k_8y_mimo_20261008",
]

# The title family's first stage: round 20261012's incentive-draft arms, the
# 100k pair and the 500k arm. A later lane on the announcement titles over the
# same eight years inherits all three; the console reads them only once they
# have closed, so such an arm waits for the last of them.
INCDRAFT_LINEAGE = [
    "incdraft_100k_8y_qwen_20261012",
    "incdraft_100k_8y_mimo_20261012",
    "incdraft_500k_8y_qwen_20261012",
]
# Plus the two lanes that read the same titles beside the drafts and have
# closed: round 20261012d's commitment-event map and round 20261012c's title
# ranker. The title family through its second stage (round 20261013).
TITLE_LINEAGE = [
    *INCDRAFT_LINEAGE,
    "titlemap_100k_8y_mimo_20261012",
    "titlerank_100k_8y_qwen_20261012",
]
# Plus round 20261013's board-matched rerun and instrument split of the drafts:
# the company-commitment titles through their third stage (round 20261014).
COMMITMENT_LINEAGE = [*TITLE_LINEAGE, "incdraft2_100k_8y_qwen_20261013"]
# Plus round 20261014's commitment arm, which selected the ESOP-draft book: the
# searches that chose both Paper books, `b30` (incdraft_100k_8y_qwen_20261012)
# and `esop60` (open_commit_100k_8y_qwen_20261014), the fixed sleeves of round
# 20261015's ensemble arm.
SLEEVE_LINEAGE = [*COMMITMENT_LINEAGE, "open_commit_100k_8y_qwen_20261014"]

# Every arm that searched the fundamentals domain over the eight years as a
# source of its own edge and recorded a non-control trial (rounds 20261005 to
# 20261010): the open, event, industry-cycle and learning lanes. Left out:
# fund_learn2_100k_8y_mimo_20261007, stopped before it recorded a trial, and
# round 20261006's fundamentals tower, a block on the sequence bag that belongs
# to the bag's lineage.
FUNDAMENTALS_LINEAGE = [
    "fund_open_100k_8y_qwen_20261005",
    "fund_open_100k_8y_mimo_20261005",
    "fund_event_100k_8y_qwen_20261005",
    "fund_event_100k_8y_mimo_20261005",
    "fund_event3_100k_8y_qwen_20261010",
    "fund_event3_100k_8y_mimo_20261010",
    "indcycle_100k_8y_qwen_20261009",
    "indcycle_100k_8y_mimo_20261009",
    "fund_learn_100k_8y_qwen_20261005",
    "fund_learn_100k_8y_mimo_20261005",
    "fund_learn2_100k_8y_qwen_20261007",
]
