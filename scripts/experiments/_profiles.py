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
