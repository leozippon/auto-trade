"""One launcher for every checked-in round definition.

A round is data: which arms it starts, which reference pack and exploration
directive each arm gets, which PIT view seed and dataset selection they share,
and the handful of parameters that round decides differently from the console
creation form. Everything around that data -- merging the console defaults,
refusing a default drift, running the console's own create-time validation
offline, reporting a dry-run and POSTing the create requests -- is the same for
every round and lives here, so a round file is its arms plus a two-line entry
point.

Two decisions are shared rather than per-round because the console runs rounds
side by side and their verdicts only compare while they agree: BASE_OVERRIDES
carries the research geometry, the account, the cost stress and the
per-session budgets, and BASE_EXPECTED_DEFAULTS pins the console creation
defaults every round relies on -- above all the model roles, which no round
overrides, so a rename of the local model must stop the launcher rather than
silently move an arm onto a hosted stream. A round states what it decides for
itself in `overrides`, and anything it states there stops being an expected
default. What several rounds
send alike -- a geometry, the graduation bars, a model pair's hosted roles, a
lineage -- lives in `_profiles.py`; an open round imports from there and from
here, never from another round file.

A round's id is the day it is launched, in Beijing time, with a letter suffix
(b, c, ...) when that id is already taken -- by an earlier round of the same
day, or by one of the rounds 20261008 to 20261015 that were named ahead of the
calendar between 2026-10-03 and 10-08 -- and its arms' ids end in it. Nothing
reads an order into these ids: the queue is the arm order of a round file,
lineages name their arms, and RETIRED_IDS and the reuse check compare whole
ids.

A round whose arms have all been created is marked `closed`, and so is one
whose remaining arms are withdrawn: they leave its arm list and their ids go
to RETIRED_IDS. Its `main` refuses every mode: once an arm is archived its
directory is gone, and the queue would read it as pending and create it again.
Once every arm of a closed round has ended, its file leaves the tree (git
history keeps it) and its ids join RETIRED_IDS.

`normalize` runs the request-level checks the console applies on POST
/api/experiments (ExperimentManager.create_experiment's closed, unknown,
required and id rules, then the worker's own resolve_worker_options pre-flight,
which type-checks every knob, refuses a malformed research geometry and, for a
named PIT view seed, refuses a tree whose cache format or snapshot
configuration is not this round's or whose prebuild has not finished,
and reads the research release the seed was built from -- the release every
arm will pin -- refusing it unless it is published here and reaches Held-out).
The console's deployment-state checks -- an experiment directory that already
exists and a free running slot -- can only be decided against the live server
and stay at POST time.

`--dry-run` judges the round before its arms: a probe request carrying only
what every arm shares goes through the same pre-flight, so the geometry and the
seed contract are read even for a round that has no arms yet. Such a round can
be dry-run but not created. An arm that names `lineage_arms` passes the same
pre-flight only when every one of them exists, researched the same period and
recorded a non-control trial, and its dry-run reading adds a `lineage:` line:
the trials and effective trials the console will record for it, and the IR bar
at that count alone.

`--fill` reads the arm list as an ordered queue and keeps the console's running
slots occupied: it asks /api/health how many are free under each of the two
running-arm limits and which GPUs a GPU arm could take, skips the arms whose
experiment directory already exists -- running, completed and failed alike, the
console's own rule -- and creates the next pending ones in file order through
the same validation and POST path. A local arm reached while the local-model
limit is full, and a GPU arm reached while too few cards are free, stays
pending without stopping the queue, so the arms behind it still take the free
slots; so does an arm whose PIT view seed has no contract yet or whose
prebuild has not finished, which the next run creates once the seed reports
status ok. Nothing pending or nothing free is the steady state and exits 0, so
the mode is idempotent and safe on a timer; only a creation that was attempted
and refused exits non-zero. A timer run writes one timestamped summary line --
slots under both limits and the free GPUs, and how many arms were skipped,
created, refused and are still pending, and how many of those the local-model
limit, the free GPUs and an unfinished seed hold back -- plus one line per arm
it created or that was refused; with `--dry-run` it also lists each pending
arm and whether it takes a free slot. The research cron
(`ops/cron/research_fill.cron`) runs it over several round files in turn.

Before an arm is created, and on every dry-run reading of it, `smoke` replays
its pack's starter through the host's own evaluator on the arm's seed: a few
days from each research-year start inside the full span, and the full span
once. A seed whose views break the replay (round 20261015's five-year seed
failed every replay crossing 2022-07-01, the untouched starter included)
refuses the arm with the failure instead of letting a session discover it.
The outcome is recorded under logs/research/smoke/, keyed by the seed, its
contract, the geometry and the starter's bytes: a pass is not replayed again,
and a failure keeps refusing until its record is deleted.

RETIRED_IDS records the experiment ids that have been used and archived, or
withdrawn from a round's queue, or defined by a round whose file has left the
tree, so a new round cannot quietly reuse one.
`logs/archive/` is not part of the repository, which is why the list is
checked in rather than read from disk; `archived_ids` reads the archive
where it exists so the two can be compared.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import rmtree_keeping_file_modes
from autotrade.pipelines.calendar import FULL_SPAN
from autotrade.pipelines.config import (
    SNAPSHOT_CACHE_FORMAT_VERSION,
    AcceptanceRules,
    ArtifactRevision,
    acceptance_for,
    research_span,
)
from autotrade.pipelines.experiment import lineage_summary
from autotrade.pipelines.hitl_state import (
    WEB_CLOSED_PARAMS,
    WEB_CREATE_DEFAULTS,
    WEB_INTERNAL_PARAMS,
    WEB_REQUIRED_PARAMS,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.lineage import extract_lineage
from autotrade.pipelines.pit_views_seed import UNFINISHED_BUILD_MARKER
from autotrade.pipelines.verdict import information_ratio_bar
from autotrade.pipelines.worker import build_experiment_pipeline, resolve_worker_options

# The console's own id, local-arm and GPU-request rules; importing them keeps
# this module from growing a second copy of the create contract.
from autotrade.webui.manager import _ID as EXPERIMENT_ID_RE
from autotrade.webui.manager import gpu_request, uses_local_model

EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
ARCHIVE_ROOT = REPO_ROOT / "logs" / "archive"
# Where an arm's reference pack lives when the arm does not name one.
PACKS_DIR = "configs/workspace_refs"
# The id the round-level dry-run validates under. Never created: it only
# carries the parameters every arm shares through the pre-flight.
PROBE_ID = "round_dry_run_probe"

# Console defaults every round relies on. Values, not commentary: if the console
# changes any of them the round has to be re-decided, not silently re-run.
BASE_EXPECTED_DEFAULTS: dict[str, object] = {
    "include_fundamentals": True,
    "include_macro": True,
    "include_events": True,
    "include_text": True,
    "include_intraday": False,
    "screen_exclude_st": False,
    "screen_exclude_new_listed_days": 0,
    "screen_boards": (),
    "screen_min_circ_mv_yi": None,
    "screen_max_circ_mv_yi": None,
    # Both the packs and the directives promise the Agent this many host null
    # controls over the research session.
    "max_null_controls": 12,
    # Replay-years the research session may spend on validations; the packs
    # size their rounds against it.
    "max_replay_years": 96,
    # Derived from SandboxLimits.fit_timeout_seconds; packs promise the Agent a
    # fit budget of this size.
    "strategy_fit_timeout_seconds": 3600,
    "operating_memory": "curated+graduated",
    "initial_control_mode": "auto",
    "reasoning_effort": "xhigh",
    "inference_time": "08:30",
    "strategy_period": "day",
    # Every model role. An arm on the local model overrides none of them (a
    # hosted arm overrides its own, `_profiles.MIMO`), so the console default
    # is what actually decides them; spelled out as literals on purpose, since
    # a rename of the local model is exactly the drift this has to catch.
    "model": "qwen-3.8-27b-fp8",
    "subagent_model": "qwen-3.8-27b-fp8",
    "nl_model": "qwen-3.8-27b-fp8",
    "compact_model": "qwen-3.8-27b-fp8",
    # Left to the console, which stamps the boards the arm's initial cash
    # qualifies for (`broker.default_permitted_boards`); a round that wants
    # others names them.
    "permitted_boards": None,
}

# What every round decides the same way. Values, not commentary.
BASE_OVERRIDES: dict[str, object] = {
    # Research on four July-June years, the twelve forward months after them,
    # and a Held-out quarter the replay clips to the release end. Required by
    # the console, and the geometry the PIT view seeds are planned over.
    "research_start": "20210701",
    "research_end": "20250630",
    "forward_end": "20260630",
    "heldout_end": "20260930",
    # Sixty months behind the research-end decision view reach 2020-07, just
    # inside the 2020 floor of fundamentals and macro; part of every seed's
    # snapshot configuration.
    "window_months": 60,
    # No GPU. The request travels with the experiment to the Agent session
    # sandbox and the strategy container of every replay alike; an arm that
    # really trains on a card overrides this with 1.
    "gpu_count": 0,
    # The researcher's real account, where the CNY 5 minimum commission and lot
    # sizes are a real cost rather than a rounding error.
    "initial_cash": 100_000,
    # The forward verdict's cost stress. The drawdown limits and the tracking
    # mandate are not stated here: an arm gets a mandate only by naming its own
    # `tracking_error_cap` (config.acceptance_for). Unnamed drawdowns stay
    # 0.45 / 0.30; an arm that wants other bars names them.
    "cost_stress_multiplier": 2.0,
    # The one research session's budgets, spent across every attempt.
    "max_research_minutes": 2400,
    "max_llm_calls": 6400,
}

# Reported once per round on --dry-run: what every arm shares.
ROUND_REPORT_KEYS: tuple[str, ...] = (
    "research_start",
    "research_end",
    "forward_end",
    "heldout_end",
    "pit_views_seed",
    "benchmark_index",
    "window_months",
    "include_fundamentals",
    "fundamental_datasets",
    "include_macro",
    "macro_datasets",
    "include_events",
    "events_datasets",
    "include_text",
    "text_datasets",
    "include_intraday",
    "operating_memory",
    "max_research_minutes",
    "max_replay_years",
    "max_llm_calls",
    "max_null_controls",
    "strategy_fit_timeout_seconds",
    "initial_cash",
    # null: the console stamps the boards the initial cash qualifies for.
    "permitted_boards",
    "cost_stress_multiplier",
    "gpu_count",
    "reasoning_effort",
    "initial_control_mode",
    "inference_time",
    "strategy_period",
    "model",
    "subagent_model",
    "nl_model",
    "compact_model",
)

# Experiment ids that were used and archived, or withdrawn from a round's
# queue. An id is never reused: the console keys the experiment directory,
# the sandbox work root, the Docker image tag and the archive path on it, so
# a second run under an old name would be indistinguishable from the first
# in every record that survives it.
RETIRED_IDS: frozenset[str] = frozenset(
    {
        "alpha158_lgbm_20260920",
        "alt_events_ranker_20260916",
        "alt_events_ranker_20260917",
        "analyst_revision_20260916",
        "analyst_revision_20260917",
        "cb_linkage_20260914",
        "corner_cases_20260907",
        "corner_cases_20260910",
        "defensive_quality_20260918",
        "defensive_quality_20260920",
        "earnings_surprise_20260918",
        "explore_github_strategies_20260910",
        "explore_platform_strategies_20260910",
        "factor_cs_20260910",
        "factor_cs_allflash_20260910",
        # Created by round 20261007, withdrawn from it and deleted unarchived.
        "fund_event2_100k_8y_mimo_20261007",
        "fund_event2_100k_8y_qwen_20261007",
        # Queued by rounds 20261008, 20261009 and 20261011b and withdrawn on
        # 2026-10-05 before any recorded a validation; the five that had been
        # created were deleted unarchived. The round files say why.
        "fund_open2_100k_8y_mimo_20261011",
        "fund_open2_100k_8y_qwen_20261011",
        "seqbag2_cost_100k_8y_mimo_20261008",
        "seqbag2_cost_100k_8y_qwen_20261008",
        "seqbag2_pool_100k_8y_mimo_20261008",
        "seqbag2_pool_100k_8y_qwen_20261008",
        "seqhold_beta_100k_8y_mimo_20261009",
        "seqhold_beta_100k_8y_qwen_20261009",
        "github_confirm_20260917",
        "gru_ranker_20260920",
        "margin_flow_20260916",
        "margin_flow_20260917",
        "ml_ranker_20260910",
        "open_mechanism_20260910",
        "open_research_20260920",
        "order_flow_ranker_20260917",
        "site_visits_20260914",
        "value_regime_20260914",
        # Every arm of the closed rounds 20260916 to 20261010, whose files were
        # removed from the tree on 2026-10-06 (last in 0b1b07e); 211 of them were
        # archived on 2026-10-04, the rest finished and stay on disk.
        "abturn_chinext_1m_20260926",
        "age1k_alla_100k_20260927",
        "ageout_alla_100k_20260927",
        "alpha158_lgbm_20260921",
        "amp_momentum_csi500_1m_20260926",
        "amtac_alla_100k_20260927",
        "amtcv_alla_100k_20260927",
        "amtind_alla_100k_20260927",
        "amtlag_alla_100k_20260927",
        "at1k_alla_100k_20260927",
        "atcy_alla_100k_20260927",
        "atlag_alla_100k_20260927",
        "aturn_alla_100k_20260927",
        "beta1k_alla_100k_20260927",
        "beta60_alla_100k_20260927",
        "big500_alla_100k_20260927",
        "bigind_alla_100k_20260927",
        "board_block_1m_20260923",
        "body_alla_100k_20260927",
        "boundary_alla_100k_20260927",
        "broad_alla_100k_20260927",
        "capital_aware_100k_concentrated_20260921",
        "capital_aware_100k_pv_20260921",
        "capital_aware_1m_enhanced_20260921",
        "capital_aware_1m_riskmodel_20260921",
        "cashdiv_pool_100k_20260921d",
        "cashmix_alla_100k_20260927",
        "census_gbdt_alla_8y_20261001",
        "census_gbdt_csi1000_100k_8y_20261001",
        "census_lin_csi1000_100k_8y_deepseek_20261004",
        "census_lin_csi1000_100k_8y_mimo_20261002",
        "census_lin_csi1000_100k_8y_qwen_20261002",
        "ch3bmk_alla_100k_20260927",
        "champ1000_alla_100k_20260927",
        "champ300_alla_100k_20260927",
        "champ_alla_100k_20260927",
        "champbmk_alla_100k_20260927",
        "chdelta_alla_100k_20260927",
        "cma_overlay_1m_20260921d",
        "csi500_shape_1m_20260923",
        "csk1k_alla_100k_20260927",
        "decay_alla_1m_20260926",
        "decouple_alla_100k_20260927",
        "defensive_quality_20260921",
        "delay1k_alla_100k_20260927",
        "delay60_alla_100k_20260927",
        "dn1000_alla_100k_20260927",
        "dn20_alla_100k_20260927",
        "dn60_alla_100k_20260927",
        "dvy_csi1000_100k_20260924",
        "dvy_csi1000_100k_b3_20260925",
        "dvy_csi1000_100k_excy_20260925",
        "dvy_csi1000_100k_q_20260925",
        "dvy_csi1000_100k_s20_20260925",
        "dvy_csi1000_100k_s8_20260925",
        "dvy_csi1000_1m_20260924",
        "dvy_csi1000_1m_b25_20260925",
        "dvy_csi1000_1m_excy_20260925",
        "dvy_csi1000_1m_q_20260925",
        "dvy_csi1000_1m_s12_20260925",
        "dvy_csi1000_1m_s30_20260925",
        "dvy_csi300_8y_20260927",
        "dvy_csi500_100k_20260924",
        "dvy_csi500_1m_20260924",
        "dvy_csi500_1m_q_20260925",
        "earnac_alla_100k_20260927",
        "entry_alla_100k_20260927",
        "ep_csi500_8y_20260927",
        "event_screens_20260917",
        "exit1000_alla_100k_20260927",
        "exit300_alla_100k_20260927",
        "exit500_alla_100k_20260927",
        "exit50_alla_100k_20260927",
        "exitbmk_alla_100k_20260927",
        "exitcy_alla_100k_20260927",
        "exitwide_alla_100k_20260927",
        "eyield_concentrated_100k_20260921b",
        "flip_alla_8y_20261001",
        "flow_block_1m_20260923",
        "freefloat_alla_100k_20260927",
        "freegrow_alla_100k_20260927",
        "freeshare_alla_100k_20260927",
        "fresh_book_1m_20260925",
        "fresh_book_csi500_1m_20260925",
        "fullcash_overlay_1m_20260921b",
        "fund_event3_100k_8y_mimo_20261010",
        "fund_event3_100k_8y_qwen_20261010",
        "fund_event_100k_8y_mimo_20261005",
        "fund_event_100k_8y_qwen_20261005",
        "fund_learn2_100k_8y_mimo_20261007",
        "fund_learn2_100k_8y_qwen_20261007",
        "fund_learn_100k_8y_mimo_20261005",
        "fund_learn_100k_8y_qwen_20261005",
        "fund_open_100k_8y_mimo_20261005",
        "fund_open_100k_8y_qwen_20261005",
        "gnn_csi1000_8y_20260925",
        "gnn_relation_1m_20260922",
        "gp_overlay_1m_20260921d",
        "gru_csi1000_8y_20260925",
        "gru_ranker_20260921",
        "high52_overlay_1m_20260921c",
        "iat1k_alla_100k_20260927",
        "indact_alla_100k_20260927",
        "indcalm_alla_100k_20260927",
        "indcycle_100k_8y_mimo_20261009",
        "indcycle_100k_8y_qwen_20261009",
        "inddisp_alla_100k_20260927",
        "index_migration_csi500_1m_20260926",
        "index_relative_100k_20260919",
        "index_relative_1m_20260919",
        "index_relative_1m_b_20260919",
        "indflow_alla_100k_20260927",
        "indmv_alla_100k_20260927",
        "indneutral_value_1m_20260921b",
        "indpb_alla_100k_20260927",
        "indpbl_alla_100k_20260927",
        "indrev_alla_100k_20260927",
        "indrise_alla_100k_20260927",
        "indto_alla_100k_20260927",
        "intraday_stats_block_1m_20260924",
        "iroe_alla_100k_20260927",
        "iwadd_alla_100k_20260927",
        "label_axis_100k_20260923",
        "lagpool_alla_8y_20260926",
        "leader_alla_100k_20260927",
        "learner_axis_1m_20260924",
        "lgbm_csi1000_8y_20260925",
        "lgbm_meta_100k_20260922",
        "limdn_alla_100k_20260927",
        "liqco_alla_100k_20260927",
        "listed_alla_100k_20260927",
        "lock1000_alla_100k_20260927",
        "lottery_reverse_100k_20260921c",
        "lowvol_bucketed_20260917",
        "lowvol_quality_20260916",
        "lstm_alla_100k_20260926",
        "lstm_alla_8y_20260926",
        "market_state_block_1m_20260924",
        "mgnlag_alla_100k_20260927",
        "mid1000_alla_100k_20260927",
        "mid300_alla_100k_20260927",
        "mid3bmk_alla_100k_20260927",
        "mid50_alla_100k_20260927",
        "midbmk_alla_100k_20260927",
        "midcy_alla_100k_20260927",
        "midflat_alla_100k_20260927",
        "midnew_alla_100k_20260927",
        "mlp_alla_100k_20260926",
        "mlp_alla_8y_20260925",
        "nearvw_alla_100k_20260927",
        "net_issuance_100k_20260921c",
        "new300_alla_100k_20260927",
        "new500_alla_100k_20260927",
        "newbig_alla_100k_20260927",
        "newbmk_alla_100k_20260927",
        "newcy_alla_100k_20260927",
        "newmem_alla_100k_20260927",
        "next1k_alla_100k_20260927",
        "nextlo_alla_100k_20260927",
        "nextmb_alla_100k_20260927",
        "nlead_alla_100k_20260927",
        "noa_index_100k_20260921d",
        "northbound_holdings_1m_20260925",
        "open_research_100k_8y_deepseek_r1_20261004",
        "open_research_100k_8y_deepseek_r2_20261004",
        "open_research_100k_8y_deepseek_r3_20261004",
        "open_research_100k_8y_mimo_20261002",
        "open_research_100k_8y_mimo_r2_20261003",
        "open_research_100k_8y_mimo_r3_20261004",
        "open_research_100k_8y_qwen_20261002",
        "open_research_100k_8y_qwen_r2_20261003",
        "open_research_100k_8y_qwen_r3_20261004",
        "open_research_20260916",
        "open_research_20260917",
        "open_research_20260917b",
        "open_research_20260917c",
        "open_research_20260917d",
        "open_research_20260917e",
        "open_research_20260917f",
        "open_research_20260917g",
        "open_research_20260917h",
        "open_research_20260917i",
        "open_research_20260917j",
        "open_research_20260921",
        "open_research_8y_20261001",
        "pacc_index_100k_20260921c",
        "patchmlp_alla_100k_20260926",
        "paylag_alla_100k_20260927",
        "pb1000_alla_100k_20260927",
        "pb_alla_100k_20260927",
        "pblagm_alla_100k_20260927",
        "pe1000_alla_100k_20260927",
        "pe_alla_100k_20260927",
        "peerlag_alla_100k_20260927",
        "peerlead_alla_100k_20260927",
        "peerliq_alla_100k_20260927",
        "peersize_alla_100k_20260927",
        "pelagm_alla_100k_20260927",
        "pond_alla_100k_20260927",
        "ps1000_alla_100k_20260927",
        "ps1ind_alla_100k_20260927",
        "ps1sz_alla_100k_20260927",
        "ps300_alla_100k_20260927",
        "ps500_alla_100k_20260927",
        "ps_alla_100k_20260927",
        "psacc_alla_100k_20260927",
        "pscy_alla_100k_20260927",
        "psgap_alla_100k_20260927",
        "psgem_alla_100k_20260927",
        "psind_alla_100k_20260927",
        "psl10_alla_100k_20260927",
        "psl300_alla_100k_20260927",
        "psl42_alla_100k_20260927",
        "psl500_alla_100k_20260927",
        "pslag_alla_100k_20260927",
        "pslage_alla_100k_20260927",
        "pslagm_alla_100k_20260927",
        "pslbig_alla_100k_20260927",
        "pslboth_alla_100k_20260927",
        "pslcy_alla_100k_20260927",
        "pslly_alla_100k_20260927",
        "psloutm_alla_100k_20260927",
        "pslsml_alla_100k_20260927",
        "psltail_alla_100k_20260927",
        "pslthen_alla_100k_20260927",
        "psly1k_alla_100k_20260927",
        "psly500_alla_100k_20260927",
        "psly_alla_100k_20260927",
        "psmain_alla_100k_20260927",
        "psmly_alla_100k_20260927",
        "psnew_alla_100k_20260927",
        "psout_alla_100k_20260927",
        "pssize_alla_100k_20260927",
        "psstay_alla_100k_20260927",
        "psxfin_alla_100k_20260927",
        "pv_exposure_budget_20260917",
        "pvcorr_alla_100k_20260927",
        "quality_lowrisk_20260916",
        "quality_overlay_1m_20260921c",
        "range20_csi1000_100k_20260926",
        "range20_csi1000_1m_20260926",
        "range20_csi300_8y_20260927",
        "range20_csi500_1m_20260926",
        "range20_csi500_8y_20260927",
        "reldn_alla_100k_20260927",
        "resid_momentum_100k_20260921b",
        "rmax_overlay_1m_20260921c",
        "roe1k_alla_100k_20260927",
        "roeout_alla_100k_20260927",
        "room_alla_100k_20260927",
        "runner_alla_100k_20260927",
        "sale1000_alla_100k_20260927",
        "sale500_alla_100k_20260927",
        "salem_alla_100k_20260927",
        "seasonality_csi500_1m_20260926",
        "seed_bag_1m_20260926",
        "seq_gru_persist_1m_20260922",
        "seqbag2_bag_100k_8y_mimo_20261008",
        "seqbag2_bag_100k_8y_qwen_20261008",
        "seqbag_alla_100k_8y_deepseek_20261004",
        "seqbag_alla_100k_8y_mimo_20261003",
        "seqbag_alla_100k_8y_qwen_20261003",
        "seqbag_alla_8y_20261001",
        "seqfund_alla_100k_8y_mimo_20261006",
        "seqfund_alla_100k_8y_qwen_20261006",
        "seqhold_clock_100k_8y_qwen_20261009",
        "seqlabel_alla_100k_8y_mimo_20261006",
        "seqlabel_alla_100k_8y_qwen_20261006",
        "shortmlp_alla_8y_20260926",
        "sleeve_blend_20260918",
        "spec_alla_100k_20260926",
        "ssch1000_alla_100k_20260927",
        "stay500_alla_100k_20260927",
        "swpe_alla_100k_20260927",
        "sz1k_alla_100k_20260927",
        "tattn_alla_8y_20260926",
        "tcn_alla_8y_20260925",
        "tom1k_alla_100k_20260927",
        "volag_alla_100k_20260927",
        "wdelta_alla_100k_20260927",
        "wresid_alla_100k_20260927",
        "wtheavy_alla_100k_20260927",
        "wtstab_alla_100k_20260927",
        "xs_transformer_1m_20260922",
        "xsattn_csi1000_8y_20260925",
        # Every arm of the closed rounds 20261011, 20261012, 20261012b,
        # 20261012c and 20261012d, whose files were removed from the tree on
        # 2026-10-06 (last in 6a0f487; 20261012b's file and pack in 0144b7c;
        # 20261013's file and pack in 315e7c5); all finished and stay on disk
        # or in the archive.
        "incdraft2_100k_8y_qwen_20261013",
        "incdraft_100k_8y_mimo_20261012",
        "incdraft_100k_8y_qwen_20261012",
        "incdraft_500k_8y_qwen_20261012",
        "seqfresh_100k_8y_mimo_20261011",
        "seqfresh_100k_8y_qwen_20261011",
        "seqnovel_cl_100k_8y_qwen_20261012",
        "seqnovel_ssm_100k_8y_mimo_20261012",
        "titlemap_100k_8y_mimo_20261012",
        "titlerank_100k_8y_qwen_20261012",
        # Every arm of the closed round 20261015, whose file and packs left the
        # tree with round 20261008b's launch; all ended no_edge and stay on disk.
        "open_chips_100k_5y_mimo_20261015",
        "open_cluster_100k_8y_mimo_20261015",
        "open_distress_100k_8y_qwen_20261015",
        "open_ensemble_100k_8y_qwen_20261015",
        "open_micro_100k_8y_mimo_20261015",
        "open_tables_100k_5y_qwen_20261015",
    }
)


def archived_ids() -> set[str]:
    """Experiment ids with an archived tree under ``logs/archive/<batch>/<id>``.

    Empty where the archive was never created -- it is a local operator
    artifact, not part of the repository -- so a caller must treat an empty
    result as "no evidence", never as "nothing was archived".
    """

    if not ARCHIVE_ROOT.is_dir():
        return set()
    return {
        arm.name
        for batch in ARCHIVE_ROOT.iterdir()
        if batch.is_dir() and not batch.is_symlink()
        for arm in batch.iterdir()
        if arm.is_dir() and not arm.is_symlink()
    }


def normalize(params: dict[str, object]) -> dict[str, object]:
    """Run the console's create-time validation offline and return params.json.

    Same order and same request-level checks as
    ExperimentManager.create_experiment, then the worker's own
    resolve_worker_options pre-flight. The console's duplicate-directory and
    running-slot checks need the live deployment and stay at POST time.
    """
    closed = sorted(set(params) & WEB_CLOSED_PARAMS)
    if closed:
        raise ValueError("console-managed parameters are not accepted: " + ", ".join(closed))
    unknown = sorted(set(params) - set(WEB_CREATE_DEFAULTS))
    if unknown:
        raise ValueError("unknown experiment parameters: " + ", ".join(unknown))
    merged = {**WEB_CREATE_DEFAULTS, **params}
    missing = sorted(key for key in WEB_REQUIRED_PARAMS if merged.get(key) in (None, ""))
    if missing:
        raise ValueError("missing required experiment parameters: " + ", ".join(missing))
    experiment_id = str(params.get("experiment_id") or "").strip()
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError("experiment_id must match [A-Za-z0-9][A-Za-z0-9_-]{0,99}")
    merged.update(
        {
            **WEB_INTERNAL_PARAMS,
            "experiment_id": experiment_id,
            "experiments_root": str(EXPERIMENTS_ROOT),
            "work_root": str(REPO_ROOT / ".runtime/sandboxes"),
            "_creation_surface": "webui",
        }
    )
    resolve_worker_options(
        merged,
        experiment_dir=EXPERIMENTS_ROOT / experiment_id,
        repo_root=REPO_ROOT,
        preflight=True,
    )
    return merged


# Trading days each research-year start is replayed for by the smoke.
SMOKE_DAYS = 3
# What decides a starter's replay besides its own bytes and its seed's contract.
SMOKE_KEYS = (
    "pit_views_seed",
    "research_start",
    "research_end",
    "window_months",
    "benchmark_index",
    "initial_cash",
    "permitted_boards",
)


def smoke(merged: Mapping[str, object]) -> str:
    """Replay an arm's starter on its seed before the arm is created; "" when it
    passed, else why not.

    ``merged`` is the arm's validated params.json. The pack's ``starter/`` is
    replayed by the host's own evaluator as a session's smoke_backtest and
    Validation replay it: over the full research span, a few days from each
    research-year start (the late-start path a defective seed breaks first),
    then the whole span. The outcome is recorded once per seed contract,
    geometry and starter bytes, so a pass is not replayed by the next timer run
    and a failure is not retried until its record is deleted.
    """

    starter = REPO_ROOT / str(merged["workspace_reference"]) / "starter"
    if not (starter / "main.py").is_file():
        return f"the pack has no starter to smoke: {starter} lacks main.py"
    digest = hashlib.sha256(
        json.dumps({key: merged.get(key) for key in SMOKE_KEYS}, sort_keys=True, default=str).encode("utf-8")
    )
    seed = str(merged.get("pit_views_seed") or "")
    if seed:
        digest.update((REPO_ROOT / seed / "provider.json").read_bytes())
    for path in sorted(starter.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(starter)).encode("utf-8") + b"\0" + path.read_bytes())
    record_path = REPO_ROOT / "logs" / "research" / "smoke" / f"{digest.hexdigest()[:20]}.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["passed"]:
            return ""
        return f"{record['failure']} (recorded {record_path.relative_to(REPO_ROOT)}; delete it to smoke again)"
    started = time.monotonic()
    failure = _smoke_replays(merged, starter)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "experiment_id": merged["experiment_id"],
                "starter": str(starter.relative_to(REPO_ROOT)),
                **{key: merged.get(key) for key in SMOKE_KEYS},
                "smoke_days": SMOKE_DAYS,
                "passed": not failure,
                "failure": failure,
                "seconds": round(time.monotonic() - started, 1),
                "at": _now(),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return failure


def _smoke_replays(merged: Mapping[str, object], starter: Path) -> str:
    """The replays of :func:`smoke`, in a scratch experiment assembled exactly
    as the worker and incubate.py assemble one (its seed hardlinked in, the
    configured sandbox image, no Agent run); "" or the first failure."""

    scratch = REPO_ROOT / ".runtime" / "round_smoke" / uuid.uuid4().hex
    experiment_dir = scratch / str(merged["experiment_id"])
    experiment_dir.mkdir(parents=True)
    try:
        options = resolve_worker_options(
            {**merged, "experiments_root": str(scratch)}, experiment_dir=experiment_dir, repo_root=REPO_ROOT
        )
        pipeline = build_experiment_pipeline(
            options,
            ledger=ExperimentLedger(experiment_dir / "ledgers" / "experiment_ledger.jsonl"),
            store=FilesystemArtifactStore(experiment_dir / "artifacts" / "strategy"),
            ref_store=AgentRefStore(experiment_dir),
        ).pipeline
        output = scratch / "output"
        shutil.copytree(starter, output, ignore=shutil.ignore_patterns("__pycache__"))
        _, years = pipeline.research_inputs()
        request = research_span(years, FULL_SPAN).request(
            ArtifactRevision("smoke", output),
            schedule=options.rolling.schedule,
            broker_profile=options.rolling.broker_profile,
        )
        runs = [(year.label, year.start, SMOKE_DAYS) for year in years] + [(FULL_SPAN, None, None)]
        for label, start, days in runs:
            try:
                pipeline.evaluator.evaluate(request, max_days=days, start_day=start)
            except Exception as exc:  # noqa: BLE001 - the exception text IS the result
                where = f"{days} days from {start}" if start else f"the full span {request.start}..{request.end}"
                return f"{label}, {where}: {type(exc).__name__}: {exc}"
        return ""
    except Exception as exc:  # noqa: BLE001 - an arm that cannot even be assembled fails its smoke
        return f"the smoke could not be assembled: {type(exc).__name__}: {exc}"
    finally:
        rmtree_keeping_file_modes(scratch)


def seed_ready(seed: str) -> bool:
    """Whether a named PIT view seed has its contract and a finished prebuild;
    the console default (no seed) always is. Readiness alone: whether the tree
    is this arm's is the pre-flight's to judge."""

    root = REPO_ROOT / seed
    return not seed or ((root / "provider.json").is_file() and not (root / UNFINISHED_BUILD_MARKER).exists())


def _lineage_reading(merged: Mapping[str, object], rules: AcceptanceRules) -> dict[str, object]:
    """What an arm's ``lineage_arms`` would add to its freeze gate.

    The figures the console records when it creates the arm, and the IR bar at
    the lineage's effective trial count over the longest lineage series: a
    floor, since each trial of the arm's own can only raise it.
    """
    extraction = extract_lineage(
        EXPERIMENTS_ROOT,
        list(merged["lineage_arms"]),  # type: ignore[call-overload]
        research_start=str(merged["research_start"]),
        research_end=str(merged["research_end"]),
    )
    summary = lineage_summary(extraction)
    days = max((len(item["daily"]) for item in extraction["series"]), default=0)  # type: ignore[attr-defined]
    return {
        # The arms themselves are on the line above, as `lineage_arms`.
        **{key: value for key, value in summary.items() if key != "arms"},
        "information_ratio_bar_floor": (
            information_ratio_bar(
                float(summary["effective_trials"]), days, rules.min_dsr_probability  # type: ignore[arg-type]
            )
            if days >= 2
            else None
        ),
        "bar_days": days,
    }


def _now() -> str:
    """Local time with its offset: the cron log's runs are told apart by it."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def post(port: int, params: dict[str, object]) -> bool:
    """POST one create request; print one line and report whether the console
    accepted it."""
    experiment_id = params["experiment_id"]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/experiments",
        data=json.dumps(params).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Flushed: stdout and stderr share the cron log, and a buffered line would
    # land after the refusals that followed it.
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read(400).decode("utf-8", "replace")
            print(f"{experiment_id}: created, HTTP {response.status} {body}", flush=True)
            return True
    except urllib.error.HTTPError as exc:
        body = exc.read(800).decode("utf-8", "replace")
        print(f"{experiment_id}: refused, HTTP {exc.code} {body}", file=sys.stderr)
    except urllib.error.URLError as exc:
        # No console on that port, or it dropped the connection: an operator
        # error, not a traceback.
        print(f"{experiment_id}: not sent, console unreachable: {exc.reason}", file=sys.stderr)
    return False


# What --fill reads from /api/health.
HEALTH_KEYS = (
    "max_running_experiments",
    "max_running_local_experiments",
    "running",
    "running_local",
    "gpus_free",
)


def health(port: int) -> dict[str, object]:
    """The console's running roster, its two running-arm limits and its free GPUs.

    Read rather than assumed: the limits are the console's constants and the
    roster and the cards change under the operator's hands, so a fill that
    cannot read them -- a console too old to report them included -- refuses
    instead of creating blind against limits it guessed.
    """
    url = f"http://127.0.0.1:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            record = json.loads(response.read())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{_now()} console health unreadable at {url}: {exc}") from exc
    missing = [key for key in HEALTH_KEYS if key not in record]
    if missing:
        raise SystemExit(
            f"{_now()} console health at {url} lacks {', '.join(missing)}: "
            "restart the console onto this checkout"
        )
    return record


@dataclass(frozen=True)
class Round:
    """One round definition: its arms and what it decides differently.

    ``arms`` maps experiment id to the per-arm part of the create request --
    normally ``research_directive`` (the pack defaults to the one named after
    the arm, see :meth:`request_params`), plus any parameter that arm alone
    changes. ``overrides`` is what the whole round decides on top of
    BASE_OVERRIDES -- normally its dataset selection, and any default every arm
    shares -- and ``pit_views_seed`` the prebuilt view tree every arm hardlinks
    from. ``closed`` says every arm has been created: the file is then only
    the record of what was run, and :meth:`main` refuses it.
    """

    arms: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    overrides: Mapping[str, object] = field(default_factory=dict)
    # Empty means the console default seed, which carries the default dataset
    # selection.
    pit_views_seed: str = ""
    # Stated, not read from disk: an archived arm has no directory and would
    # look pending.
    closed: bool = False

    def __post_init__(self) -> None:
        reused = sorted(set(self.arms) & RETIRED_IDS)
        if reused:
            raise ValueError(
                "experiment ids that were already used and archived cannot be "
                "reused: " + ", ".join(reused)
            )

    @property
    def common_overrides(self) -> dict[str, object]:
        """What every arm of this round sends, seed included."""
        overrides = {**BASE_OVERRIDES, **self.overrides}
        if self.pit_views_seed:
            overrides["pit_views_seed"] = self.pit_views_seed
        return overrides

    @property
    def expected(self) -> dict[str, object]:
        """Console defaults this round still relies on.

        Anything the round decides for itself stops being a default it depends
        on, so it is dropped here rather than pinned twice.
        """
        common = self.common_overrides
        return {key: value for key, value in BASE_EXPECTED_DEFAULTS.items() if key not in common}

    def check_console_defaults(self) -> None:
        drift = {
            key: (value, WEB_CREATE_DEFAULTS[key])
            for key, value in self.expected.items()
            if WEB_CREATE_DEFAULTS[key] != value
        }
        if drift:
            raise SystemExit(
                "console creation defaults drifted from what this round assumes; "
                "re-decide the round before creating: "
                + json.dumps(
                    {k: {"round": str(v[0]), "console": str(v[1])} for k, v in drift.items()},
                    ensure_ascii=False,
                )
            )

    def request_params(self, experiment_id: str) -> dict[str, object]:
        """The create request body: console defaults, the round's decisions, the id.

        ``PROBE_ID`` stands for the part every arm shares; any other id must be
        one of the round's arms. An arm's pack is the one named after it,
        ``configs/workspace_refs/<id>``, unless the arm or the round names
        another (``""`` for none); the pre-flight refuses one that is missing.
        """
        base = {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in WEB_CREATE_DEFAULTS.items()
        }
        common = self.common_overrides
        arm = {} if experiment_id == PROBE_ID else dict(self.arms[experiment_id])
        if experiment_id != PROBE_ID and "workspace_reference" not in {*arm, *common}:
            arm["workspace_reference"] = f"{PACKS_DIR}/{experiment_id}"
        return {**base, **common, **arm, "experiment_id": experiment_id}

    def validated(self, experiment_id: str) -> tuple[dict[str, object] | None, str]:
        """params.json for one arm (or the probe), or ``(None, reason)``.

        The refusal is returned rather than raised because a dry-run should
        read every arm before it stops.
        """
        try:
            return normalize(self.request_params(experiment_id)), ""
        except ValueError as exc:
            reason = f"{experiment_id}: parameters rejected, nothing was sent: {exc}"
            if "unfinished build" in str(exc):
                reason += (
                    "\n  the tree is there and its contract is this round's, but its"
                    " prebuild has not finished: wait for a running"
                    " scripts/data/prebuild_pit_views_seed.py, or re-run one that"
                    " failed, until it reports status ok, then re-run --"
                    " no parameter needs changing"
                )
            elif "pit_views_seed" in str(exc) or "view seed" in str(exc):
                reason += (
                    f"\n  every arm shares one prebuilt view tree ({self.pit_views_seed});"
                    " build it with scripts/data/prebuild_pit_views_seed.py for exactly"
                    " this dataset selection and research geometry before creating or"
                    " dry-running the round"
                )
            return None, reason

    def seed_status(self) -> str:
        """One line on the shared seed, printed before anything is validated.

        Says whether the tree's contract can be read at all and which release
        every arm will pin; whether it matches this round's selection, whether
        its build finished and whether that release is published and reaches
        Held-out is the pre-flight's own verdict below.
        """
        if not self.pit_views_seed:
            return (
                "seed: this round names none, so its arms use the console default tree"
                " and cold-build anything it does not carry"
            )
        provider = REPO_ROOT / self.pit_views_seed / "provider.json"
        if not provider.is_file():
            return (
                f"seed {self.pit_views_seed}: provider.json absent -- the prebuild has not"
                " written its contract yet, so the round is refused below"
            )
        try:
            record = json.loads(provider.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return f"seed {self.pit_views_seed}: provider.json unreadable ({exc}); the round is refused below"
        version = record.get("schema_version")
        if version != SNAPSHOT_CACHE_FORMAT_VERSION:
            return (
                f"seed {self.pit_views_seed}: built under snapshot cache format {version!r}"
                f" but this code writes {SNAPSHOT_CACHE_FORMAT_VERSION} -- the pre-flight"
                " refuses it; rebuild the seed under a new directory"
            )
        return (
            f"seed {self.pit_views_seed}: contract present (snapshot cache format {version},"
            f" release {record.get('generation_id')}, which every arm pins); the pre-flight"
            " below compares this round's selection against it, refuses a tree whose prebuild"
            " has not finished and checks that release is published and reaches Held-out."
        )

    def fill(
        self, port: int, *, dry_run: bool
    ) -> tuple[list[str], list[str], dict[str, list[str]], str]:
        """The arms to create now, every pending arm, the pending arms held
        back by each reason that clears by itself, and the slot part of the
        summary line.

        The queue is the arm order of the round file. An arm whose experiment
        directory exists has been created already -- the console refuses a
        second one whatever state it reached -- so only the rest are pending.
        They take the console's free slots in order, except that an arm the
        console would refuse for a reason that clears by itself is held: it
        stays pending and the arms behind it go ahead. That is a local arm (the
        console's own ``uses_local_model``) reached while the local-model limit
        is full, an arm asking for more GPUs (``gpu_request``) than the console
        has free cards for, and an arm whose seed has no contract yet or an
        unfinished prebuild (``seed_ready``). A dry-run lists the pending arms;
        a timer run only counts them.
        """
        record = health(port)
        running = sorted(str(name) for name in record["running"])
        running_local = sorted(str(name) for name in record["running_local"])
        cap = int(record["max_running_experiments"])
        local_cap = int(record["max_running_local_experiments"])
        free = max(cap - len(running), 0)
        local_free = max(local_cap - len(running_local), 0)
        cards = [str(device) for device in record["gpus_free"]]  # type: ignore[attr-defined]
        gpus_free = len(cards)
        slots = (
            f"slots {len(running)}/{cap} in use ({', '.join(running) or 'none'}), {free} free; "
            f"local-model slots {len(running_local)}/{local_cap} in use "
            f"({', '.join(running_local) or 'none'}), {local_free} free; "
            f"GPUs {gpus_free} free ({', '.join(cards) or 'none'})"
            + (f", unreadable: {record['gpu_error']}" if record.get("gpu_error") else "")
        )
        pending = [arm for arm in self.arms if not (EXPERIMENTS_ROOT / arm).exists()]
        chosen: list[str] = []
        held: dict[str, list[str]] = {"local": [], "gpu": [], "seed": []}
        for experiment_id in pending:
            if len(chosen) == free:
                break
            params = self.request_params(experiment_id)
            if not seed_ready(str(params.get("pit_views_seed") or "")):
                held["seed"].append(experiment_id)
                continue
            local = uses_local_model(params)
            if local and local_free == 0:
                held["local"].append(experiment_id)
                continue
            gpus = gpu_request(params)
            if gpus > gpus_free:
                held["gpu"].append(experiment_id)
                continue
            local_free -= local
            gpus_free -= gpus
            chosen.append(experiment_id)
        if dry_run:
            waits = {
                "local": "waits for a local-model slot",
                "gpu": "waits for a free GPU",
                "seed": "waits for its seed's prebuild to finish",
            }
            for experiment_id in pending:
                slot = next(
                    (waits[reason] for reason, arms in held.items() if experiment_id in arms),
                    "takes a free slot" if experiment_id in chosen else "waits for a free slot",
                )
                print(f"{experiment_id}: pending, {slot}")
        return chosen, pending, held, slots

    def main(self, argv: list[str], usage: str | None = None) -> int:
        """`<port> [--dry-run] [--fill] [experiment_id ...]`, shared by every round file."""
        if self.closed:
            raise SystemExit(
                "this round is closed: every arm it defines has been created, and the file"
                " is only the record of what was run"
            )
        # A mistyped flag must never fall through to the real POST path: without
        # this, --dryrun is read as an experiment-id filter and creates the round.
        mistyped = [
            arg for arg in argv[1:] if arg.startswith("--") and arg not in ("--dry-run", "--fill")
        ]
        if mistyped:
            print("unknown option: " + ", ".join(mistyped), file=sys.stderr)
            return 2
        if len(argv) < 2 or not argv[1].isdigit():
            raise SystemExit(usage or self.main.__doc__)
        port = int(argv[1])
        dry_run = "--dry-run" in argv
        fill = "--fill" in argv
        wanted = {arg for arg in argv[2:] if not arg.startswith("--")}
        unknown_ids = sorted(wanted - set(self.arms))
        if unknown_ids:
            raise SystemExit("not in this round: " + ", ".join(unknown_ids))
        if fill and wanted:
            raise SystemExit("--fill reads the queue itself; name no experiment id")
        if not dry_run and not self.arms:
            raise SystemExit("this round has no arms to create; --dry-run validates its geometry and seed")
        self.check_console_defaults()
        if not fill or dry_run:
            # A timer run's log carries its summary line, not the seed's
            # contract every ten minutes.
            print(self.seed_status())
        shared, reason = self.validated(PROBE_ID)
        if shared is None:
            # Every arm sends these parameters, so no arm can pass either.
            print(f"{_now()} {reason}" if fill else reason, file=sys.stderr)
            return 1
        if dry_run:
            print(json.dumps({key: shared[key] for key in ROUND_REPORT_KEYS}, ensure_ascii=False))
        if fill:
            # The queue decides the selection; --dry-run still decides whether
            # anything is sent.
            selected, pending, held, slots = self.fill(port, dry_run=dry_run)
        else:
            selected = [arm for arm in self.arms if not wanted or arm in wanted]
        created: list[str] = []
        failed: list[str] = []
        for experiment_id in selected:
            merged, reason = self.validated(experiment_id)
            if merged is None:
                # On a dry-run the refusal is a reading, so the remaining arms
                # are read too and the exit code still reports it; on the POST
                # path nothing more is sent after a rejected arm.
                print(reason, file=sys.stderr)
                failed.append(experiment_id)
                if dry_run:
                    continue
                break
            failure = smoke(merged)
            if failure:
                print(
                    f"{experiment_id}: the starter's smoke failed on {merged.get('pit_views_seed')},"
                    f" nothing was sent: {failure}",
                    file=sys.stderr,
                )
                failed.append(experiment_id)
                if dry_run:
                    continue
                break
            if dry_run:
                # What this arm decides for itself: everything it sends that
                # differs from the round it belongs to.
                own = {
                    key: value
                    for key, value in merged.items()
                    if key != "research_directive" and value != shared.get(key)
                }
                directive = str(merged["research_directive"])
                print(json.dumps(own, ensure_ascii=False))
                print("  smoke: the starter replayed from every research-year start and over the full span")
                # The gates this arm would be judged by: the defaults with
                # whatever the arm names, mandate included.
                rules = acceptance_for(merged)
                print("  acceptance:", json.dumps(rules.to_record(), ensure_ascii=False))
                if merged.get("lineage_arms"):
                    print("  lineage:", json.dumps(_lineage_reading(merged, rules), ensure_ascii=False))
                print(f"  directive: {len(directive.splitlines())} lines, {len(directive)} chars")
                for line in directive.splitlines():
                    print("   |", line)
                continue
            if post(port, self.request_params(experiment_id)):
                created.append(experiment_id)
            else:
                failed.append(experiment_id)
        if fill:
            done = len(selected) - len(failed) if dry_run else len(created)
            print(
                f"{_now()} fill {Path(argv[0]).stem}: {slots}; queue {len(self.arms)}: "
                f"{len(self.arms) - len(pending)} skipped (created already), "
                f"{done} {'would be created' if dry_run else 'created'}, "
                f"{len(failed)} refused, {len(pending) - done - len(failed)} pending "
                f"({len(held['local'])} held by the local-model limit, {len(held['gpu'])} by the free GPUs,"
                f" {len(held['seed'])} by an unfinished seed)",
                flush=True,
            )
        if failed:
            print(("refused: " if dry_run else "not created: ") + ", ".join(failed), file=sys.stderr)
            return 1
        return 0
