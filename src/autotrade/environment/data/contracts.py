from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

CN_TZ = ZoneInfo("Asia/Shanghai")

# ---------------------------------------------------------------------------
# Raw-lake research contract. The environment owns these values (they define
# how PIT consumers interpret the lake); the ingest adapter under
# ``autotrade.data_sources`` imports them so both sides cannot drift.
# ---------------------------------------------------------------------------

# Raw-lake generation stamp published by the cron updater after each fully
# successful mutation run. One schema, one committed-acceptance rule: the
# updater writes it, and every PIT consumer (snapshot builds, research-release
# pinning) must apply the same strictness when reading it back.
RAW_GENERATION_FILENAME = ".raw_generation.json"
GENERATION_SCHEMA_VERSION = 2
GENERATION_COMMITTED = "committed"
GENERATION_IN_PROGRESS = frozenset({"updating", "dirty"})


def require_committed_generation(payload: dict[str, object]) -> None:
    """Fail unless ``payload`` is an explicit schema-v2 committed record.

    The updater always writes ``schema_version`` and ``state``; a stamp
    missing either is malformed, not implicitly committed.
    """
    if payload.get("schema_version") != GENERATION_SCHEMA_VERSION or "state" not in payload:
        raise RuntimeError("raw generation is not an explicit schema-v2 committed record")
    state = str(payload.get("state"))
    if state != GENERATION_COMMITTED:
        transaction = payload.get("transaction") or {}
        job = transaction.get("job", "unknown") if isinstance(transaction, dict) else "unknown"
        raise RuntimeError(f"raw lake generation is not committed: state={state} job={job}")


def read_committed_raw_generation(raw_dir: str | Path | None) -> dict[str, object] | None:
    """Read the raw-lake generation stamp; ``None`` for unstamped lakes
    (manual/test raw dirs). A stamp that exists must pass
    :func:`require_committed_generation`."""
    if raw_dir is None:
        return None
    path = Path(raw_dir) / RAW_GENERATION_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid raw generation record: {path}: {exc}") from exc
    try:
        require_committed_generation(payload)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc} path={path}") from None
    return payload


# TuShare amount/vol can differ from the quoted cent price through rounding.
# Half one stock tick accepts the full local history while rejecting grossly
# inconsistent clearing truth.
STK_AUCTION_PRICE_ABS_TOLERANCE = 0.005

# Formal data-quality status reports, one per snapshot consumption domain
# ("independent consumption, independent failure isolation"): audit producers
# write them, snapshot gates read them per enabled domain, research releases
# pin them. Owned here so producer and consumers cannot drift.
DOMAIN_STATUS_FILES: dict[str, str] = {
    "daily": "core_market_status.json",
    "intraday_1min": "intraday_minutes_status.json",
    "fundamentals_raw": "fundamental_raw_status.json",
    "events": "event_flow_status.json",
    "board_trading": "board_trading_status.json",
    "macro": "macro_context_status.json",
    "text": "text_evidence_status.json",
    "fundamentals": "fundamental_events_status.json",
}
DOMAIN_REPORT_TYPES: dict[str, str] = {
    "daily": "core_market",
    "intraday_1min": "intraday_minutes",
    "fundamentals_raw": "fundamental_raw",
    "events": "event_flow",
    "board_trading": "board_trading",
    "macro": "macro_context",
    "text": "text_evidence",
    "fundamentals": "fundamental_events",
}
# Frozen research releases are self-describing (verified against their own
# manifest set), so retired filenames must stay resolvable to their report
# type for as long as a pinned release may reference them.
LEGACY_STATUS_REPORT_TYPES: dict[str, str] = {
    "base_research_status.json": "base_research",
}

# Datasets forming the board-trading (打板) research domain in the raw lake.
# This is the DOWNLOAD and AUDIT scope, not a selectability list: `hm_list` is
# still fetched and audited even though it can never enter a snapshot (it has no
# historical PIT stamp), so the snapshot's board gate simply never matches it.
BOARD_TRADING_DATASETS = [
    "kpl_list",
    "kpl_concept_cons",
    "dc_index",
    "dc_member",
    "limit_step",
    "limit_cpt_list",
    "limit_list_d",
    "limit_list_ths",
    "top_list",
    "top_inst",
    "hm_list",
    "hm_detail",
    "ths_hot",
    "dc_hot",
]


# Close-published daily tables: the row is public on its data date, after the
# close at the latest, so the daily core and the same-day macro tables below
# share one stamp (docs/data-documentation.md §3.3).
CLOSE_PUBLISHED_TIME = time(17, 30)


@dataclass(frozen=True)
class DatasetContract:
    dataset: str
    partition_key: str
    available_time: time
    lag_days: int = 0
    pit_notes: str = ""

    def available_at(self, partition_date: date) -> datetime:
        return datetime.combine(partition_date + timedelta(days=self.lag_days), self.available_time, tzinfo=CN_TZ)

    @property
    def rule(self) -> str:
        """``available_at_rule`` label for rows stamped by this contract."""
        lag = f"+{self.lag_days}d" if self.lag_days else ""
        return f"contract_{self.available_time:%H%M}{lag}_from:{self.partition_key}"


def default_tushare_contracts() -> dict[str, DatasetContract]:
    return {
        "daily": DatasetContract(
            dataset="daily",
            partition_key="trade_date",
            available_time=CLOSE_PUBLISHED_TIME,
            pit_notes="Use for close-to-close research or next-trade-date decisions, not same-day 09:25 decisions.",
        ),
        "daily_basic": DatasetContract(
            dataset="daily_basic",
            partition_key="trade_date",
            available_time=time(18, 0),
            pit_notes="Valuation and share fields are available after market close; use next trade date for decisions.",
        ),
        "adj_factor": DatasetContract(
            dataset="adj_factor",
            partition_key="trade_date",
            available_time=time(9, 30),
            pit_notes="Raw trade_date alone is not enough for intraday PIT; conservative daily replay should use prior close factors.",
        ),
        "stk_limit": DatasetContract(
            dataset="stk_limit",
            partition_key="trade_date",
            available_time=time(8, 45),
            pit_notes="Can be used before the trading session if the source timestamp is trusted.",
        ),
        "suspend_d": DatasetContract(
            dataset="suspend_d",
            partition_key="trade_date",
            available_time=time(8, 45),
            pit_notes="Use as a trading constraint; zero rows mean no suspended names for that partition.",
        ),
    }


# Same-day published macro tables. The ingest adapter stamps every macro row
# on the conservative date-EOD placeholder (23:59:59), which the 23:35 evening
# node can only release a day late (T rows first visible at T+2). These tables
# are public on their data date -- exchange index/industry bars, exchange
# statistics and market money flow, repo and derivative settlements after the
# close, SHIBOR/LPR fixings before noon -- and the same night's evening job
# (cn_evening_full, macro tier) lands them, so the snapshot re-stamps their rows
# on the daily core's close contract: T rows roll in at T+1, exactly as Paper
# sees them once that job has landed. Registries stamp on the listing day; the
# listing notice precedes it, so its close is still conservative. Delayed
# releases (monthly/quarterly statistics, broker_recommend), announcement
# tables keyed by ann_date (cb_call) and the global tier (its data date closes
# after the evening launch and lands a night later) keep the adapter's stamp.
MACRO_DATASET_CONTRACTS: dict[str, DatasetContract] = {
    dataset: DatasetContract(
        dataset=dataset,
        partition_key=partition_key,
        available_time=CLOSE_PUBLISHED_TIME,
        pit_notes="Published on the data date; visible from the next trading day's pre-open.",
    )
    for dataset, partition_key in (
        ("shibor", "date"),
        ("shibor_quote", "date"),
        ("shibor_lpr", "date"),
        ("repo_daily", "trade_date"),
        ("index_daily", "trade_date"),
        ("index_dailybasic", "trade_date"),
        ("sw_daily", "trade_date"),
        ("ci_daily", "trade_date"),
        ("ths_daily", "trade_date"),
        ("daily_info", "trade_date"),
        ("sz_daily_info", "trade_date"),
        ("moneyflow_mkt_dc", "trade_date"),
        ("fut_basic", "list_date"),
        ("fut_mapping", "trade_date"),
        ("fut_daily", "trade_date"),
        ("opt_basic", "list_date"),
        ("opt_daily", "trade_date"),
        ("cb_basic", "list_date"),
        ("cb_daily", "trade_date"),
    )
}


# ---- Timeview refresh nodes (docs/environment-design.md) ----
#
# The Timeview replays the real local-DB refresh cadence: a dataset row
# is visible only once the cron job that lands it has finished writing. Each node
# below mirrors one data-landing job in configs/tushare_update_schedule.json:
# ``start`` is the installed crontab launch time (Asia/Shanghai), and
# ``duration_minutes`` is the measured refresh cost, so the view does not see the
# data until ``ready_at = start + duration_minutes``. Audit-only jobs (the nightly
# full audit, the daily text audit, the revision sentinel, the 09:20 event-flow
# audit) land no new data
# and are deliberately NOT nodes. ``test_*`` drift guards assert every node name is
# a real cron job and that the audit jobs stay excluded.


@dataclass(frozen=True)
class RefreshNode:
    """One data-landing refresh job: launches at ``start`` and is queryable from
    ``ready_at = start + duration_minutes`` (possibly the next calendar day)."""

    name: str
    start: time
    duration_minutes: int

    def start_at(self, day: date) -> datetime:
        return datetime.combine(day, self.start, tzinfo=CN_TZ)

    def ready_at(self, day: date) -> datetime:
        return self.start_at(day) + timedelta(minutes=self.duration_minutes)


REFRESH_NODES: dict[str, RefreshNode] = {
    # Evening rolling-window update: A-share daily core, minute history, money flow,
    # block trade, holders/repurchase/float/top-list and macro.
    # Real dispatches have exceeded the former 150-minute estimate (one reached
    # 169 minutes). Historical replays have no per-run completion ledger, so use
    # a conservative 210-minute fallback (03:05) rather than knowingly exposing
    # rows before the observed job completed.
    "cn_evening_full": RefreshNode("cn_evening_full", time(23, 35), 210),
    # Bulk text-evidence landing, every calendar evening (see TEXT_NODE below).
    # The tier step measured 177s inside the evening update; 15 min keeps the
    # boundary conservative and still clears the 23:35 evening launch.
    "cn_nightly_text_full": RefreshNode("cn_nightly_text_full", time(23, 15), 15),
    # Ann-date disclosure tables (holder counts/trades, top-10 holders,
    # repurchases), every calendar evening: ~one row in ten announces on a
    # weekend, and waiting for the next trading evening left the structured
    # rows two days behind their own announcement text.
    "cn_nightly_disclosure_full": RefreshNode("cn_nightly_disclosure_full", time(23, 5), 10),
    # Global tier (overseas indices, FX, US rates, economic calendar), every
    # calendar evening: these sources publish through A-share holidays, and
    # the trading-day evening job alone left the first post-holiday open up to
    # seven overseas sessions stale (measured National Day 2025).
    "cn_nightly_global_full": RefreshNode("cn_nightly_global_full", time(23, 25), 15),
    # Fundamental PIT event index build (financial filings become queryable).
    # Full-window rebuild+audit measured 18m13s on first run (2026-08-08 run,
    # 20200101..20260807); 30 min keeps the boundary conservative.
    "cn_nightly_pit_event_build": RefreshNode("cn_nightly_pit_event_build", time(3, 35), 30),
    # Pre-open board-trading backfill (kpl_list etc.).
    "cn_preopen_board_backfill_0850": RefreshNode("cn_preopen_board_backfill_0850", time(8, 50), 5),
    # Pre-open short-text backfill (cctv_news / news).
    "cn_preopen_text_backfill_0855": RefreshNode("cn_preopen_text_backfill_0855", time(8, 55), 5),
    # Same-day margin_secs (shortable universe) first attempt + retry.
    "cn_preopen_margin_secs_backfill_0903": RefreshNode("cn_preopen_margin_secs_backfill_0903", time(9, 3), 2),
    "cn_preopen_margin_secs_retry_0913": RefreshNode("cn_preopen_margin_secs_retry_0913", time(9, 13), 2),
    # Previous-day margin / margin_detail first attempt + retry.
    "cn_preopen_margin_backfill_0905": RefreshNode("cn_preopen_margin_backfill_0905", time(9, 5), 2),
    "cn_preopen_margin_retry_0915": RefreshNode("cn_preopen_margin_retry_0915", time(9, 15), 2),
    # Same-day exact opening-auction capture. Agent visibility uses each
    # partition's observed row-level available_at; the node remains the cron
    # drift/lifecycle record and covers the 09:27 polling window.
    "cn_open_auction_capture_0927": RefreshNode("cn_open_auction_capture_0927", time(9, 27), 4),
}

EVENING_NODE = "cn_evening_full"
PIT_EVENT_NODE = "cn_nightly_pit_event_build"
# Text publishes on weekends and holidays, so its landing job uses a natural-day
# window and runs every calendar evening: no NODE_ACTIVE_WEEKDAYS entry, and the
# boundary it defines is a genuine daily one.
TEXT_NODE = "cn_nightly_text_full"
# The evening job resolves its end date to the latest SSE open day and skips an
# already-successful identical range. Weekend launches therefore do not create
# visibility boundaries. Weekday exchange holidays still need a future trading-
# calendar/observed-completion ledger for exact modelling.
NODE_ACTIVE_WEEKDAYS: dict[str, frozenset[int]] = {
    EVENING_NODE: frozenset(range(5)),
    # The job audits the previous A-share trading day. Tuesday through
    # Saturday produce a new range; Sunday and Monday resolve to Friday again
    # and are skipped. Weekday exchange holidays remain covered by the
    # documented conservative fallback until per-run completion history exists.
    PIT_EVENT_NODE: frozenset(range(1, 6)),
}

# Whole-domain node assignment (a domain file's rows all roll on the same node).
DOMAIN_REFRESH_NODES: dict[str, tuple[str, ...]] = {
    "daily": (EVENING_NODE,),
    "intraday_1min": (EVENING_NODE,),
    # The macro file mixes A-share series with the global tier, whose landing
    # jobs run on different calendars; the replay view therefore gates macro
    # rows PER DATASET (MACRO_DATASET_REFRESH_NODES below). A domain-level
    # node union is a proven early-visibility bias: domestic macro rows carry
    # weekend available_at stamps (month-end EODs, weekend repo/shibor dates)
    # that the weekend global runs never ingest, yet a union cutoff would
    # expose them a day and a half before the Monday evening landing. This
    # whole-domain entry stays the conservative evening node for any
    # domain-level consumer.
    "macro": (EVENING_NODE,),
    # Auction is deliberately not node-gated: each row carries the uniform
    # official 09:29 publish stamp and Timeview advances on that timestamp.
    "auction": (),
    "fundamentals": (PIT_EVENT_NODE,),
}

# Per-dataset overrides inside the events domain (default = cn_evening_full).
EVENT_DATASET_REFRESH_NODES: dict[str, tuple[str, ...]] = {
    "margin_secs": ("cn_preopen_margin_secs_backfill_0903", "cn_preopen_margin_secs_retry_0913"),
    "margin": ("cn_preopen_margin_backfill_0905", "cn_preopen_margin_retry_0915"),
    "margin_detail": ("cn_preopen_margin_backfill_0905", "cn_preopen_margin_retry_0915"),
    # Board-trading sources publishing next-day ~08:30: the 08:50 pre-open
    # backfill is their real landing job (the evening node refines backfills).
    "kpl_list": (EVENING_NODE, "cn_preopen_board_backfill_0850"),
    "kpl_concept_cons": (EVENING_NODE, "cn_preopen_board_backfill_0850"),
    "limit_step": (EVENING_NODE, "cn_preopen_board_backfill_0850"),
    "limit_cpt_list": (EVENING_NODE, "cn_preopen_board_backfill_0850"),
    # limit_list_ths / ths_hot / dc_hot / hm_detail / hm_list land in the
    # evening window only — the default node is already correct for them.
    # Ann-date disclosure tables land via their own natural-day job (weekend
    # announcements become visible the following pre-open, matching live).
    "top10_holders": ("cn_nightly_disclosure_full",),
    "top10_floatholders": ("cn_nightly_disclosure_full",),
    "stk_holdernumber": ("cn_nightly_disclosure_full",),
    "stk_holdertrade": ("cn_nightly_disclosure_full",),
    # The sell-side forecast slice is the text dataset of the same name read
    # by the events domain: its rows land with the natural-day text job, so
    # the numbers and the title of one report become visible together.
    "report_rc": (TEXT_NODE,),
}

# Per-dataset overrides inside the text domain (default = TEXT_NODE). The
# pre-open backfill refines the same natural day's short text before the open.
TEXT_DATASET_REFRESH_NODES: dict[str, tuple[str, ...]] = {
    "cctv_news": (TEXT_NODE, "cn_preopen_text_backfill_0855"),
    "news": (TEXT_NODE, "cn_preopen_text_backfill_0855"),
}

# Per-dataset overrides inside the macro domain (default = cn_evening_full).
# The global tier lands via its own natural-day job; every other macro dataset
# stays on the trading-evening node so its weekend-stamped rows (month-end
# EODs and the like) wait for the run that actually ingests them.
MACRO_DATASET_REFRESH_NODES: dict[str, tuple[str, ...]] = {
    "index_global": ("cn_nightly_global_full",),
    "fx_daily": ("cn_nightly_global_full",),
    "us_tycr": ("cn_nightly_global_full",),
    "us_trycr": ("cn_nightly_global_full",),
    "us_tbr": ("cn_nightly_global_full",),
    "us_tltr": ("cn_nightly_global_full",),
    "eco_cal": ("cn_nightly_global_full",),
}


def node_visible_cutoff(node: RefreshNode, when: datetime) -> datetime | None:
    """Latest daily instance of ``node`` already finished by ``when``; returns its
    ``start`` instant (the availability cutoff) or None if none has completed yet.

    A row is visible under this node when its ``available_at`` is at or before the
    returned cutoff. Searching back ten days crosses weekends and ordinary
    holiday gaps.
    """
    if when.tzinfo is None:
        when = when.replace(tzinfo=CN_TZ)
    base_day = when.astimezone(CN_TZ).date()
    active_weekdays = NODE_ACTIVE_WEEKDAYS.get(node.name)
    for delta in range(0, 10):
        day = base_day - timedelta(days=delta)
        if active_weekdays is not None and day.weekday() not in active_weekdays:
            continue
        if node.ready_at(day) <= when:
            return node.start_at(day)
    return None


def visible_cutoff(node_names: tuple[str, ...], when: datetime) -> datetime | None:
    """Availability cutoff under any of ``node_names`` at ``when``: the most recent
    completed node's start instant (later node refines an earlier one)."""
    cutoffs = [
        cutoff
        for name in node_names
        if (cutoff := node_visible_cutoff(REFRESH_NODES[name], when)) is not None
    ]
    return max(cutoffs) if cutoffs else None


def next_visible_boundary(node_names: tuple[str, ...], when: datetime) -> datetime | None:
    """Earliest future ``ready_at`` among ``node_names``.

    Timeview uses this as a fast gate between refreshes. Boundaries are strict:
    a node ready exactly at ``when`` has already been processed and the next
    scheduled instance is returned.
    """
    if not node_names:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=CN_TZ)
    local_when = when.astimezone(CN_TZ)
    base_day = local_when.date()
    candidates = [
        ready_at
        for name in node_names
        for delta in range(-1, 11)
        if (
            (active_weekdays := NODE_ACTIVE_WEEKDAYS.get(name)) is None
            or (base_day + timedelta(days=delta)).weekday() in active_weekdays
        )
        if (ready_at := REFRESH_NODES[name].ready_at(base_day + timedelta(days=delta))) > local_when
    ]
    return min(candidates) if candidates else None


def domain_visible_cutoff(domain: str, when: datetime) -> datetime | None:
    """Timeview availability cutoff for a whole-domain file at ``when``."""
    return visible_cutoff(DOMAIN_REFRESH_NODES.get(domain, (EVENING_NODE,)), when)


def domain_next_visible_boundary(domain: str, when: datetime) -> datetime | None:
    """Next Timeview refresh boundary for a whole-domain file."""
    return next_visible_boundary(DOMAIN_REFRESH_NODES.get(domain, (EVENING_NODE,)), when)


def event_dataset_visible_cutoff(dataset: str, when: datetime) -> datetime | None:
    """Timeview availability cutoff for one events-domain dataset at ``when``."""
    return visible_cutoff(EVENT_DATASET_REFRESH_NODES.get(dataset, (EVENING_NODE,)), when)


def event_dataset_next_visible_boundary(dataset: str, when: datetime) -> datetime | None:
    """Next Timeview refresh boundary for one events-domain dataset."""
    return next_visible_boundary(EVENT_DATASET_REFRESH_NODES.get(dataset, (EVENING_NODE,)), when)


def macro_dataset_visible_cutoff(dataset: str, when: datetime) -> datetime | None:
    """Timeview availability cutoff for one macro-domain dataset at ``when``."""
    return visible_cutoff(MACRO_DATASET_REFRESH_NODES.get(dataset, (EVENING_NODE,)), when)


def macro_dataset_next_visible_boundary(dataset: str, when: datetime) -> datetime | None:
    """Next Timeview refresh boundary for one macro-domain dataset."""
    return next_visible_boundary(MACRO_DATASET_REFRESH_NODES.get(dataset, (EVENING_NODE,)), when)


def text_dataset_visible_cutoff(dataset: str, when: datetime) -> datetime | None:
    """Timeview availability cutoff for one text-domain dataset at ``when``."""
    return visible_cutoff(TEXT_DATASET_REFRESH_NODES.get(dataset, (TEXT_NODE,)), when)


def text_dataset_next_visible_boundary(dataset: str, when: datetime) -> datetime | None:
    """Next Timeview refresh boundary for one text-domain dataset."""
    return next_visible_boundary(TEXT_DATASET_REFRESH_NODES.get(dataset, (TEXT_NODE,)), when)
