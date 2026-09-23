#!/usr/bin/env python
"""The 2026-09-25 round: the same score, a book that follows it.

The dataset selection and the prebuilt seed are the 2026-09-19 round's -- the
2026-09-20 selection plus `index_weight` -- imported rather than restated so
the rounds that mount it cannot drift apart by a typo; the two fresh-book arms
read exactly what `learner_axis_1m_20260924` read. Every arm needs it: each
universe is a decision day's visible index section, and the host's zero-skill
panel matches its replacements on constituent membership, so an arm without
the section would be graded against a different floor than it trades. Neither
arm below reads a dataset outside that selection; the third arm, once
appended, selects one more events dataset and names its own seed. Minute data
stays unmounted; `include_intraday` keeps the console default `False`, which
`check_console_defaults` pins for every round.

What this round is. The 20260923 and 20260924 rounds held the Alpha158 +
LightGBM carrier fixed and moved seven levers -- four feature blocks, the
label, the index, the learner -- and all seven arms closed `no_edge`. The
register (docs/research-lessons.md) keeps what they settled: the carrier's
rank IC is real, but its monthly 50-seat book reads an active IR of
0.077-0.189 over five host validations, inside the zero-skill sampling error,
and an increment below about 4 pp/yr can be resolved neither offline nor in one
host batch, so stacking blocks on a fixed carrier is closed as a method. The
packs' census, run on each research year's exact views, located where the IC
is lost. The long end barely earns: the top decile beats the equal-weight pool
by +0.13 % per 10 days on CSI 300 (t 0.69) and +0.11 % on CSI 500 (t 0.74).
The book does not hold the score (H2): the monthly book hits its swap cap at
every review and transfers 0.22 / 0.18 of the day's score. Tilts dominate the
active variance (H3): 73-77 % of it is systematic. The flat long end is not a
construction problem and no arm here claims to fix it; this round tests only
construction, on the score unchanged. `f1` tracks the score -- no swap cap,
keep band 1.2, a review cadence set by the score's own IC half-life (three
weeks on CSI 300, a month on CSI 500), empty seats refilled -- and `f2` is
`f1` with beta, size, volatility and industry pulled back towards the pool:
the register's untested cadence lever and the unmandated variant of its
exposure pull-back. Batch 1 is `c_base` + `f1` + `f2`, one full-span
validation of 12 replay-years, and in the common case it is the whole arm:
unless `f1` or `f2` beats the same-batch `c_base` by at least +2 pp/yr of
active excess, positive in at least 3 of 4 years at no more than 20x annual
turnover, the arm closes `no_edge`.

The round's three arms:

- `fresh_book_1m_20260925`: the construction test on CSI 300 constituents.
- `fresh_book_csi500_1m_20260925`: the same design and starter on CSI 500
  constituents, graded against 000905.SH, with no CSI 300 leg in any batch.
- `northbound_holdings_1m_20260925`, not yet appended (see ARMS): information
  the repository has never read, in its most transparent form -- the
  northbound (Stock Connect) row of the quarterly top-ten holders tables as a
  standalone score on CSI 300 constituents. It waits for its own seed.

The CSI 1000 column was read offline only: its top decile earns 1.85x CSI
300's, but at t 1.39 it misses the slot condition (t >= 2), so it has no arm.

`benchmark_index` is a create-time parameter (console default `000300.SH`) and
is stated on every arm rather than inherited, as in the rounds before.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
        "initial_cash": <the account this arm runs>,
        "benchmark_index": <the index the host grades this arm against>,
        "gpu_count": <stated; every arm this round is CPU-only>,
        "max_drawdown" / "active_max_drawdown":
            stated so `--dry-run` prints the gates this arm is judged by,
    }

`_round.Round.request_params` merges the arm last. No arm carries a tracking
mandate this round, so every one states the rules' own drawdowns 0.45 / 0.30
and names no `tracking_error_cap`. Every arm also names the statistical bars
(IR, DSR, positive-year share, full-span validations, forward confidence,
recency, activity, Held-out z) so the create request records a choice rather
than inheriting a hidden pair of packages.

The fresh-book packs carry no offline gate. Their census readings are
registered in `families.md` as facts: a session may re-verify them in round 0
without counting a trial, must report a discrepancy, and does not re-derive or
declare them. The first batch is the test itself, and it is also the one
full-span validation `finish_session(no_edge)` needs.

Contamination, per arm. Operator-side only.

- Round-level. The decision to spend this round on construction was taken
  after all seven 20260923 / 20260924 arms closed `no_edge`; those closes and
  the census behind the design are research-period readings, not forward
  information. The operator has read the shared-forward discards that
  preceded them.
- `fresh_book_1m_20260925`. The carrier's lineage is the one whose
  calibration arm's forward verdict the operator has read; its score runs in
  every leg and its monthly book runs as `c_base`, the required first leg of
  every batch, so the contamination reaches the comparison rather than the
  construction increment being claimed. `f1`'s cadence and the offline
  `f1 - c_base` delta (+4.70 pp/yr, 4 of 4 years) were read on research-period
  views and are registered as facts, not as an edge; the literature on
  transfer coefficients and trading bands is used for direction only.
- `fresh_book_csi500_1m_20260925`. Same carrier statement. The CSI 500
  section has carried one book, `csi500_shape_1m_20260923`, closed `no_edge`
  on research-period readings; no forward reading exists on this shape. The
  CSI 300 readings are evidence, never an in-batch control.

The third arm states its own when it is appended.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260925.py <port> [--dry-run] [--fill] [experiment_id ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Appended, not prepended: the repository root carries directories named `data`,
# `logs` and `results`, which must never shadow an installed package. The
# launcher pins this repo's `src/` itself.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from scripts.experiments._round import Round

# The selection and the prebuilt view tree are the benchmark round's, imported
# rather than restated: `index_weight` is what both the in-index universes and
# the panel's membership matching depend on.
from scripts.experiments.create_round_20260919 import (
    EVENTS_DATASETS,
    FUNDAMENTAL_DATASETS,
    MACRO_DATASETS,
    PIT_VIEWS_SEED,
    TEXT_DATASETS,
)

# The console default, stated so an arm that keeps it records a choice.
CSI300 = "000300.SH"
CSI500 = "000905.SH"

# Create-time graduation bars, named on every arm so the request records a
# choice. These are today's defaults; an arm that wants other bars overrides
# them the same way it overrides drawdowns.
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

ARMS: dict[str, dict[str, object]] = {
    # Construction only: the carrier's score, label, fits and 50 equal-cash
    # seats unchanged; the book follows the score on a three-week cadence.
    "fresh_book_1m_20260925": {
        "workspace_reference": "configs/workspace_refs/fresh_book_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI300,
        "gpu_count": 0,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只动构造：载体 Alpha158 + LightGBM 的分数、标签、训练与 50 个等额座位一字不改。"
            "作包普查已量出的事实（顶端十分位几乎不挣、f1 的离线 Δ 落在约 4 pp/年 的噪声底上、主动方差多为倾斜）"
            "登记在 families.md，不是门：可以复核、有不一致就报，不重推、不申报。"
            "第 1 批 c_base（月度、每次最多换 8 只、保留带 2.0，control: true、永不可提名）"
            "+ f1（不设上限、保留带 1.2、每 3 周复核、补空座位）+ f2（f1 加贪心敞口拉回）同批检验 H2 与 H3；"
            "f1 与 f2 都没过线（同批 Δ 主动超额 ≥ +2 pp/年 且 ≥ 3/4 年为正、年换手 ≤ 20 倍）就 no_edge。"
            "全臂 host trial ≤ 6、回放 ≤ 32 年，筛过未提交的配置如实申报 offline_trials。"
        ),
    },
    # The same design and starter on the CSI 500 section, reviewed monthly;
    # graded against its own index, so no CSI 300 leg may enter a batch.
    "fresh_book_csi500_1m_20260925": {
        "workspace_reference": "configs/workspace_refs/fresh_book_csi500_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂与沪深 300 的新鲜书臂同一设计、同一份起步包，只换池：决策日在册的中证 500 成分。"
            "benchmark_index 必须是 000905.SH 且与 INDEX_CODE 一致，不一致立即停；批次里不许有沪深 300 的腿。"
            "作包普查已量出的事实登记在 families.md（顶端十分位几乎不挣，f1 的离线 Δ 只有两年为正），"
            "不是门：可以复核、有不一致就报，不重推、不申报。"
            "第 1 批 c_base（control: true、永不可提名）+ f1（不设上限、保留带 1.2、月度复核、补空座位）"
            "+ f2（加贪心敞口拉回）同批检验 H2 与 H3；"
            "都没过线（同批 Δ 主动超额 ≥ +2 pp/年 且 ≥ 3/4 年为正、年换手 ≤ 20 倍）就 no_edge。"
            "全臂 host trial ≤ 6、回放 ≤ 32 年，offline_trials 如实申报。"
        ),
    },
    # --- THIRD ARM, NOT YET APPENDED ----------------------------------------
    # `northbound_holdings_1m_20260925`
    #     "workspace_reference": "configs/workspace_refs/northbound_holdings_20260925",
    #     "initial_cash": 1_000_000,
    #     "benchmark_index": CSI300,
    #     "events_datasets": [*EVENTS_DATASETS, "top10_holders"],
    #     "pit_views_seed": "data/pit_views_seed_research_20260925_holders",
    # belongs at the end of this table with the same gpu_count, drawdowns and
    # GATES as the arms above, its own directive, and its contamination
    # statement in the module docstring. It needs a different snapshot
    # selection from the two above and names its own seed so they keep the
    # 2026-09-19 tree they pin.
    #
    # DO NOT add the entry before that seed's prebuild has finished for
    # exactly that selection. A round's snapshot configuration is the contract
    # its prebuilt seed was built under, so the create pre-flight refuses an
    # arm whose selection does not match a finished tree -- and a refused
    # creation inside a `--fill` run exits non-zero, which would abort the fill
    # for the arms that are ready.
    # ------------------------------------------------------------------------
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=PIT_VIEWS_SEED,
    overrides={
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
