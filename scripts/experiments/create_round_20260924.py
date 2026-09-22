#!/usr/bin/env python
"""The 2026-09-24 round: the same carrier, two more levers.

The dataset selection and the prebuilt seed are the 2026-09-19 round's -- the
2026-09-20 selection plus `index_weight` -- imported rather than restated so
the rounds that mount it cannot drift apart by a typo. Every arm needs it:
each universe is a decision day's visible CSI 300 section, and the host's
zero-skill panel matches its replacements on constituent membership, so an arm
without the section would be graded against a different floor than it trades.
Neither arm below reads a dataset outside that selection; the third arm, once
appended, selects two more events datasets and names its own seed. Minute data
stays unmounted; `include_intraday` keeps the console default `False`, which
`check_console_defaults` pins for every round.

What this round is. The 20260923 round held the confirmed carrier (Alpha158 +
LightGBM on CSI 300 constituents, beta-residual 10-day label; active IR 0.4642
as the required control of the 20260922 round, below the 0.75 freeze bar) and
moved one lever per arm: information, label, index. This round moves two levers
that round left untouched: the LEARNER itself on the carrier's own features,
label and parameter point (`learner_axis`), and MARKET STATE as a feature of the
cross-sectional ranker (`market_state_block`) -- the register's open item that
state-as-a-feature has never been separated from the closed portfolio-level
timing family. The carrier and the 50-seat monthly book are otherwise unchanged,
which is what makes each arm's own control a control.

`benchmark_index` is a create-time parameter (console default `000300.SH`) and
is stated on every arm rather than inherited, as in the 20260923 round.

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

Each pack carries its own offline gate (`G-INC2` in `families.md`), which must
pass BEFORE any full-span batch. It is offline: it spends no replay-year and
counts no trial, which matters because each completed validation, whatever its
span, raises this arm's own deflated-Sharpe bar.

Contamination, per arm. Operator-side only.

- Round-level. The decision to spend two arms on these levers was taken after
  all four 20260922 arms closed `no_edge` and while the 20260923 arms were
  still open; the closes behind it are research-period readings of their own
  controls, not forward information. The operator has read the shared-forward
  discards that preceded them.
- `learner_axis_1m_20260924`. The carrier's lineage is the one whose
  calibration arm's forward verdict the operator has read; the carrier runs
  here as `c_base`, the required first leg of every batch, so the
  contamination reaches the comparison rather than the increment being
  claimed. DoubleEnsemble and a top-weighted ranking objective have never
  carried a book here, and the pack's external evidence is used for the sign of
  a same-feature comparison only, never for a level.
- `market_state_block_1m_20260924`. Same carrier statement. The one positive
  prior it cites is a controlled gate-on minus gate-off difference inside
  `xs_transformer_1m_20260922`, a research-period reading on a carrier that
  pack never allowed to be nominated; the arm re-measures that sign on a
  nominable carrier. Portfolio-level timing is a closed family, and any leg
  that lets state move gross exposure, cash, seats or the review calendar is
  that family, not this arm (the pack's P6).

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260924.py <port> [--dry-run] [--fill] [experiment_id ...]
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

# The console default, stated so an arm that keeps it records a choice. Every
# arm of this round is graded against it.
CSI300 = "000300.SH"

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
    # Lever: learner. Same features, label, parameter point and book.
    "learner_axis_1m_20260924": {
        "workspace_reference": "configs/workspace_refs/learner_axis_20260924",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI300,
        "gpu_count": 0,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只换学习器：载体特征、标签、参数点与月频 50 席书一字不改；c_base 同批第一条腿永不可提名，"
            "c_cold 拆开集成与不续训。先过离线 G-INC2（Δ主动超额与 Δ秩IC 两年皆正，秩IC ≥ +0.005 且超种子差），"
            "全臂不超过 6 个完整期 trial；Qlib 水平不可搬，只有同特征比较的方向可搬。"
        ),
    },
    # Lever: information, a block the carrier already reads the inputs of.
    # State enters the booster only; the book never moves with it.
    "market_state_block_1m_20260924": {
        "workspace_reference": "configs/workspace_refs/market_state_block_20260924",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI300,
        "gpu_count": 0,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是消融臂：载体 Alpha158 + LightGBM 在沪深 300 成分上一个字不改，只把 12 列日期级市场状态"
            "（基准 1 / 5 / 20 / 60 日收益与波动、均线距离、回撤、成交额 z、截面离散度、广度，m1）"
            "或 12 列状态 × 个股交互（m_int）喂给排序器，c_base 永远是同批第一条腿、永不可提名。"
            "书的形状与 trade.py 一字不改、始终满仓 50 席月频复核，任何按状态改仓位、现金、席位或复核频率的腿"
            "都属于已关闭的择时家族（P6 检查）。先过离线 G-INC2——由 Δ 主动超额与逐年符号决定，"
            "增益占比对日期级列几乎自动满足、只作报告；第四研究年有 42.6 % 的决策日至少一列状态落在"
            "该年首次拟合的训练范围之外，须按登记读数逐年汇报。"
        ),
    },
    # --- THIRD ARM, NOT YET APPENDED ----------------------------------------
    # `intraday_stats_block_1m_20260924`
    #     "workspace_reference": "configs/workspace_refs/intraday_stats_block_20260924",
    #     "initial_cash": 1_000_000,
    #     "benchmark_index": CSI300,
    #     "events_datasets": [*EVENTS_DATASETS, "intraday_flow", "intraday_stats"],
    #     "pit_views_seed": "data/pit_views_seed_research_20260924",
    # belongs at the end of this table with the same gpu_count, drawdowns and
    # GATES as the arms above. It needs a different snapshot selection from the
    # two above, in exactly that dataset order, and names its own seed so they
    # keep the 2026-09-19 tree they pin.
    #
    # DO NOT add the entry before that seed is built for exactly that
    # selection. A round's snapshot configuration is the contract its prebuilt
    # seed was built under, so the create pre-flight refuses an arm whose
    # selection does not match a staged tree -- and a refused creation inside a
    # `--fill` run exits non-zero, which would abort the fill for the arms that
    # are ready.
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
