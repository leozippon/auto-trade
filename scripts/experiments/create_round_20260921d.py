#!/usr/bin/env python
"""The 2026-09-21d round: four openings under the same panel, two accounts.

The dataset selection and the prebuilt seed are the 2026-09-19 round's -- the
2026-09-20 selection plus `index_weight` -- imported rather than restated so
the rounds that mount it cannot drift apart by a typo. All four arms need it:
the overlay and in-index books build their universe from the constituent
cross-section, the cash-dividend pool still needs the host's zero-skill panel
to match replacements on CSI 300 membership, and every arm reports tracking
the same way.

Every arm mounts `configs/workspace_refs/capital_aware_20260921d`. What differs
per arm is the account, whether a tracking mandate is in force, and the
registered opening. `initial_cash` and the tracking mandate are independent
create parameters; no capital turns a mandate on.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
        "initial_cash": <the account this arm runs>,
        "tracking_error_cap": <only on a mandated arm>,
        "max_drawdown" / "active_max_drawdown" / "beta_min" / "beta_max":
            stated so `--dry-run` prints the gates this arm is judged by,
    }

`_round.Round.request_params` merges the arm last. The two mandated arms
state the 8 % cap and their own drawdowns 0.35 / 0.15 and beta 0.85-1.15.
The two un-mandated arms state the rules' own 0.45 / 0.30 and do not name
a cap. Every arm also names the statistical bars (IR, DSR, positive-year
share, full-span validations, forward confidence, recency, activity,
Held-out z) so the create request records a choice rather than inheriting
a hidden pair of packages.

The four openings, and nothing else, in this order so an empty slot takes
the first pending arm. Half the arms run CNY 100k, half CNY 1M.
`cma_overlay` / `cashdiv_pool` are innovative explorations. The 21c closes
are written into the reopen ban: quality three-pillar overlay, 52-week-high
ratio, percent accrual. lottery / rmax / issuance are still running -- do
not open another arm in the same direction. The 21b openings stay closed:
value overlay, industry-neutral value, residual momentum, eyield. Bounded
tilt, a risk-model TE budget, 20-day reversal, and in-index weekly
price-volume stay closed. Bounded in-mandate tilt falsifies the box, not
the mandate. In-index daily price-volume at 100k froze on a research-period
active IR of 1.23 and died on the shared forward F2 (lower bound -5.1 %):
a research-period IR above 1 is not graduation.

Contamination, per arm. Operator-side only.

- Round-level. The decision to spend four arms on these openings was taken
  after the 21c quality / high52 / pacc closes, while lottery / rmax /
  issuance were still running. The operator has read the shared-forward
  discards. Every reading behind the four choices is research-period,
  except the F2 close of the one freeze the 21-round produced.
- `gp_overlay_1m_20260921d`. Cites the close of the 21c quality-mix
  opening. The construction is new (gross-profit / assets only, not the
  three-pillar mix, no EP/BP). Forward verdict is partly a pipeline
  calibration of a mandate the operator already had reason to keep trying.
- `noa_index_100k_20260921d`. Cites the close of percent accrual. The
  score is −NOA / lagged assets, not (NI − OCF) / |NI|. Clean apart from
  the round-level statement.
- `cma_overlay_1m_20260921d`. Innovative. The falsifier (rank correlation
  against the closed quality book's investment pillar above 0.85 and
  active IR in the same band) is written from that research-period close.
- `cashdiv_pool_100k_20260921d`. Innovative. Must not become earnings
  yield or OCF yield; those 21b openings are closed. Clean apart from
  the round-level statement.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260921d.py <port> [--dry-run] [--fill] [experiment_id ...]
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

PACK = "configs/workspace_refs/capital_aware_20260921d"

# Create-time graduation bars, named on every arm so the request records a
# choice. These are today's defaults; an arm that wants other bars overrides
# them the same way it overrides drawdowns.
GATES: dict[str, object] = {
    "min_active_ir": 0.75,
    "min_dsr_probability": 0.90,
    "min_positive_year_share": 0.75,
    "min_full_span_validations": 2,
    "forward_confidence": 0.80,
    "recency_months": 6,
    "min_mean_gross": 0.50,
    "min_round_trips_per_month": 1,
    "heldout_tolerance_z": 1.28,
}

ARMS: dict[str, dict[str, object]] = {
    "gp_overlay_1m_20260921d": {
        "workspace_reference": PACK,
        "initial_cash": 1_000_000,
        "tracking_error_cap": 0.08,
        "max_drawdown": 0.35,
        "active_max_drawdown": 0.15,
        "beta_min": 0.85,
        "beta_max": 1.15,
        **GATES,
        "research_directive": (
            "本臂账户 100 万元，带跟踪授权。第一件事：把 trade.BOOK 改成 gp_overlay，再 smoke。"
            "强制近满仓（仓位≥0.96），席位 80–100。"
            "股票池是决策日可见的沪深 300 成分，按时点从宏观域读，不得回退全市场。"
            "分数只是毛利率（收入−营业成本）/资产，不要 EP/BP，不要质量三柱混合（那条已关）。"
            "同批必须跑同一分数等额、以及纯基准权重。"
            "第一件要量的是仓位、β、跟踪误差，然后才是主动 IR。"
            "先读 refs/README.md 与 refs/families.md。改完 BOOK 再 smoke_backtest，然后完整研究期验证。"
            "方向被证伪而预算有余量时换下一个预登记方向。"
        ),
    },
    "noa_index_100k_20260921d": {
        "workspace_reference": PACK,
        "initial_cash": 100_000,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂账户 10 万元，不设跟踪授权。第一件事：把 trade.BOOK 改成 noa_index，再 smoke。"
            "席位 12。股票池是决策日可见的沪深 300 成分，不得回退全市场。"
            "指数内按行业权重配座，分数是 −NOA/滞后资产，不是 percent accrual（那条已关）。"
            "完整研究期之前按主动回撤 30% 预筛。P2：单一申万一级行业时间加权权重 ≤ 0.30。"
            "先读 refs/README.md 与 refs/families.md。改完 BOOK 再 smoke_backtest，然后完整研究期验证。"
            "方向被证伪而预算有余量时换下一个预登记方向。"
        ),
    },
    "cma_overlay_1m_20260921d": {
        "workspace_reference": PACK,
        "initial_cash": 1_000_000,
        "tracking_error_cap": 0.08,
        "max_drawdown": 0.35,
        "active_max_drawdown": 0.15,
        "beta_min": 0.85,
        "beta_max": 1.15,
        **GATES,
        "research_directive": (
            "本臂账户 100 万元，带跟踪授权。第一件事：把 trade.BOOK 改成 cma_overlay，再 smoke。"
            "强制近满仓（仓位≥0.96），席位 80–100。"
            "股票池是决策日可见的沪深 300 成分，不得回退全市场。"
            "创新性探索：只用 −资产增长做无界 overlay。不要叠盈利/应计。"
            "同批必须跑同一分数等额、以及纯基准权重。第一件要量的是仓位、β、跟踪误差。"
            "证伪：与已关质量书的投资柱秩相关 > 0.85 且主动 IR 同档。"
            "先读 refs/README.md 与 refs/families.md。改完 BOOK 再 smoke_backtest，然后完整研究期验证。"
            "方向被证伪而预算有余量时换下一个预登记方向。"
        ),
    },
    "cashdiv_pool_100k_20260921d": {
        "workspace_reference": PACK,
        "initial_cash": 100_000,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂账户 10 万元，不设跟踪授权。第一件事：把 trade.BOOK 改成 cashdiv_pool，再 smoke。"
            "可见现金分红/收盘价，席位 12，每业最多 2。不要做成盈利收益率或 OCF 收益率。"
            "完整研究期之前按主动回撤 30% 预筛。价格上限由座位金额自算。"
            "先读 refs/README.md 与 refs/families.md。改完 BOOK 再 smoke_backtest，然后完整研究期验证。"
            "方向被证伪而预算有余量时换下一个预登记方向。"
        ),
    },
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
