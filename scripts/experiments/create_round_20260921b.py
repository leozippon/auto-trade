#!/usr/bin/env python
"""The 2026-09-21b round: four openings under the same panel, two accounts.

The dataset selection and the prebuilt seed are the 2026-09-19 round's -- the
2026-09-20 selection plus `index_weight` -- imported rather than restated so
the rounds that mount it cannot drift apart by a typo. All four arms need it:
two build their universe from the constituent cross-section, and all four need
the host's zero-skill panel to match replacements on CSI 300 membership.

Every arm mounts `configs/workspace_refs/capital_aware_20260921b`. What differs
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

`_round.Round.request_params` merges the arm last. The two mandated arms state
the 8 % cap and their own drawdowns 0.35 / 0.15 and beta 0.85-1.15. The two
un-mandated arms state the rules' own 0.45 / 0.30 and do not name a cap.
Every arm also names the statistical bars (IR, DSR, positive-year share,
full-span validations, forward confidence, recency, activity, Held-out z)
so the create request records a choice rather than inheriting a hidden pair
of packages.

The four openings, and nothing else. Bounded in-mandate tilt is closed (no
edge inside the mandate; the edge sat on an unbounded book whose TE/β broke
the band) -- that falsifies the box, not the mandate. In-index daily
price-volume at 100k froze on a research-period active IR of 1.23 and died on
the shared forward F2 (lower bound -5.1 %): a research-period IR above 1 is
not graduation. 20-day reversal at 100k is closed (active IR negative at
N = 8/10/12/15). Value/low-vol as a 100k in-index opening stays closed.

Contamination, per arm. Operator-side only.

- Round-level. The decision to spend four arms on these openings was taken
  after six frozen books were discarded on the shared forward window, which
  the operator has read. Every reading behind the four choices is
  research-period, except the F2 close of the one freeze this round produced.
- `fullcash_overlay_1m_20260921b`. Cites the close of the bounded-tilt arm.
  That arm's forward verdict was never taken; its research-period
  `no_deliverable` is. The construction is new (unbounded overlay, no name or
  industry box). Forward verdict is partly a pipeline calibration of a
  mandate the operator already had reason to keep trying.
- `indneutral_value_1m_20260921b`. The family was closed twice in the old
  pool; it has not submitted a book under the active panel plus
  `index_weight`. Clean apart from the round-level statement.
- `resid_momentum_100k_20260921b`. Skips 20-day reversal because that opening
  read negative active IR at every seat in 8-15. No frozen lineage. Clean
  apart from the round-level statement.
- `eyield_concentrated_100k_20260921b`. Avoids PB because a sibling arm is
  still measuring it, and avoids in-index daily price-volume because that
  freeze died on F2. The F2 reading is operator-side contamination of the
  choice, not of the family's research-period evidence.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260921b.py <port> [--dry-run] [--fill] [experiment_id ...]
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

PACK = "configs/workspace_refs/capital_aware_20260921b"

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
    "fullcash_overlay_1m_20260921b": {
        "workspace_reference": PACK,
        "initial_cash": 1_000_000,
        "tracking_error_cap": 0.08,
        "max_drawdown": 0.35,
        "active_max_drawdown": 0.15,
        "beta_min": 0.85,
        "beta_max": 1.15,
        **GATES,
        "research_directive": (
            "本臂账户 100 万元，带跟踪授权。第一件事：确认 output 里 trade.BOOK 为 "
            "index_overlay（起步包默认就是它），不要改成有界盒子。"
            "强制近满仓（仓位≥0.96），席位 80–100。"
            "股票池是决策日可见的沪深 300 成分，按时点从宏观域读，不得回退全市场。"
            "从基准权重出发，用无界分数做 overlay，不要名字数或行业硬盒子。"
            "同批必须跑同一分数等额、以及纯基准权重。"
            "第一件要量的是仓位、β、跟踪误差，然后才是主动 IR。"
            "先读 refs/README.md 与 refs/families.md。改完 BOOK 再 smoke_backtest，然后完整研究期验证。"
            "方向被证伪而预算有余量时换下一个预登记方向。"
        ),
    },
    "indneutral_value_1m_20260921b": {
        "workspace_reference": PACK,
        "initial_cash": 1_000_000,
        "tracking_error_cap": 0.08,
        "max_drawdown": 0.35,
        "active_max_drawdown": 0.15,
        "beta_min": 0.85,
        "beta_max": 1.15,
        **GATES,
        "research_directive": (
            "本臂账户 100 万元，带跟踪授权。第一件事：把 trade.BOOK 改成 index_indneutral，再 smoke。"
            "股票池是决策日可见的沪深 300 成分，不得回退全市场。"
            "书的行业权重按构造对齐指数行业权重（用来控 TE/β），行业内按价值（EP/BP）选名；"
            "这不是事后行业帽，也不是桶填位。近满仓，席位 80–100。"
            "同批跑同一分数等额与纯基准权重。"
            "先读 refs/README.md 与 refs/families.md。改完 BOOK 再 smoke_backtest，然后完整研究期验证。"
            "方向被证伪而预算有余量时换下一个预登记方向。"
        ),
    },
    "resid_momentum_100k_20260921b": {
        "workspace_reference": PACK,
        "initial_cash": 100_000,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂账户 10 万元，不设跟踪授权。第一件事：把 trade.BOOK 改成 pool_momentum，再 smoke。"
            "席位 8–12，价格上限由座位金额自算。"
            "开局是 60–120 日相对市场的残差动量，禁止 20 日反转开局。周度复核。"
            "完整研究期之前按主动回撤 30% 预筛。"
            "先读 refs/README.md 与 refs/families.md。改完 BOOK 再 smoke_backtest，然后完整研究期验证。"
            "方向被证伪而预算有余量时换下一个预登记方向。"
        ),
    },
    "eyield_concentrated_100k_20260921b": {
        "workspace_reference": PACK,
        "initial_cash": 100_000,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂账户 10 万元，不设跟踪授权。第一件事：把 trade.BOOK 改成 pool_eyield，再 smoke。"
            "席位 8–12，价格上限由座位金额自算。"
            "开局是盈利收益率与现金流收益率，不要 PB，不要指数内日频量价。周度复核。"
            "完整研究期之前按主动回撤 30% 预筛。"
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
