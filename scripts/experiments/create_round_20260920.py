#!/usr/bin/env python
"""The 2026-09-20 round: the first research-forward arms, on one seed.

No arm is defined yet. The round already fixes what its arms will share: the
research geometry and budgets of `_round.BASE_OVERRIDES`, and the dataset
selection the seed below was prebuilt for. A dataset whose rows do not span
2020-07 through the Held-out replay end is left out of it: `intraday_flow`, the
text datasets other than `report_rc`, and the vendor variants that start after
2020.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
    },

A reference pack carried over from a Fold-era arm must state its contamination
in this docstring when an arm names it: which of its probes read rows after
research end.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260920.py <port> [--dry-run] [experiment_id ...]
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

# The one prebuilt view tree every arm hardlinks from, built by
# scripts/data/prebuild_pit_views_seed.py for exactly the selection below and
# the default research geometry; gitignored, so an operator precondition.
PIT_VIEWS_SEED = "data/pit_views_seed_research_20260920"

FUNDAMENTAL_DATASETS = [
    "income_vip",
    "balancesheet_vip",
    "cashflow_vip",
    "fina_indicator_vip",
    "forecast_vip",
    "express_vip",
    "dividend",
    "fina_audit",
    "fina_mainbz_vip",
    "disclosure_date",
]
MACRO_DATASETS = [
    "cn_gdp",
    "cn_cpi",
    "cn_ppi",
    "cn_pmi",
    "cn_m",
    "sf_month",
    "shibor",
    "shibor_lpr",
    "index_daily",
    "index_dailybasic",
    "sw_daily",
    "fut_basic",
    "fut_mapping",
    "fut_daily",
    "opt_basic",
    "opt_daily",
    "cb_basic",
    "cb_daily",
    "cb_call",
]
EVENTS_DATASETS = [
    "margin",
    "margin_detail",
    "margin_secs",
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
    "top10_floatholders",
    "report_rc",
]
TEXT_DATASETS = ["report_rc"]

ARMS: dict[str, dict[str, object]] = {
    # The one GPU arm. Its pack was written from probes that read no row after
    # research end (logs/notes/review_20260915/NA1_next_arm_research.md), so the
    # forward period is unseen for this direction; the pack discloses that its
    # research-period evidence sits below the Alpha158 + LightGBM ranker.
    "gru_ranker_20260920": {
        "workspace_reference": "configs/workspace_refs/gru_ranker_20260920",
        # fit trains on a card in every replay: the starter has no CPU path.
        "gpu_count": 1,
        "research_directive": (
            "本臂是固定方向的臂：在 GPU 上训练的 GRU 日频量价序列排序器。先读 refs/README.md 与 "
            "refs/families.md——主候选、允许的变体轴、报告用对照与本臂终止规则只以 families.md 为准。"
            "从 refs/starter 起步（fit 内按季度在滚动的最近三年上重训、固定种子、每周复核 15 只、"
            "每次最多换 2 只），先 smoke_backtest，再做完整研究期验证。策略必须在 CUDA 上训练和打分，"
            "不写 CPU 路径。Alpha158 + LightGBM 排序器是必须同批汇报的对照，不是冻结门槛；方向研究里"
            "本家族的研究期读数低于它，这是已披露的起点。没有候选证明边际时按 families.md 以 "
            "no_edge 结束，不换家族。"
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
