#!/usr/bin/env python
"""The 2026-09-20 round: the first research-forward arms, on one seed.

The arms carry 20260921 ids: the first launch of these four directions was
retired on 2026-09-15, when research became one session per arm, and an id is
never reused. The reference packs keep their own directory names.

Four arms share the research geometry and budgets of `_round.BASE_OVERRIDES`
and the dataset selection the seed below was prebuilt for. A dataset whose rows
do not span 2020-07 through the Held-out replay end is left out of it:
`intraday_flow`, the text datasets other than `report_rc`, and the vendor
variants that start after 2020.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
    },

Contamination, per arm. Operator-side only: the packs themselves carry no
reading from after research end, so none of this reaches the Agent, but it
decides how much each forward verdict is worth.

- gru_ranker_20260921: forward unseen. Its direction probes
  (logs/notes/review_20260915/NA1_next_arm_research.md) read no row after
  research end.
- open_research_20260921: forward unseen for its direction, which nobody has
  chosen yet. Its controls come from the same research-only NA1 study; some of
  its lessons and closed families were drawn from Fold-era probes that read
  later rows, and the pack keeps them only as prohibitions, without readings.
- alpha158_lgbm_20260921: the graduate lineage (github_confirm_20260917). Its
  selection used rows after research end and the operator has seen its forward
  behaviour, so its forward verdict is a pipeline baseline, not evidence.
- defensive_quality_20260921 (pack carried over from the Fold-era arm
  defensive_quality_20260918): its direction probes
  (logs/notes/review_20260912/DIR4_more_arms.md, DIR6) read rows through
  mid-2026, the whole forward period. The pack keeps only calendar-year readings
  up to 2024, but the choice of its two legs is contaminated, so its forward
  verdict is not independent evidence either.

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
    # The one GPU arm; the pack discloses that its research-period evidence sits
    # below the Alpha158 + LightGBM ranker.
    "gru_ranker_20260921": {
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
    # The reference arm: its forward verdict calibrates the pipeline.
    "alpha158_lgbm_20260921": {
        "workspace_reference": "configs/workspace_refs/alpha158_lgbm_20260920",
        "research_directive": (
            "本臂是参考臂：旧流程一个冻结产物里的 Alpha158 + LightGBM 截面排序器，放到研究会话流程里。"
            "先读 refs/README.md 与 refs/families.md——主候选 l1、它与冻结产物的差异、允许的变体轴、"
            "提名条件与本臂终止规则只以 families.md 为准。从 refs/starter 起步（它就是 l1：fit 内按季度"
            "在滚动的最近三年上重训、每周复核 15 只、每次最多换 2 只），先 smoke_backtest，再按 "
            "refs/exploration-plan.md 做完整研究期验证。训练在容器 CPU 上完成，fit 失败直接报错，不设退路"
            "打分器；一轮只改一个常量。没有候选满足提名条件时按 families.md 以 no_edge 结束，不换家族。"
        ),
    },
    "open_research_20260921": {
        "workspace_reference": "configs/workspace_refs/open_research_20260920",
        "research_directive": (
            "本臂是开放方向的臂：不指定机制家族，在研究期上找一个只做多、15–30 只、有经济解释并在书本"
            "尺度上成立的日频策略。先读 refs/README.md 与 refs/standards.md——证据标准、每个候选都要比的"
            "对照、提名条件与收尾规则只以 standards.md 为准。refs/starter 的 o1 只是能跑的基线，不是推荐"
            "的机制：先 smoke_backtest；方向筛选在离线做（refs/exploration-plan.md），回放只用来确认。"
            "不重开 standards.md 列出的已关闭家族，不以 Alpha158 + LightGBM 排序器或序列网络作候选。"
            "预算用尽前仍没有候选满足提名条件时以 no_edge 结束。"
        ),
    },
    "defensive_quality_20260921": {
        "workspace_reference": "configs/workspace_refs/defensive_quality_20260918",
        "research_directive": (
            "本臂是固定方向的臂：防御型质量——现金流盈利质量（低应计、高经营现金流/资产，取首版报表）"
            "与低 60 日残差波动两腿等权合成的月度只做多篮子，不训练。先读 refs/README.md 与 "
            "refs/families.md——主候选 s1、两条可提名的消融腿、三个对照、硬门与本臂终止规则只以 "
            "families.md 为准。从 refs/starter 起步，先 smoke_backtest，再按 refs/exploration-plan.md "
            "的两轮批次做完整研究期验证；关闭一条腿要两份独立证据。两腿都关或没有候选证明边际时按 "
            "families.md 以 no_edge 结束，不换家族。"
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
