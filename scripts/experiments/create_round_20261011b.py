#!/usr/bin/env python
"""The 2026-10-11 round's second pair: the open lane on the fundamentals pack, last in the queue.

Round 20261011 defined two pairs. This one is a second open pair on the
unchanged fundamentals pack (`FUND_PACK`), CPU only: the open lane under
today's rules (taxed Broker, the raw-excess freeze condition, the late-year
and split-half readings). It inherits every earlier open-lane arm on the 100k
eight-year surface that is still on disk, and its directive names what those
sessions closed, so its bar starts high.

It is the lowest-priority work in the queue. The fill reads the round files
one by one, so in its own file, listed last in the research cron, the pair
takes only the slots the GPU arms of rounds 20261011, 20261009 and 20261008
leave. The ids keep the round's date: nothing was created from them.

The round-level parameters are round 20261008's (the fundamentals pack's
latest, round 20261010, shares them value for value), on the same taxed seed.
Text-evidence scoring stays on the local model in both arms.

The pair (`fund_open2_100k_8y_qwen` / `_mimo`) was withdrawn on 2026-10-05
before either arm recorded a validation: the inherited trials put its bar at
1.739 before its own first trial, against a best open-lane row of 0.777 on
this surface. Nothing was created from this file that remains; its ids are
retired and the round is closed.

Usage: create_round_20261011b.py <port> [--dry-run] [--fill] [experiment_id ...]
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

from scripts.experiments._profiles import (
    FUND_EIGHT_YEAR_100K,
    FUND_PACK,
    TAXED_FUND_PIT_VIEWS_SEED,
)
from scripts.experiments._round import Round

# Every open-lane arm on the 100k eight-year surface still on disk: the nine
# daily-only sessions of rounds 20261002-20261004 and the two fundamentals
# sessions of round 20261005.
OPEN2_LINEAGE = [
    "open_research_100k_8y_qwen_20261002",
    "open_research_100k_8y_mimo_20261002",
    "open_research_100k_8y_qwen_r2_20261003",
    "open_research_100k_8y_mimo_r2_20261003",
    "open_research_100k_8y_deepseek_r1_20261004",
    "open_research_100k_8y_deepseek_r2_20261004",
    "open_research_100k_8y_deepseek_r3_20261004",
    "open_research_100k_8y_mimo_r3_20261004",
    "open_research_100k_8y_qwen_r3_20261004",
    "fund_open_100k_8y_qwen_20261005",
    "fund_open_100k_8y_mimo_20261005",
]

OPEN2_DIRECTIVE = (
    "本臂走开放线的第二场：在 10 万元账户上、从挂上财务域的八年数据面里自己提出家族，成批预登记并检验，"
    "以冻结或 no_edge 收尾；只用 CPU，不做逐股序列网络。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停。"
    "此前十一场开放会话测过并关闭的不要重走，它们的试验已由宿主并入本臂的门槛，重开不再申报："
    "只读日线与估值列的九场收敛到销售规模（流通市值除以市销率取最大）、指数移出与成分进出的集合、"
    "负端剔除筛（下跌日成交额占比、换手、β 等剔除，多数输给同数量的随机剔除），以及低波动、估值修正、送转除权与新股窗口；"
    "挂上财务域的上一场测过事件意外、现金流收益率、报表红旗、披露完整性、应计、财务特征上的逻辑回归、符号窗与报表价值，"
    "最好的报表红旗整期前半成立、后半反转，报表价值被换列对照追平，都离门槛很远。"
    "事件线冻结过的分红实施书计税后整期主动 IR 为负，不要以任何形式重走。"
    "至少两个结构不同的家族，其中至少一个建在财务域上；每个分数带自己的 c_shuf，"
    "主张落在构造、筛子或事件时点上再带随机化的安慰剂，事件书与学习型排序器带 refs/README.md 规定的对照；"
    "先 Y1..Y4 探针、再全期；全期胜过自己对照的家族至少再细化一步。"
    "Broker 按持有期扣红利税：买入后不满一个月卖出，按持有期间到账现金分红的 20% 在卖出时扣（满一个月不满一年 10%），"
    "跨除权日的书把它算进每个回合。"
    "提名的行还要过运行事实里的原始超额条件：成本压力下自己的权益要赢过基准；"
    "提名前读最后两个研究年与前后两半的原始、主动与朴素选名读数（书减零技能面板，不回归）以及书对基准的市场载荷，"
    "都从行里取，写进 reason。"
    "杀死线只结束候选或家族，不结束本臂；至多四批，登记过的轴没测完就收尾要在 reason 里说明；"
    "门槛读每行的 information_ratio_bar，不自己算；重开 refs/families.md 里其余家族照 README「什么算重复」申报 offline_trials。"
    "学习型候选的训练种子写成一行 SEED_BASE 整数常量，复现只改这一行，冻结时把其他种子上的整期行登记为 finish_session 的 seed_replicates。"
)


OPEN2_ARM: dict[str, object] = {
    "workspace_reference": FUND_PACK,
    "lineage_arms": OPEN2_LINEAGE,
    "research_directive": OPEN2_DIRECTIVE,
}

# The pair (`fund_open2_100k_8y_qwen`, and `_mimo` with the `MIMO` roles) was
# withdrawn and its ids retired; `OPEN2_ARM` stays as the record of what was
# queued.
ARMS: dict[str, dict[str, object]] = {}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=TAXED_FUND_PIT_VIEWS_SEED,
    overrides=FUND_EIGHT_YEAR_100K,
    closed=True,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
