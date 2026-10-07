#!/usr/bin/env python
"""The 2026-10-15 round: six open-exploration arms away from the occupied commitment families.

Round 20261014 gave six arms broad directions; three of them re-derived the
one known edge (incentive-plan drafts), the one new mechanism (employee
shareholding-plan drafts) came from the narrowest pack, and agent-written
controls decided two outcomes wrongly (logs/notes/review_20261008/). Both
books are now in Paper, so their families are occupied: a book that overlaps
either earns nothing. This round keeps the open format -- a direction, the
shared rules and the budget, no prescribed legs -- and changes the rules the
audit found wanting. One references/open-rules.md, byte-identical in the six
packs (configs/workspace_refs/open_*_20261015), adds: the occupied families
and their overlap test; a re-open priced at the family's lineage trial count;
standard controls shipped as configs/starter_lib/controls.py (late entry,
shuffled assignment, random skip, static version), of which every nomination
reports the one its construction needs; probes and offline screens on the
first half of the research years only; freezing as soon as a qualifying node
passes the gate, and a written reason for ending with more than half the
replay budget unused.

| Arm | Direction | Geometry | Model | Lineage |
| --- | --- | --- | --- | --- |
| `open_tables_100k_5y_qwen` | commitments at a disclosed price in the events tables: insider and controller purchases, block trades at a premium, early scheduled disclosure, pre-announcements | five years, events domain | local | none |
| `open_chips_100k_5y_mimo` | ownership structure and exclusion: holder counts, top-ten holders, chip distribution, lock-up expiries, margin, dragon-tiger and limit lists | five years, events domain | MiMo | none |
| `open_ensemble_100k_8y_qwen` | a low-correlation sleeve beside the two Paper books, nominated as the combined book | eight years | local | `SLEEVE_LINEAGE` (7 arms, 90 trials, bar floor about 1.57) |
| `open_distress_100k_8y_qwen` | regulatory and distress titles: inquiry letters, investigations, warnings, delisting-risk warnings and their removal | eight years | local | none |
| `open_cluster_100k_8y_mimo` | lead-lag diffusion within industries and index membership; no concept table is point-in-time over a research period, so eight years | eight years | MiMo | none |
| `open_micro_100k_8y_mimo` | daily-bar microstructure: overnight against intraday, price limits, gaps, volume-conditioned reversal | eight years | MiMo | none |

Two geometries. The eight-year arms share round 20261014's surface and seed
(`FUND_TITLE_PIT_VIEWS_SEED`): daily bars, universe, four macro tables, the
ten fundamentals datasets and the `anns_d` titles. The five-year arms research
2020-07..2025-06, the years the lake's events domain covers, on that surface
plus seventeen events datasets (`_profiles.FIVE_YEAR`); their seed must be
prebuilt for exactly that selection (`FIVE_YEAR_PIT_VIEWS_SEED`, built with
scripts/data/prebuild_pit_views_seed.py, logs/data/seed_5y_full_20261015/).
A lineage must research the same period, so no five-year arm inherits one.
The gates are the same: a positive-year share of 0.75 asks for four of five
years, and the deflation bar rises by itself on fewer days.

No arm claims a GPU (BASE_OVERRIDES' `gpu_count` 0 stands): the record's
trained books have not held up, the starters drop the device helper, and a CPU
model is priced as a trial like any other screen. With nothing held for a
card, the fill creates the arms as fast as the running-arm limits allow, in
the order of the table. The boards are pinned to main board and ChiNext and
the rule set is the current generation's, which the launcher pins.

Fill order (ids end in `_20261015`): `open_tables_100k_5y_qwen`,
`open_chips_100k_5y_mimo`, `open_ensemble_100k_8y_qwen`,
`open_distress_100k_8y_qwen`, `open_cluster_100k_8y_mimo`,
`open_micro_100k_8y_mimo`.

Usage: create_round_20261015.py <port> [--dry-run] [--fill] [experiment_id ...]
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
    FIVE_YEAR_100K,
    FIVE_YEAR_PIT_VIEWS_SEED,
    FUND_TITLE_EIGHT_YEAR_100K,
    FUND_TITLE_PIT_VIEWS_SEED,
    MIMO,
    SLEEVE_LINEAGE,
)
from scripts.experiments._round import Round

PACKS = "configs/workspace_refs"

# What every open arm is told, after its own direction line. The packs hold the
# rules; this only points at them and names the round-0 checks.
COMMON = (
    "本臂是开放臂：包只给方向、约束、让结果诚实的对照与预算，机制、数据、特征、模型、构造与迭代由你决定。"
    "refs/README.md 写方向、挂载的数据与这个方向上已经知道的东西，refs/references/open-rules.md 写各开放臂共同的规则，两者都是合同；"
    "refs/families.md 是这份记录上已经占用与关闭的方向，不是禁令。"
    "第 0 轮核对 broker_replay.initial_cash 为 10 万、benchmark_index 为 000852.SH、"
    "broker_replay.permitted_boards.boards 恰好是 main 与 gem、没有 GPU、研究期与 refs/README.md 一致、"
    "refs/README.md 列出的数据域都已挂载，任何一条不符就停。"
    "激励计划草案书与员工持股计划草案书在 Paper 上，与它们重叠的书不算产出（open-rules.md §3 第 5 条）。"
    "每个打算提名的候选在验证之前写一句主张，并带 lib/controls.py 里适用于它的标准对照，在同一 span 上；"
    "探针与离线筛选只用研究期的前半段；离线筛过的配置如实申报 offline_trials；策略不调用 context.nl()。"
    "提名照 refs/references/open-rules.md §5：合格且过冻结门就立即冻结；合格而只差去膨胀，就以 no_edge 结束并在 reason 第一句点名它；"
    "没有合格的，以 no_edge 写清学到了什么。门槛读每行的 information_ratio_bar，冻结门与毕业条件只看运行事实。"
)


def directive(direction: str) -> str:
    return f"方向：{direction}{COMMON}"


# The five-year arms name their geometry, selection and seed; the round's own
# overrides and seed are the eight-year surface.
FIVE_YEAR_ARM: dict[str, object] = {**FIVE_YEAR_100K, "pit_views_seed": FIVE_YEAR_PIT_VIEWS_SEED}

ARMS: dict[str, dict[str, object]] = {
    "open_tables_100k_5y_qwen_20261015": {
        **FIVE_YEAR_ARM,
        "workspace_reference": f"{PACKS}/open_tables_100k_5y_20261015",
        "research_directive": directive(
            "事件域的表里，有人按披露的价格押上钱的地方：控股股东与高管的增持、回购的执行、溢价的大宗交易、"
            "按披露价格认购的定增、提前预约的定期报告披露，以及作为回应一侧的业绩预告与快报。"
        ),
    },
    "open_chips_100k_5y_mimo_20261015": {
        **MIMO,
        **FIVE_YEAR_ARM,
        "workspace_reference": f"{PACKS}/open_chips_100k_5y_20261015",
        "research_directive": directive(
            "事件域里的持有结构与剔除：股东户数与集中度、前十大股东、筹码分布、解禁、两融、龙虎榜与游资、涨停与彩票型名字；"
            "证据支持时作做多信号，否则作一个真的、有对照的核心上的剔除层。这些表在记录里只在空头一侧读出过信息，"
            "整本书的主动 IR 下限是 0.75。"
        ),
    },
    "open_ensemble_100k_8y_qwen_20261015": {
        "workspace_reference": f"{PACKS}/open_ensemble_100k_8y_20261015",
        "research_directive": directive(
            "Paper 上的激励计划草案书与员工持股计划草案书作固定的袖子，找一只与它们相关低的新袖子，"
            "使合并书的主动 IR 更高、主动回撤更小；提名合并书，证据是对没有新袖子的合并书之差。"
            "给定袖子的逻辑不改，只改份额与席位。"
        ),
        "lineage_arms": SLEEVE_LINEAGE,
    },
    "open_distress_100k_8y_qwen_20261015": {
        "workspace_reference": f"{PACKS}/open_distress_100k_8y_20261015",
        "research_directive": directive(
            "监管与困境公告：问询函与关注函、立案调查与处罚、监管函与警示函、退市风险警示与摘帽、非标准审计意见、诉讼、担保与违约；"
            "困境解除之后的反向做多、之前与之中的剔除，以及各自的时点。"
        ),
    },
    "open_cluster_100k_8y_mimo_20261015": {
        **MIMO,
        "workspace_reference": f"{PACKS}/open_cluster_100k_8y_20261015",
        "research_directive": directive(
            "同一行业与指数成分里的信息扩散：大市值带小市值、成分带非成分、同行公告与业绩预告向还没公告的公司外溢；"
            "跨名字的先后关系，不是给行业排名。"
        ),
    },
    "open_micro_100k_8y_mimo_20261015": {
        **MIMO,
        "workspace_reference": f"{PACKS}/open_micro_100k_8y_20261015",
        "research_directive": directive(
            "日线的开高低收、量与额在排序因子之外说了什么：隔夜与日内收益的分解、涨跌停的动力学、公告前后的开盘跳空、"
            "以成交量为条件的反转与延续、振幅与影线；每个候选是一条写得出来的规则，不是学出来的价量排序，先算它的换手成本。"
        ),
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=FUND_TITLE_PIT_VIEWS_SEED,
    overrides={**FUND_TITLE_EIGHT_YEAR_100K, "permitted_boards": ["main", "gem"]},
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
