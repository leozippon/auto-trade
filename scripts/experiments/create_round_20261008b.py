#!/usr/bin/env python
"""The 2026-10-08 round (id 20261008b): five open arms on commitments, exclusion and a third sleeve.

Round 20261015's six arms all ended `no_edge` (logs/notes/review_20261008/
round_20261015_review.md): its five-year seed typed some events columns two
ways, so both five-year arms could read only Y1..Y2; the ensemble arm starved
its given sleeves at 72 seats on 100k and read the result as dilution; four
arms argued their exhaustion from the information-ratio bar; and the standard
screen control redrew whole books at every review. This round re-runs the
blocked commitments direction on the rebuilt seed, gives the exclusion layers
a core with an edge of its own, replaces the pre-announcement arm (that drift
is closed in the register) by the company's own guidance tested by its own
later numbers, takes the capital-structure commitments the two title screens
did not reach, and re-runs the ensemble at 16 + 16 seats after a baseline
replay showed the given sleeves keep their exposure there. Rules v3, one
references/open-rules.md byte for byte in the five packs
(configs/workspace_refs/open_*_20261008b): at most 30 seats a book (36 for the
ensemble), at least 12; a shuffled control keeps the candidate's holding
period (configs/starter_lib/controls.py `held_shuffle`) and every control's
turnover lies within 25 % of the candidate's; an exhaustion claim may not rest
on the bar and needs four full-span validations of distinct constructions.

| Arm | Direction | Geometry | Model | Lineage |
| --- | --- | --- | --- | --- |
| `open_guidance_100k_8y_qwen` | a company's own earnings guidance against its own later numbers: beaten, kept or missed, and its own revisions | eight years | local | none |
| `open_capital_100k_8y_mimo` | capital-structure commitments: rights issues, spin-offs, placement approval and subscribers, convertible redemption choices, buyback execution | eight years | MiMo | none |
| `open_ensemble_100k_8y_qwen` | a third sleeve beside `b30` and `esop60`, the given sleeves at 16 seats each, 36 in total | eight years | local | `ENSEMBLE_LINEAGE` (8 arms) |
| `open_tables_100k_5y_qwen` | commitments at a disclosed price in the events and fundamentals tables (re-run) | five years, events domain | local | none |
| `open_exclude_100k_5y_mimo` | exclusion layers from the events tables on a core with its own edge, judged as one book | five years, events domain | MiMo | none |

Two geometries. The eight-year arms share round 20261014's surface and seed
(`FUND_TITLE_PIT_VIEWS_SEED`). The five-year arms research 2020-07..2025-06 on
that surface plus seventeen events datasets (`_profiles.FIVE_YEAR`), on the
seed rebuilt after the type fix (`FIVE_YEAR_FIXED_PIT_VIEWS_SEED`); until that
prebuild reports status ok the fill holds them pending, and the timer creates
them once it does. Every arm's starter is replayed on its seed before the arm
is created (`_round.smoke`).

No arm claims a GPU. The boards are pinned to main board and ChiNext.

Fill order (ids end in `_20261008b`): `open_guidance_100k_8y_qwen`,
`open_capital_100k_8y_mimo`, `open_ensemble_100k_8y_qwen`,
`open_tables_100k_5y_qwen`, `open_exclude_100k_5y_mimo`.

Usage: create_round_20261008b.py <port> [--dry-run] [--fill] [experiment_id ...]
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
    ENSEMBLE_LINEAGE,
    FIVE_YEAR_100K,
    FIVE_YEAR_FIXED_PIT_VIEWS_SEED,
    FUND_TITLE_EIGHT_YEAR_100K,
    FUND_TITLE_PIT_VIEWS_SEED,
    MIMO,
)
from scripts.experiments._round import Round

PACKS = "configs/workspace_refs"

# What every open arm is told, after its own direction line. The packs hold the
# rules; this only points at them and names the round-0 checks.
COMMON = (
    "本臂是开放臂：包只给方向、约束、让结果诚实的对照与预算，机制、数据、特征、模型、构造与迭代由你决定。"
    "refs/README.md 写方向、挂载的数据与这个方向上已经知道的东西，refs/references/open-rules.md 写本轮各开放臂共同的规则（第三版），两者都是合同；"
    "refs/families.md 是这份记录上已经占用与关闭的方向，不是禁令。"
    "第 0 轮核对 broker_replay.initial_cash 为 10 万、benchmark_index 为 000852.SH、"
    "broker_replay.permitted_boards.boards 恰好是 main 与 gem、没有 GPU、研究期与 refs/README.md 一致、"
    "refs/README.md 列出的数据域都已挂载，任何一条不符就停。"
    "激励计划草案书与员工持股计划草案书在 Paper 上，与它们重叠的书不算产出（open-rules.md §3 第 5 条）。"
    "每本书同时至少 12 只、至多 30 只（集成臂几只袖子合计至多 36 只）。"
    "每个打算提名的候选在验证之前写一句主张，并带 lib/controls.py 里适用于它的标准对照，在同一 span 上；"
    "对照的年换手要在候选的 ±25 % 以内，候选永远不登记为对照。"
    "探针与离线筛选只用研究期的前半段；离线筛过的配置如实申报 offline_trials；策略不调用 context.nl()。"
    "提名照 refs/references/open-rules.md §5：合格且过冻结门就立即冻结；合格而只差去膨胀，就以 no_edge 结束并在 reason 第一句点名它；"
    "没有合格的，以 no_edge 写清学到了什么。information_ratio_bar 只决定能不能冻结，不是收尾的理由；"
    "以方向已经穷尽收尾之前，要有四次结构不同的构造做过整期验证，并写明剩下的回放年（§4）。"
)


def directive(direction: str) -> str:
    return f"方向：{direction}{COMMON}"


# The five-year arms name their geometry, selection and seed; the round's own
# overrides and seed are the eight-year surface.
FIVE_YEAR_ARM: dict[str, object] = {**FIVE_YEAR_100K, "pit_views_seed": FIVE_YEAR_FIXED_PIT_VIEWS_SEED}

ARMS: dict[str, dict[str, object]] = {
    "open_guidance_100k_8y_qwen_20261008b": {
        "workspace_reference": f"{PACKS}/open_guidance_100k_8y_20261008b",
        "research_directive": directive(
            "公司自己的业绩承诺被自己检验：业绩预告给出的区间与后来快报或定期报告里的数字相比是超出、落在其中还是没达到，"
            "以及公司在正式数字之前自己修正区间。预告之后的漂移本身已经关闭（refs/families.md 财务域一行），"
            "本臂比的是结果相对公司自己的承诺。"
        ),
    },
    "open_capital_100k_8y_mimo_20261008b": {
        **MIMO,
        "workspace_reference": f"{PACKS}/open_capital_100k_8y_20261008b",
        "research_directive": directive(
            "资本结构上的承诺：配股（控股股东承诺认购）、分拆上市、定增的批准这一步与按认购人和折价拆开的定增、"
            "可转债触发赎回条件之后强赎与不强赎的选择、按总股本下降量的回购执行。"
            "两次离线筛选已经走过的家族（refs/README.md 的表）不重走，要重开就照价申报。"
        ),
    },
    "open_ensemble_100k_8y_qwen_20261008b": {
        "workspace_reference": f"{PACKS}/open_ensemble_100k_8y_20261008b",
        "research_directive": directive(
            "Paper 上的激励计划草案书与员工持股计划草案书作固定的袖子，起步各 16 席，找一只与它们相关低的新袖子，"
            "使合并书的主动 IR 更高、主动回撤更小，三只袖子合计至多 36 席。提名合并书，证据是对没有新袖子的合并书之差；"
            "「给定袖子各自单独」与「没有新袖子的合并书」两条对照起步包已经接好，第一个探针批就跑。给定袖子的逻辑不改，只改份额与席位。"
        ),
        "lineage_arms": ENSEMBLE_LINEAGE,
    },
    "open_tables_100k_5y_qwen_20261008b": {
        **FIVE_YEAR_ARM,
        "workspace_reference": f"{PACKS}/open_tables_100k_5y_20261008b",
        "research_directive": directive(
            "事件域与财务域的表里，公司或内部人按披露的价格押上钱的地方：控股股东与高管在大跌之后的增持、回购的实际执行、"
            "溢价的大宗交易、按披露价格认购的定增、提前预约的定期报告披露，以及作为回应一侧的业绩预告与快报（它们在财务域）。"
            "这是上一轮被种子缺陷挡住的同一方向的重跑；五年上现实的产出是只差去膨胀的合格节点，试验要少、主张要先写。"
        ),
    },
    "open_exclude_100k_5y_mimo_20261008b": {
        **MIMO,
        **FIVE_YEAR_ARM,
        "workspace_reference": f"{PACKS}/open_exclude_100k_5y_20261008b",
        "research_directive": directive(
            "一个自己有边、不被占用的做多核心（股息率核心、表里的承诺核心或你提出的核心），加上从事件表来的剔除层"
            "（彩票型名字、限售股解禁、融资拥挤、龙虎榜、筹码套牢盘、审计意见下调），整本书作为一个整体被评判："
            "核心对它自己的标准对照，整本书对同一核心、同数的随机跳过。剔除单独成书从没过过主动 IR 0.75 的下限。"
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
