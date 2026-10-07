#!/usr/bin/env python
"""The 2026-10-14 round: six open-exploration arms on statements and titles together.

The packs before this round pre-registered batches, legs and kill lines, and
the sessions became careful executors of a designer's plan. This round gives
the Agent a broad direction and leaves the strategy space to it: each pack
(configs/workspace_refs/open_*_100k_8y_20261014) states a direction, what the
record already closed near it and a few external leads, and shares one
contract with the other four byte for byte -- the account, the controls that
make a result honest (a one-sentence claim before validating, the candidate's
own placebo on the same span, honest offline-screen declarations, no model
queries from a strategy), the budget's shape and a nomination rule: freeze a
qualifying node that passes the gate, name one that misses only the deflation
first in a `no_edge` reason, and otherwise end with what was learned. No legs,
batches, seat counts or signals are prescribed.

| Arm | Direction | Model | Lineage |
| --- | --- | --- | --- |
| `open_free_100k_8y_{qwen,mimo}` | anything on the surface | local, MiMo | none |
| `open_adapt_100k_8y_mimo` | adaptive constructions, beating their own static version and a shuffled-state or shifted-signal placebo | MiMo | none |
| `open_earnings_100k_8y_qwen` | earnings and financial information | local | `FUNDAMENTALS_LINEAGE` (11 arms, 72 trials, bar floor about 1.55) |
| `open_commit_100k_8y_qwen` | company-commitment titles beyond the incentive drafts | local | `COMMITMENT_LINEAGE` (6 arms, 66 trials, bar floor about 1.53) |
| `open_avoid_100k_8y_mimo` | long books whose edge is exclusion | MiMo | none |

The surface is new: daily bars, universe, the four macro tables, the ten
fundamentals datasets and the `anns_d` titles on one arm, which no seed carried
before. Its seed must be prebuilt for exactly this selection before the round
is dry-run or filled (`FUND_TITLE_PIT_VIEWS_SEED`, built with
scripts/data/prebuild_pit_views_seed.py). The boards are pinned to main board
and ChiNext, as round 20261013 pinned them, and the rule set is the current
generation's (R4), which the launcher pins. Every arm claims one GPU, so the
Agent may train a model if it chooses: the fill queue holds a GPU arm while
too few cards are free and creates the arms behind it, so the order below puts
the arms likeliest to train first. Each pack's starter probes the device and
keeps the CPU path the contract requires.

Fill order (ids end in `_20261014`): `open_free_100k_8y_qwen`,
`open_free_100k_8y_mimo`, `open_adapt_100k_8y_mimo`,
`open_earnings_100k_8y_qwen`, `open_commit_100k_8y_qwen`,
`open_avoid_100k_8y_mimo`.

Usage: create_round_20261014.py <port> [--dry-run] [--fill] [experiment_id ...]
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
    COMMITMENT_LINEAGE,
    FUND_TITLE_EIGHT_YEAR_100K,
    FUND_TITLE_PIT_VIEWS_SEED,
    FUNDAMENTALS_LINEAGE,
    MIMO,
)
from scripts.experiments._round import Round

PACKS = "configs/workspace_refs"

# What every open arm is told, after its own direction line. The packs hold the
# rules; this only points at them and names the round-0 checks.
COMMON = (
    "本臂是开放臂：包只给方向、约束、让结果诚实的对照与预算，机制、数据、特征、模型、构造与迭代由你决定。"
    "refs/README.md 写方向与这个方向上已经知道的东西，refs/references/open-rules.md 写各开放臂共同的规则，两者都是合同；"
    "refs/families.md 是这份记录上已经关闭的方向，不是禁令。"
    "第 0 轮核对 broker_replay.initial_cash 为 10 万、benchmark_index 为 000852.SH、"
    "broker_replay.permitted_boards.boards 恰好是 main 与 gem、财务域与公告标题都已挂载，任何一条不符就停。"
    "每个打算提名的候选在验证之前写一句主张，并带它自己在同一 span 上的对照；离线筛过的配置如实申报 offline_trials；"
    "策略不调用 context.nl()。"
    "提名照 refs/references/open-rules.md §5：合格且过冻结门就冻结；合格而只差去膨胀，就以 no_edge 结束并在 reason 第一句点名它；"
    "没有合格的，以 no_edge 写清学到了什么。门槛读每行的 information_ratio_bar，冻结门与毕业条件只看运行事实。"
)


def directive(direction: str) -> str:
    return f"方向：{direction}{COMMON}"


FREE = directive("在日线、指数与行业、财务报表与公告标题这块数据面上为 10 万账户找一条只做多的选股边，方向之内不设限。")

ARMS: dict[str, dict[str, object]] = {
    "open_free_100k_8y_qwen_20261014": {
        "workspace_reference": f"{PACKS}/open_free_100k_8y_20261014",
        "research_directive": FREE,
    },
    "open_free_100k_8y_mimo_20261014": {
        **MIMO,
        "workspace_reference": f"{PACKS}/open_free_100k_8y_20261014",
        "research_directive": FREE,
    },
    "open_adapt_100k_8y_mimo_20261014": {
        **MIMO,
        "workspace_reference": f"{PACKS}/open_adapt_100k_8y_20261014",
        "research_directive": directive(
            "随行情改变的构造——在尾部数据上重拟合或重选、按决策时可见的市场状态取条件；"
            "提名的候选要胜过它自己的静态版本与一条打乱状态或平移信号的安慰剂（refs/README.md）。"
        ),
    },
    "open_earnings_100k_8y_qwen_20261014": {
        "workspace_reference": f"{PACKS}/open_earnings_100k_8y_20261014",
        "research_directive": directive(
            "盈利与财务信息：带时点的季度报表、业绩预告与快报、分红政策的变化、PEAD 一类的漂移，以及它们与公告事件的组合。"
        ),
        "lineage_arms": FUNDAMENTALS_LINEAGE,
    },
    "open_commit_100k_8y_qwen_20261014": {
        "workspace_reference": f"{PACKS}/open_commit_100k_8y_20261014",
        "research_directive": directive(
            "股权激励计划草案之外的公司承诺类公告：计划、完成、注销或终止、事件组合、持有结构与事件质量；"
            "草案书是参考答案，不要重新发现它。"
        ),
        "lineage_arms": COMMITMENT_LINEAGE,
    },
    "open_avoid_100k_8y_mimo_20261014": {
        **MIMO,
        "workspace_reference": f"{PACKS}/open_avoid_100k_8y_20261014",
        "research_directive": directive(
            "一本只做多的书，边来自剔除，或来自剔除与一个中性或事件驱动的核心的组合；剔除的主张要胜过同样多的随机剔除。"
        ),
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=FUND_TITLE_PIT_VIEWS_SEED,
    # One card per arm, claimed by the console when it starts the worker.
    overrides={**FUND_TITLE_EIGHT_YEAR_100K, "permitted_boards": ["main", "gem"], "gpu_count": 1},
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
