#!/usr/bin/env python
"""The 2026-10-02 round: 100k arms, each pack run by two models at once.

The first wave of the 2026-10-01 round closed three arms without a deliverable
(logs/notes/round_20261001/RESULTS_first_wave.md). On the 100k shape the tree
ranker over the 44 census inputs read a full-span active IR of 0.815 with five
positive years of eight, while its no-model control on the same book, the
equal-weight composite of the per-date ranks with each sign fitted on the
trailing window, read 0.849 with seven of eight and an active drawdown of
0.205. A control cannot be nominated, and no session refined after a full-span
reading: 52 to 76 of 96 replay-years stayed unused.

This round keeps every arm on a 100k account and changes two things
(logs/notes/round_20261002/DESIGN.md). A kill line ends a candidate, not the
arm: the packs allow four batches and say which registered axes remain after
each kind of failure. And each pack is run twice at the same time, once by the
local model and once by the hosted `mimo-v2.6-flash`, with the same budgets,
lineage and compaction threshold, so the two sessions can be compared on
identical work. A pair looks at one family twice; if both arms of a pair
freeze the same family that is one result.

The queue, in fill order (ids end in `_20261002`):

- `open_research_100k_8y_qwen` / `_mimo`: no family fixed; the session
  proposes its own on the eight-year data surface, pool and seats open.
- `census_lin_csi1000_100k_8y_qwen` / `_mimo`: the sign-fitted linear composite
  as the candidate, against a shuffled score and the same composite with its
  signs frozen at the first fit. It inherits the 100k tree arm, and its first
  batch declares the two control readings that selected the family.

Hosted arms keep text-evidence scoring on the local model, so nothing from the
forward or Held-out replays leaves the machine.

Usage: create_round_20261002.py <port> [--dry-run] [--fill] [experiment_id ...]
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

# The hosted arm of a pair, MIMO, first stated here, is shared with the rounds
# after it.
from scripts.experiments._profiles import MIMO
from scripts.experiments._round import Round
from scripts.experiments.create_round_20260927 import (
    CSI1000,
    DIVIDEND_PIT_VIEWS_SEED,
    EIGHT_YEAR,
    GATES,
)

OPEN_PACK = "configs/workspace_refs/open_research_100k_8y_20261002"
CENSUS_LIN_PACK = "configs/workspace_refs/census_lin_csi1000_100k_8y_20261002"

OPEN_DIRECTIVE = (
    "本臂不定机制家族：你在 10 万元账户上、从八年数据面里自己提出家族、成批预登记并检验，"
    "以冻结或 no_edge 收尾；只用 CPU，不做逐股序列网络。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停。"
    "池与席位是开放轴，取值照 refs/README.md；起步包的 p0 按构造不含信息，不是候选。"
    "至少两个结构不同的家族；每个分数带自己的 c_shuf，主张落在构造或筛子上再带随机化的安慰剂；"
    "先 Y1..Y4 探针、再全期；全期胜过自己对照的家族至少再细化一步。"
    "杀死线只结束候选或家族，不结束本臂；至多四批，登记过的轴没测完就收尾要在 reason 里说明。"
    "重开 refs/families.md 里的家族照 README「什么算重复」申报 offline_trials。"
    "数据合同读挂载的运行记忆，不再派子代理重推。"
)

CENSUS_LIN_DIRECTIVE = (
    "本臂的家族是普查 44 个输入的符号拟合线性合成（中证 1000 成分、10 万）。"
    "基线 l1 即上一臂对照 c_lin，原样不改（12 席、双周）；c_shuf、c_fix 每批同书必跑。"
    "benchmark_index 须为 000852.SH、initial_cash 10 万，与 knobs 一致，否则停。"
    "第一批 offline_trials = 2；门槛读每行 information_ratio_bar。"
    "只开席位、符号窗、加权、复核四轴。"
    "杀死线只结束候选；l1 胜过两对照时整期之后至少细化一步；其余按 refs/README.md。"
)

CENSUS_LIN_LINEAGE = ["census_gbdt_csi1000_100k_8y_20261001"]

ARMS: dict[str, dict[str, object]] = {
    "open_research_100k_8y_qwen_20261002": {
        "workspace_reference": OPEN_PACK,
        "research_directive": OPEN_DIRECTIVE,
    },
    "open_research_100k_8y_mimo_20261002": {
        **MIMO,
        "workspace_reference": OPEN_PACK,
        "research_directive": OPEN_DIRECTIVE,
    },
    "census_lin_csi1000_100k_8y_qwen_20261002": {
        "workspace_reference": CENSUS_LIN_PACK,
        "lineage_arms": CENSUS_LIN_LINEAGE,
        "research_directive": CENSUS_LIN_DIRECTIVE,
    },
    "census_lin_csi1000_100k_8y_mimo_20261002": {
        **MIMO,
        "workspace_reference": CENSUS_LIN_PACK,
        "lineage_arms": CENSUS_LIN_LINEAGE,
        "research_directive": CENSUS_LIN_DIRECTIVE,
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=DIVIDEND_PIT_VIEWS_SEED,
    overrides={
        **EIGHT_YEAR,
        **GATES,
        "max_replay_years": 96,
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
    },
    closed=True,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
