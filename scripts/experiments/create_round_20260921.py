#!/usr/bin/env python
"""The 2026-09-21 round: four directions under the panel grading, two accounts.

The dataset selection and the prebuilt seed are the 2026-09-19 round's -- the
2026-09-20 selection plus `index_weight` -- imported rather than restated so
the two rounds that mount it cannot drift apart by a typo. All four arms need
it: two build their universe from the constituent cross-section, and all four
need the host's zero-skill panel to match replacements on CSI 300 membership
rather than falling back to the float-cap decile, which inherits the
candidate's own size tilt.

Every arm mounts `configs/workspace_refs/capital_aware_20260920`, whose one
registered claim is that the account shapes the book. What differs per arm is
the account, whether a tracking mandate is in force, and the opening family.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
        "initial_cash": <the account this arm runs>,
        "tracking_error_cap": <only on a mandated arm>,
    }

`initial_cash` and the acceptance limits are ordinary create parameters,
so an arm entry overrides the round's value for that arm alone
(`_round.Round.request_params` merges the arm last). **The tracking mandate is
a manual setting and no capital turns it on**: it exists exactly where an arm
names `tracking_error_cap`. A cap without a beta band takes 0.85 / 1.15;
drawdowns stay 0.45 / 0.30 unless an arm names them. The two 1M arms name
only the cap, so `--dry-run` prints 0.45 / 0.30 plus the cap and default
beta. The two arms without a cap have their tracking error measured and
reported, not graded. Statistical bars that no arm names take today's
create-time defaults.

One experiment, one account, one strategy: each arm runs a single account and
the artifact it freezes serves that account alone.

Why both CNY 1M arms carry the mandate
(`logs/notes/review_20260921/D3_four_directions.md` section 3). It is entirely
unmeasured -- of the sixteen materially different books on record, none
satisfies it and fifteen fail on beta alone (0.51-0.85 against a 0.85-1.15
band) -- so an un-mandated 1M arm would re-run exactly the shape
`index_relative_1m_20260919` and `index_relative_1m_b_20260919` already ran. It
is also satisfiable by construction: a zero-skill benchmark-weight book at 1M
reads its own tracking error 5.5 / 4.7 / 4.6 % at 50 / 80 / 100 seats at beta
0.94-0.98, leaving about 6.5 %/yr of active tracking error to spend under an
8 % cap, so active IR 0.75 needs only 4.9 %/yr of active excess against the
7.9-14.5 %/yr the in-index record actually carries. And its price is
recoverable inside each arm for four replay-years, because the pack already
requires a same-batch mechanism control and here that control is literally
"the same score, unconstrained, equal cash at the same seats".

CSI 300 itself fell 39.6 % over the research period, and of twenty zero-skill
benchmark-weight books 40 % breach 35 % at 50 seats, 15 % at 80 and none at
100. These two mandated arms do not name a tighter equity limit, so they are
held to the rules' 45 %; the directives still steer toward the safer seat
range because a full-beta tracker remains a large absolute drawdown.

Contamination, per arm. Operator-side only: the pack carries no reading from
after research end, so none of this reaches the Agent, but it decides how much
each forward verdict is worth.

- Round-level, all four arms. The grading change itself -- every statistic
  moved onto the active series against a host-drawn zero-skill panel -- and the
  decision to spend four arms on these directions were taken after five frozen
  strategies were discarded on the shared forward window, which the operator
  has read. Every reading behind the four choices is research-period
  (2021-07-01..2025-06-30): the zero-skill panels of A10 section 1.1, the
  sixteen-node re-grading of D3 section 1, the seat and cost arithmetic of A11
  section 1.2.
- A1 (`capital_aware_1m_enhanced_20260921`). Its stated prior cites `d2` of
  `index_relative_1m_20260919`, an arm whose forward verdict -- discarded --
  the operator has read, so that lineage is contaminated and A1's forward
  verdict is partly a pipeline calibration. The two nodes that motivate the
  construction most, `d1_range20cap` and `d1_iv` of
  `index_relative_1m_b_20260919`, were never forward-tested and are clean.
- A2 (`capital_aware_1m_riskmodel_20260921`). The construction is new and cites
  no frozen lineage; `sw_daily` has never carried a book, so no forward reading
  exists for it. Clean apart from the round-level statement.
- B1 (`capital_aware_100k_concentrated_20260921`). The seat-count ladder that
  motivates it (15 -> 20 -> 30 names) comes from the old-flow NX1 probe, whose
  sample extended past research end; the pack keeps only its conclusion, not
  its readings, and the extrapolation below 15 seats uses research-period
  panels alone. No frozen lineage is cited, so B1's forward verdict is
  independent evidence.
- B2 (`capital_aware_100k_pv_20260921`). Both arms whose readings it revives
  (`index_relative_100k_20260919`, `index_relative_1m_b_20260919`) closed
  `no_edge` and were never forward-tested, and the re-grading that revives them
  was computed on research-period data alone. B2 is the cleanest of the four:
  its forward verdict is independent evidence, subject only to the round-level
  statement.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260921.py <port> [--dry-run] [--fill] [experiment_id ...]
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

PACK = "configs/workspace_refs/capital_aware_20260920"

ARMS: dict[str, dict[str, object]] = {
    "capital_aware_1m_enhanced_20260921": {
        "workspace_reference": PACK,
        "initial_cash": 1_000_000,
        "tracking_error_cap": 0.08,
        "research_directive": (
            "本臂是指数增强方向的臂：账户 100 万元，带跟踪授权。股票池是决策日可见的沪深 300 成分，"
            "按时点从宏观域读出来，不得回退全市场池。开局家族是**基准权重锚定的增强书**：以成分权重为底，"
            "按你自己选的分数给每个名字加一个有界的主动权重，并按申万一级行业把主动权重的净敞口也限住；"
            "席位 80–100 只由你定并写进预登记，只数越少绝对回撤越难过。先读 refs/README.md 与 refs/families.md——"
            "提名条件、必跑对照与终止规则只以它为准。跟踪误差上限与 β 区间是约束不是证据："
            "同批必须跑同一分数不设界的等额前 N 版本，把授权的代价量出来。先 smoke_backtest，再做完整研究期验证。"
            "这一族被证伪而预算还有余量时，换下一个预登记方向，不要就此收尾。"
        ),
    },
    "capital_aware_1m_riskmodel_20260921": {
        "workspace_reference": PACK,
        "initial_cash": 1_000_000,
        "tracking_error_cap": 0.08,
        "research_directive": (
            "本臂是指数增强方向的臂，开局家族与同资金的另一条臂不同：账户 100 万元，带跟踪授权，"
            "股票池是决策日可见的沪深 300 成分，不得回退全市场池。开局做**显式风险模型下的跟踪误差预算**："
            "用 sw_daily 的申万一级行业收益与 daily_basic/daily 的规模、估值、波动做一个简单的因子协方差，"
            "把多信号分数在事前主动跟踪误差预算下配成权重，席位 50–100 只由你定；"
            "绝对回撤上限对一本满仓的跟踪书是紧的，席位偏少时零技能书就有相当比例越线，选席位时把这条算进去。"
            "先读 refs/README.md 与 refs/families.md——提名条件、必跑对照与终止规则只以它为准。"
            "第一件要量的不是超额，是事前预算与实测主动跟踪误差的比值；对不上就直接关掉这条方向。"
            "同批跑同一分数等额前 N 的版本。先 smoke_backtest，再做完整研究期验证。"
            "方向被证伪而预算还有余量时换下一个预登记方向。"
        ),
    },
    "capital_aware_100k_concentrated_20260921": {
        "workspace_reference": PACK,
        "initial_cash": 100_000,
        "research_directive": (
            "本臂是小账户高集中方向的臂：账户 10 万元，不设跟踪授权，跟踪误差只汇报不设门。"
            "股票池是全市场（非北交所与科创板、非 ST、有行有量）里本账户买得起的那部分——"
            "价格上限由席位金额自己算出来，不要沿用固定的 30 元上限。开局家族是**席位数本身**："
            "在 8 到 12 只之间选定并写进预登记，同批必须跑同一分数 15 只的版本，赢不过它就说明集中不是杠杆。"
            "排序分数由你选，已关闭家族表是先验不是围栏。先读 refs/README.md 与 refs/families.md——"
            "证据标准、提名条件与收尾规则只以它为准。先 smoke_backtest，再做完整研究期验证。"
            "主动回撤与收益集中度是这条臂最可能的死法，第一轮之前先把这两笔账算清楚。"
            "方向被证伪而预算还有余量时换下一个预登记方向。"
        ),
    },
    "capital_aware_100k_pv_20260921": {
        "workspace_reference": PACK,
        "initial_cash": 100_000,
        "research_directive": (
            "本臂是小账户指数内方向的臂：账户 10 万元，不设跟踪授权，跟踪误差只汇报不设门。"
            "股票池是决策日可见的沪深 300 成分里本账户买得起的那部分（15 只时约六成名字、六成权重），"
            "按时点从宏观域读出来，不得回退全市场池。开局家族是**日频量价截面算子分数 + 周度复核**，"
            "规则式打分，不训练、不用 GPU；席位 15–30 只由你定。价值与低波这两族在这个池子里已经量过，"
            "不得作为开局家族重测。先读 refs/README.md 与 refs/families.md——提名条件、必跑对照与终止规则只以它为准。"
            "先 smoke_backtest，再做完整研究期验证。最低佣金随席位数与复核频率一起涨，第一轮之前先算清这笔账。"
            "方向被证伪而预算还有余量时换下一个预登记方向。"
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
