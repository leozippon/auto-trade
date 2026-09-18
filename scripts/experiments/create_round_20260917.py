#!/usr/bin/env python
"""The 2026-09-17 round: the queue the console's free running slots are filled from.

Naming follows the previous round: this file, the reference packs and the
experiment ids all carry the date the round was written. Which arms ran
together is the round, and the round is this file; real start and end times
live in the ledger's `recorded_at` and the run manifest.

Every arm shares the research geometry and budgets of `_round.BASE_OVERRIDES`
and the dataset selection the seed below was prebuilt for -- byte for byte the
previous rounds', so the same prebuilt view tree serves all of them.

ARMS is an ordered queue rather than a batch. `--fill` asks the console how
many running slots are free, skips the arms it has already created and creates
the next pending ones in the order written here; the operator keeps the slots
occupied by appending arms and their reference packs to the end of the list.
Two things about that order are decisions rather than accidents:
`pv_exposure_budget_20260917` is last of the arms the round opened with because
it shares an information family with `alpha158_lgbm_20260921`, which is still
running, and the three open arms ahead of it take a slot that frees early for an
unconstrained search. The open seeds appended below it were appended after it had
been created, so nothing waits behind it.

`open_research_20260917` and its `b`, `c`, `d` and `e` seeds are five
independent seeds of one search, not five variants of one design: they share a
single reference pack and a single directive and differ only in their experiment
id, because what an open session finds depends on which branch it takes first.
The pack names none of them, so each session reads as the only one. `d` and `e`
were appended once `b` and `c` had ended `no_edge` and the queue had run dry:
they run under the same pack and the same standards, unchanged.

What `lowvol_bucketed_20260917` registers is a CONSTRUCTION, not a data face. Its ranking column
-- 60-day residual volatility -- is the one the sibling arm
`lowvol_quality_20260916` ranks on, and its pool is that pack's pool byte for
byte, so the arm's own `c_uncapped` control IS the sibling's main candidate.
The registered mechanism is what happens after the score: selecting inside
SW-L1-industry and float-cap-tercile buckets instead of off the whole pool.

`event_screens_20260917` registers POOL MEMBERSHIP, on the borrowed pool, score
and buckets of the arm above: an audit-opinion, a share-unlock and a
revenue-concentration screen, three point-in-time faces this repository holds,
covers fully and no arm has ever read. Its `c_noscreen` control is that arm's
main candidate, so the two cross-check each other.

`pv_exposure_budget_20260917` registers an EXPOSURE BUDGET on the score-to-basket
step of the one construction whose research-period information ratio passes 1:
fifteen seats filled inside industry and size buckets rather than taken off the
whole ranking.

The five open arms register NO FAMILY. Each looks for a long-only 15-30 name
daily strategy on the research period under the refreshed evidence standard of
its pack -- information ratio 0.95, four of four positive research years, and
an exposure budget as a hard gate rather than a reading.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
    },

Contamination, per arm. Operator-side only: the packs themselves carry no
reading from after research end, so none of this reaches the Agent, but it
decides how much each forward verdict is worth.

- lowvol_bucketed_20260917: the direction was chosen after this operator had
  seen that two frozen strategies failed the 2025-07..2026-06 forward test, and
  after reading why -- both books were concentrated exposures that the verdict's
  two-regressor neutralisation does not measure, and the forward window reversed
  the size factor by roughly 45 pp/yr
  (logs/notes/review_20260917/A4_profitability_assessment.md §B). That knowledge
  is what makes an exposure budget the arm's registered mechanism and what set
  the industry-concentration and size-tilt bounds of its nomination gate. The
  same recommendation is reproducible without it: A4 §C.1 derives D1 from the
  failure-mode analysis, from `quality_lowrisk_20260916`'s own research-period
  readings and from NX1 §0.2/§3.2, every citation measured on or before research
  end, and reports that the contaminated and uncontaminated paths agree on the
  structure and differ only in bounding |size tilt| symmetrically. The pack is
  written from the uncontaminated path alone. The honest consequence stands:
  this arm's forward verdict is a pipeline baseline plus a weak edge test, not
  independent evidence that an exposure budget makes an edge persist.
- The arm also inherits the quality family's contamination through its pool: the
  accrual screen and the pool floor come from the same construction study and the
  same family whose forward year this operator has already read
  (see scripts/experiments/create_round_20260916.py). The pool is held fixed
  precisely so that the arm's reading is about the buckets and not about the
  pool.

- event_screens_20260917: the direction was chosen after this operator had read the forward and Held-out
  verdicts of both frozen strategies and the two `no_edge` closures of 2026-09-17. What that knowledge
  contributed to this arm is the **exposure budget** — the symmetric `|size_tilt| <= 0.30` bound and the
  `top_industry_weight <= 0.30` bound — and the 4-of-4-positive-years condition. The direction itself is
  reproducible without it: it is A4 §C.1 D2, resting on the open pack's own "cheap but never tried" list
  (`configs/workspace_refs/open_research_20260916/standards.md`) and on DIR4 §8
  (`logs/notes/review_20260912/DIR4_more_arms.md`), both written from research-period evidence, and on the
  margin-crowding screen result of NX1 §3.6, measured on research-period rows only. The pack is written from
  the uncontaminated path alone and carries no reading from after research end.
- The arm also inherits the quality family's contamination through its borrowed pool: the accrual screen and
  the pool floor come from the construction study and the family whose forward year this operator has already
  read (see `scripts/experiments/create_round_20260916.py`). The pool, the score and the buckets are held
  fixed precisely so that the arm's reading is about the screens and not about them.
- The base book is the one a concurrent arm (`lowvol_bucketed_20260917`) is testing at the same time, and this
  arm's `c_noscreen` is that arm's main candidate. The two cross-check each other, but a shared regime failure
  will show in both, so their forward verdicts are not independent.
- The honest consequence stands: this arm's forward verdict is a pipeline baseline plus a weak edge test on
  three never-read data faces, not independent evidence that disclosed-risk exclusion screens make an edge
  persist. The pack's own research-period census already shows the screens move two or three names of fifteen,
  and that the audit screen alone moves none.

- open_research_20260917, open_research_20260917b, open_research_20260917c,
  open_research_20260917d, open_research_20260917e (one shared pack): This
  direction was queued after the operator had read the forward and Held-out verdicts of both
  frozen arms. The exposure budget and the 4/4-year condition were added after the reviewer had
  seen both forward verdicts, and the reviewer knows which style reversed. Both are reproducible
  from research-period readings alone: the previous open arm's five full-span validations sat at
  45.8-86.6 % single-industry weight with `size_tilt` 0.79-0.91, and the other frozen book read
  `size_tilt` 0.447-0.479 -- the two bounds follow from those numbers without any forward
  information. The closed-family table quotes research-period readings only; families whose
  closure rests on later rows keep the prohibition without the number. No dataset, cadence,
  basket size or mechanism in this pack was chosen because of what happened after 2025-06-30.

  The five open arms share one `workspace_reference` and differ only in their experiment id:
  they are independent seeds of a search whose outcome depends on which branch the Agent takes
  first, not five variants of one design. The pack names none of them and says nothing about
  the others, so each session reads as the only one.

- pv_exposure_budget_20260917: 本臂**按设计就是被污染的，它的前推裁决是一次流水线校准，不是独立证据**。这个家族承自旧流程的一个冻结产物，那条谱系的选择用到过研究期之后的数据，运营方已经部分知道它此后的样子，而 `alpha158_lgbm_20260921` 此刻正在同一个前推窗口上跑同一个家族。选择在这里测敞口预算，还是在看过两条冻结产物都因一次规模与板块摆动而前推失败之后做出的。本臂仍能干净回答的是一个研究期问题：在本项目唯一一个研究期信息比率超过 1 的构造上，一层敞口预算要付多少信息比率与换手代价。这个包必须写成「那个答案不依赖前推裁决」的样子。

Exposure-budget amendment, 2026-09-17, after every arm below had been created.
Two of the budget's three bounds are relaxed in all four packs and in the
directives above: `|size_tilt|` moves from 0.30 to 0.70 and the beta band
[0.7, 1.2] becomes `beta <= 1.2`. `top_industry_weight <= 0.30` is unchanged --
it is the one bound with a mechanism the verdict statistic cannot see, and
every diversified book on record sits at 0.09-0.15 against it.

The amendment was decided from evidence outside the running arms, not from
anything they produced: every scored book this repository has ever measured
reads `size_tilt` above 0.30, so the bound excluded the whole record rather
than a demonstrated failure; the two frozen books that read 0.44 and 0.447 had
opposite forward outcomes, so at that level the statistic separates nothing;
and the loading that moves the verdict -- the size beta -- is already regressed
out of every figure the freeze gate and the forward test read, which is why a
book's own size position is a money question rather than a verdict question.
The beta floor guarded the same already-regressed-out quantity; `mean_gross >=
0.5` still blocks a closet-cash book, and the 1.2 ceiling still guards a raw
drawdown.

The four arms created before the amendment were sent the pre-amendment
directive, which their `hitl/params.json` keeps as the record of what they were
told; each was given the new numbers through an operator message and reads the
amended pack text in `refs/` after its restart.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260917.py <port> [--dry-run] [--fill] [experiment_id ...]

  `--fill` creates as many pending arms as the console has free slots; naming an
  id creates just that one.
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

# The seed and the selection are not this round's decisions: the arm hardlinks
# from the tree the earlier prebuild produced, and the create-time pre-flight
# refuses any selection that is not the one that tree was built for. Re-listing
# 47 dataset names here would be a second copy of a contract the seed already
# owns, and a copy that drifts is refused rather than caught.
from scripts.experiments.create_round_20260920 import (
    EVENTS_DATASETS,
    FUNDAMENTAL_DATASETS,
    MACRO_DATASETS,
    PIT_VIEWS_SEED,
    TEXT_DATASETS,
)

ARMS: dict[str, dict[str, object]] = {
    "lowvol_bucketed_20260917": {
        "workspace_reference": "configs/workspace_refs/lowvol_bucketed_20260917",
        "research_directive": (
            "本臂是固定方向的臂，不训练，登记的机制是「分数 → 篮子」这一层的构造，不是一条新的数据面："
            "在同一个股票池里先按流通市值截到较大的一半、再剔掉应计最差的一成，然后按 60 日残差波动升序，"
            "在每个申万一级行业至多 2 只、每个流通市值三分位至少 1 只的约束下选出 15 只等额持有，季度复核。"
            "假说是：整池排名会连带选中一个行业格与一个规模格，而这部分不持续；桶内选名保留截面信息、去掉这个连带。"
            "三个对照必须同批同跨度跑：同分数同池但不设桶的 c_uncapped（机制归因）、同样的桶换成固定随机分数的 "
            "c_rand（证明桶本身不是边际）、无分数的等距池篮子 c_pool（组合尺度底线）。提名硬门包括完整研究期"
            "信息比率 ≥ 0.95、四个研究年的中性化超额全为正、最大回撤 ≤ 22 %、stats.benchmark.top_industry_weight "
            "≤ 0.30、|stats.benchmark.size_tilt| ≤ 0.70，并且在信息比率上严格高于三个对照——判据、变体轴与本臂"
            "终止规则只以 refs/families.md 为准，先读它与 refs/README.md。会话开始时 output/ 就是 refs/starter "
            "这份代码，直接在它上面改（四条腿只差 main.py 的 CANDIDATE 一行），先 smoke_backtest，再按 "
            "refs/exploration-plan.md 做完整研究期验证。families.md 的禁止表逐条带着关掉它的研究期读数，不得重开；"
            "对照不得被提名冻结。没有候选满足提名硬门时按 families.md 以 no_edge 结束，不换家族。"
        ),
    },
    "event_screens_20260917": {
        "workspace_reference": "configs/workspace_refs/event_screens_20260917",
        "research_directive": (
            "本臂是固定方向的臂，不训练，登记的机制是股票池的成员资格，不是一条新的排序分数："
            "在与姊妹臂逐字相同的池、60 日残差波动分数、中性化与行业/规模桶构造上，先剔掉最新审计意见"
            "不是无保留意见的名字、剔掉未来 90 个自然日内已公告解禁合计占总股本 ≥ 3 % 的名字、再剔掉"
            "收入最集中的一成，然后照原样选出 15 只等额持有、季度复核。三个对照必须同批同跨度跑："
            "不筛的同一个篮子 c_noscreen（机制归因）、把筛子换成同样比例永久随机排除的 c_placebo"
            "（证明要紧的是被剔掉的那些名字而不是选择集变小）、无分数的等距池篮子 c_pool。提名硬门包括"
            "完整研究期信息比率 ≥ 0.95、四个研究年的中性化超额全为正、最大回撤 ≤ 22 % 且严格低于 "
            "c_noscreen、stats.benchmark.top_industry_weight ≤ 0.30、|stats.benchmark.size_tilt| ≤ 0.70，"
            "在信息比率上严格高于三个对照，并且每个筛子每次复核剔除 < 筛前池的 15 %——判据、变体轴与"
            "本臂终止规则只以 refs/families.md 为准，先读它与 refs/README.md；三个面的可见时点、单位与"
            "去重陷阱只以 refs/pit-field-map.md 为准。诚实的先验是弱的：包里带着本仓库同形状筛子的反向"
            "读数，以及本包自己的研究期普查——三个筛子合起来只换两三只名字，审计筛子单开时一只都不换。"
            "会话开始时 output/ 就是 refs/starter 这份代码，直接在它上面改（四条腿只差 CANDIDATE 一行），"
            "先 smoke_backtest，再按 refs/exploration-plan.md 做完整研究期验证。筛子只排除不准入，"
            "读不到一行不等于不合格。families.md 的禁止表逐条带着关掉它的研究期读数，不得重开；"
            "没有候选满足提名硬门时按 families.md 以 no_edge 结束，不换家族。"
        ),
    },
    "open_research_20260917": {
        "workspace_reference": "configs/workspace_refs/open_research_20260917",
        "research_directive": (
            "本臂是开放方向的臂：不指定机制家族，在研究期上找一个只做多、15–30 只、有经济解释并在组合"
            "尺度上成立的日频策略。先读 refs/README.md 与 refs/standards.md——证据标准、已关闭家族表、"
            "每个候选都要比的对照、敞口预算、提名条件与收尾规则只以 standards.md 为准：完整研究期信息"
            "比率 ≥ 0.95、四个研究年的中性化超额全为正、单一申万一级行业时间加权权重 ≤ 0.30、"
            "|size_tilt| ≤ 0.70、β ≤ 1.2 都是硬门，空对照只作诊断。筛子与叠加规则和排序分数"
            "一样是一等候选。每个方向在花任何回放预算之前，必须先离线读出它在 15 只与 30 只上的逐研究年"
            "组合尺度读数（refs/exploration-plan.md）。refs/starter 的 o1 只是能跑的基线，不是推荐的"
            "机制：先 smoke_backtest。standards.md 点名禁止的家族不得重开，其中包括 Alpha158 一类的"
            "日频价量排序器与序列网络。预算用尽前仍没有候选满足提名条件时以 no_edge 结束。"
        ),
    },
    "open_research_20260917b": {
        "workspace_reference": "configs/workspace_refs/open_research_20260917",
        "research_directive": (
            "本臂是开放方向的臂：不指定机制家族，在研究期上找一个只做多、15–30 只、有经济解释并在组合"
            "尺度上成立的日频策略。先读 refs/README.md 与 refs/standards.md——证据标准、已关闭家族表、"
            "每个候选都要比的对照、敞口预算、提名条件与收尾规则只以 standards.md 为准：完整研究期信息"
            "比率 ≥ 0.95、四个研究年的中性化超额全为正、单一申万一级行业时间加权权重 ≤ 0.30、"
            "|size_tilt| ≤ 0.70、β ≤ 1.2 都是硬门，空对照只作诊断。筛子与叠加规则和排序分数"
            "一样是一等候选。每个方向在花任何回放预算之前，必须先离线读出它在 15 只与 30 只上的逐研究年"
            "组合尺度读数（refs/exploration-plan.md）。refs/starter 的 o1 只是能跑的基线，不是推荐的"
            "机制：先 smoke_backtest。standards.md 点名禁止的家族不得重开，其中包括 Alpha158 一类的"
            "日频价量排序器与序列网络。预算用尽前仍没有候选满足提名条件时以 no_edge 结束。"
        ),
    },
    "open_research_20260917c": {
        "workspace_reference": "configs/workspace_refs/open_research_20260917",
        "research_directive": (
            "本臂是开放方向的臂：不指定机制家族，在研究期上找一个只做多、15–30 只、有经济解释并在组合"
            "尺度上成立的日频策略。先读 refs/README.md 与 refs/standards.md——证据标准、已关闭家族表、"
            "每个候选都要比的对照、敞口预算、提名条件与收尾规则只以 standards.md 为准：完整研究期信息"
            "比率 ≥ 0.95、四个研究年的中性化超额全为正、单一申万一级行业时间加权权重 ≤ 0.30、"
            "|size_tilt| ≤ 0.70、β ≤ 1.2 都是硬门，空对照只作诊断。筛子与叠加规则和排序分数"
            "一样是一等候选。每个方向在花任何回放预算之前，必须先离线读出它在 15 只与 30 只上的逐研究年"
            "组合尺度读数（refs/exploration-plan.md）。refs/starter 的 o1 只是能跑的基线，不是推荐的"
            "机制：先 smoke_backtest。standards.md 点名禁止的家族不得重开，其中包括 Alpha158 一类的"
            "日频价量排序器与序列网络。预算用尽前仍没有候选满足提名条件时以 no_edge 结束。"
        ),
    },
    "pv_exposure_budget_20260917": {
        "workspace_reference": "configs/workspace_refs/pv_exposure_budget_20260917",
        "research_directive": (
            "本臂是固定方向的臂，登记的机制是「分数 → 篮子」这一层的敞口预算，不是一条新的数据面："
            "分数是借来的、固定不动的日频量价学习型排序器（158 列量价特征 + 梯度提升树，截面 10 日标签，"
            "按季度在滚动窗口上重训），把它的 15 个座位放进「每个申万一级行业至多 2 只、每个流通市值"
            "三分位至少 1 只」的桶里填，等额、每周复核、2 倍保留带、每次最多换 2 只。三个对照必须同批"
            "同跨度跑：去掉桶的同分数篮子 c_uncapped（机制归因）、同样的桶换成固定随机分数的 c_rand"
            "（证明赢的不是分散化）、无分数的等权池篮子 c_pool；c_rand 与 c_pool 不训练，要把 fit 与 "
            "REFIT_PERIOD 整段删掉。提名硬门包括完整研究期信息比率 ≥ 0.95、四个研究年的中性化超额全为正、"
            "stats.benchmark.top_industry_weight ≤ 0.30、|stats.benchmark.size_tilt| ≤ 0.70、"
            "β ≤ 1.2、2 倍滑点后超额 > 0，在信息比率上严格高于三个对照，并且换手不得超过 "
            "c_uncapped 的 1.5 倍；回撤门取流水线自己的 25 %，因为本臂的机制作用在集中度上而不是回撤上"
            "——判据、变体轴与本臂终止规则只以 refs/families.md 为准，先读它与 refs/README.md。fit 在"
            "容器 CPU 上完成，失败或样本不足直接报错，不设退路打分器；提交任何整期批次之前，必须按运行"
            "事实 smoke_backtest_probe_note 在研究期靠后的一天再冒烟量一次 fit。会话开始时 output/ 就是 "
            "refs/starter 这份代码，先 smoke_backtest，再按 refs/exploration-plan.md 做完整研究期验证。"
            "本臂的前推裁决只用来校准流水线，交付物是研究期上那个代价读数；没有候选满足提名硬门时按 "
            "families.md 以 no_edge 结束，不换家族。"
        ),
    },
    "open_research_20260917d": {
        "workspace_reference": "configs/workspace_refs/open_research_20260917",
        "research_directive": (
            "本臂是开放方向的臂：不指定机制家族，在研究期上找一个只做多、15–30 只、有经济解释并在组合"
            "尺度上成立的日频策略。先读 refs/README.md 与 refs/standards.md——证据标准、已关闭家族表、"
            "每个候选都要比的对照、敞口预算、提名条件与收尾规则只以 standards.md 为准：完整研究期信息"
            "比率 ≥ 0.95、四个研究年的中性化超额全为正、单一申万一级行业时间加权权重 ≤ 0.30、"
            "|size_tilt| ≤ 0.70、β ≤ 1.2 都是硬门，空对照只作诊断。筛子与叠加规则和排序分数"
            "一样是一等候选。每个方向在花任何回放预算之前，必须先离线读出它在 15 只与 30 只上的逐研究年"
            "组合尺度读数（refs/exploration-plan.md）。refs/starter 的 o1 只是能跑的基线，不是推荐的"
            "机制：先 smoke_backtest。standards.md 点名禁止的家族不得重开，其中包括 Alpha158 一类的"
            "日频价量排序器与序列网络。预算用尽前仍没有候选满足提名条件时以 no_edge 结束。"
        ),
    },
    "open_research_20260917e": {
        "workspace_reference": "configs/workspace_refs/open_research_20260917",
        "research_directive": (
            "本臂是开放方向的臂：不指定机制家族，在研究期上找一个只做多、15–30 只、有经济解释并在组合"
            "尺度上成立的日频策略。先读 refs/README.md 与 refs/standards.md——证据标准、已关闭家族表、"
            "每个候选都要比的对照、敞口预算、提名条件与收尾规则只以 standards.md 为准：完整研究期信息"
            "比率 ≥ 0.95、四个研究年的中性化超额全为正、单一申万一级行业时间加权权重 ≤ 0.30、"
            "|size_tilt| ≤ 0.70、β ≤ 1.2 都是硬门，空对照只作诊断。筛子与叠加规则和排序分数"
            "一样是一等候选。每个方向在花任何回放预算之前，必须先离线读出它在 15 只与 30 只上的逐研究年"
            "组合尺度读数（refs/exploration-plan.md）。refs/starter 的 o1 只是能跑的基线，不是推荐的"
            "机制：先 smoke_backtest。standards.md 点名禁止的家族不得重开，其中包括 Alpha158 一类的"
            "日频价量排序器与序列网络。预算用尽前仍没有候选满足提名条件时以 no_edge 结束。"
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
