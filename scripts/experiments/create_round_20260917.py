#!/usr/bin/env python
"""The 2026-09-17 round: one fixed-direction arm on within-bucket selection.

Naming follows the previous round: this file, the reference pack and the
experiment id all carry the date the round was written. Which arms ran together
is the round, and the round is this file; real start and end times live in the
ledger's `recorded_at` and the run manifest.

The arm shares the research geometry and budgets of `_round.BASE_OVERRIDES` and
the dataset selection the seed below was prebuilt for -- byte for byte the
previous rounds', so the same prebuilt view tree serves it. Earlier rounds
reserved a running slot for `alpha158_lgbm_20260921`; that reservation is a
scheduling fact of those rounds and not a property of this one, which brings a
single arm and takes whichever slot is free when the operator creates it.

What this arm registers is a CONSTRUCTION, not a data face. Its ranking column
-- 60-day residual volatility -- is the one the sibling arm
`lowvol_quality_20260916` ranks on, and its pool is that pack's pool byte for
byte, so the arm's own `c_uncapped` control IS the sibling's main candidate.
The registered mechanism is what happens after the score: selecting inside
SW-L1-industry and float-cap-tercile buckets instead of off the whole pool.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
    },

Contamination, per arm. Operator-side only: the pack itself carries no reading
from after research end, so none of this reaches the Agent, but it decides how
much the forward verdict is worth.

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

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260917.py <port> [--dry-run] [experiment_id ...]
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
            "≤ 0.30、|stats.benchmark.size_tilt| ≤ 0.30，并且在信息比率上严格高于三个对照——判据、变体轴与本臂"
            "终止规则只以 refs/families.md 为准，先读它与 refs/README.md。会话开始时 output/ 就是 refs/starter "
            "这份代码，直接在它上面改（四条腿只差 main.py 的 CANDIDATE 一行），先 smoke_backtest，再按 "
            "refs/exploration-plan.md 做完整研究期验证。families.md 的禁止表逐条带着关掉它的研究期读数，不得重开；"
            "对照不得被提名冻结。没有候选满足提名硬门时按 families.md 以 no_edge 结束，不换家族。"
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
