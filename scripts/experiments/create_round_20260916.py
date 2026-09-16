#!/usr/bin/env python
"""The 2026-09-16 round: three fixed-and-open arms after the construction study.

Naming. Every name here -- this file, the three reference packs and the three
experiment ids -- carries the date the round was actually written, which is also
the date the packs were written. Earlier rounds used the date as a round label
instead, so their ids sit a day or two ahead of their own files; nothing is
renamed, and from here on a suffix is a creation date. Which arms ran together
is the round, and the round is this file. Real start and end times live where
they cannot drift: `recorded_at` in the ledger and the run manifest.

Three arms share the research geometry and budgets of `_round.BASE_OVERRIDES`
and the dataset selection the seed below was prebuilt for -- byte for byte the
previous round's, so the same prebuilt view tree serves them. The fourth slot
stays with `alpha158_lgbm_20260921`, which is still running: it is the
pipeline's own calibration and relaunching it would only reset that.

The two quality arms supersede `defensive_quality_20260921`, which the operator
retires at restart time; its id then joins `_round.RETIRED_IDS`. The pack it
ran, `configs/workspace_refs/defensive_quality_20260918`, is not edited: its
shape -- one composite over two averaged legs -- is what the construction study
falsified, so it is replaced rather than patched, once per leg order.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
    },

Contamination, per arm. Operator-side only: the packs themselves carry no
reading from after research end, so none of this reaches the Agent, but it
decides how much each forward verdict is worth.

- quality_lowrisk_20260916 and lowvol_quality_20260916: the cash-flow quality
  family's entire forward year is already visible to this operator, printed
  quarter by quarter in logs/notes/review_20260913/DIR6_learned_survivors.md
  §2.2, and the family's two legs were selected by DIR4, which also read rows
  past research end. The construction rule both packs pre-register was chosen
  from research-period probes only
  (logs/notes/review_20260916/NX1_next_directions.md §3, every row filtered to
  trade dates on or before research end), so the construction is untainted --
  but the family is not. Either arm's forward verdict is therefore a pipeline
  baseline plus a weak edge test, not independent evidence about this family.
  The pair still answers a question no reading answers: which of the two legs
  carries at book size once the book's risk is shaped.
- open_research_20260916: forward unseen for its direction, which nobody has
  chosen yet. Its standards and its closed-family table come from studies that
  did read later rows; the pack keeps them only as prohibitions and quotes a
  number only where that number was measured inside the research period.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260916.py <port> [--dry-run] [experiment_id ...]
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

# The seed and the selection are not this round's decisions: the arms hardlink
# from the tree the previous round's prebuild produced, and the create-time
# pre-flight refuses any selection that is not the one that tree was built for.
# Re-listing 47 dataset names here would be a second copy of a contract the
# seed already owns, and a copy that drifts is refused rather than caught.
from scripts.experiments.create_round_20260920 import (
    EVENTS_DATASETS,
    FUNDAMENTAL_DATASETS,
    MACRO_DATASETS,
    PIT_VIEWS_SEED,
    TEXT_DATASETS,
)

ARMS: dict[str, dict[str, object]] = {
    "quality_lowrisk_20260916": {
        "workspace_reference": "configs/workspace_refs/quality_lowrisk_20260916",
        "research_directive": (
            "本臂是固定方向的臂，不训练：现金流盈利质量（低应计、高经营现金流/资产，取首版报表）排名，"
            "风险成形的构造——先把股票池按流通市值截到较大的一半，再在质量前 30 名里取 60 日残差波动"
            "最低的 15 只，季度复核。先读 refs/README.md 与 refs/families.md——主候选 s_ql、三个必须同批"
            "跑的对照、提名硬门、允许的变体轴与本臂终止规则只以 families.md 为准。从 refs/starter 起步"
            "（四条腿只差 main.py 的 CANDIDATE 一行），先 smoke_backtest，再按 refs/exploration-plan.md "
            "做完整研究期验证。篮子放宽到 20 只以上、行业上限、月度复核、按反转/动量/换手择时、融资"
            "拥挤筛除、与价量分数混合、以及对存活列做学习型合成，都已在 families.md 的禁止表里各自"
            "带着关掉它的读数，不得重开。没有候选满足提名硬门时按 families.md 以 no_edge 结束，不换家族。"
        ),
    },
    "lowvol_quality_20260916": {
        "workspace_reference": "configs/workspace_refs/lowvol_quality_20260916",
        "research_directive": (
            "本臂是固定方向的臂，不训练，是质量臂的镜像：60 日残差波动排名，质量做筛子——先把股票池"
            "按流通市值截到较大的一半，再剔掉应计最差的一成，然后取残差波动最低的 15 只，季度复核。"
            "先读 refs/README.md 与 refs/families.md——主候选 s_lq、三个必须同批跑的对照、提名硬门、"
            "允许的变体轴与本臂终止规则只以 families.md 为准。从 refs/starter 起步（四条腿只差 main.py "
            "的 CANDIDATE 一行），先 smoke_backtest，再按 refs/exploration-plan.md 做完整研究期验证。"
            "两条腿的顺序是本臂的全部内容：把质量改回排名信号就是另一条臂，不得在本臂里做。families.md "
            "的禁止表逐条带着关掉它的读数，不得重开。没有候选满足提名硬门时按 families.md 以 no_edge "
            "结束，不换家族。"
        ),
    },
    "open_research_20260916": {
        "workspace_reference": "configs/workspace_refs/open_research_20260916",
        "research_directive": (
            "本臂是开放方向的臂：不指定机制家族，在研究期上找一个只做多、15–30 只、有经济解释的日频"
            "策略，按信息比率与回撤提名，不按整窗超额提名。先读 refs/README.md 与 refs/standards.md——"
            "证据标准、已关闭家族表、每个候选都要比的对照、提名条件与收尾规则只以 standards.md 为准。"
            "筛子与叠加规则和排序分数一样是一等候选。每个方向在花任何回放预算之前，必须先离线读出它"
            "在 15 只与 30 只上的逐研究年组合尺度读数（refs/exploration-plan.md）。refs/starter 的 o1 "
            "只是能跑的基线，不是推荐的机制：先 smoke_backtest。standards.md 点名禁止的家族不得重开，"
            "其中包括 Alpha158 一类的日频价量排序器与序列网络。预算用尽前仍没有候选满足提名条件时以 "
            "no_edge 结束。"
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
