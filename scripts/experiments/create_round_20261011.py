#!/usr/bin/env python
"""The 2026-10-11 round: how fresh the 100k sequence bag's training is, on four seeds per head.

The book the owner trades is the 100k sequence bag: a rank average of four
sequence heads on 12 seats, at most 4 swaps a weekly review, no
residualisation, a 10-day beta-residual label, a trailing three-year window
refitted quarterly. Every lane on it so far held that training recipe fixed.
In the research window the bag's own plain selection (book minus zero-skill
panel) stayed strong into the last research years on every seed base, while
three related books (the single-head MLP and LSTM arms and the 1m bag) read a
last-year selection below their own earlier mean, and the label variant that
raised the bag's whole-window reading was already below its baseline in the
last research year on all three seed bases (logs/notes/round_20261011/
DESIGN.md). Nobody has asked whether the recipe matters for the late years.

The lane was written against the frozen bag, two seeds per head. The two arms
first created from it were deleted within the hour, before either recorded a
validation, and the pack was revised in place, once round 20261008's bag pair
had read the seed count on the taxed seed (logs/notes/late_period_20261005/
F_state_digest_2.md section 2.2): active IR on seed bases 1000 / 2000 of
1.578 / 0.909 with two seeds per head, 1.311 / 1.523 with four and 1.390 /
1.243 with six, against a bar of 1.479. A model-training nominee is judged on
its seed bases together, so a recipe variant measured against the two-seed
bag would be measured against a seed draw. The baseline is therefore
`c_bag4`, the bag with four seeds per head, whose starter reproduces that
lane's four-seed nodes: the same members, cold-refit schedule, training dates
and training caches on the Y1, Y5 and Y8 decision views, and the same orders
from the same scores (logs/notes/round_20261011/revision_bag4/).

`seqfresh` asks the question on one pack (configs/workspace_refs/
seqfresh_alla_100k_8y_20261011): a two- or five-year window and a one-year
recency half-life on the training loss, each a single-axis leg against
`c_bag4` on the same seed base in every batch, then the best single axis and
the combination of the best two. Monthly refit, registered in the first
draft, is dropped: on four seeds a monthly full-span leg runs about 25-43
hours against 14-15 for a quarterly one, and with a finalist on three seed
bases it would hold the arm's card one to two days longer (three to five
with the five-year window). The claim is paired and about time structure:
against the same-batch, same-seed `c_bag4` on seed bases 1000, 2000 and 3000,
plain selection higher on each half of the research window and not lower in
its last two years on every base, and a mean active-IR difference of at least
+0.20. A finalist whose claim holds is frozen on its seed-1000 node with its
2000 and 3000 rows registered as seed replicates, and the arm's facts decide
the rest.

The plan: the three single axes probed on the second half of the window on a
fourth seed base, 4000 (16 replay-years, about 13-15 hours at the batch's
three-way concurrency); the finalists and `c_bag4` full span on 1000 and
2000 (48, about 23-30 hours, 36-42 with a five-year finalist); the finalists
whose claim held there, and `c_bag4`, on 3000 (16-24, about 15-27 hours). At
most 88 of the 96 replay-years, so the default budget stands. Each arm owns
one card; the five-year window peaks at 14.5 GiB or more, so a batch holds
at most two such legs (the pack's finalist rule ensures it).

The local model and `mimo-v2.6-flash` each run the lane, with the sequence
bag's lineage up to its last closed stage: every arm on the frozen bag plus
round 20261008's bag pair (`SEQBAG4_LINEAGE`, 83 trials, an IR bar from
about 1.52 before the arm's own trials). The round-level parameters are
round 20261008's, on the same taxed seed. Text-evidence scoring stays on the
local model in both arms.

Fill order (ids end in `_20261011`): `seqfresh_100k_8y_qwen`,
`seqfresh_100k_8y_mimo`. The ids are the deleted arms' own: nothing of those
arms was archived, and only their worker logs remain.

Usage: create_round_20261011.py <port> [--dry-run] [--fill] [experiment_id ...]
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
    MIMO,
    SEQBAG4_LINEAGE,
    TAXED_FUND_PIT_VIEWS_SEED,
)
from scripts.experiments._round import Round

SEQFRESH_PACK = "configs/workspace_refs/seqfresh_alla_100k_8y_20261011"

SEQFRESH_DIRECTIVE = (
    "本臂走训练新鲜度线：基线 c_bag4 是每头四个种子的 10 万序列袋（四个头各训四个种子，12 席、每次至多换 4 只、每周复核、不残差化、按季重训）；"
    "冻结过的那本书每头只有两个种子，换一个种子基数整期主动 IR 就差 0.67，所以本臂从四个种子起比。"
    "头、每头的种子数、输入、标签、重训节奏与书都不动，只开头从多少历史学、怎样给近期加权——"
    "尾部窗口 2 年或 5 年（c_bag4 3 年）、窗口内按一年半衰期给近期的训练日加权，以及前两名的组合；每条腿在每个种子基数上都是一个试验。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停；"
    "gpu_count 为 1，没有 CUDA 就停，不要改成 CPU。每一批都带同 span、同种子基数的 c_bag4。"
    "主张是时间结构上的、按种子基数配对的：在 refs/README.md 写定的三个种子基数上各与同一批、同种子的 c_bag4 比，"
    "候选的朴素选名（每个研究年书的收益减同一年零技能面板的收益，不回归，从行里逐年取）"
    "在 Y1..Y4 与 Y5..Y8 两半的平均都更高、在 Y7..Y8 的平均不更低，三个种子基数上主动 IR 之差的平均至少 +0.20。"
    "先在第四个种子基数上用 Y5..Y8 探三根单轴，再按 refs/README.md 的规则选决赛者上整期；整期批次之前先单独冒烟 win5 一次，量 fit 的秒数与显存。"
    "主张不成立的决赛者即使过了门也不提名；配方对研究期后几年没有影响也是要写清的结果，reason 里带上每个种子基数上两半与 Y7..Y8 的朴素选名之差。"
    "提名哪一行、登记哪几行种子复现照 refs/README.md §9。"
    "每一行都报主动 IR、正年、主动回撤、逐年朴素选名、书对基准的市场载荷与权益口径的扣费年化收益。"
    "门槛读每行的 information_ratio_bar，不自己算；第一批 offline_trials = 0；至多四批；"
    "批次计划、决赛者规则与杀死线照 refs/README.md。"
)

SEQFRESH_ARM: dict[str, object] = {
    "workspace_reference": SEQFRESH_PACK,
    "gpu_count": 1,
    "lineage_arms": SEQBAG4_LINEAGE,
    "research_directive": SEQFRESH_DIRECTIVE,
}

ARMS: dict[str, dict[str, object]] = {
    "seqfresh_100k_8y_qwen_20261011": dict(SEQFRESH_ARM),
    "seqfresh_100k_8y_mimo_20261011": {**MIMO, **SEQFRESH_ARM},
}

# The seed, the geometry and every round-level parameter are round 20261008's.
ROUND = Round(
    arms=ARMS,
    pit_views_seed=TAXED_FUND_PIT_VIEWS_SEED,
    overrides=FUND_EIGHT_YEAR_100K,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
