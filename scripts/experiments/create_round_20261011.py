#!/usr/bin/env python
"""The 2026-10-11 round: how fresh the frozen 100k sequence bag's training is.

The book the owner trades is the 100k sequence bag: a rank average of four
sequence heads on 12 seats, at most 4 swaps a weekly review, no
residualisation, a 10-day beta-residual label, a trailing three-year window
refitted quarterly. Every lane on it so far (label, fundamentals as inputs,
the swap clock, more seeds per head, and the queued turnover, pool and
index-beta lanes) held that training recipe fixed, and the packs told the
Agent not to touch it. In the research window the bag's own plain selection
(book minus zero-skill panel) stayed strong into the last research year on
every seed base, while three related books (the single-head MLP and LSTM
arms and the 1m bag) read a last-year selection below their own earlier
mean, and the label variant that raised the bag's whole-window reading was already
below its baseline in the last research year on all three seed bases
(logs/notes/round_20261011/DESIGN.md). Nobody has asked whether the recipe
matters for the late years.

`seqfresh` asks it, on one pack (configs/workspace_refs/
seqfresh_alla_100k_8y_20261011): a two- or five-year window, monthly refit,
and a one-year recency half-life on the training loss, each a single-axis leg
against `c_base` (the frozen bag on the same seed base, in every batch), then
the best single axis and the combination of the best two. The claim is paired
and about time structure: against the same-seed `c_base`, on seed bases 1000
and the second one, plain selection higher on each half of the research
window and not lower in its last two years, with a mean active-IR difference
of at least +0.20; a row that clears the gate without the claim is not
nominated, because the baseline is already frozen. The four single axes are
probed on the second half of the window on a third seed base (20
replay-years), and two finalists go full span on the two claim bases (48), so
the plan spends at most 68 of the round's 96 replay-years and the default
budget stands. A monthly-refit leg refits three times as often and its
full-span replay takes about 13-22 hours at the batch's three-way
concurrency, against 5-9 for a quarterly one (a five-year window about
9-15); the pack prices every batch with these figures, measured on the host
(logs/notes/round_20261011/DESIGN.md §5). The local model and
`mimo-v2.6-flash` each run the lane on one GPU, with the lineage round
20261008's queued cost and pool pairs carry: that round's own plus the five
arms that have since closed on this baseline (the label and
fundamentals-input pairs and the clock arm).

The round-level parameters are round 20261008's, on the same taxed seed.
Text-evidence scoring stays on the local model in both arms. The round's
second pair, the open lane, sits in `create_round_20261011b.py` at the end
of the queue.

Fill order (ids end in `_20261011`): `seqfresh_100k_8y_qwen`,
`seqfresh_100k_8y_mimo`.

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
    SEED_REPLICATES,
    SEQBOOK_LINEAGE,
    TAXED_FUND_PIT_VIEWS_SEED,
)
from scripts.experiments._round import Round

SEQFRESH_PACK = "configs/workspace_refs/seqfresh_alla_100k_8y_20261011"

SEQFRESH_DIRECTIVE = (
    "本臂走训练新鲜度线：c_base 是已冻结的 10 万序列袋（四个头各两个种子，12 席、每次至多换 4 只、每周复核、不残差化），"
    "头、输入、标签与书都不动，只开头从多少历史学、多久重训、怎样给近期加权——"
    "尾部窗口 2 年或 5 年（c_base 3 年）、按月重训（c_base 按季，开关是 main.py 的 REFIT_PERIOD）、"
    "窗口内按一年半衰期给近期的训练日加权，以及前两名的组合；每条腿在每个种子基数上都是一个试验。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停；"
    "gpu_count 为 1，没有 CUDA 就停，不要改成 CPU。每一批都带同 span、同种子基数的 c_base。"
    "主张是时间结构上的、按种子基数配对的：在 refs/README.md 写定的两个种子基数上各与同种子的 c_base 比，"
    "候选的朴素选名（每个研究年书的收益减同一年零技能面板的收益，不回归，从行里逐年取）"
    "在 Y1..Y4 与 Y5..Y8 两半的平均都更高、在 Y7..Y8 的平均不更低，两个种子基数上主动 IR 之差的平均至少 +0.20；"
    "提名的那一行还要过它自己的 information_ratio_bar 与冻结门。"
    "先在第三个种子基数上用 Y5..Y8 探四根单轴，再按 refs/README.md 的规则选两个决赛者上整期；"
    "按月重训的整期腿约是按季的两到三倍长，整期批次之前先冒烟量 fit 的秒数、内存与显存。"
    "主张不成立的行即使过了门也不提名、以 no_edge 收尾——c_base 已经冻结；"
    "配方对研究期后几年没有影响也是要写清的结果，reason 里带上每个种子基数上两半与 Y7..Y8 的朴素选名之差。"
    "一个候选失败就测下一根登记的轴，它们不是参数邻域，本臂不换家族。"
    "每一行都报主动 IR、正年、主动回撤、逐年朴素选名、书对基准的市场载荷与权益口径的扣费年化收益。"
    "门槛读每行的 information_ratio_bar，不自己算；第一批 offline_trials = 0；杀死线只结束候选，至多四批；"
    "批次计划与决赛者规则照 refs/README.md。"
    + SEED_REPLICATES
)

# The lineage the queued cost and pool pairs of round 20261008 carry: that
# round's own, plus the label, fundamentals-input and clock arms that have
# since closed on the same baseline.
SEQFRESH_ARM: dict[str, object] = {
    "workspace_reference": SEQFRESH_PACK,
    "gpu_count": 1,
    "lineage_arms": SEQBOOK_LINEAGE,
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
