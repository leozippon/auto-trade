#!/usr/bin/env python
"""The 2026-10-12 title round: a ranker trained from scratch on announcement titles, one GPU arm.

The incentive-draft lane (round 20261012) showed on the host that the vendor's
announcement titles carry drift: a hand-written rule, the first restricted-stock
or option plan draft after 90 quiet days filling equal seats oldest first,
reads an active IR of 0.99 / 1.32 / 1.41 / 1.15 at 16 / 20 / 24 / 30 seats on
the eight years. The designer's title screens say more than one category
drifts and that the sign turns on the kind and wording of an announcement:
plan drafts, grants and holding-increase plans up, contract wins and finished
employee-plan purchases down (logs/notes/round_20261012/DESIGN.md §3, §8,
DESIGN_2_expanded_data.md §3, §6 G1). This lane asks whether a model that
reads titles and nothing else can rank every fresh announcement better than
the one rule we trust, and whether that needs a neural net: the
character-level ranker must beat its linear twin or the twin is the
candidate. Its information set is new to the platform, its lineage is only
the three incentive arms, and its honest prior of a graduate is about five
per cent.

One pack (configs/workspace_refs/titlerank_alla_100k_8y_20261012):

- Unit: one row per (name, weekly review), the distinct titles that became
  visible for the name since the previous review, company names and digits
  stripped, companion copies merged, at most 16. The book acts at reviews, so
  every title fresh at one review shares one entry and one label; a row per
  title or per day would repeat that label and weigh a name by how much it
  files. The vendor's index carries one code a row, so a document filed for
  several names is a unit of each.
- Label: the name's close-to-close return over the book's 40-day hold from
  the review's close, minus the equal-weight mean of its float-cap quintile
  (the designer's size matching), ranked across the review's units; only
  labels whose last bar is visible at the fit.
- Models, refitted at each calendar quarter on the trailing three years, the
  last 13 labelled reviews as validation and 40 trading days of embargo:
  `nn`, a character CNN in plain torch (embedding 64, widths 2-4 x 128
  channels, max over positions and over the unit's titles, a two-layer head),
  early-stopped on the validation rank IC, seeded by one integer line `SEED`,
  CUDA only; `ridge`, the mandatory twin, a ridge on 2^18 hashed character
  1-3-grams with alpha from 1 / 10 / 100 on the same validation, deterministic.
- Book: an event book, as the memos favour for turnover. `fit` runs daily: it
  refits at a quarter's first decision and, at each weekly review, records the
  fresh units scoring at or above the 99th percentile of the model's
  validation scores; those events fill a 20-seat equal-cash book highest score
  first, bought at the 15:00 close, sold at 09:30 of the first review 40
  trading days on, one lot must fit, at most 4 names an industry. At 100k a
  seat is about 4,850 yuan, so the 5-yuan minimum is about 10 bp a side and a
  round trip about 36 bp; six turns a year cost about 2.2 % of the account.
  Each review is scored once and recorded, so a holding keeps the event it
  was bought on through later refits.
- Controls: `c_shuf`, labels permuted across each review's units before
  fitting; `c_blind`, the same pipeline reading each unit's size decile, board
  and document count instead of its titles; `c_rule`, the incentive rule in
  this book, which a unit test shows trades order for order as the incdraft
  book at 20 seats. The random-name copy is the host's zero-skill panel and is
  not run again.

The claim, paired with same-batch, same-seed controls: plain selection higher
than `c_shuf` and `c_blind` on both halves and the last two research years
on every paired seed base, active IR at least 0.25 above each on average, and
an active IR not below `c_rule` with a higher eight-year plain selection. The
pack states its claim and plan and points to the arm's facts for the gate
conditions; it waives none. Among rows that would clear the gate, it
nominates the one with the higher equity return after costs, not the higher
IR: the owner wants small-account lanes to pursue larger returns.

Batches: a `Y5..Y8` probe of `nn` and `c_shuf` on a fourth seed base and
`ridge` (12 replay-years); `nn` on seed bases 1000 and 2000 with `c_shuf`,
`c_blind`, `ridge` and `c_rule` over the full span (48); then `nn`, `c_shuf`
and `c_blind` on seed base 3000, or the twin's own controls and seed
replicate if the net lost to it (24): 84 of the 96 replay-years. Measured
through the real pipeline on a free L20 (six days from Y1's start and from
Y8's last May week; logs/notes/round_20261012c/_pack/): a cold refit of the
net takes 34 s on Y1's 68,288 training units and 121 s on Y8's 297,228
alone, 323-342 s with a second net training on the same card, at 3.6 GiB of
card memory each; the ridge's 130 s on the CPU; a review's scoring about 2 s
and any other day's `fit` 0.2 s. The net trains the same weights byte for
byte alone and beside other legs, and `c_rule`'s 61 orders over 70 days of
Y1 are the incdraft starter's at 20 seats. At three legs at a time on the
arm's one card the batches take about 2-2.5, 7-8 and 2.5-5 hours (the card
held), 14-16 card-hours in all and under ten a batch, about 10 of them
training on it.

The lineage is the incentive family (`INCDRAFT_LINEAGE`): the same titles on
the same eight years. The console reads a lineage arm only after it closes,
so the pre-flight refuses this arm by name until the last of the three has
closed, and a fill that reaches it before then reports that refusal. The
title-event map (round 20261012d, `titlemap_100k_8y_mimo`) runs beside it on
the same titles with the same lineage; neither inherits the other, the
accepted cost of running them side by side. The lineage's 54 declared screens
count one independent trial each here, so the bar starts near 1.5 rather
than low. The arm declares in its first batch the four title screens of the
second memo that shaped this lane and that no lineage arm declared (grants,
finished employee-plan purchases, placement plans, no-sale commitments); the
other six title screens of both memos reach its family through the lineage.

Text-evidence scoring stays on the local model; the book makes no evidence
call. Fill order (ids end in `_20261012`): `titlerank_100k_8y_qwen`.

Usage: create_round_20261012c.py <port> [--dry-run] [--fill] [experiment_id ...]
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
    INCDRAFT_LINEAGE,
    TITLE_EIGHT_YEAR_100K,
    TITLE_PIT_VIEWS_SEED,
)
from scripts.experiments._round import Round

TITLERANK_PACK = "configs/workspace_refs/titlerank_alla_100k_8y_20261012"

DIRECTIVE = (
    "本臂走公告标题上从零训练的排序器，refs/README.md 是合同，refs/families.md 是近邻与血缘。"
    "单元是一只股票在一次周复核时新可见的全部标题；标签是它在书的持有期上相对同流通市值五分位的收益在同一次复核内的秩。"
    "候选是字符级卷积排序器 nn（在卡上训练，训练种子是 knobs 的 SEED 一行），必备基线是散列字符 n-gram 上的岭回归 ridge："
    "nn 在两个种子基数的整期主动 IR 上都赢过 ridge 才是决赛者，否则 ridge 是决赛者。"
    "书是 20 席事件书：分数过阈值的新单元按分数填空席，持满 40 个交易日后卖，每次复核的分数由 fit 记下，书只读这些记录。"
    "对照三个，都登记为对照：标签在同一次复核内打乱的 c_shuf，不读标题、只读规模十分位、板块与文件数的 c_blind，"
    "同一本书上的激励草案规则 c_rule；随机名副本就是零技能面板，不另跑。"
    "第 0 轮核对 initial_cash 为 10 万、benchmark_index 为 000852.SH、文本域可用、gpu_count 为 1，否则停；没有 CUDA 就停，不要改成 CPU。"
    "主张、批次计划、决赛者规则与杀死线照 refs/README.md；第一批 offline_trials 报 4，之后报 0。"
    "门槛读每行的 information_ratio_bar，不自己算；冻结门、种子复现与毕业条件只看运行事实。"
    "过门的行不止一条时，提名权益口径扣费年化收益更高的那一行，不按主动 IR 挑：主人要求小账户追求更大的收益。"
    "每一行都报主动 IR、正年、主动回撤、逐年朴素选名、两半与最后两年、市场载荷与权益口径的扣费年化收益。"
)

ARMS: dict[str, dict[str, object]] = {
    "titlerank_100k_8y_qwen_20261012": {
        "workspace_reference": TITLERANK_PACK,
        "gpu_count": 1,
        "lineage_arms": INCDRAFT_LINEAGE,
        "research_directive": DIRECTIVE,
    },
}

# The seed, the geometry and every round-level parameter are round 20261012's.
ROUND = Round(
    arms=ARMS,
    pit_views_seed=TITLE_PIT_VIEWS_SEED,
    overrides=TITLE_EIGHT_YEAR_100K,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
