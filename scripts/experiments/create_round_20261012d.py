#!/usr/bin/env python
"""The 2026-10-12 title-event map: other title-defined company commitments, one category per candidate.

Across some fifty long-side screens on the research window the designer found
drift only after announcements in which a company or its insiders commit
money or pay: incentive-plan drafts, then grants, holding-increase plans and
buyback plans, in that order of strength; salient news such as contract wins
is followed by underperformance (logs/notes/round_20261012/DESIGN.md §3-§4,
DESIGN_2_expanded_data.md §3-§4). Round 20261012 runs the drafts. This round
runs the same event book over the three next commitment categories, each a
registered candidate with its own title rule, so that each category's drift is
measured on the host against its own late-entry control and priced as a host
trial rather than chosen offline. The honest prior is about five per cent.

One pack (configs/workspace_refs/titlemap_alla_100k_8y_20261012) carries the
incentive book's machinery unchanged: the first title of a category a stock
carries after 90 quiet days is visible on the first trading day after the
local date of `available_at`, and the weekly review fills 20 equal-cash seats
with the events that became visible since the previous review, oldest first,
selling each the first review HOLD trading days after it was bought (sell at
09:30, buy at 15:00, the 100k account). Candidates: incentive grants
(`grant`, the baseline; the designer's rule without the unlock notices that
opened 12 % of its events), holding-increase plans (`increase`, the
designer's rule) and buyback plans (`buyback`, a rule written for the titles,
since the designer read buybacks from the structured table). Employee-
ownership drafts and completed purchases, no-reduction commitments and
placement plans read flat or weak offline and are declared screens (4
offline trials in the first batch); contract wins are short-side only,
tender offers too sparse (104 in eight years), buyback completions flat and
reduction plans a selling signal, so none of them enters the host.

The control is the incentive lane's `c_late` for each category: the same
events entered 120 trading days late on the same clock. The claim, per
category and against the same-shape `c_late`: plain selection higher on each
half of the research window and on its last two years, and an active IR at
least 0.25 higher. Batch 1 is the three categories and their controls (48
replay-years, about three hours at three-way concurrency, as the incentive
arms' first batches took); batch 2 the finalist's 60-day hold with its
control and, when two or more categories qualify, their union under one
quiet period with its control (16 or 32). Among the rows whose claim holds
and that clear the gate, the pack nominates the one with the higher raw
return after costs (`stats.annualized_return`), not the higher IR: the
owner asks small-account lanes for larger returns. The pack points to the
arm's facts for every gate condition and waives none. No `fit`, no seed
replicates, CPU only.

`titlemap_100k_8y_mimo` runs on `mimo-v2.6-flash` (the local-model slots are
nearly full); text-evidence scoring stays on the local model and the book
makes no evidence call. Seed, geometry and round-level parameters are round
20261012's. The lineage is the three incentive-draft arms (`INCDRAFT_LINEAGE`):
the same family of title-defined commitment events on the same seed. A
lineage arm is read from its research session's record, so the pre-flight
refuses this arm by name until all three have closed.

Fill order (ids end in `_20261012`): `titlemap_100k_8y_mimo`.

Usage: create_round_20261012d.py <port> [--dry-run] [--fill] [experiment_id ...]
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
    MIMO,
    TITLE_EIGHT_YEAR_100K,
    TITLE_PIT_VIEWS_SEED,
)
from scripts.experiments._round import Round

TITLEMAP_PACK = "configs/workspace_refs/titlemap_alla_100k_8y_20261012"

DIRECTIVE = (
    "本臂走标题定义的公司承诺事件图，refs/README.md 是合同，refs/families.md 是血缘与已关的近邻。"
    "三个登记的类目各是一条候选：激励授予（grant，基线）、增持计划（increase）、回购计划（buyback）；"
    "标题规则、九十天静默期与可见日写在 lib/titles.py 与 refs/README.md §3，不改，也不登记别的类目。"
    "书在每个 ISO 周的第一个决策日把上次复核以来可见的新事件按先后填进 20 个等额空席，一手可买，"
    "09:30 卖、15:00 买，持满登记的交易日数后在复核日卖出；不在事件之间排序，不训练、没有种子、只用 CPU。"
    "第 0 轮核对 broker_replay.initial_cash 为 10 万、benchmark_index 为 000852.SH、文本域可用，否则停。"
    "每一类配自己的晚进场对照 c_late：同一批事件晚 120 个交易日进场、同类目同持有，登记为对照；"
    "零技能面板本身就是同日、同数、同额的随机名安慰剂，不另跑。"
    "主张逐类与同形 c_late 配对：朴素选名（每个研究年书的收益减同年零技能面板的收益）"
    "在 Y1..Y4、Y5..Y8 与 Y7..Y8 的平均都更高，主动 IR 至少高 0.25。"
    "批 1 是三类与各自的 c_late；几类的并集只在批 2，只由有资格的类目组成。"
    "批次计划、决赛者规则与杀死线照 refs/README.md；门槛读每行的 information_ratio_bar，不自己算；"
    "冻结门与毕业条件只看运行事实。"
    "在主张成立、冻结门通过的行之间提名扣费年化收益（stats.annualized_return）最高的一行，"
    "不是主动 IR 最高的一行：小账户的持有人要更高的绝对收益。"
    "第一批 offline_trials 报 4，之后报 0。每一行都报主动 IR、正年、主动回撤、逐年朴素选名、"
    "扣费年化收益、市场载荷与平均总仓位。"
)

ARMS: dict[str, dict[str, object]] = {
    "titlemap_100k_8y_mimo_20261012": {
        **MIMO,
        "workspace_reference": TITLEMAP_PACK,
        "research_directive": DIRECTIVE,
        "lineage_arms": INCDRAFT_LINEAGE,
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
