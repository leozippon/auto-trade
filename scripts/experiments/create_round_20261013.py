#!/usr/bin/env python
"""The 2026-10-13 round: the incentive-draft book again, graded against the boards it can buy.

Round 20261012's 100k draft book (the first restricted-stock or option plan
draft after 90 quiet days, filling equal seats oldest first at the weekly
review, sold after 40 trading days) read an active IR of 0.99 / 1.32 / 1.41 /
1.15 at 16 / 20 / 24 / 30 seats on the eight years. Those readings were taken
against a zero-skill panel that swaps in names from every board, while the
book buys only main-board and ChiNext names, so part of them may be the
boards rather than the drafts. This round asks one question in three parts
(logs/notes/round_20261013/A_incdraft2_design.md): does the family survive a
board-matched panel (batch 1, the predecessor's legs byte for byte); which
drafts carry the drift, those whose titles name a stock option or the rest
(batch 2); and does the favoured side also lead on the titles before the
research period (offline), which alone admits a refined book (batch 3).

One pack (configs/workspace_refs/incdraft2_alla_100k_8y_20261013) is the
contract. Its starter is the predecessor's byte for byte, so `b20` and `b30`
add no trial to the lineage, and its stage 2 adds the instrument split and a
three-day delay control and nothing else. The account's boards are pinned in
the request, main board and ChiNext, rather than left to the capital (which
stamps the same two at CNY 100k today): the pack's round-0 check stops the
session unless the run facts name exactly these two, and the panel draws
only from the boards the Broker permits. No `fit`, no seed replicates, CPU
only.

The arm runs on the local model in every role. Seed, geometry and every other
round-level parameter are round 20261012's, and the acceptance rules are the
current generation's, which the launcher pins. The lineage is the title
family's five closed arms (`TITLE_LINEAGE`): 60 trials, about 57 effective,
an IR bar near 1.52 before the arm's own trials -- above the predecessor's
best reading of 1.41, so the arm's value is the three answers rather than a
graduate (the design note puts a freeze near three per cent).

Fill order (ids end in `_20261013`): `incdraft2_100k_8y_qwen`.

Usage: create_round_20261013.py <port> [--dry-run] [--fill] [experiment_id ...]
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
    TITLE_EIGHT_YEAR_100K,
    TITLE_LINEAGE,
    TITLE_PIT_VIEWS_SEED,
)
from scripts.experiments._round import Round

INCDRAFT2_PACK = "configs/workspace_refs/incdraft2_alla_100k_8y_20261013"

DIRECTIVE = (
    "本臂在板块一致的零技能面板上重跑上一场的股权激励计划草案书，并按标题把草案分成写明股票期权的一侧与其余一侧；"
    "refs/README.md 是合同、唯一权威表述，refs/families.md 是血缘与已关的近邻。"
    "第 0 轮核对 broker_replay.initial_cash 为 10 万、benchmark_index 为 000852.SH、"
    "broker_replay.permitted_boards.boards 恰好是 main 与 gem、文本域可用，任何一条不符就停。"
    "批 1 的四条腿只从未改过的 refs/starter 改 refs/README.md 开头那张表写明的一两行得来，与上一场逐字节相同；"
    "批 1 之后把 refs/stage2 整份拷进 output/ 一次，之后只改它的 knobs，不跑表外的腿。"
    "事件规则、可见日、复核、填席、持有期、等额、执行与行业上限都不改；不在事件之间排序，不训练、没有种子、只用 CPU。"
    "主张、分侧与精炼的规则、研究期之前的检查、批次计划、收尾线与提名照 refs/README.md；"
    "门槛读每行的 information_ratio_bar，不自己算；冻结门与毕业条件只看运行事实。"
    "offline_trials 照 refs/README.md §4 报：批 1 报 4，批 2 报 0，跑了研究期之前的检查后批 3 报 1。"
    "收尾 reason 要写的读数照 refs/README.md §8，连同本臂答不了的问题。"
)

ARMS: dict[str, dict[str, object]] = {
    "incdraft2_100k_8y_qwen_20261013": {
        "workspace_reference": INCDRAFT2_PACK,
        "research_directive": DIRECTIVE,
        "lineage_arms": TITLE_LINEAGE,
    },
}

# The seed, the geometry and every other round-level parameter are round
# 20261012's; the boards are this round's own decision.
ROUND = Round(
    arms=ARMS,
    pit_views_seed=TITLE_PIT_VIEWS_SEED,
    overrides={**TITLE_EIGHT_YEAR_100K, "permitted_boards": ["main", "gem"]},
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
