#!/usr/bin/env python
"""The 2026-10-26 round: round 20261025b's retrieval direction on the local console, with gemini-4-argon, gpt-6-astra and claude-fable-5-1 and an emphasis on exploration.

The three arms run round 20261025b's experiment as it stands -- the pack
`retrieval_100k_8y_20261025b`, the eight-year fundamentals-plus-titles
geometry on a 100k account graded against CSI 1000
(`FUND_TITLE_EIGHT_YEAR_100K`), main board and ChiNext, the 10-24 seed,
sub-agents and compaction on DeepSeek-V4.1-Flash with the compaction threshold
pinned at 221,184, xhigh, the console's default budgets and batch width, no
lineage -- with four differences: the main models, the relay's alternate key,
two cards per arm, and a directive that is the pack's direction paragraph
word for word followed by one sentence asking for exploration inside it.

| Arm | Main session | Sub-agents / compaction |
| --- | --- | --- |
| `retrieval_100k_8y_argon_20261026` | gemini-4-argon | DeepSeek-V4.1-Flash |
| `retrieval_100k_8y_astra_20261026` | gpt-6-astra | DeepSeek-V4.1-Flash |
| `retrieval_100k_8y_fable_20261026` | claude-fable-5-1 | DeepSeek-V4.1-Flash |

Key. Every role of every arm calls the relay with `LIGHTING_API_KEY_ALT`
first (`llm_key` alternate, kept in the arm's params.json), with
`LIGHTING_API_KEY` as the fallback when the alternate is rejected or runs dry.

Cards. Each arm claims two L20s, whichever free ones the console takes when it
starts the worker (`/api/health` lists them as `gpus_free`), so the round
needs six free cards. A batch's three legs take the arm's two cards in turn,
so two of them share one; a replay that runs alone attaches both.

Usage, against the local console on port 38888:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261026.py 38888 [--dry-run] [experiment_id ...]

`--dry-run` reads the seed contract and the template's smoke record and
creates nothing.
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

from scripts.experiments._profiles import FUND_TITLE_EIGHT_YEAR_100K, FUND_TITLE_PIT_VIEWS_SEED
from scripts.experiments._round import Round

PACK = "configs/workspace_refs/retrieval_100k_8y_20261025b"
ARGON = "gemini-4-argon"
ASTRA = "gpt-6-astra"
FABLE = "claude-fable-5-1"
FLASH = "DeepSeek-V4.1-Flash"
COMPACT_TOKEN_THRESHOLD = 221_184

# The owner's direction, word for word the paragraph under the pack README's
# 「方向」, then the exploration this round asks for inside it
# (tests/unit/test_retrieval_pack.py holds the first part equal to the README).
DIRECTIVE = (
    "在八年研究期上，检索与当前情形相似的历史价量形态与自然语言信息（种子里的公告标题等文本），"
    "用检索到的历史案例选股，选出的书要相对零技能面板有真实的边。"
    "什么算「相似」、怎样使用检索到的案例、是否以及怎样用到任何模型（语言模型或其他），都由会话决定。"
    "在这个方向之内要多探索：试几种不同的「相似」、不同的检索对象与使用检索案例的不同方式，"
    "不要在第一个出现的信号上过早收敛；去膨胀门槛决定一个节点能不能冻结，不决定什么时候停止探索。"
)

ARMS: dict[str, dict[str, object]] = {
    f"retrieval_100k_8y_{name}_20261026": {
        "model": model,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "llm_key": "alternate",
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    }
    for name, model in (("argon", ARGON), ("astra", ASTRA), ("fable", FABLE))
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=FUND_TITLE_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_EIGHT_YEAR_100K,
        "permitted_boards": ["main", "gem"],
        # Two L20s per arm, claimed by the console when it starts the worker.
        "gpu_count": 2,
        "reasoning_effort": "xhigh",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
