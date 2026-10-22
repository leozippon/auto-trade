#!/usr/bin/env python
"""The 2026-10-22 round: case retrieval over a point-in-time memory, two models on one pack, locally.

Both arms mount `retrieval_lm_100k_8y_20261022`: a book that scores each name
by the outcomes of its nearest matured cases in a memory of price-volume
patterns and announcement titles, kept inside each replay and grown daily,
against the pack's quarterly cold-refit carrier on the same keys (C1). The
starter embeds titles in process with Qwen3-Embedding-0.6B; the served-model
path (in-process vLLM over a shortlist of retrieved cases) is an example in
the pack that the starter does not call. Models reach a strategy only in
process, from the pinned pretrained store; no network and no hosted model is
reachable from a strategy container.

| Arm | Main session | Sub-agents / compaction |
| --- | --- | --- |
| `retrieval_lm_100k_8y_opus_20261022` | claude-opus-5-5 | DeepSeek-V4.1-Flash |
| `retrieval_lm_100k_8y_sol_20261022` | gpt-6.1-sol | DeepSeek-V4.1-Flash |

The surface is round 20261019's: the eight-year fundamentals-plus-titles
geometry on a 100k account graded against CSI 1000
(`FUND_TITLE_EIGHT_YEAR_100K`), main board and ChiNext, on the 10-22 seed.
Each arm claims two cards and validates two candidates at a time, one leg per
card; a replay that runs alone attaches both, which a leg that wants both
(tensor-parallel, the 27B in bf16) needs. A fit may take three hours instead
of one: the starter's first fit takes up to about four minutes and one with
the 27B example about seven, but a first fit that also runs a bulk,
point-in-time pass of the 27B over the memory's history needs the room (the
pack README states the measured costs). Decisions keep the default 360 s:
the pack puts every model call in `fit`. Hosted spend is capped at 50 USD,
the compaction threshold pinned as in round 20261019. The direction is a new
family: no lineage. Neither arm counts against the local-model limit.

Fill order (ids end in `_20261022`): `retrieval_lm_100k_8y_opus`,
`retrieval_lm_100k_8y_sol`.

Usage: create_round_20261022.py <port> [--dry-run] [--fill] [experiment_id ...]
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

PACK = "configs/workspace_refs/retrieval_lm_100k_8y_20261022"
OPUS = "claude-opus-5-5"
SOL = "gpt-6.1-sol"
FLASH = "DeepSeek-V4.1-Flash"
COMPACT_TOKEN_THRESHOLD = 221_184

DIRECTIVE = (
    "验证「检索相似历史案例并据此预测」：把每只股票每个决策日的价量形态与公告标题存成按时点到期的案例记忆，"
    "每天按相似度检索已到期的历史案例、用它们的结局预测未来，能否胜过同一组特征上按季冷重训的参数模型（C1）；"
    "随后分清相似性对随机邻居、持续更新对冻结记忆、文本对纯形态，以及进程内大模型对检索案例的上下文推断"
    "对同一短名单与打乱检索的安慰剂各贡献多少。记忆怎样存、怎样检索、模型怎样部署与推断由会话设计。"
)

ARMS: dict[str, dict[str, object]] = {
    "retrieval_lm_100k_8y_opus_20261022": {
        "model": OPUS,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
    "retrieval_lm_100k_8y_sol_20261022": {
        "model": SOL,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=FUND_TITLE_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_EIGHT_YEAR_100K,
        "permitted_boards": ["main", "gem"],
        # Two cards per arm, claimed by the console when it starts the worker;
        # the two legs of a batch take one card each.
        "gpu_count": 2,
        "batch_validate_max_concurrency": 2,
        # Three hours per fit (doubled while a two-leg batch runs): room for a
        # first fit that loads a 27B engine and precomputes over the memory.
        "strategy_fit_timeout_seconds": 10_800,
        "max_api_spend_usd": 50,
        "reasoning_effort": "xhigh",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
