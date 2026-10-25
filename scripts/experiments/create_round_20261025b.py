#!/usr/bin/env python
"""The 2026-10-25 b round: one focused direction, stock selection by retrieving similar historical price-volume patterns and text, on the A100 instance.

Both arms mount `retrieval_100k_8y_20261025b`, which states the owner's
direction -- retrieve similar historical price-volume patterns and
natural-language information (the seed's announcement titles and other text)
and select stocks with the retrieved cases -- and leaves the method to the
session: what counts as similar, how the retrieved cases are used, and whether
and how any model takes part. Beyond the direction it holds only the facts a
session cannot discover for itself -- the card, text visibility and
anonymisation, the case-memory boundary, the data's size and what a full-span
leg costs, the pretrained store for an arm that chooses to use it, the account
and its costs, the seed -- with the record of closed families as information.
It ships no starter, so `output/` begins as the contract template, and no case
definition, registered pairs, claim thresholds, budget plan or stop-loss: what
binds an arm beyond the direction is the harness's own (point-in-time data,
costs, hard caps, the freeze gate and honest reporting).

| Arm | Main session | Sub-agents / compaction |
| --- | --- | --- |
| `retrieval_100k_8y_opus_zgca_20261025b` | claude-opus-5-5 | DeepSeek-V4.1-Flash |
| `retrieval_100k_8y_sol_zgca_20261025b` | gpt-6.1-sol | DeepSeek-V4.1-Flash |

It replaces round 20261025, whose two arms carried a fully open question and
were withdrawn before their first leg. The surface is round 20261022's: the
eight-year fundamentals-plus-titles geometry on a 100k account graded against
CSI 1000 (`FUND_TITLE_EIGHT_YEAR_100K`), main board and ChiNext, on the 10-24
seed. Each arm claims one of the instance's two A100s, so the two run side by
side, and the candidates of a batch share that card. Budgets, batch width and
fit timeout are the console defaults; the compaction threshold is pinned as in
the earlier rounds. No lineage: the arms start from the template, not from a
shape the 20261022 arms validated, and the pack's record of closed families
lists that round's construction as closed. The arms pin the pretrained store
the instance holds when they are created; the pack points to the contract's
model section instead of listing it.

The host must name its card model (`AUTOTRADE_GPU_NAME=A100`) in the
environment of both the console and this launcher, which `ops/zgca/zgca.py
run` does; one launcher run naming both ids creates them:

    python ops/zgca/zgca.py run scripts/experiments/create_round_20261025b.py retrieval_100k_8y_opus_zgca_20261025b retrieval_100k_8y_sol_zgca_20261025b

(`--dry-run` before it reads the seed contract and smokes the template.)

Usage: create_round_20261025b.py <port> [--dry-run] [--fill] [experiment_id ...]
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
OPUS = "claude-opus-5-5"
SOL = "gpt-6.1-sol"
FLASH = "DeepSeek-V4.1-Flash"
COMPACT_TOKEN_THRESHOLD = 221_184

# The owner's direction, word for word the paragraph under the pack README's
# 「方向」 (tests/unit/test_retrieval_pack.py holds the two equal).
DIRECTIVE = (
    "在八年研究期上，检索与当前情形相似的历史价量形态与自然语言信息（种子里的公告标题等文本），"
    "用检索到的历史案例选股，选出的书要相对零技能面板有真实的边。"
    "什么算「相似」、怎样使用检索到的案例、是否以及怎样用到任何模型（语言模型或其他），都由会话决定。"
)

ARMS: dict[str, dict[str, object]] = {
    "retrieval_100k_8y_opus_zgca_20261025b": {
        "model": OPUS,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
    "retrieval_100k_8y_sol_zgca_20261025b": {
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
        # The one card, claimed by the console when it starts the worker.
        "gpu_count": 1,
        "reasoning_effort": "xhigh",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
