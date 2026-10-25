#!/usr/bin/env python
"""The 2026-10-25 round: one open question on every data domain and the in-process models, on the A100 instance.

Both arms mount `open_lm_100k_8y_20261025`, which states the owner's question
and only the facts a session cannot discover for itself -- the card, the
model store and in-process vLLM, text visibility and anonymisation, the
account and its costs, what a full-span leg costs -- with the record of
closed families as information. It ships no starter, so `output/` begins as
the contract template, and no registered pairs, claim thresholds, budget plan
or stop-loss: what binds an arm is the harness's own (point-in-time data,
costs, hard caps, the freeze gate and honest reporting).

| Arm | Main session | Sub-agents / compaction |
| --- | --- | --- |
| `open_lm_100k_8y_opus_zgca_20261025` | claude-opus-5-5 | DeepSeek-V4.1-Flash |
| `open_lm_100k_8y_sol_zgca_20261025` | gpt-6.1-sol | DeepSeek-V4.1-Flash |

The surface is round 20261022's: the eight-year fundamentals-plus-titles
geometry on a 100k account graded against CSI 1000
(`FUND_TITLE_EIGHT_YEAR_100K`), main board and ChiNext, on the 10-24 seed.
Each arm claims one of the instance's two A100s, so the two run side by side,
and the candidates of a batch share that card. Budgets, batch width and fit
timeout are the console defaults; the compaction threshold is pinned as in the
earlier rounds. The direction is new: no lineage. The arms pin the pretrained
store the instance holds when they are created, which must be the
three-model library the pack describes.

The host must name its card model (`AUTOTRADE_GPU_NAME=A100`) in the
environment of both the console and this launcher, which `ops/zgca/zgca.py
run` does; one launcher run naming both ids creates them:

    python ops/zgca/zgca.py run scripts/experiments/create_round_20261025.py open_lm_100k_8y_opus_zgca_20261025 open_lm_100k_8y_sol_zgca_20261025

(`--dry-run` before it reads the seed contract and smokes the template.)

Usage: create_round_20261025.py <port> [--dry-run] [--fill] [experiment_id ...]
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

PACK = "configs/workspace_refs/open_lm_100k_8y_20261025"
OPUS = "claude-opus-5-5"
SOL = "gpt-6.1-sol"
FLASH = "DeepSeek-V4.1-Flash"
COMPACT_TOKEN_THRESHOLD = 221_184

# The owner's question, word for word as the pack's README states it.
DIRECTIVE = (
    "在八年研究期上设计一个新颖的选股策略，让它相对零技能面板有真实的边。"
    "沙箱里的一切都可以用：日线、财务、种子里的公告标题，"
    "以及进程内的 `Qwen3.8-27B-FP8`、`Qwen3.5-2B` 与 `Qwen3-Embedding-0.6B`。"
)

ARMS: dict[str, dict[str, object]] = {
    "open_lm_100k_8y_opus_zgca_20261025": {
        "model": OPUS,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
    "open_lm_100k_8y_sol_zgca_20261025": {
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
