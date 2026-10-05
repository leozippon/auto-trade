#!/usr/bin/env python
"""The 2026-10-04 round: more sessions per model, and DeepSeek as a third model.

The first paired comparison of the local model against the hosted
`mimo-v2.6-flash` did not show one clearly better: the hosted session used its
budget and followed the contract more fully, the local one was more restrained
about trials, and both cited figures or drew inferences their readings did not
support (logs/notes/round_20261002/R15_model_comparison.md). One finished pair
cannot separate what a model does from what one run happened to do, so this
round adds sessions on the same three 100k packs, scored on the rubric fixed in
logs/notes/round_20261003/DESIGN.md §3, and brings in `deepseek-flash`.

The queue, in fill order (ids end in `_20261004`); hosted arms come first
because they do not draw on the local inference service:

- `open_research_100k_8y_deepseek_r1` / `_r2`: DeepSeek on the open 100k pack,
  matching the two sessions the other two models already have.
- `census_lin_csi1000_100k_8y_deepseek`: DeepSeek on the pack-driven linear
  composite, with the lineage its pair had.
- `open_research_100k_8y_mimo_r3` / `_deepseek_r3` / `_qwen_r3`: a third
  session per model on the open pack.
- `seqbag_alla_100k_8y_deepseek`: DeepSeek on the sequence bag, on one GPU,
  with the lineage its pair has.

The open sessions carry no lineage, so every session of the comparison faces
the same conditions. That leaves their search outside each other's trial
count: a freeze from one of them is read against the pooled trials of all the
open 100k arms before it is treated as a result. Text-evidence scoring stays
on the local model in every arm.

Usage: create_round_20261004.py <port> [--dry-run] [--fill] [experiment_id ...]
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
from scripts.experiments.create_round_20260927 import (
    CSI1000,
    DIVIDEND_PIT_VIEWS_SEED,
    EIGHT_YEAR,
    GATES,
)
from scripts.experiments.create_round_20261002 import (
    CENSUS_LIN_DIRECTIVE,
    CENSUS_LIN_LINEAGE,
    CENSUS_LIN_PACK,
    MIMO,
    OPEN_DIRECTIVE,
    OPEN_PACK,
)
from scripts.experiments.create_round_20261003 import (
    SEQBAG_DIRECTIVE,
    SEQBAG_LINEAGE,
    SEQBAG_PACK,
)

# A hosted arm on DeepSeek: main session, sub-agents and compaction on its Flash
# model, the compaction threshold pinned to the local arms' value. `nl_model`
# is left on the local default.
DEEPSEEK: dict[str, object] = {
    "model": "deepseek-flash",
    "subagent_model": "deepseek-flash",
    "compact_model": "deepseek-flash",
    "compact_token_threshold": 221_184,
}

OPEN_ARM: dict[str, object] = {
    "workspace_reference": OPEN_PACK,
    "research_directive": OPEN_DIRECTIVE,
}

ARMS: dict[str, dict[str, object]] = {
    "open_research_100k_8y_deepseek_r1_20261004": {**DEEPSEEK, **OPEN_ARM},
    "open_research_100k_8y_deepseek_r2_20261004": {**DEEPSEEK, **OPEN_ARM},
    "census_lin_csi1000_100k_8y_deepseek_20261004": {
        **DEEPSEEK,
        "workspace_reference": CENSUS_LIN_PACK,
        "lineage_arms": CENSUS_LIN_LINEAGE,
        "research_directive": CENSUS_LIN_DIRECTIVE,
    },
    "open_research_100k_8y_mimo_r3_20261004": {**MIMO, **OPEN_ARM},
    "open_research_100k_8y_deepseek_r3_20261004": {**DEEPSEEK, **OPEN_ARM},
    "open_research_100k_8y_qwen_r3_20261004": dict(OPEN_ARM),
    "seqbag_alla_100k_8y_deepseek_20261004": {
        **DEEPSEEK,
        "workspace_reference": SEQBAG_PACK,
        "gpu_count": 1,
        "lineage_arms": SEQBAG_LINEAGE,
        "research_directive": SEQBAG_DIRECTIVE,
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=DIVIDEND_PIT_VIEWS_SEED,
    overrides={
        **EIGHT_YEAR,
        **GATES,
        "max_replay_years": 96,
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
    },
    closed=True,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
