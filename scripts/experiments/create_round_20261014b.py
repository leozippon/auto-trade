#!/usr/bin/env python
"""The 2026-10-14 second round: two arms on hierarchical continual memory.

Both arms mount one pack (configs/workspace_refs/hcm_100k_8y_20261014b) and
receive one directive: test whether a stock-selection model that is never
re-initialised, updates its parameters on several time scales and carries an
external episodic memory keeps accumulating selection skill across market
regimes better than periodic cold retraining. A claim needs the same
architecture retrained cold every quarter and frozen as its controls, and
ablations of the state, the external memory and the per-level learning rates.
The arms differ only in the sub-agent model; the main session runs on
gpt-6.1-sol, while text-evidence scoring and compaction stay on the local model
(the defaults), since scoring also runs in the forward and Held-out replays,
whose data must not leave the machine.

| Arm | Main session | Sub-agents |
| --- | --- | --- |
| `hcm_100k_8y_sol_qwen_20261014b` | gpt-6.1-sol | local Qwen |
| `hcm_100k_8y_sol_dsf_20261014b` | gpt-6.1-sol | DeepSeek-V4.1-Flash |

The surface is the eight-year fundamentals geometry of round 20261005 on a
100k account graded against CSI 1000 (`FUND_EIGHT_YEAR_100K`: daily bars,
universe, the four macro tables and the ten default fundamentals datasets, no
events and no text), on the main board and ChiNext. Its seed was prebuilt for
exactly this selection under the current cache format from research release
62f3e8edb0b4424e828fbf3df0af8adf (logs/data/seed_8y_fund_20261014b/). Each arm
claims one GPU: the hypothesis is a trained model, and the pack's starter keeps
the CPU path the contract requires. The Qwen arm counts against the local-model
limit; the DeepSeek arm does not.

Fill order (ids end in `_20261014b`): `hcm_100k_8y_sol_qwen`,
`hcm_100k_8y_sol_dsf`.

Usage: create_round_20261014b.py <port> [--dry-run] [--fill] [experiment_id ...]
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

from scripts.experiments._profiles import FUND_EIGHT_YEAR_100K
from scripts.experiments._round import Round

PACK = "configs/workspace_refs/hcm_100k_8y_20261014b"
PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_fund_20261014b"
MAIN_MODEL = "gpt-6.1-sol"

DIRECTIVE = (
    "验证「分层持续记忆」——一个从不重新初始化、在多个时间尺度上更新参数、并带有外部情景记忆的选股模型，"
    "能否比周期性冷重训更能跨市况持续累积选股能力；"
    "必须对照同架构的季度冷重训与冻结模型，并做状态/外部记忆/分层学习率的消融。"
)

ARMS: dict[str, dict[str, object]] = {
    "hcm_100k_8y_sol_qwen_20261014b": {
        "model": MAIN_MODEL,
        "subagent_model": "qwen-3.8-27b-fp8",
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
    "hcm_100k_8y_sol_dsf_20261014b": {
        "model": MAIN_MODEL,
        "subagent_model": "DeepSeek-V4.1-Flash",
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=PIT_VIEWS_SEED,
    # One card per arm, claimed by the console when it starts the worker.
    overrides={**FUND_EIGHT_YEAR_100K, "permitted_boards": ["main", "gem"], "gpu_count": 1},
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
