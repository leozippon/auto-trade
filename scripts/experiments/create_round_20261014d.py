#!/usr/bin/env python
"""The 2026-10-14 fourth round: round 20261014b's two HCM arms, restarted.

Round 20261014b's arms were terminated after 36 minutes. Their starter lost
the slow parameter groups' pending gradients at every checkpoint, so the
multi-timescale updates the study is about never accumulated (fixed in commit
3c7b844), and the main model filled the `agent` tool's optional parameters
with placeholders, so all 469 sub-agent launches were rejected (fixed in
483b16d: optional tool parameters are nullable). Both arms keep their records
and resolved issue reports; this round reruns the same study on the fixed
pack and code.

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
whose data must not leave the machine. The main model's window is far larger
than the local models', so the compaction threshold is pinned to the local
arms' value, as `_profiles.MIMO` does, to keep each call's prompt bounded.

| Arm | Main session | Sub-agents |
| --- | --- | --- |
| `hcm_100k_8y_sol_qwen_20261014d` | gpt-6.1-sol | local Qwen |
| `hcm_100k_8y_sol_dsf_20261014d` | gpt-6.1-sol | DeepSeek-V4.1-Flash |

The surface is the eight-year fundamentals geometry of round 20261005 on a
100k account graded against CSI 1000 (`FUND_EIGHT_YEAR_100K`: daily bars,
universe, the four macro tables and the ten default fundamentals datasets, no
events and no text), on the main board and ChiNext. Its seed was prebuilt for
exactly this selection under the current cache format from research release
62f3e8edb0b4424e828fbf3df0af8adf (logs/data/seed_8y_fund_20261014b/). Each arm
claims one GPU: the hypothesis is a trained model, and the pack's starter keeps
the CPU path the contract requires. The Qwen arm counts against the local-model
limit; the DeepSeek arm does not.

Fill order (ids end in `_20261014d`): `hcm_100k_8y_sol_qwen`,
`hcm_100k_8y_sol_dsf`.

Usage: create_round_20261014d.py <port> [--dry-run] [--fill] [experiment_id ...]
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
COMPACT_TOKEN_THRESHOLD = 221_184

DIRECTIVE = (
    "验证「分层持续记忆」——一个从不重新初始化、在多个时间尺度上更新参数、并带有外部情景记忆的选股模型，"
    "能否比周期性冷重训更能跨市况持续累积选股能力；"
    "必须对照同架构的季度冷重训与冻结模型，并做状态/外部记忆/分层学习率的消融。"
)

ARMS: dict[str, dict[str, object]] = {
    "hcm_100k_8y_sol_qwen_20261014d": {
        "model": MAIN_MODEL,
        "subagent_model": "qwen-3.8-27b-fp8",
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
    "hcm_100k_8y_sol_dsf_20261014d": {
        "model": MAIN_MODEL,
        "subagent_model": "DeepSeek-V4.1-Flash",
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
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
