#!/usr/bin/env python
"""The 2026-10-14 third round: two HCM arms on the A100 development instance.

The arms run round 20261014b's study unchanged -- the same pack
(configs/workspace_refs/hcm_100k_8y_20261014b), directive, eight-year
fundamentals geometry on a 100k account graded against CSI 1000, boards and
seed -- on a host with one A100 instead of the L20 workstation. They differ
only in the main session: DeepSeek-V4.1-Flash, or gpt-6.1-sol. Sub-agents and
compaction run on DeepSeek-V4.1-Flash on the LightingTheWord relay in both,
since the instance has no local model, with the compaction threshold pinned to
the local arms' value as `_profiles.MIMO` does for a 1M window. Text-evidence
scoring stays on the local default; the pack's starter never calls
`context.nl()` and the pack's rules forbid it, so that gateway is built but
not called.

| Arm | Main session | Sub-agents | Compaction |
| --- | --- | --- | --- |
| `hcm_100k_8y_dsf_zgca_20261014c` | DeepSeek-V4.1-Flash | DeepSeek-V4.1-Flash | DeepSeek-V4.1-Flash |
| `hcm_100k_8y_sol_dsf_zgca_20261014c` | gpt-6.1-sol | DeepSeek-V4.1-Flash | DeepSeek-V4.1-Flash |

The host must name its card model (`AUTOTRADE_GPU_NAME=A100`) in the
environment of both the console and this launcher. Each arm claims the one
GPU, so they run one after the other, and neither counts against the
local-model limit.

Usage: create_round_20261014c.py <port> [--dry-run] [--fill] [experiment_id ...]
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
MODEL = "DeepSeek-V4.1-Flash"
SOL = "gpt-6.1-sol"

DIRECTIVE = (
    "验证「分层持续记忆」——一个从不重新初始化、在多个时间尺度上更新参数、并带有外部情景记忆的选股模型，"
    "能否比周期性冷重训更能跨市况持续累积选股能力；"
    "必须对照同架构的季度冷重训与冻结模型，并做状态/外部记忆/分层学习率的消融。"
)

ARMS: dict[str, dict[str, object]] = {
    "hcm_100k_8y_dsf_zgca_20261014c": {
        "model": MODEL,
        "subagent_model": MODEL,
        "compact_model": MODEL,
        "compact_token_threshold": 221_184,
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
    "hcm_100k_8y_sol_dsf_zgca_20261014c": {
        "model": SOL,
        "subagent_model": MODEL,
        "compact_model": MODEL,
        "compact_token_threshold": 221_184,
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=PIT_VIEWS_SEED,
    # One card, claimed by the console when it starts the worker.
    overrides={**FUND_EIGHT_YEAR_100K, "permitted_boards": ["main", "gem"], "gpu_count": 1},
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
