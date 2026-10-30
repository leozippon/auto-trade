#!/usr/bin/env python
"""The 2026-10-31 round: memory on a ten-year window, gemini-4-argon and gpt-6-astra on the local console.

Two arms mount `memory_100k_rolling_20261031`. The pack states the direction
and carries the owner's memory note; this file does not copy either. Research
runs ten whole August-July years, 2015-08-01..2025-07-31, the twelve forward
months after them end 2026-07-31, and Held-out reaches 2026-10-31, which the
replay clips to the release's last trading day. Decision views reach
`window_months` 132 back from the research end: the ten research years plus
the one year of history in front of their start, where the eight-year rounds'
108 would stop a year short of the research start. Every data domain follows
that one window; none is overridden, and a domain whose release history is
shorter carries what the release has rather than a claimed 132 months of raw
data. The
dataset selection, the 100k account graded against CSI 1000 and the
main-board-plus-ChiNext universe are unchanged (`FUND_TITLE_EIGHT_YEAR_100K`:
daily, the ten default fundamentals, the four macro datasets and the
announcement titles).

No earlier candidate is copied and no per-arm lineage is declared. Shared
operating memory keeps the existing policy; old Forward/Held-out readings and
verdicts are not added to the new reference pack. Prior searches observed
overlapping history, so shifting the dates does not restore independence.

The shared seed is `data/pit_views_seed_research_10y_fundanns_20261031`.
Creation requires a complete prebuild and a published release covering the
requested dates. The release is read from the seed's `provider.json` at
pre-flight, not selected from the mutable lake.

| Arm | Main session | Sub-agents / compaction |
| --- | --- | --- |
| `memory_100k_10y_argon_local_20261031` | gemini-4-argon | DeepSeek-V4.1-Flash |
| `memory_100k_10y_astra_local_20261031` | gpt-6-astra | DeepSeek-V4.1-Flash |

Sub-agents and compaction stay on DeepSeek-V4.1-Flash, the compaction
threshold at 221,184, reasoning at xhigh. `llm_key` alternate starts every relay
role on `LIGHTING_API_KEY_ALT`; the primary key remains the relay's existing
fallback.

The figures below are this round's create defaults, not a host limit. An arm's
own entry overrides any of them, including `gpu_count`,
`batch_validate_max_concurrency`, `strategy_cpus` and `strategy_memory`, and
that entry is what the existing console create accepts. This round's default is
one card and one replay at a time, with the same ceilings as rounds 20261028
and 20261030: 288 replay-years, 36 null controls, USD 150, 19,200 model calls,
7,200 minutes, 10,800 s per `fit` and 1,080 s per `generate_orders`. The
strategy container default is 8 CPUs and 64 GiB. GPU placement uses the existing
free-memory threshold and claim rules; this file names no device.

Create through the existing console round entry, not a separate launcher:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261031.py <port> [--dry-run] [--fill]

`--dry-run` reads the seed contract and, unless a smoke record already covers
this template, replays the template starter. It creates nothing, and it cannot
pass before the seed's prebuild has finished.
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

from scripts.experiments._profiles import FUND_TITLE_EIGHT_YEAR_100K
from scripts.experiments._round import Round

PACK = "configs/workspace_refs/memory_100k_rolling_20261031"
SEED = "data/pit_views_seed_research_10y_fundanns_20261031"
ARGON = "gemini-4-argon"
ASTRA = "gpt-6-astra"
FLASH = "DeepSeek-V4.1-Flash"
COMPACT_TOKEN_THRESHOLD = 221_184

# Point at the pack. Do not restate it, name a post-research reading, add a
# gate, or send the session to a repository document it cannot mount.
DIRECTIVE = (
    "读 `refs/README.md` 与 `refs/memory-framework.md`。"
    "要同时成立的是一种有明确新意的记忆机制，以及可复现、扣费后为正的净收益。"
    "机制、对照、是否使用预训练模型或 GPU，由会话自己决定；参考包里的建议可以不采纳。"
    "针对某一处记忆机制的主张，要用同输入、同标签、只差那一处的对照来支撑；几处同时变的比较只能说明整套方案。"
    "判定依据是宿主回放与现有裁决流程，不要另加统计门槛，也不要自建撮合或成本模型去顶替宿主。"
)

ARMS: dict[str, dict[str, object]] = {
    "memory_100k_10y_argon_local_20261031": {
        "model": ARGON,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "llm_key": "alternate",
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
    "memory_100k_10y_astra_local_20261031": {
        "model": ASTRA,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "llm_key": "alternate",
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=SEED,
    overrides={
        **FUND_TITLE_EIGHT_YEAR_100K,
        # Ten August-July research years; the forward period is the twelve
        # months after them and Held-out ends after it.
        "research_start": "20150801",
        "research_end": "20250731",
        "forward_end": "20260731",
        "heldout_end": "20261031",
        # Ten research years plus the year their start reads, measured back
        # from the research end. The per-domain windows stay unset so each
        # inherits this one.
        "window_months": 132,
        "permitted_boards": ["main", "gem"],
        # This round's defaults. An arm entry replaces any of them at create.
        "gpu_count": 1,
        "batch_validate_max_concurrency": 1,
        "reasoning_effort": "xhigh",
        "max_replay_years": 288,
        "max_null_controls": 36,
        "max_api_spend_usd": 150.0,
        "max_llm_calls": 19_200,
        "max_research_minutes": 7_200,
        "strategy_fit_timeout_seconds": 10_800,
        "strategy_inference_timeout_seconds": 1_080,
        "strategy_cpus": 8,
        "strategy_memory": "64g",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
