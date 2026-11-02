#!/usr/bin/env python
"""The 2026-11-02 round: free exploration around the memory framework on a ten-year window at a CNY 200k account, gpt-6-astra on the local console.

Three experiments, identical but for their ids, mount
`memory_200k_rolling_20261102`. The pack states the direction and carries the
owner's memory note; this file does not copy either.
The note is a reference to think with, not a specification to build: the
session may take the idea it finds useful, argue against it, or propose a
strategy and a learning scheme that keeps none of its structure, and it is not
failed for leaving a write or retrieval mechanism unimplemented. What it owes
is scientific novelty and a reproducible, after-cost net profit, read on the
host's replay and verdict. Research runs ten whole August-July years,
2015-08-01..2025-07-31, the twelve forward months after them end 2026-07-31,
and Held-out reaches 2026-10-31, which the replay clips to the release's last
trading day. Decision views reach
`window_months` 132 back from the research end: the ten research years plus
the one year of history in front of their start. Every data domain follows
that one window; none is overridden, and a domain whose release history is
shorter carries what the release has rather than a claimed 132 months of raw
data. The dataset selection, the CSI 1000 benchmark, the main-board-plus-ChiNext
universe and the bars are round 20261031's
(`FUND_TITLE_EIGHT_YEAR_100K`: daily, the ten default fundamentals, the four
macro datasets and the announcement titles); only the account is overridden.

The financial setup differs from round 20261031 only in its
`initial_cash` of 200,000; the research framing above is also new. It gives no holding count and no single-name weight, so
both stay at the console default null and the session decides how many names it
holds, how heavy each is, how often it trades, which features it builds and
what architecture carries them. No earlier candidate is copied and no lineage
is declared. Shared operating memory keeps the existing policy; old
Forward/Held-out readings and verdicts are not added to the new reference pack.
Prior searches observed overlapping history, so another account or shifted
dates do not restore independence.

The shared seed is `data/pit_views_seed_research_10y_fundanns_20261031`.
Creation requires a complete prebuild and a published release covering the
requested dates. The release is read from the seed's `provider.json` at
pre-flight, not selected from the mutable lake.

Each experiment runs gpt-6-astra, the main session the operator starts on this
machine; their ids are `memory_200k_10y_astra_local_20261102` and the same with
`b` and `c` appended. Sub-agents and compaction stay on DeepSeek-V4.1-Flash,
the compaction threshold at 221,184, reasoning at xhigh. `llm_key` alternate
starts every relay role on `LIGHTING_API_KEY_ALT`; the primary key remains the
relay's existing fallback. All three were created on 2026-11-02. Nothing
queues this round: the fill cron lists no round file.

The figures below are this round's create defaults, not a host limit. An
experiment's own entry overrides any of them, including `gpu_count`,
`batch_validate_max_concurrency`, `strategy_cpus` and `strategy_memory`, and
that entry is what the existing console create accepts. This round's default is
one card and one replay at a time, with the same ceilings as rounds 20261028,
20261030 and 20261031: 288 backtest-years, 36 null controls, USD 150, 19,200
model calls, 7,200 minutes, 10,800 s per `fit` and 1,080 s per
`generate_orders`. The strategy container default is 8 CPUs and 64 GiB. GPU
placement uses the existing free-memory threshold and claim rules; this file
names no device.

Create through the existing console round entry, not a separate launcher:

    ~/miniconda3/envs/quant/bin/python scripts/experiments/create_round_20261102.py <port> [--dry-run] [--fill]

`--dry-run` reads the seed contract and, unless a smoke record already covers
this template, replays the contract template from every research-year start
and over the full span. The template reads nothing and places no orders, so
that smoke shows only that the seed's views build for every decision, the
strategy container starts and the contract holds; it exercises no `fit`, no
data read by a strategy and no order or fill. It creates nothing, and it
cannot pass before the seed's prebuild has finished.
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

PACK = "configs/workspace_refs/memory_200k_rolling_20261102"
SEED = "data/pit_views_seed_research_10y_fundanns_20261031"
ASTRA = "gpt-6-astra"
FLASH = "DeepSeek-V4.1-Flash"
COMPACT_TOKEN_THRESHOLD = 221_184

# Point at the pack. Do not restate it, name a post-research reading, add a
# gate, or send the session to a repository document it cannot mount.
DIRECTIVE = (
    "读 `refs/README.md` 与 `refs/memory-framework.md`。"
    "将框架作为思想启发和可质疑的工作假说，而非实现清单；"
    "自由提出策略与学习方案，不要求沿用原文结构或复现具体机制。"
    "目标是有明确新意、可复现且扣费后能够盈利的策略，方法与资源使用由会话自主决定。"
    "数据边界、对照归因与验收要求见参考包；以宿主真实记账验证，不另加门槛。"
)


def _arm() -> dict[str, object]:
    return {
        "model": ASTRA,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "llm_key": "alternate",
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    }


ARMS: dict[str, dict[str, object]] = {
    "memory_200k_10y_astra_local_20261102": _arm(),
    "memory_200k_10y_astra_local_20261102b": _arm(),
    "memory_200k_10y_astra_local_20261102c": _arm(),
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=SEED,
    overrides={
        **FUND_TITLE_EIGHT_YEAR_100K,
        "initial_cash": 200_000,
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
        # This round's defaults. An experiment entry may override them at create.
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
