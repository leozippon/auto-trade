#!/usr/bin/env python
"""The 2026-10-19 second round: the two proposals of round 20261019 on the A100 instance, models swapped.

The same two packs, directives, geometry, boards and seed as round 20261019
(`create_round_20261019.py`), on the ZGCA development instance with two
A100s (auto-trade-dev-a100x2), with the main-session models exchanged so
that each proposal is researched by both models across the two rounds.
Sub-agents and compaction run on DeepSeek-V4.1-Flash in both, since the
instance has no local model.

| Arm | Pack | Main session | Sub-agents / compaction |
| --- | --- | --- | --- |
| `hcm_lm_100k_8y_sol_zgca_20261019b` | `hcm_lm_100k_8y_20261019` | gpt-6.1-sol | DeepSeek-V4.1-Flash |
| `egcu_lm_100k_8y_opus_zgca_20261019b` | `egcu_lm_100k_8y_20261019` | claude-opus-5-5 | DeepSeek-V4.1-Flash |

The host must name its card model (`AUTOTRADE_GPU_NAME=A100`) in the
environment of both the console and this launcher, which `ops/zgca/zgca.py
run` does. Each arm claims one of the two cards, so the two run together and
one launcher run naming both ids creates them:

    python ops/zgca/zgca.py run scripts/experiments/create_round_20261019b.py hcm_lm_100k_8y_sol_zgca_20261019b egcu_lm_100k_8y_opus_zgca_20261019b

(`--dry-run` before it reads the seed contract and smokes the starter.)

Usage: create_round_20261019b.py <port> [--dry-run] [--fill] [experiment_id ...]
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

from scripts.experiments._profiles import FUND_TITLE_EIGHT_YEAR_100K, HCM_LINEAGE
from scripts.experiments._round import Round

PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_fundanns_20261022"
OPUS = "claude-opus-5-5"
SOL = "gpt-6.1-sol"
FLASH = "DeepSeek-V4.1-Flash"
COMPACT_TOKEN_THRESHOLD = 221_184

HCM_DIRECTIVE = (
    "验证「分层持续记忆」在语言模型载体上的主张——一个读匿名化公告标题的预训练语言模型加日线特征的选股模型，"
    "从回放第一天起不再重新初始化、在多个时间尺度上更新状态与参数（可带外部记忆），"
    "能否比同架构的季度冷重训与冻结模型更能跨市况持续累积选股能力；"
    "必做的配对比较：候选对冷重训与冻结各两个种子基，以及候选与冷重训各自的 text-off 孪生；机制怎样实现由会话设计。"
)
EGCU_DIRECTIVE = (
    "验证「证据引导的持续更新」——同一个读匿名化公告标题的预训练语言模型加日线特征的选股模型，"
    "由一条统一的更新规则按新证据的可信度、现有知识的可靠度与新旧模式的冲突共同决定是否更新、更新多少与何时回滚，"
    "能否胜过天天无条件更新、季度冷重训，以及提交时刻随机但频率相同的安慰剂；规则怎样实现由会话设计。"
)

ARMS: dict[str, dict[str, object]] = {
    "hcm_lm_100k_8y_sol_zgca_20261019b": {
        "model": SOL,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "workspace_reference": "configs/workspace_refs/hcm_lm_100k_8y_20261019",
        "research_directive": HCM_DIRECTIVE,
        "lineage_arms": HCM_LINEAGE,
    },
    "egcu_lm_100k_8y_opus_zgca_20261019b": {
        "model": OPUS,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        "workspace_reference": "configs/workspace_refs/egcu_lm_100k_8y_20261019",
        "research_directive": EGCU_DIRECTIVE,
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_EIGHT_YEAR_100K,
        "permitted_boards": ["main", "gem"],
        # The one card, claimed by the console when it starts the worker; up
        # to three candidates of a batch share it.
        "gpu_count": 1,
        "batch_validate_max_concurrency": 3,
        "max_api_spend_usd": 50,
        "reasoning_effort": "xhigh",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
