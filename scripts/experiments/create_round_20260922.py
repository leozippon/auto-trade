#!/usr/bin/env python
"""The 2026-09-22 round: four learned rankers on the index shape, one panel.

The dataset selection and the prebuilt seed are the 2026-09-19 round's -- the
2026-09-20 selection plus `index_weight` -- imported rather than restated so
the rounds that mount it cannot drift apart by a typo. All four arms need it:
every universe is the decision day's visible CSI 300 section, and the host's
zero-skill panel matches its replacements on constituent membership, so an arm
without the section would be graded against a different floor than it trades.

Unlike the 21x rounds, each arm mounts its OWN reference pack: the four
openings are four model families, not four scores inside one family. What else
differs per arm is the account and whether the arm trains on a card. No arm
carries a tracking mandate this round, so every one of them states the rules'
own drawdowns 0.45 / 0.30 and names no `tracking_error_cap`.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
        "initial_cash": <the account this arm runs>,
        "gpu_count": <only when it differs from the round default 0>,
        "max_drawdown" / "active_max_drawdown":
            stated so `--dry-run` prints the gates this arm is judged by,
    }

`_round.Round.request_params` merges the arm last. Every arm also names the
statistical bars (IR, DSR, positive-year share, full-span validations, forward
confidence, recency, activity, Held-out z) so the create request records a
choice rather than inheriting a hidden pair of packages.

The four openings, and nothing else, in this order so an empty slot takes the
first pending arm. Three arms run CNY 1M on a card; the meta-labeling arm runs
CNY 100k on the container's CPUs, because that is the account where the 5 CNY
commission floor prices its turnover.

Every pack carries its own offline incremental-information gate (`G-INC` in
`families.md`), which must pass BEFORE any full-span batch. It is offline: it
spends no replay-year and counts no trial, which matters because each completed
validation, whatever its span, raises this arm's own deflated-Sharpe bar.

Not in this round: the 1M + tracking-mandate index-enhancement shape. Three
arms and about 90 replay-years (`capital_aware_1m_enhanced_20260921`,
`capital_aware_1m_riskmodel_20260921`, `rmax_overlay_1m_20260921c`) have now
measured that the mandate's beta band admits only near-pure-index books, and
`capital_aware_20260921d/families.md` carries bounded in-mandate tilt and a
risk-model TE budget on its reopen ban. The evidence and the one untested lever
are in `logs/notes/round_20260922/C_index_tilt_dropped.md`; nothing here
reopens it.

Contamination, per arm. Operator-side only.

- Round-level. The decision to spend four arms on learned rankers was taken
  after all sixteen 21b/21c/21d arms closed `no_edge`, and the operator has
  read the shared-forward discards of the five freezes that preceded them.
  Every reading behind the four choices is research-period, except those
  forward closes.
- `gnn_relation_1m_20260922`. Relation message passing has never carried a
  book here, so no forward reading exists for it. Its required control is the
  Alpha158 + LightGBM ranker, whose calibration arm's forward verdict the
  operator has read; the control is reported, never nominated, so the
  contamination reaches the comparison rather than the candidate.
- `xs_transformer_1m_20260922`. Same: cross-name attention is new, the control
  is the same read lineage. Its gating mechanism is the one piece with an
  external prior the pack could not reproduce locally, and `families.md`
  requires gate and attention to be reported apart (P8) for exactly that
  reason.
- `seq_gru_persist_1m_20260922`. Cites `gru_ranker_20260921`, an arm whose
  forward verdict -- discarded, active IR -2.44 against its own panel -- the
  operator has read, so that lineage is contaminated and this arm's forward
  verdict is partly a pipeline calibration. The four changes that make it a
  different arm (universe, residual label, warm-started persistence, the
  incremental gate) are research-period reasoning.
- `lgbm_meta_100k_20260922`. Cites both `github_confirm_20260917`, the only
  graduate, and `alpha158_lgbm_20260921`, whose positive forward point excess
  read -0.30 active against its own panel; the operator has read both, so the
  primary model's lineage is contaminated. The second stage itself is new --
  meta-labeling is recorded as never considered, not as deprioritised -- and
  the arm's whole claim is the increment over `c_primary`, which runs in the
  same batch.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260922.py <port> [--dry-run] [--fill] [experiment_id ...]
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

# The selection and the prebuilt view tree are the benchmark round's, imported
# rather than restated: `index_weight` is what both the in-index universes and
# the panel's membership matching depend on.
from scripts.experiments.create_round_20260919 import (
    EVENTS_DATASETS,
    FUNDAMENTAL_DATASETS,
    MACRO_DATASETS,
    PIT_VIEWS_SEED,
    TEXT_DATASETS,
)

# Create-time graduation bars, named on every arm so the request records a
# choice. These are today's defaults; an arm that wants other bars overrides
# them the same way it overrides drawdowns.
GATES: dict[str, object] = {
    "min_active_ir": 0.75,
    "min_dsr_probability": 0.90,
    "min_positive_year_share": 0.75,
    "min_full_span_validations": 2,
    "forward_confidence": 0.80,
    "recency_months": 6,
    "min_mean_gross": 0.50,
    "min_round_trips_per_month": 1,
    "heldout_tolerance_z": 1.28,
}

ARMS: dict[str, dict[str, object]] = {
    # Relation message passing. fit trains on a card in every replay.
    "gnn_relation_1m_20260922": {
        "workspace_reference": "configs/workspace_refs/gnn_relation_20260922",
        "initial_cash": 1_000_000,
        "gpu_count": 1,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "在沪深 300 成分截面上做关系消息传递的学习型排序器：行业共同归属图 + 滚动收益相关 kNN，"
            "Alpha158 节点特征、基准残差化前向标签、同特征同标签的 LightGBM 对照，"
            "GPU 训练、跨季度重训热启动。整期批次之前必须先过 families.md 的 G-INC 增量信息门。"
        ),
    },
    # Cross-name attention. fit trains on a card in every replay.
    "xs_transformer_1m_20260922": {
        "workspace_reference": "configs/workspace_refs/xs_transformer_20260922",
        "initial_cash": 1_000_000,
        "gpu_count": 1,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "在沪深 300 成分截面上做跨名字注意力的学习型排序器：小时序编码器 + 基准状态门控 + "
            "同一决策日成分之间的一层注意力，Alpha158 特征、基准残差化前向标签、同特征同标签的 "
            "LightGBM 对照，GPU 训练、跨季度重训热启动。门控与注意力必须分开汇报（P8）。"
        ),
    },
    # The sequence family's second attempt, four changes from the arm that died
    # forward. fit trains on a card in every replay.
    "seq_gru_persist_1m_20260922": {
        "workspace_reference": "configs/workspace_refs/seq_gru_persist_20260922",
        "initial_cash": 1_000_000,
        "gpu_count": 1,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "在沪深 300 成分截面上做持久化的序列排序器：14 条价量 / 换手 / 触板 / 两融通道的 60 日窗口"
            "喂三个固定种子的 GRU，标签是基准残差化的 10 日开盘到开盘收益秩，对照是读同一份 window() "
            "张量、同一标签的 LightGBM；权重写进 state_dir，季度重训热启动、每第四次冷启动。"
            "整期批次之前必须先过 families.md 的 G-INC 增量信息门——起步包原样实测 ρ 中位 0.77 已经越过"
            "那一半，所以第一件事是轴 a 的通道消融或轴 c 的标签，不是拿原样起步包去花 12 个回放年。"
        ),
    },
    # The meta stage on the one graduated lineage. CPU only: gpu_count stays at
    # the round default 0.
    "lgbm_meta_100k_20260922": {
        "workspace_reference": "configs/workspace_refs/lgbm_meta_20260922",
        "initial_cash": 100_000,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "给 Alpha158 + LightGBM 主模型接一个元标注第二阶段：主模型在沪深 300 成分上按基准残差化的 "
            "10 日标签训练，训练窗切成 4 折、折间留 10 个交易日隔离带，用离折主分数取每日前 10 % 作为"
            "元阶段的训练集；元分类器只用 11 列状态量，不得加入任何新的收益预测列，默认用它去筛、变体才"
            "用它去配权。10 万元 12 席、每业最多 3 只，按 refs/references/capital-arithmetic.md 的整手与"
            "佣金算术自算价格上限。主模型与元模型的 booster 都写进 state_dir，季度重训用 init_model 续训、"
            "每第四次冷启动。整期批次之前必须先过 families.md 的 G-INC 增量信息门。"
            "第 0 轮先按起步包实测的年换手约 17 倍（双边）重算一遍账户算术表，不要按预期的 6 倍——"
            "提名条件 7 的年换手 ≤ 12 倍是本臂 P3 的真实风险点。"
        ),
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=PIT_VIEWS_SEED,
    overrides={
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
