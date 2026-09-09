#!/usr/bin/env python
"""The 2026-09-10 round, as it now stands: three arms, two still running.

`factor_cs_20260910`, `explore_platform_strategies_20260910` and
`corner_cases_20260910` were retired and archived and are gone from this file;
their ids stay in `_round.RETIRED_IDS` so no later round can reuse one.
`explore_github_strategies_20260910` and `ml_ranker_20260910` are still
running, `open_mechanism_20260910` has finished and was discarded at Held-out,
so this file remains the definition the three were created from.

The round decides nothing beyond the shared launcher: it takes the common
account, gates, schedule and budgets and the console creation defaults as they
are, names no PIT view seed of its own, and differs arm by arm only in
direction, reference pack and GPU request. `ml_ranker` is the one arm with
`gpu_count=1`; the request travels with the experiment, reaching the Agent's
session sandbox and the strategy container of every formal replay, so `fit()`
may really train on the device, and because the card is shared and the same
strategy must replay unchanged without one, its directive demands a device
probe and a working CPU path.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \
      scripts/experiments/create_round_20260910.py <port> [--dry-run] [experiment_id ...]
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

from scripts.experiments._round import PARENT_CONTROL_LINE, ROBUSTNESS_LINE, Round

# From experiments/open_mechanism_20260903/hitl/params.json
# ("fold_exploration_directive"), copied here because that experiment was
# archived out of the repository before this round started; only the drawdown
# clause is updated, since the cap is now enforced at graduation.
OPEN_MECHANISM_PRIOR_DIRECTIVE = (
    "跳开因子库、横截面打分、线性排序和常规技术指标堆叠。本实验无参考仓库。只根据当前 PIT 可见结构自行提出"
    "一种不同机制，写成最小可执行策略，并用完整 Validation 证伪。事件状态、微观结构、制度约束、行为路径、"
    "非对称执行只是类型，不是指定答案。本轮以机制新颖与可证伪为目标，不要求稳定或正收益；证伪后如实结束该方向。"
    "仍须遵守 PIT、禁止硬编码股票或日期、真实回测 ABI 与诚实失败。本指令不放宽提交合同、毕业裁决的回撤上限或 finish_fold。"
)

ARMS: dict[str, dict[str, object]] = {
    "explore_github_strategies_20260910": {
        "workspace_reference": "configs/workspace_refs/github_20260912",
        "fold_exploration_directive": "\n".join(
            [
                "方向：在本项目 ABI 内重写 Qlib 风格的 Alpha158 特征 + LightGBM 截面排序器作为有据可查的强基线，"
                "再在它之上做 refs 登记的创新家族（标签口径、特征中性化、跨重训日集成、WorldQuant-101 与国泰君安-191 算子族），"
                "一次只验证一个改动；是重写，不是移植代码。",
                "先读 refs/README.md、exploration-plan.md、alpha158.md 与 families.md；源仓库的数据层、训练器、Recorder 与回测器"
                "一律不搬，撮合、T+1、费用与涨跌停属于环境，策略只发订单意图。不要把 Qlib 文档里的收益表当成预期，"
                "只用本项目的 Validation 复现其行为。",
                "拟合全部放在 fit(context) 内，标签只用推断时已实现的开盘到开盘收益并留禁运，超参网格预先写死并在 fit 内选点；"
                "股票池未经任何筛选，可交易性、停牌与涨跌停由策略自理。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
                "禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足 available_at <= 推断时点。"
                "可执行指纹必须不同于父策略。",
            ]
        ),
    },
    "ml_ranker_20260910": {
        # The only arm with a GPU. The request reaches the Agent's session
        # sandbox and the strategy container (fit worker included) of every
        # formal replay, so fit() may train on the device; the card is shared
        # with concurrent replays and the strategy must replay unchanged in an
        # experiment without one, so the directive demands a device probe and
        # a working CPU path.
        "gpu_count": 1,
        "workspace_reference": "configs/workspace_refs/ml_ranker_20260912",
        "fold_exploration_directive": "\n".join(
            [
                "方向：用机器学习与深度学习做截面排序器：先在 fit(context) 里训出 LightGBM 基线，再比较 MLP 与序列模型"
                "（GRU 或小型 Transformer，读 20–60 日 K 线与特征序列），每一步只改一个可分离的组件（模型族、标签口径、"
                "中性化、集成、换手控制），正式产物写在 output/ 包内。",
                "先读 refs/README.md、exploration-plan.md、models.md 与 protocol.md。本臂挂了一块 GPU：开发沙箱用它做"
                "离线筛查与超参预选，正式回放的策略容器（含 fit worker）同样能看到它，fit 可以真的在 GPU 上训练；"
                "但这块卡与并发回放共享，同一份策略还要能在没有 GPU 的实验里原样回放，因此代码必须探测设备并保留"
                "可用的 CPU 路径，fit 的墙钟先在沙箱实测、落在 fit 预算内，模型状态只能以 NumPy 数组或 booster 文件"
                "写入 context.state_dir。",
                "反过拟合是硬纪律：滚动重训、带禁运的清洗时间序列验证、预先写死的小网格并在 fit 内选点、按候选数读去膨胀 Sharpe；"
                "标签只用推断时已实现的开盘到开盘收益。持有期与篮子按 refs 的手数与成本纪律。",
                "股票池不做任何 ST、板块、次新、市值或价格筛选；可交易性、停牌与涨跌停由策略自己处理并说明理由。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
                "禁止克隆父策略、禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足"
                " available_at <= 推断时点。可执行指纹必须不同于父策略。",
            ]
        ),
    },
    "open_mechanism_20260910": {
        # No workspace_reference and no inherit_from: the arm starts from the
        # empty template with nothing mounted but the operating memory.
        "fold_exploration_directive": "\n".join(
            [
                OPEN_MECHANISM_PRIOR_DIRECTIVE,
                "本轮环境提供 output/ 包结构、可选的 fit(context) 与并排跑完整 Validation 的 batch_validate，"
                "可按需使用；它们只是手段，不改变本方向对机制新颖与可证伪的要求。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
            ]
        ),
    },
}

ROUND = Round(arms=ARMS)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
