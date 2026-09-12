#!/usr/bin/env python
"""The 2026-09-18 round: one arm on the 2026-09-17 round's shared seed.

`earnings_surprise` is the one direction the DIR3 study
(logs/notes/review_20260912/DIR3_new_arms.md) found to survive every control
the earlier arms exhausted: the gap between a firm's reported earnings and the
sell-side consensus formed before the announcement, plus the quarterly
seasonal-random-walk earnings change, carried for a few weeks after the
announcement. It reads only datasets the 2026-09-17 selection already carries
(`report_rc` in events, the ten fundamental datasets), so it hard-links the
same prebuilt view tree and cold-builds nothing; it inherits no artifact and
no memory. Its hard gate is an attribution control against the same report's
yoy growth -- the growth axis `explore_github` already carries -- on the same
skeleton: losing to it closes the mechanism.

The calendar, the dataset selection and the deployment-adjustment slot are
imported from the 2026-09-17 round rather than retyped: a seed's identity is
the whole snapshot configuration, and the point of sharing one is that the two
round files cannot drift apart.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260918.py <port> [--dry-run] [experiment_id ...]
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
from scripts.experiments.create_round_20260917 import ROUND as SHARED_ROUND

_DIRECTIVE = (
    "方向：把公司实际业绩相对事前预期的缺口做成 08:30 决策的只做多截面信号——年度归母净利润相对"
    "卖方一致预期（income_vip/express_vip 对 report_rc.np）的缺口，加上单季归母净利润相对去年同季的变化，"
    "按公告后的新鲜度加权，只测 refs 登记的意外面及其变体，正式产物写在 output/ 包内。"
    "硬门只有一条：每个候选必须同批带同一骨架只用同一份报告同比增速的归因对照（c_growth），"
    "在中性化超额与空对照分位两项上都严格高于它；比不过就立即关闭本机制，那说明一致预期缺口只是成长的另一种写法。"
    "同批还必须带只按公告新鲜度排序的安慰剂（c_timing），输给它的候选不可提名。"
    "先读 README 与 families.md，按 pit-field-map.md 核对：判可见只看 available_at；只取每个报告期最早的一版；"
    "一致预期窗口严格结束于公告日之前；quarter 是财年标签；np 是万元而 n_income_attr_p 是元。"
    "任何回测前先做不占预算的普查：三张表在快照视图与验证视图各自的行数与覆盖必须分别读，"
    "逐月事件数与本折四个季度的旺季/淡季标记，单位量级核对，复核日的可选池与新鲜名字数。"
    "信息流是季节性的：淡季按 families.md 的政策持有上一季尾巴、不清仓，淡季子窗按门 4 的季节读法只汇报不计分。"
    "主线不训练：没有 fit，每次决策是有界读取加一次最小二乘；登记学习器变体时等权合成是它的对照。"
    "只做多的那条腿比空头腿薄，成本门写死在 families.md：成本后不为正就如实判不可交易并弃权，"
    "不得靠放宽门槛或调滑点假设救活。通用价量打分不得作候选，只能以中性化列的身份出现。"
)

ARMS: dict[str, dict[str, object]] = {
    "earnings_surprise_20260918": {
        "workspace_reference": "configs/workspace_refs/earnings_surprise_20260918",
        "fold_exploration_directive": f"{_DIRECTIVE}\n{PARENT_CONTROL_LINE}\n{ROBUSTNESS_LINE}",
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=SHARED_ROUND.pit_views_seed,
    overrides=dict(SHARED_ROUND.overrides),
    expected_defaults=dict(SHARED_ROUND.expected_defaults),
    report_keys=SHARED_ROUND.report_keys,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
