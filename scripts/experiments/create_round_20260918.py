#!/usr/bin/env python
"""The 2026-09-18 round: two arms on the 2026-09-17 round's shared seed.

`earnings_surprise` is the one direction the DIR3 study
(logs/notes/review_20260912/DIR3_new_arms.md) found to survive every control
the earlier arms exhausted: the gap between a firm's reported earnings and the
sell-side consensus formed before the announcement, plus the quarterly
seasonal-random-walk earnings change, carried for a few weeks after the
announcement. Its hard gate is an attribution control against the same
report's yoy growth -- the growth axis `explore_github` already carries -- on
the same skeleton: losing to it closes the mechanism.

`defensive_quality` is the one direction the DIR4 study
(logs/notes/review_20260912/DIR4_more_arms.md) found after that: two
independent "boring is under-priced" legs -- cash-flow earnings quality (low
accruals, high operating cash flow over assets, from the first announced
statement) and low 60-day residual volatility -- rank-averaged into one
estimator-free monthly book. Its low-vol leg consciously reopens the
`factor_cs` `f2_ivol60` family, which survived its two folds and was never
falsified, so the arm carries two independent attribution gates: the closed
20-day low-vol face closes the low-vol leg, and the growth face or the running
earnings-surprise composite closes the quality leg.

Both arms read only datasets the 2026-09-17 selection already carries
(`report_rc` in events, the fundamental datasets, `index_daily` in macro), so
they hard-link the same prebuilt view tree and cold-build nothing; neither
inherits an artifact or a memory. The calendar, the dataset selection and the
deployment-adjustment slot are imported from the 2026-09-17 round rather than
retyped: a seed's identity is the whole snapshot configuration, and the point
of sharing one is that the round files cannot drift apart.

The arms were not created together: `earnings_surprise_20260918` was created
first and `defensive_quality_20260918` fills the slot the retired
`order_flow_ranker_20260917` leaves. Name the arm to create on the command
line -- the launcher creates only the ids it is given -- so a re-run never
tries to recreate the arm that already exists.

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

_DEFENSIVE_DIRECTIVE = (
    "方向：把「无聊的股票」做成 08:30 决策的只做多月度截面信号——现金流盈利质量（首版报表的低应计与"
    "高经营现金流/总资产）与低 60 日残差波动（个股日收益对沪深 300 回归的残差标准差）各自截面秩等权合成，"
    "只测 refs 登记的两腿合成、两条可提名的消融腿及其变体，正式产物写在 output/ 包内。"
    "低波腿有意重开 factor_cs 登记过、从未证伪的 ivol_60 家族，因此硬门是两条互相独立的归因对照："
    "s1 与 s_lowvol 必须在中性化超额与空对照分位两项上都严格高于只用 20 日波动的 c_vol20，输了即低波腿关闭；"
    "s1 与 s_cfq 必须同样严格高于只用同比增速的 c_growth 和按盈利意外臂规则重建的 c_es，输了即质量腿关闭；"
    "两腿都关即本臂关闭。s1 只有同时高于两条消融腿才可提名，否则提名更好的那条腿。"
    "先读 README 与 families.md，按 pit-field-map.md 核对：判可见只看 available_at；"
    "三张报表只取每个报告期最早的一版且 report_type 为 1；年初至今流量按季度序号年化；"
    "index_daily 的 pct_chg 是百分数，缺沪深 300 时用总波动并写进买单元数据；对照不得对自己的面中性化。"
    "任何回测前先做不占预算的普查：三张报表与沪深 300 在快照视图与验证视图各自的行数与覆盖必须分别读，"
    "复核日的可选池、两腿非空率与波动窗口日数，单位量级核对。"
    "主线不训练：没有 fit，每次决策是有界读取、一次 60 日滚动回归加一次最小二乘；登记学习器变体时 s1 是它的对照。"
    "防御型篮子在宽幅高波动上涨的子窗里原始超额接近零，按中性化超额与规模倾斜一起读；"
    "成本门写死在 families.md：成本后不为正就如实判不可交易并弃权，不得靠放宽门槛或调滑点假设救活。"
    "通用价量打分不得作候选，只能以中性化列或 c_vol20 对照的身份出现。"
)

ARMS: dict[str, dict[str, object]] = {
    "earnings_surprise_20260918": {
        "workspace_reference": "configs/workspace_refs/earnings_surprise_20260918",
        "fold_exploration_directive": f"{_DIRECTIVE}\n{PARENT_CONTROL_LINE}\n{ROBUSTNESS_LINE}",
    },
    "defensive_quality_20260918": {
        "workspace_reference": "configs/workspace_refs/defensive_quality_20260918",
        "fold_exploration_directive": f"{_DEFENSIVE_DIRECTIVE}\n{PARENT_CONTROL_LINE}\n{ROBUSTNESS_LINE}",
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
