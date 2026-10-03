# 财务域：本包补充的事实

财务域的读法——怎样确认本臂挂了它、十个数据集与单位、可见时间与周一的例外、Y1 取得到的历史、版本与 `update_flag`、年初至今累计、事后回填的未来字段、读取的内存上限——只写在运行记忆 `memory/curated/research-8y-data-surface/SKILL.md` 的「财务域」一节，先读它；起步包的 `lib/fund.py` 是照它写好的读取函数。本文只补那一节没写、而本包三条线要用的东西。数字都是包作者在本臂的种子上实读的。

## 覆盖与每年的行数

覆盖 = 研究年锚点上，全 A / 中证 1000 成分里看得到上一财年（`end_date` 为前一年 1231）那一行的比例；行数 = 那个研究年回放里新发布的行。

| 数据集 | 各条线用得上的列 | 行数 Y1 → Y8 | 覆盖 2017-06-30 → 2024-06-30（全 A / 中证 1000） |
|---|---|---|---|
| `income_vip` | `total_revenue`、`revenue`、`operate_profit`、`n_income_attr_p`、`basic_eps`、`f_ann_date` | 17,409 → 23,956 | 0.97 / 0.97 → 0.98 / 0.99 |
| `balancesheet_vip` | `total_assets`、`total_hldr_eqy_exc_min_int`、`money_cap` | 16,467 → 23,933 | 0.97 / 0.96 → 1.00 / 1.00 |
| `cashflow_vip` | `n_cashflow_act`、`c_fr_sale_sg` | 17,197 → 23,656 | 0.99 / 0.98 → 0.98 / 0.99 |
| `fina_indicator_vip` | `roe`、`roe_dt`、`q_roe`、`grossprofit_margin`、`debt_to_assets`、`netprofit_yoy`、`tr_yoy`、`q_sales_yoy`、`q_op_qoq`、`eps`、`bps`、`ocfps` | 19,351 → 25,544 | 1.00 / 1.00 → 1.00 / 1.00 |
| `forecast_vip` | `type`、`p_change_min` / `max`、`net_profit_min` / `max`、`last_parent_net`、`first_ann_date`、`summary`、`change_reason` | 9,497 → 7,918 | 0.72 / 0.79 → 0.52 / 0.50 |
| `express_vip` | `revenue`、`n_income`、`diluted_eps`、`diluted_roe`、`bps`、`yoy_net_profit` | 2,312 → 1,549 | 0.49 / 0.59 → 0.22 / 0.24 |
| `dividend` | `div_proc`、`cash_div`、`cash_div_tax`、`stk_div`、`stk_bo_rate`、`stk_co_rate`、`record_date`、`ex_date`、`pay_date`、`imp_ann_date` | 6,681 → 26,462 | 0.79 / 0.83 → 0.99 / 1.00 |
| `fina_audit` | `audit_result`、`audit_fees`、`audit_agency` | 4,718 → 5,868 | 1.00 / 1.00 → 1.00 / 1.00 |
| `fina_mainbz_vip` | `bz_code`、`bz_item`、`bz_sales`、`bz_profit`、`bz_cost`（多为半年与全年） | 139,431 → 164,270 | 0.96 / 0.95 → 0.96 / 0.97 |
| `disclosure_date` | `pre_date`、`actual_date`、`ann_date` | 16,018 → 19,792 | 0.99 / 1.00 → 1.00 / 1.00 |

预告与快报是自愿或按条件披露的：发预告的公司从约七成降到约五成，发快报的从约五成降到两成多，事件集的成分因此逐年在变；数一数每个复核日有几只，比看覆盖更要紧。

## 事件线用得上的口径

- **业绩预告。** `type` 取 预增 / 略增 / 续亏 / 预减 / 首亏 / 扭亏 / 略减 / 续盈 / 不确定 / 其他；同一期可以有多份预告，每份按自己的公告日成一行。`p_change_min` / `max` 与 `(net_profit_min ÷ last_parent_net − 1) × 100` 一致（中位绝对差 0.003），所以对去年同期的惊喜直接读变动幅度即可。
- **分红。** 同一次分配按 `div_proc` 的阶段（预案、股东大会通过、实施……）各成一行；`cash_div` 与 `cash_div_tax` 是元 / 股。
- **主营构成。** 没有自己的公告日，盖的是同一报告期报表最晚一次公告的时间：报表一更正，整期的分部表就推迟到更正日才可见。内容是发布时供应商给的版本，不是当年的历史版本（供应商在 2026-07 整体重述过这张表），所以基于分部的特征带着今天的分部口径。
- **上市前的报告期**（招股书里的数字）按发布时盖章，远晚于报告期；只影响当时还不在 `universe` 里的名字。
- 表里有全部板块，`.BJ` 也在；池的过滤照旧是策略的事。
