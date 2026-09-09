# 08:30 字段图

先核对本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json`；`events` 的 `datasets` 必须同时列出 `report_rc`、`margin_detail`、`block_trade`、`cyq_perf`、`stk_holdernumber`、`stk_holdertrade`、`top10_floatholders`、`share_float_complete`，任何一张缺失时对应的特征族整块记 `NaN` 并在结果笔记里声明，**不得用其他字段冒充**；缺三张以上时整臂判「不可测」，用 `finish_fold(outcome="no_edge")` 如实弃权。`report_rc`、`top10_floatholders` 默认不加载，靠创建实验时的 `events_datasets` 显式选入；其余六张默认加载。

## 读法

```text
daily    = pd.read_parquet(context.asof_dir + "/daily",
                           columns=["ts_code","trade_date","open","high","low","close","pre_close",
                                    "vol","amount","pct_chg","adj_factor","turnover_rate","circ_mv",
                                    "pe_ttm","pb","up_limit","down_limit","is_suspended"],
                           filters=[("trade_date", ">=", start)])
ev       = pd.read_parquet(context.asof_dir + "/events",
                           columns=["dataset","available_at","ts_code", ...该族自己的列...],
                           filters=[("dataset", "=", "margin_detail")])
ev       = ev[pd.to_datetime(ev["available_at"]) <= pd.Timestamp(context.inference_at)]
universe = pd.read_parquet(context.asof_dir + "/universe")   # ts_code、name、list_date、l1_code/l1_name
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名，pandas 自己拼接分片），`snapshot_dir` 下是平铺文件；`context.asof_dir + "/events.parquet"` 不存在，读失败**不得**回退 `snapshot_dir`（那会把停在决策时刻的冻结快照当成滚动视图，是 PIT 违规而不是恢复）。

事件域是多个数据集的列并集：同名列在不同数据集里含义不同（`name`、`amount`、`close`、`ann_date` 都是），**必须先按 `dataset` 过滤再谈字段与单位**，并把 `dataset` 列进 `columns` 以便审计。同一个 `ts_code` 在不同数据集下的行不能用同一个日期索引串起来：`report_rc` 只有 `report_date`，`margin_detail`/`block_trade`/`cyq_perf`/`top_list` 只有 `trade_date`，`stk_holdernumber`/`stk_holdertrade`/`top10_floatholders`/`share_float_complete` 只有 `ann_date`（后者还有 `float_date`）。

## 逐表可见边界（本地用 `contracts.event_dataset_visible_cutoff` 实测，2025-03-05 周三 08:30 / 2025-03-03 周一 08:30）

| 数据集 | 行级 `available_at` 规则 | 放行节点 | 周三 08:30 的可见截止 | 周一 08:30 的可见截止 | 本包用的最新行 |
|---|---|---|---|---|---|
| `cyq_perf` | `official_19_from:trade_date`（19:00） | `cn_evening_full`（23:35，交易日） | 2025-03-04 23:35 | 2025-02-28 23:35 | **T-1**（周一是上周五） |
| `block_trade` | `official_21_from:trade_date`（21:00） | `cn_evening_full` | 2025-03-04 23:35 | 2025-02-28 23:35 | **T-1** |
| `top_list` / `top_inst` | `official_20_from:trade_date`（20:00） | `cn_evening_full` | 2025-03-04 23:35 | 2025-02-28 23:35 | **T-1**（本包只作诊断） |
| `margin_detail` | `official_next_day_09_from:trade_date`（**次日** 09:00） | `cn_preopen_margin_backfill_0905` + `_retry_0915` | 2025-03-04 09:15 | 2025-03-02 09:15 | **T-2** |
| `report_rc`（数值切片） | `source:create_time` 或 `conservative_from:report_date`（22:00） | `cn_nightly_text_full`（23:15，自然日） | 2025-03-04 23:15 | 2025-03-02 23:15 | **T-1** |
| `stk_holdertrade` | `official_19_from:ann_date`（19:00） | `cn_nightly_disclosure_full`（23:05，自然日） | 2025-03-04 23:05 | 2025-03-02 23:05 | **T-1** |
| `stk_holdernumber` | `conservative_date_eod`（23:59:59） | `cn_nightly_disclosure_full`（23:05） | 2025-03-04 23:05 | 2025-03-02 23:05 | **T-2**（23:59:59 > 23:05） |
| `top10_floatholders` | `conservative_date_eod`（23:59:59） | `cn_nightly_disclosure_full`（23:05） | 2025-03-04 23:05 | 2025-03-02 23:05 | **T-2** |
| `share_float_complete` | `conservative_date_eod`（23:59:59） | `cn_evening_full`（23:35） | 2025-03-04 23:35 | 2025-02-28 23:35 | 公告 **T-2**；`float_date` 是未来日，前瞻合法 |

两条容易漏掉的后果：**同一个「日终盖章」在不同节点下给出不同的滞后**——`stk_holdertrade` 的 19:00 赶得上当晚 23:05 的披露节点，`stk_holdernumber` 的 23:59:59 赶不上，两张表相差整整一天，尽管数据文档把它们写在同一行。**披露节点走自然日、晚间节点走交易日**——周一 08:30 时披露类表的截止是周日 23:05（周末公告已可见），而 `cyq_perf`/`block_trade` 的截止是上周五 23:35。判可见永远只看 `available_at`，不要从日期列推。

## 逐表字段与单位

| 数据 | 本包所需字段 | 单位与注意 |
|---|---|---|
| 合并日线 `daily` | `open/high/low/close/pre_close`、`vol`、`amount`、`pct_chg`、`adj_factor`、`turnover_rate`、`circ_mv`、`pe_ttm`、`pb`、`up_limit`/`down_limit`、`is_suspended` | 快照已归一：价元/股、`vol` 股、`amount` 与 `circ_mv` 元、`pct_chg`/`turnover_rate` 小数。当日行 17:30 才可见，08:30 只有 T-1；`adj_factor` 当日 09:30 盖章 |
| `universe` | `ts_code`、`name`（含 ST 标记）、`list_date`、`l1_code`/`l1_name`、`market` | 决策日冻结；行业、ST、上市年龄只从这里取，不从 `events` 的 `name` 列取，也不回填今天的状态 |
| `events.report_rc` | `ts_code`、`report_date`、`create_time`、`org_name`、`author_name`、`quarter`、`eps`、`np`、`rating`、`report_type`、`classify` | `eps` 元/股；`np`/`op_rt`/`op_pr`/`tp` 万元（`tp` 是**利润总额**不是目标价）；`pe`/`ev_ebitda` 是券商**前瞻**倍数；`rd` 是预测**股息率**（百分数）不是研发费用；`quarter` 是**财年**标签（98.8% 以 `Q4` 结尾，`2024Q4` = FY2024）；切片里没有 `report_title`/`name`/`imp_dg` |
| `events.margin_detail` | `ts_code`、`trade_date`、`rzye`、`rqye`、`rzmre`、`rzche`、`rqyl` | `rzye/rqye/rzmre/rzche/rzrqye` 元；`rqyl/rqmcl/rqchl` 股。两类不得相加（`rzrqye` 已是和）。`rzye/(circ_mv)` 中位 3.85%，把 `circ_mv` 误当万元读会得到 385 这种不可能的比值 |
| `events.block_trade` | `ts_code`、`trade_date`、`price`、`vol`、`amount`、`buyer`、`seller` | `price` 元/股、`vol` **万股**、`amount` **万元**；`close`/`pre_close` 对大宗行恒为空，折溢价分母只能从 `daily` 按同一 `(ts_code, trade_date)` 关联并用未复权价 |
| `events.cyq_perf` | `ts_code`、`trade_date`、`winner_rate`、`cost_15pct`、`cost_50pct`、`cost_85pct`、`weight_avg` | `winner_rate` 百分数 0–100（实测最大 100.55，用前截断）；`cost_*`/`weight_avg`/`his_low`/`his_high` 是元/股的**成本价位**，不是比例 |
| `events.stk_holdernumber` | `ts_code`、`ann_date`、`end_date`、`holder_num` | `holder_num` 是计数。只有 50.3% 的行落在报告期端点 |
| `events.stk_holdertrade` | `ts_code`、`ann_date`、`holder_name`、`holder_type`、`in_de`、`change_vol`、`change_ratio`、`after_ratio` | `change_vol`/`after_share`/`total_share` 股（`total_share` 是**该股东**交易后持仓，不是公司股本）；`change_ratio`/`after_ratio` 百分数；`avg_price` 元/股且 34.4% 缺失；`in_de` 取 `IN`/`DE`；`holder_type` 取 `C`（公司）/`G`（高管）/`P`（个人） |
| `events.top10_floatholders` | `ts_code`、`ann_date`、`end_date`、`hold_amount`、`hold_float_ratio`、`hold_change` | `hold_amount`/`hold_change` 股；`hold_ratio`/`hold_float_ratio` 百分数 |
| `events.share_float_complete` | `ts_code`、`ann_date`、`float_date`、`float_share`、`float_ratio`、`holder_name`、`share_type` | `float_share` **股**（不是万股：存在 386 股的解禁行）；`float_ratio` 是该笔占**总股本**的**百分数**，99.3% 的行小于 1 只是因为多数批次很小——按小数比例读会错 100 倍 |

## 三条本包特有的去重/聚合规则

这三条是本包最容易写错的地方，实现与本地核对面板必须共用同一个函数。

1. **`report_rc`**：`available_at <= inference_at` → 对业务键 `(ts_code, report_date, org_name, author_name, quarter)` 取 `create_time` 最大的一条（该键**不唯一**，实测重复率 0.919%，其中 1,812 组的 `eps` 真的变了）→ 保守盖章行的时刻完全相同（22:00:00），并列再按 `(report_date, org_name, author_name)` 打破 → 按 `quarter` 切财年 → 窗口内每券商只取最新一条。
2. **`stk_holdernumber` / `top10_floatholders`**：先 `drop_duplicates()`（`stk_holdernumber` 的 18,864 个重复键组里 `holder_num` 全部一致，是逐字节重复）→ 对每个 `(ts_code, end_date)` 取**可见的最大 `ann_date`**（14,088 组有两个版本、429 组三个）→ 只保留 `end_date` 的 MMDD ∈ {0331, 0630, 0930, 1231} 的行再做相邻期差分（`stk_holdernumber` 只有 50.3% 的行满足，`top10_floatholders` 有 5.2% 落在 349 个非报告期 MMDD 上）。
3. **`share_float_complete`**：`drop_duplicates()` → 对每个 `(ts_code, float_date)` 只保留**可见的最大 `ann_date`** 那一版 → 在该版内对逐持有人行的 `float_ratio` 求和。**跳过第二步会得到 360% 这种不可能的解禁比例**：这张表是多次公告的并集，同一次解禁会被 2–5 个 `ann_date` 反复重述（每次按送转调整过股数），混版求和就是重复计数。正确规则下逐事件解禁比例中位 7.84%、p75 24.6%、p95 66.6%，残余尾部仍有少数 > 100%（不同 `ann_date` 的持有人清单只部分重叠），一律截到 100 并汇报被截比例。

`block_trade` 的两级聚合（笔 → `(ts_code, trade_date)` 事件 → 窗口）与 `margin_detail` 的「只有有余额的名字才出现」两条见 `families.md`；三张表的原始事实与更多分位数在 `configs/workspace_refs/analyst_revision_20260916/`、`margin_flow_20260916/`、`block_trade_20260916/` 的同名文件里，本包只按指针引用，不重述。

## 复权与截面

推断时冻结前复权锚：`anchor = adj_factor(T-1)`，`qfq(t) = raw(t) * adj_factor(t) / anchor`；标签与所有价格类特征在同一个锚下算，跨除权日直接用原始收盘会造出假收益。折溢价（`block_trade`）与成本分布（`cyq_perf`）用同日**未复权**价，因为分子分母同属一天。成交额不复权。`920627.BJ` 在 2022 年的 `adj_factor` 逐日反复跳变（数据文档 §4），本包整体剔除 `.BJ`，但仍不得假设 `adj_factor` 逐日非降。

截面只含当时已上市、未退市、T-1 未停牌、值有限的股票；`is_suspended=True` 的 Bar 一律拒单，且该标志含复牌日，「真停牌」按日线缺行判断。

## 执行

- 最长回看：日线 60 个交易日 + 预热，事件窗最长 180 个自然日（`insider_net_180`），解禁前瞻 90 个自然日。先按股票取尾窗，再留最新截面；模块级缓存只键控 `context.asof_version`，冷启动必须得到相同订单；`fit` 会在不重启 worker 的情况下替换 `state_dir` 的文件，所以模型要在 `generate_orders` 内部加载，不能加载进跨重训存活的缓存。
- 卖先于买：同一调仓日把卖单挂 `09:30`、买单挂 `15:00`，Broker 按时间戳顺序处理；`context.account` 是调用入口的快照，不会在 `generate_orders` 运行中变化，买入预算按本地递减的余额算并留 3% 费用缓冲，向下取整到 100 股。非再平衡日返回 `[]`。
- 涨跌停：买单成交价 ≥ 当日 `up_limit` 拒单、卖单 ≤ `down_limit` 拒单；当日限价 08:30 不可见，只能用 T-1 代理估计。拒单是结果，逐折汇报拒单率。

## 常见失败

- 用 `trade_date`/`ann_date`/`report_date` 判可见，或假定 `margin_detail` 有 T-1 行。
- 把 `stk_holdernumber`/`top10_floatholders` 当成 T-1：它们的 23:59:59 盖章赶不上 23:05 的披露节点。
- 不去重就对 `report_rc` 的行取均值，把同一份预测的新旧两版一起算进一致预期；或不按财年切片，把 FY0/FY1/FY2 的 `eps` 混在一个窗口均值里。
- 对 `share_float_complete` 直接按 `(ts_code, float_date)` 求和而不先取最新公告版本。
- 跳过 `(ts_code, trade_date)` 这一级直接对大宗笔加权，让一个拆成 340 笔的名称-日主导排序。
- 混单位：`block_trade.amount`（万元）直接与 `daily.amount`（元）相比、`rqyl`（股）当金额、`float_ratio` 当小数比例、`winner_rate` 当小数、`change_ratio` 当小数。
- 用事件域的 `name` 判 ST，或用今天的 `stk_basic` 状态回填上市板块与行业。
- 把「没有事件行」填成 0 与「事件值恰为 0」混为一谈（大宗与解禁两族尤其致命）。
- 在 `generate_orders` 里每天全量重读事件域并重算全历史滚动量。
- 原始湖与快照的单位不同：`data/raw/daily` 的 `amount` 是千元，快照的 `daily.amount` 是元。会话里做离线普查时用错会把整个可选池筛成空集。
