# 08:30 字段图

先核对本轮 `data_summary.json`、manifest 和单位表；`macro` 的 `datasets` 必须列出 `cb_daily`、`cb_basic`、`cb_call`，否则本包所有家族「不可测」，不要用其他字段冒充。下表是仓库常见合同，某一折仍可能缺表或为空。

## 读法

```text
cb     = pd.read_parquet(context.asof_dir + "/macro",
           columns=["dataset", "ts_code", "trade_date", "close", "pct_chg", "vol", "amount", "cb_value", "cb_over_rate", "bond_value", "bond_over_rate", "available_at"],
           filters=[("dataset", "==", "cb_daily"), ("trade_date", ">=", start)])
basic  = pd.read_parquet(context.asof_dir + "/macro",
           columns=["dataset", "ts_code", "stk_code", "cb_type", "issue_size", "list_date", "conv_start_date", "conv_end_date", "maturity_date", "first_conv_price", "issue_rating", "call_clause", "reset_clause", "available_at"],
           filters=[("dataset", "==", "cb_basic")])
calls  = pd.read_parquet(context.asof_dir + "/macro",
           columns=["dataset", "ts_code", "call_type", "is_call", "ann_date", "call_date", "call_reg_date", "payment_date", "call_price", "call_vol", "call_amount", "available_at"],
           filters=[("dataset", "==", "cb_call")])
daily  = pd.read_parquet(context.asof_dir + "/daily", columns=[...], filters=[("trade_date", ">=", start)])
index  = pd.read_parquet(context.asof_dir + "/macro", columns=[...], filters=[("dataset", "==", "index_daily")])
universe = pd.read_parquet(context.asof_dir + "/universe")   # 决策日冻结：ts_code、name、list_date、l1_code/l1_name
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；不要把 `asof_dir` 读失败回退到 `snapshot_dir`。`macro` 文件是各数据集列的并集，同名列在不同数据集里单位不同（`amount` 在 `cb_daily` 是万元，在指数表里以单位表为准），先按 `dataset` 过滤再谈单位；每次读都再按 `available_at <= context.inference_at` 过滤一次。

## 转债三表

| 数据 | 本包所需字段与单位 | `available_at` 规则 | 08:30 的后果 |
| --- | --- | --- | --- |
| `macro.cb_daily` | `ts_code`（转债代码，110/111/113/118 .SH，123/127/128 .SZ）、`trade_date`、`close`/`pre_close`/`open`/`high`/`low`（元/100 面值）、`pct_chg`（百分数）、`vol`（手，1 手 = 10 张 = 1,000 元面值）、`amount`（万元）、`cb_value`（转股价值，元/100 面值）、`cb_over_rate`（转股溢价率，百分数）、`bond_value`（纯债价值，元/100 面值）、`bond_over_rate`（纯债溢价率，百分数） | `conservative_date_eod`：`trade_date` 当日 23:59:59 | T-1 行可见、当日行不可见，与 `daily` 同节奏；受 `macro_window_months`（默认随 `window_months`，24 个月）窗口，回放期间新行按日进入 |
| `macro.cb_basic` | `ts_code`、`stk_code`（正股代码，与 `daily.ts_code` 同格式）、`cb_type`（`CB` 可转债 / `EB` 可交换债）、`issue_size`（元，推断）、`list_date`、`conv_start_date`/`conv_end_date`/`maturity_date`（静态条款，上市即知，未来日期 PIT 合法）、`first_conv_price`（元/股，初始转股价）、`coupon_rate`（百分数）、`issue_rating`（发行时评级，静态）、`call_clause`/`reset_clause`/`put_clause`（条款文本） | `conservative_date_eod` 按 `list_date`：上市日 23:59:59 | 上市次日起可见；决策快照里是全生命周期注册表，不受月窗截断。`conv_price`、`remain_size`、`newest_rating`、`delist_date` 是每晚刷新的当前状态，已被快照剔除（读不到，也不得用替代来源）；无 `list_date` 的 24 行没有 `available_at`、不进快照；3 行无 `stk_code` |
| `macro.cb_call` | `ts_code`、`call_type`（`强赎`/`到赎`）、`is_call`（`公告提示强赎`/`公告实施强赎`/`公告不强赎`/`已满足强赎条件`/`公告到期赎回`）、`ann_date`、`call_date`/`call_reg_date`/`payment_date`（公告里的未来日程，PIT 合法）、`call_price`（元/100 面值）、`call_vol`（张）、`call_amount`（万元） | `conservative_date_eod` 按 `ann_date`：公告日 23:59:59 | T-1 晚间公告在 T 08:30 可见，T 09:30 开盘是公告后首个可成交价；同一只券的各状态是不同公告行，不能 `drop_duplicates(ts_code)`；按 `available_at` 受 24 个月窗口（不是注册表） |

`stk_code` 到正股：1,163 只里 1,160 只非空，99% 能在股票列表里找到，其余是无正股或老三板代码，直接丢弃；27 只 `EB` 的 `stk_code` 是被交换的股票而非发行人，剔除。一只正股同时有多只存续转债的情况 2024 年末有 10 例。

## 正股与指数

| 数据 | 本包所需字段 | 08:30 可见边界与单位 |
| --- | --- | --- |
| 合并日线 `daily` | `open/high/low/close`、`vol`（股）、`amount`（元）、`pct_chg`（小数）、`adj_factor`、`turnover_rate`（小数）、`circ_mv`（元）、`pb`、`up_limit`/`down_limit`、`is_suspended` | 当日行 17:30 才可见，08:30 只有 T-1 及更早；`adj_factor` 当日 09:30 盖章，同样只能用 T-1；反推转股价用未复权 `close` |
| `macro.index_daily` | `000300.SH` 的 `pct_chg`（百分数）、`close` | T-1 可见；用于 β |
| `universe` | `ts_code`、`name`（当时名称，含 ST 标记）、`list_date`、`l1_code`/`l1_name` | 决策日冻结；匹配对照的行业从这里取，不回填今天的行业 |

本包不用两融（`margin_detail` 是 T-2）、资金流、筹码或任何 T-2 数据集；唯一的 T-2 情形是转债或正股 T-1 停牌时用最近一行，且要把「最近一行的日期」当作特征的一部分记录。

## 覆盖（本地核对）

- `cb_daily`：2018-01-02 起，2,104 个交易日；季末在市转债 2020Q4 348、2021Q4 391、2022Q4 474、2023Q4 558、2024Q4 508、2025Q4 387、2026Q2 314 只。`cb_value`/`cb_over_rate`/`bond_over_rate` 缺失率 ≤ 0.4%，`amount == 0` 约 0.5%。
- `cb_basic`：1,163 只（`CB` 1,136、`EB` 27）；年末存续可转债 2020–2025 为 334/387/473/552/519/395 只，对应 330/382/464/540/509/389 只不同正股。`conv_start_date` 在上市后约 166 天（中位）。
- `cb_call`：3,071 行、856 只券；2020–2025 年各 232/562/743/290/184/455 行，其中 `公告实施强赎` 72/68/53/41/55/133 行；`已满足强赎条件` 在 2020 与 2024 年为 0 行（供应商缺口，不作信号）；`call_date − ann_date` 中位 24 天。
- 2024-06-28 基准日：519 只发行人，流通市值中位 39 亿元（p10 14、p90 220），68% 日成交额 ≥ 3,000 万元，95% 收盘 ≤ 40 元，科创板 8%，无北交所；转债发行额/正股流通市值中位 19%。2022-06-30：407 只，流通市值中位 58 亿元，86% 日成交额 ≥ 3,000 万元。

## 复权与截面

推断时冻结前复权锚：`anchor = adj_factor(T-1)`，`qfq(t) = raw(t) * adj_factor(t) / anchor`，正股收益用 qfq 收盘；转股价值与转股价用未复权 `close`（分红除权时转股价同步调整，`cb_value` 与未复权价一致）；转债价格不复权，付息日 `pct_chg` 有约 −1% 的跳变。

截面处理顺序见 `families.md`。截面只含当时已上市、未退市、T-1 未停牌、值有限且 T-1 有 `cb_daily` 行的发行人；行业中性与匹配对照只用本次 PIT `universe`。

## 执行

- 最长回看 60 个真实交易观测（`beta_60`、`mean(Amt, 60)`）加 30 日触发计数；先按券与正股取尾窗，再留最新截面；模块级缓存只键控 `context.asof_version`，冷启动必须得到相同订单。
- 卖先于买，按 `context.account.cash` 快照本地递减预算，留 3% 费用缓冲，买入向下取整到 100 股，不把未成交卖出当成已到账。
- 涨跌停：买单成交价 ≥ 当日 `up_limit` 拒单、卖单 ≤ `down_limit` 拒单，当日限价 08:30 不可见，只能用 T-1 代理估计；转债领先的信号最强时正股次日常以涨停开盘，拒单是结果，汇报它。
- 停牌：`is_suspended=True` 的 Bar 一律拒单，且该标志含复牌日；「真停牌」按日线缺行判断。
- 强赎后转债退市，`cb_daily` 不再有行：正股离开股票池，持仓按持有期正常退出，不因转债退市而强制平仓。

## 常见失败

- 用当日 `cb_daily`、当日日线、当日复权，或用 `ann_date`/`call_date` 而非 `available_at` 判断公告可见。
- 把 `cb_daily.pct_chg`/`cb_over_rate` 的百分数与 `daily.pct_chg` 的小数混算；把 `amount` 的万元与 `daily.amount` 的元混算；把 `vol` 的手当张。
- 用 `first_conv_price` 当当前转股价（下修与分红后已变），或试图从别处补 `conv_price`。
- 用 `cb_basic` 判断存续（没有 `delist_date`），或对 `cb_call` 按 `ts_code` 去重。
- 把缺失填成中性 0，使没有转债行的正股进篮子。
- 在 `generate_orders` 里每天全量重读全历史并重算滚动量。
