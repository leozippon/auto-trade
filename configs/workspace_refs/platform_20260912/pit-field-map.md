# 08:30 字段图

先核对本轮 `data_summary.json`、manifest 和单位表；下表是仓库常见合同，某一折仍可能缺表或为空。`data_summary.json` 里 `events` 的 `datasets` 列出本轮实际加载的数据集，只有列在其中的才可读。

## 读法

```text
daily    = pd.read_parquet(context.asof_dir + "/daily", columns=[...], filters=[("trade_date", ">=", start)])
events   = pd.read_parquet(context.asof_dir + "/events", columns=["dataset", "available_at", "trade_date", "ts_code", ...])
events   = events[(events["dataset"] == "limit_list_d") & (pd.to_datetime(events["available_at"]) <= pd.Timestamp(context.inference_at))]
fund     = pd.read_parquet(context.asof_dir + "/fundamentals", columns=[...])   # dataset == "forecast_vip"
macro    = pd.read_parquet(context.asof_dir + "/macro", columns=[...])          # dataset in {"index_daily", "sw_daily"}
universe = pd.read_parquet(context.asof_dir + "/universe")                      # ts_code、name、list_date、l1_code/l1_name
```

`asof_dir` 下每个域是 parquet parts 目录，`snapshot_dir` 下是平铺文件；不要在 `asof_dir` 读失败时回退到 `snapshot_dir`。

## 本包用到的数据集

| 数据 | 有用字段 | 08:30 可见边界与单位 |
| --- | --- | --- |
| 合并日线 `daily` | `open/high/low/close`、`vol`（股）、`amount`（元）、`pct_chg`（小数）、`adj_factor`、`turnover_rate`（小数）、`circ_mv`（元）、`up_limit`/`down_limit`、`is_suspended` | 当日行 17:30 才可见，08:30 只有 T-1 及更早；`adj_factor` 当日 09:30 盖章 |
| `universe` | `ts_code`、`name`（当时名称，含 ST 标记）、`list_date`、`l1_code`/`l1_name` | 决策日冻结；行业从这里取，不回填今天的行业；核对 `l1_code` 与 `sw_daily.ts_code` 是否同一编码，不同则用成分等权自算行业收益 |
| `events.limit_list_d` | `limit`（U/D/Z）、`limit_times`、`open_times`、`fd_amount`、`amount`（元）、`pct_chg`、`turnover_ratio`（百分数） | 行级 `trade_date` 16:00 盖章，T-1 可用；覆盖自 2020-01；`limit_amount` 已被快照剥离 |
| `events.share_float_complete` | `ann_date`、`float_date`、`float_share`（股）、`float_ratio`（百分数）、`holder_name`、`share_type` | 按公告日可见，公告后 `float_date` 是 PIT 合法的未来日程；同一次解禁可能有多个公告副本，先按（股票、解禁日期、持有人、股份类型）去重 |
| `events.stk_holdertrade` | `ann_date`、`in_de`、`change_vol`（股）、`change_ratio`（百分数） | 行级公告可见；只作解禁后 cohort 的退出条件 |
| `events.block_trade` | `price`（元/股）、`vol`（股）、`amount`（元）、`buyer`、`seller` | 交易日 21:00 可见，T-1 可用；`buyer/seller` 是自由文本 |
| `events.moneyflow` | `net_mf_amount` 等（万元） | 19:00 可见；本包不作主信号，只可作机制 1 的辅助诊断 |
| `fundamentals.forecast_vip` | `type`、`p_change_min`/`p_change_max`、`ann_date` | 每个版本按自身 `ann_date` 可见；机制 3 的负面事件过滤 |
| `macro.index_daily` | `000300.SH` 的 `close`、`pct_chg`（百分数） | T-1 可见；机制 2 的均线对照 |
| `macro.sw_daily` | 申万一级指数 `ts_code`、`close`、`pct_change`（核对单位） | T-1 可见；机制 1 的行业动量对照 |

## 关键陷阱

1. **08:30 看不到任何当日行。** 当日 `stk_limit` 与 `suspend_d` 08:45 盖章、`adj_factor` 09:30、日线 17:30、精确竞价 09:29（且 2025-01-16 前无精确竞价）。是否贴板、是否停牌、开盘缺口多大，决策时只有 T-1 证据。
2. **`kpl_list` 在 08:30 不安全。** 源合同次日 08:30 盖章，恰好等于决策时点，整点调用时 T-1 榜可能尚未入视图；宽度一律从 `limit_list_d` 自算。
3. **两融 T-2。** `margin`/`margin_detail` 行级 `available_at` 是下一日 09:00，08:30 只能用 T-2；本包不用它。
4. **涨跌停拒单对称。** 买单成交价 ≥ 当日 `up_limit` 拒、卖单 ≤ `down_limit` 拒；跌停 cohort 的出场路径要写清并汇报拒单。缺 `up_limit`/`down_limit` 的 Bar 拒 `missing_daily_price_limit`。
5. **`is_suspended` 含复牌日**：快照把 `suspend_d` 的任何行都并成 True，复牌当日常买不进；「真停牌」按日线缺行判断。
6. **只有 09:30 与 15:00 两个成交时点**，本轮不含分钟数据；「盘中反应」不可交易。
7. **单位。** 归一化 `daily.amount` 为元、`pct_chg`/`turnover_rate` 为小数；`limit_list_d.pct_chg`/`turnover_ratio`、`share_float_complete.float_ratio`、`stk_holdertrade.change_ratio` 为百分数；`moneyflow.*_amount` 为万元；`macro.index_daily.pct_chg` 为百分数。跨表先对单位表。
8. **列表可以有合法重复键。** `share_float_complete`、`block_trade` 同日多行要按事件身份聚合，不能盲目 `drop_duplicates(ts_code)`。

## 本轮快照默认没有的东西

- `ths_hot`/`dc_hot`（热榜）、`moneyflow_dc`/`moneyflow_ths` 及板块资金流变体、`limit_step`、`limit_cpt_list`、`limit_list_ths`、`kpl_concept_cons`、`dc_index`/`dc_member`、`top10_holders`、`hm_detail`：存在于原始湖但默认不进快照。上一轮包的热榜与厂商资金流 playbook 在这里读不到，本包不再依赖它们。
- 论坛帖子、评论数、点赞、Level-2、封单队列、部分成交模型。
- 无 Bar 持仓的退出通道：长停牌或退市后没有日线 Bar 的持仓卖不出，入场前把这类尾部锁仓计入仓位上限。
