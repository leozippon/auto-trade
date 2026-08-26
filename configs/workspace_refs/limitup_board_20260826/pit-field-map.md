# 09:29 字段图

先核对本轮 `data_summary.json`、manifest 和单位表。下表是仓库常见合同，某一折仍可能缺表或为空。

## 读法

```text
events = pd.read_parquet(context.asof_dir + "/events", columns=[...])
events = events[(events["dataset"] == "limit_list_d") & (available_at <= inference_at)]
auction = pd.read_parquet(context.asof_dir + "/auction", columns=[...])
daily = pd.read_parquet(context.asof_dir + "/daily", columns=[...])
```

路径必须直接以 `context.asof_dir` 或 `context.snapshot_dir` 为根。不要读宿主路径，不要 import refs。

## 数据集

| 数据 | 有用字段 | 在 09:29 怎么用 |
| --- | --- | --- |
| `limit_list_d` | `limit`（U/Z/D）、`limit_times`、`open_times`、`first_time`、`fd_amount`、`amount`、`pct_chg`、`turnover_ratio`、行业 | 行级约 `trade_date` 16:00。09:29 **只能用 T-1 及更早**。日终 `first_time` / `open_times` / `fd_amount` 若当成当日成交或队列，就是前视。`limit_amount` 已被快照剥离，不要找它。 |
| `kpl_list` | 开盘啦涨停/炸板/跌停/竞价等标签 | 源合同是次日约 08:30 可见，真实落地约 08:55 预回填。09:29 通常能看到 **T-1** 榜，**不能**把它当成今日可交易性或今日实时竞价。先按 `dataset` 与 tag 解释，单位未核前不要对 `bid_*` 做算术。 |
| `limit_step` | `nums` 等连板高度 | 保守日终盖章 + 预回填。可作 T-1 高度交叉检查，**不要**当作当日可买名单或盘中天梯。高度优先从可见 `limit_list_d` 自己数。 |
| `stk_limit` | 当日涨跌停价 | 核心约束，约 08:45 可见。09:29 应用今日上下限判断竞价是否贴板，而不是用 `pct_chg≈10%` 猜。ST 幅度规则会变。 |
| `auction` | `price`、`vol`、`amount`、`pre_close`、`available_at` | 一律盖章交易日 09:29。这是研究用官方竞价，**不是** Broker 成交价。2025-01-16 起多为精确 `stk_auction`；更早常是带规则标记的 09:30 分钟代理。缺覆盖就不要假装有精确竞价。 |
| `daily` | T-1 OHLC、成交额、换手、市值、停牌 | 今日日线 17:30 才可见。09:29 用 T-1 做流动性与昨收。`adj_factor` 09:30 盖章，本推断时点仍用 T-1。 |
| `universe` | 当时代码、名称、行业 | 决策日冻结。不要回填当前行业或当前 ST 名单到历史。 |

## 关键陷阱

1. **09:29 ≠ 盘中打板。** 看不到 09:30 开盘之后的分钟，也不能循环下单。`execute_at` 用随后的 09:30。
2. **竞价可见 ≠ 已经成交。** Broker 对 09:30 用当日日线 `open`。竞价价只作缺口/贴板过滤。
3. **`kpl_list` / `limit_step` ≠ 当日可交易。** 它们是情绪/高度描述。约束层是 `stk_limit` 与停牌。
4. **日终字段前视。** `first_time`、`open_times`、`fd_amount` 描述的是那个交易日收盘后才完整的过程。用它们筛选 **T-1 队列** 可以；用它们推断「今日已经封住、今日能排到」不行。
5. **`limit_amount` 不存在于冻结输入。** 源会回写历史，快照已按列排除。
6. **`fd_amount` 不是可执行封单。** 没有 Level-2、没有队列、没有部分成交模型。涨停买可以被拒。
7. **单位。** `limit_list_d.amount`/`fd_amount`/`float_mv` 一般为元；`pct_chg`/`turnover_ratio` 常是百分数；归一化 `daily.amount` 是元。跨表先对单位表。
8. **08:30 实验的旧结论不要照搬。** 那边丢掉今日竞价是对的；这边 09:29 才允许竞价过滤，其余前视规则不变。

## 本仓库没有的东西

- 实时雪球/股吧/开盘啦 APP 评论和盘口。
- 可执行的封单队列、撤单流、买卖一档。
- 聚宽 `get_price` / `order_target` / 回测收益表。
- 把宣传「年化 xx%」当作本折预期。
