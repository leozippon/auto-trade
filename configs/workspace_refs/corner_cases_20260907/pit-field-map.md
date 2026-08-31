# 08:30 字段图

先核对本轮 `data_summary.json`、manifest 和单位表。下表是仓库常见合同，某一折仍可能缺表或为空。

## 读法

```text
daily = pd.read_parquet(context.asof_dir + "/daily", columns=[...])       # 含 up_limit/down_limit/adj_factor/is_suspended
events = pd.read_parquet(context.asof_dir + "/events", columns=[...])
events = events[(events["dataset"] == "limit_list_d") & (available_at <= inference_at)]
fund = pd.read_parquet(context.asof_dir + "/fundamentals", columns=[...]) # dividend 等按 dataset 过滤
universe = pd.read_parquet(context.asof_dir + "/universe")                # 决策日冻结
```

路径必须直接以 `context.asof_dir` 或 `context.snapshot_dir` 为根。不要读宿主路径，不要 import refs。

## 数据集

| 数据 | 有用字段 | 在 08:30 怎么用 |
| --- | --- | --- |
| 合并日线 `daily` | T-1 OHLC、成交额、换手、市值、`up_limit`/`down_limit`、`adj_factor`、`is_suspended` | 当日行 17:30 才可见，08:30 只有 T-1 及更早。这是一张区间表：可回看历史行重算涨跌停触碰、复权跳变、停牌区段与 ST 限幅比例，冷重启后可完整重算，不依赖跨日全局状态。 |
| `universe` | 当时代码、`name`（namechange 口径的当时名称）、`list_date`、申万一级行业 | 决策日冻结。ST 当前状态从当时 `name` 判断；上市天龄从 `list_date` 算。不要回填今天的名称、行业或 ST 名单到历史。 |
| `limit_list_d` | `limit`（U/Z/D）、`limit_times`、`open_times`、`first_time`、`last_time`、`fd_amount`、`amount`、`pct_chg`、`turnover_ratio`、行业、市值 | 行级 `trade_date` 16:00 盖章，08:30 只能用 **T-1 及更早**。覆盖自 2020-01。日终字段描述的是收盘后才完整的过程：筛 T-1 队列可以，推断「今日封住/能排到」不行。`limit_amount` 已被快照按列剥离。 |
| `kpl_list` | 开盘啦 涨停/自然涨停/炸板/跌停/竞价 标签，`bid_*` 等 | 源合同次日 08:30 盖章——恰好等于本轮决策时点，整点调用时 T-1 榜可能尚未入视图。宽度与板高一律从 `limit_list_d` 自算，`kpl_list` 只作可见时的交叉检查。`bid_*` 字段仅在 `tag=竞价` 行内非空且单位未核实，只能做同标签内名次/分位，不得做算术。 |
| `dividend`（fundamentals 域） | `div_proc`、`stk_div`、`cash_div`、`ex_date`、`record_date`、`imp_ann_date`/`ann_date` | 按公告日可见。公告可见之后，行内 `ex_date` 就是 PIT 合法的未来日程，可用于预期除权事件；不得反向用 `ex_date` 提前得知尚未公告的安排。区分 `div_proc`（预案/股东大会/实施），只有实施口径的日程可靠。 |
| `new_share` | `ipo_date`（申购日）、`issue_date`、`price`、`pe`、`ballot` | 整行在申购日后第二个 A 股交易日收盘后可见，上市日之前发行价已可见。新股上市首日没有任何可见 T-1 行情，规模估计只能用发行价。 |
| `share_float_complete` | `float_date`、`float_share`、`float_ratio`、持有人、股份类型 | 按公告日可见。公告可见后 `float_date` 即为 PIT 合法的未来解禁日。同一次解禁可能残留多个公告日副本：先按（股票、解禁日期、持有人、股份类型）聚合去重再计压力，不能假定一键一行。 |
| `stk_holdertrade` / `stk_holdernumber` | 增减持方向、`change_vol`/`change_ratio`；股东人数 | 按公告月份入库、行级公告可见。本包只允许作解禁/状态事件的伴随证据；「增持减解禁压力」的截面合成分数属于 explore_github 方向的 idea B，不要在这里重做（见 `sources.md`）。 |
| `index_daily`（macro 域） | 宽基指数 OHLC | T-1 可见。配合 `limit_list_d` 宽度统计定义极端市场日/修复态，仅作门控。 |

## 关键陷阱

1. **08:30 看不到任何当日行。** 当日 `stk_limit` 与 `suspend_d` 08:45 盖章、`adj_factor` 09:30、日线 17:30、精确竞价 09:29（且 2025-01-16 前无精确竞价）。是否贴板、是否停牌、是否除权，决策时只有 T-1 证据；当日真实约束由 Broker 执行，落空就是拒单。
2. **`is_suspended` ≠「当日没交易」。** 快照把 `suspend_d` 的任何行都并成 `is_suspended=True`，其中包括盘中临停（当日仍有真实成交 Bar）和复牌标记行（R，复牌当日多数有 Bar）。2023 年抽样（每 21 个交易日取一日）：停牌行 609 条中 12 条当日有 Bar，复牌行 13 条中 10 条当日有 Bar。Broker 对 `is_suspended=True` 的 Bar 一律拒单，所以**复牌当日在模拟里常常买不进**。判「真停牌」用「当日无日线 Bar」；判可入场用「T-1 已有恢复 Bar 且 `is_suspended=False`」，次日 09:30 进。
3. **涨跌停拒单是对称的。** 成交价 ≥ 当日 `up_limit` 的买单拒 `daily_price_limit`（封死的涨停买不进），成交价 ≤ `down_limit` 的卖单同样被拒（跌停出不去——持有跌停票的出场路径必须写清）。Bar 缺 `up_limit`/`down_limit` 值时拒 `missing_daily_price_limit`。
4. **新股初期的限价是数据驱动的。** 注册制窗口实测：上市初期 `stk_limit` 给哨兵宽限（`up_limit=1e6`、`down_limit=0.01`），模拟里前几日可成交、不触发限价拒单；更早窗口的首日限幅规则不同，逐窗从数据核实，不要写死制度年表。
5. **手数与费用。** 买入通常 100 股整数倍；688/689 至少 200 股、其后可 1 股递增；`.BJ` 至少 100 股、其后可 1 股递增。佣金万一（最低 5 元/笔）、卖出印花税（2023-08-28 起由 0.1% 降至 0.05%，按日期自动切换）、过户费 0.1bp、方向滑点 5bp。低价股与小篮子先算费用占比再谈超额。
6. **只有开盘与收盘两个成交时点。** 本轮不含分钟数据，`execute_at` 只有 09:30 与 15:00 有成交价，其余拒 `missing_execution_price`。事件「盘中反应」在本环境不可交易，不要改写成盘中触发。
7. **T-1 事件面板的可见时间各不相同。** `moneyflow` 当日 19:00、`top_list`/`top_inst` 20:00、`block_trade` 21:00——T-1 行在 08:30 均可见；但融资融券 `margin`/`margin_detail` 次日 09:00 才可见，08:30 只能用 **T-2**。
8. **单位。** 归一化 `daily.amount` 为元；`limit_list_d` 的 `amount`/`fd_amount`/`float_mv`/`total_mv` 为元、`pct_chg`/`turnover_ratio` 为百分数；`share_float_complete.float_share` 为股、`float_ratio` 为百分数；`stk_holdertrade.change_vol` 为股、比例为百分数；`dividend.cash_div` 为元/股、送转为股/股；`new_share.price` 为元/股、`funds` 为亿元、`ballot` 为百分数。跨表先对单位表。
9. **ST 限幅比例启发式有失效日。** 2026-07-06 起主板 ST/*ST 涨跌幅由 5% 改为 10%，`up_limit/pre_close ≈ 1.05` 的识别在该日后失效；本折各验证窗均在其之前，但结论里必须写明该边界，特征本身要从 `stk_limit` 绝对价算而不是从 `pct_chg` 猜。

## 本轮快照没有的东西

- `limit_list_ths`、`limit_step`、`limit_cpt_list`、`ths_hot`/`dc_hot`、厂商资金流变体、`top10_holders`、`hm_detail`：存在于原始湖但默认不进快照，本轮未选入，策略读不到。板高从可见 `limit_list_d` 自己数。
- `namechange` 历史表本身：只有 `universe` 里的当时名称。ST 的「切换事件」从可见日线区间里 `stk_limit` 限幅比例的跳变重算，而不是靠昨日名称缓存。
- 可执行封单队列、Level-2、盘口与部分成交模型；`fd_amount` 只是 T-1 日终描述。
- 无 Bar 持仓的退出通道：长停牌或退市后没有日线 Bar 的持仓卖不出（拒 `missing_execution_price`），权益按最后可见价滞留。入场前就要把这类尾部锁仓计入仓位上限，而不是事后当意外。
