# 验证记录的读数口径：主动回撤、换手、成本与持有人读数

验证行与 `result_ref` 记录里有几组名字相近、口径不同的读数，各字段的定义在 `batch_validate` 的说明里，哪条条件读哪个字段在运行事实 `acceptance_rules` 里；拿它们比较、外推或预估冻结门与毕业条件之前，先认清下面这些容易读错的地方。

## 回撤

主动回撤直接读 `benchmark.active_max_drawdown`：门比较的就是这个数，每一行都有，提名被拒时也会列出。不用权益回撤 `stats.max_drawdown` 或等权组合去代：两种回撤给候选的排序可以相反。

## 换手

| 读数 | 分子 | 分母 |
|---|---|---|
| `stats.turnover` | 整段回放的买入名义与卖出名义之和 | 期初资金；整段累计，不年化 |
| `stats.sub_windows[].turnover` | 该研究年的买卖名义之和 | 同样是期初资金，不是该年的期初权益 |
| 毕业条件的成本压力 | 所评那一段的买卖名义之和 | 那一段的期初权益 |

「一年换几遍」（双边、期初资金口径）直接读逐年的 `sub_windows[].turnover`，或用 `stats.turnover` 除以年数；要按平均权益算，用 `executions` 与 `equity_curve` 自己重算。几种分母都对不上某个报告值时，先查它是不是期初资金。

## 成本压力

行里有三个名字相近的成本压力读数，都由宿主算好，不用手算：

| 读数 | 序列与口径 | 谁用它 |
|---|---|---|
| `selection_statistics.graduation_activity.active_excess_at_cost_stress` | 主动序列，年化，滑点按本臂倍数 | 毕业条件的成本压力在前推期判的就是这个量 |
| `raw_readings.raw_excess_at_cost_stress` | 账户对基准指数的原始超额，整段累计，滑点按本臂倍数 | 冻结门列有 `raw_excess_at_cost_stress` 的臂，完整研究期提名判它 |
| `stats.cost_sensitivity.excess_at_2x_slippage` | 同上的原始超额，但固定两倍滑点 | 什么也不判；倍数是 2 时与上一行相等 |

主动读数赢了面板，账户却可能没赢基准：面板本身扣费后也会亏（`raw_readings.panel_return`），赢一个亏钱的面板不等于账户赚钱。

## 账户、基准与面板

账户、基准指数与零技能面板各自扣费后的收益——逐年行 `sub_windows[]` 的 `return`、`benchmark_return`、`panel_return`，本行整段与最后两个研究年（`raw_readings` 与 `raw_readings.last_two_years`）的 `strategy_return`、`benchmark_return`、`panel_return`——两两相减是三种不经回归的读法：账户减基准是持有人对基准的原始超额（`raw_excess`，逐年行里是 `excess_return`）；账户减面板是选股本身是否赢了随机名字（`plain_selection`）；面板减基准是骨架与股票池相对基准的部分，与选股无关。

`raw_readings.active_market_beta` 为负，说明账户持有的名字比随机副本更低 β：指数大涨时它落后于面板，中性化读数（`active_neutralized_*`）按载荷把这段落后记回来，于是中性化超额可以为正，而账户对面板的未回归差 `plain_selection` 为负。持有人只做多、不对冲，拿不到这笔记回。某一年的载荷用 `daily_series`（`validation/active_daily.csv`）里该年的 `active_return` 对 `benchmark_return` 回归即得。
