# 验证记录的读数口径：主动回撤、换手、成本与持有人读数

验证行与 `result_ref` 记录里有几组名字相近、口径不同的读数；拿它们比较、外推或预估冻结门与毕业条件之前，先认清各自是什么。

## 回撤

冻结门与毕业条件各读两种回撤：权益的 `stats.max_drawdown`，和主动序列的 `benchmark.active_max_drawdown`——节点日收益减去零技能面板合成收益、从 1 复利后的最大峰谷跌幅。门比较的就是这个数，每一行都有，提名被拒时也会列出。主动回撤直接读它，不用权益回撤或等权组合去代：两种回撤给候选的排序可以相反。

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
| `selection_statistics.graduation_activity.active_excess_at_cost_stress` | 主动序列，年化：滑点乘以本臂倍数后的主动中性化超额，按毕业条件的算法、在本行自己的区间上算 | 毕业条件（前推期）判的就是这个量 |
| `raw_readings.raw_excess_at_cost_stress` | 账户自身扣费收益减基准指数同期收益，整段累计，再扣（倍数 − 1）倍滑点 | 本臂的冻结门列有 `raw_excess_at_cost_stress` 时，完整研究期提名要它大于 0 |
| `stats.cost_sensitivity.excess_at_2x_slippage` | 同上的原始超额，但固定两倍滑点 | 什么也不判；倍数是 2 时与上一行相等 |

主动读数赢了面板，账户却可能没赢基准：面板本身扣费后也会亏（`raw_readings.panel_return`），赢一个亏钱的面板不等于账户赚钱。

## 账户、基准与面板

逐年行 `sub_windows[]` 并列给出 `return`（账户扣费后的年收益）、`benchmark_return`（本臂基准指数的价格收益）与 `panel_return`（零技能面板，即同一成交骨架随机换名的副本，扣费后的年收益）。三者相减就是三种不经回归的读法：`return − benchmark_return` 是持有人对基准的原始超额（即 `excess_return`），`return − panel_return` 是选股本身是否赢了随机名字，`panel_return − benchmark_return` 是骨架与股票池相对基准的部分，与选股无关。覆盖最后两个研究年的行另有 `raw_readings.last_two_years`，给出这两年合起来的同三个数。

`raw_readings.active_market_beta` 是主动序列（账户减面板）对基准指数的回归载荷，整段一个数。为负说明账户持有的名字比随机副本更低 β：指数大涨时它落后于面板，中性化读数（`active_neutralized_*`）按载荷把这段落后记回来，于是中性化超额可以为正，而账户对面板的未回归差为负。持有人只做多、不对冲，拿不到这笔记回。某一年的载荷用 `daily_series`（`validation/active_daily.csv`）里该年的 `active_return` 对 `benchmark_return` 回归即得。
