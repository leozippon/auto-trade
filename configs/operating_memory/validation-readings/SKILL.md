# 验证记录的读数口径：主动回撤、换手与成本

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

`cost_sensitivity.cost_per_bp_per_side` = `stats.turnover` × 1e−4，即每边多 1 bp 滑点吃掉期初资金的比例（整段累计）。毕业条件把滑点乘以 `acceptance_rules` 里的倍数后，要求主动中性化超额仍为正；研究期上可以先估：`benchmark.active_neutralized_excess` − (倍数 − 1) × `cost_sensitivity.slippage_bps` × `cost_per_bp_per_side` ÷ (`benchmark.n_days` ÷ 244)。权益一路增长时这个估计偏保守，缩水时偏乐观。
