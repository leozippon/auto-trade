# 验证读数里名字看不出的口径：两种回撤、换手、对照组合与噪声

`validate` 与 `test` 返回的字段含义在工具描述里，哪条登记条件读哪个字段在运行事实 `acceptance_rules.registration` 里；拿读数比较、外推或预估登记之前，先认清下面几处名字说不出来的口径。

## 两种回撤

`active.max_drawdown` 是主动序列复利路径的回撤，`book.max_drawdown` 是账户权益的回撤。登记在回撤上只比权益回撤，上限在 `acceptance_rules.registration.max_drawdown`，每次 `test` 都查；验证记录的统计量不读回撤。两种回撤给候选的排序可以相反，比哪条限就读哪个字段，不用另一种或等权组合去代。

## 换手

`book.turnover` 是整段回放的买入名义与卖出名义之和除以期初资金，整段累计、不年化：一年换几遍（双边、期初资金口径）用它除以 span 的年数，`active.excess_at_cost_stress` 加收的滑点也按它算。要按平均权益算，用 `result_ref` 记录里的 `executions` 与 `equity_curve` 自己重算；几种分母都对不上某个报告值时，先查它是不是期初资金。

## 主动、账户与对照组合

主动读数赢了对照组合，账户却可能没赢基准：对照组合本身扣费后也会亏（`book.baseline_return`），赢一个亏钱的对照组合不等于账户赚钱。`active.excess` 是主动序列逐日差的年化均值，`book.excess_vs_baseline` 是两条复利收益之差，两者同号居多，但不是同一个数。

`active.market_beta` 为负说明账户持有的名字比随机副本更低 β：指数大涨的年份主动序列落后于对照组合，大跌的年份领先；残差统计量不算三条腿能解释的部分。设了跟踪约束的实验，登记比的是策略自身收益的市场载荷与跟踪误差（`result_ref` 记录里的 `stats.benchmark.market_beta` 与 `stats.benchmark.tracking_error`，见 `acceptance_rules.registration.tracking_mandate`），不是主动序列的。

## 与策略无关的读数噪声

随机选股对照组合的随机种子只由回测区间决定，换名的对象是节点自己的成交骨架：同一份字节在同一个 span 上重跑，对照组合与读数都不变；换一个 span，或者字节变了以致成交变了，对照组合就换了。两个从同一年起跑的 span 上，确定性策略在共同那几年的日收益逐位相同，那几年逐年主动值之差就是对照组合噪声的一次实测。训练模型的策略换一个训练种子本身就会移动读数：两个配置之差小于同一配置换种子的读数差时，这个差分不开。
