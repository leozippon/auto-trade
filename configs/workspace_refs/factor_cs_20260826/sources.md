# 来源与诚实边界

这些材料只提供机制先验。对方平台或论文里的收益表不是本环境 Validation 预期。不要粘贴第三方完整源码。

## WorldQuant 101 算子

- Kakushadze, Z. (2015/2016), *101 Formulaic Alphas*, [arXiv:1601.00991](https://arxiv.org/abs/1601.00991)。公开的是公式化算子与截面 rank 习惯，不是 A 股可交易账单。
- 社区实现索引：[yli188/WorldQuant_alpha101_code](https://github.com/yli188/WorldQuant_alpha101_code)。本包只转述 `#1`、`#6`、`#12`、`#101` 的机制，禁止把该仓库拷进 workspace 或 `output`。

`#1` 在论文里是「负收益则用波动、否则用 close，再对短窗 `Ts_ArgMax` 做 rank」；本包把它当作 close-vs-delay 控制，而不是把 `close/delay(close)-1` 再包装一次。
`#6/#12` 是量价相关与量价变动符号；`#101` 是 `(c-o)/(h-l)`。本地必须用 T-1 冻结 qfq 重算。

## 因子菜单为何不搬 202 项

- TuShare [因子列表 doc 486](https://tushare.pro/document/2?doc_id=486) 与 [因子值 doc 490](https://tushare.pro/document/2?doc_id=490) 描述付费日值接口。沙箱无网络，本仓库也没有把 `factor_value` 作为正式输入。
- [stk_factor doc 296](https://tushare.pro/document/2?doc_id=296) / [stk_factor_pro doc 328](https://tushare.pro/document/2?doc_id=328) 的最新日 qfq 锚与本仓库 T-1 冻结复权不一致，且当日盘前不可用。
- 上一轮包 `configs/workspace_refs/explore_tushare_factors/` 已经把可重算候选缩到家族级流程。本轮再缩到五个可分离家族，避免重复同一菜单。

## A 股研究只作优先级

- Liu, Stambaugh and Yuan, [Size and Value in China](https://www.nber.org/papers/w24458)：中国价值常用 E/P，并排除最小市值尾部以减弱壳价值。支持先测 Value+Quality 与规模控制，不把小盘收益直接叫 alpha。
- Jacobs and Müller, [Anomalies in the China A-share market](https://pure.eur.nl/en/publications/anomalies-in-the-china-a-share-market)：价值/风险/交易类证据相对更强，规模、质量、过去收益整体较弱。支持分家族证伪，不能把海外动量先验当成事实。

样本、成本和交易约束都不同。对方平台或论文里的收益、IC、分位表一律不是本折预期。只有本环境含成本 PIT Validation 能决定保留、翻转或淘汰。

## 公开挖掘流程（先验，禁止克隆）

只借用「先分家族、再看 IC / 换手 / 分位」的流程。不要把对方表达式库、遗传搜索或因子菜单搬进 `output`，也不要按其收益表验收。

- [RndmVariableQ/AlphaAgent](https://github.com/RndmVariableQ/AlphaAgent)：A 股因子 DSL，用 IC / 换手 / 分位评估。当作「家族先立、再用 IC 证伪」的流程先验，不是可提交策略。禁止克隆其 FactorZoo 表达式或 LLM 挖因子循环；沙箱无网络，也不能按其 Tushare 抓数。其隔夜/日内算子与本包 08:30、T-1 冻结冲突，不要搬。
- [cn-vhql/FactorHub](https://github.com/cn-vhql/FactorHub)：遗传算法 + 多目标（IC / IR / 单调性）挖掘。只作「不要单看毛收益」的筛选先验。禁止 import DEAP / TA-Lib / XGBoost，禁止通达信公式引擎，禁止把内置 MA / RSI / MACD 厨房当成第五个家族。
- [wzx11223344/factor-mining](https://github.com/wzx11223344/factor-mining)：Spearman IC / ICIR，等权、IC 加权或 PCA 合成。支持先分家族再合成；本包合成只允许两个以上独立胜出家族等权。禁止 sklearn PCA，禁止把它的 14 项动量/反转/波动/流动性菜单整表复制（那会回到 202 因子厨房）。

## 本仓库

- 日线、复权、每日指标、财务报表和财务指标已落为 PIT parquet；正式策略从这些原料重算。
- 唯一总闸是 `available_at <= inference_at`。
- 正式 ABI 不挂载 `models/`，也不允许 sklearn/lightgbm/qlib。
