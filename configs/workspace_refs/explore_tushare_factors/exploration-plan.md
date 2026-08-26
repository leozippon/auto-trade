# Fold-ready 探索计划

以下 10 步每一步都应产生可执行候选或明确淘汰结论，而不是收集注释。

1. 先读本次 `data_summary.json`、daily/fundamentals/macro schema 与可见日期；固定可交易股票池、再平衡频率、持仓数、手续费和现金缓冲。
2. 在 `generate_orders` 内按 `context.asof_version` 构建一张有界尾窗特征表；同一 PIT 视图只算一次，视图变化才增量或重读有限窗口。
3. 先跑一个简单可交易基线，并记录因子覆盖率、候选数、换手、拒单、收益、Sharpe 和最大回撤；覆盖不足的因子不进入组合。
4. 所有因子先乘方向、去 1%/99% 极值，再各自做截面 rank 或 z-score；绝不把原始量纲直接相加。
5. 第一轮分别验证两个 A 股优先假说：`Value+Quality`，以及 `Reversal+Low-Risk`；每个组合只取 4–6 个去冗余因子，不混入父策略机制。
6. 第二轮再分别验证 `Medium-Term Trend`、`Growth`、`Liquidity` 和小型 `Alpha101`；趋势与同周期反转不能在同一候选里互相抵消。
7. 对幸存因子做相关性去重；有 PIT 行业字段时按行业去均值并对 `log(MV)` 残差化，否则只做规模中性化并明确记录行业限制。
8. 任一候选若靠极少股票、单一行业/小盘暴露、不可交易高成本股票、符号不稳定或覆盖骤降取得收益，就由 Validation 判为该机制失败。
9. 只把两个以上独立胜出家族做等权 family-score 组合；若含成本表现、回撤或稳定性未优于最佳单家族，就保留单家族而不是调权拟合。
10. 把本折最小完整胜出方案写到 `output/main.py` 并做最终完整 Validation；失败则留下证伪结论和下一折不同假说，不克隆父本。

## 组合模板

- `Value+Quality`：`earnings_to_price`、`book_to_market`、`roe`、`roa`、`grossprofit_margin`、负向 `debt_to_assets`；先做规模中性，另试排除最小市值尾部，避免壳价值主导。
- `Reversal+Low-Risk`：`reversal_5d` 或 `reversal_21d` 二选一，加负向 `return_std_21d/63d`、`beta_60d_000300`、`high_low_63d`；低流动性只作惩罚。
- `Medium-Term Trend`：`return_63d`、`return_126d`、`return_252d_skip21d`、`macd_hist` 中去冗余后选 2–3 个；A 股动量证据较弱，必须独立证伪。
- `Growth`：`or_yoy`、`netprofit_yoy`、`ocf_yoy`、`roe_yoy` 中选覆盖稳定者；亏损跨零时同比比率容易爆炸，先去极值。
- `Liquidity/Alpha101`：只作为边际增量；若提高换手或选出难成交股票，即使毛收益增加也淘汰。

## 标准化与中性化

稳健 z-score：`z=(clip(x,q01,q99)-mean)/std`；稳健 rank：`score=2*(rank(pct=True)-0.5)`。
中性化可用同一决策截面的 `x = intercept + log(MV) + PIT行业哑变量 + residual`，对 residual 再标准化。
截面只含当时已上市、未退市、值有限且有可见源行的股票；行业缺失单列为 unknown 或只做规模中性，不能用今天的行业回填历史。

## Validation 的明确否证条件

- 发现任何 `available_at`、T-1、复权锚或财务版本错误：结果无效，不讨论收益。
- 有效覆盖长期低于约 60%，或在决策间大幅塌缩：淘汰该因子，除非它本来就是明确的稀疏事件假说。
- 方向在相邻子窗反复翻转，或收益只来自单个极端日期/个股：淘汰该机制。
- 加手续费、滑点、涨跌停和流动性约束后优势消失：淘汰该候选。
- 规模/行业中性化后消失：否证“独立 alpha”叙述；只有明确接受该风格暴露时才能保留。
- 多家族组合不优于最佳单家族，或只降低收益却未改善回撤：回退到更简单候选。
