# 残差标签：基准指数等于本臂的基准，价格用复权价

宿主算主动读数时，回归的市场腿是本臂运行事实里的 `benchmark_index`（见 `neutralized_excess_method`），零技能面板也在这只指数成分的同一侧抽名。学习型策略若把标签写成前向收益减去 β 乘某指数收益，这个指数必须是同一只，β 与前向收益也都要用复权价算。

## 为什么

- 指数常量写成另一只时，模型学到的排序里仍带着宿主随后会减掉的暴露，主动读数被稀释，对照之间的差异也不再只来自机制。起步包或自写代码里的指数常量，先对照运行事实核对一次。
- 日线价格不复权，除权日的价格按分红送转跳低（002667.SZ 在 2018-05-02 每股送转 0.7：相邻收盘之比 −42.6 %，复权后 −2.3 %）。这一天落进 β 的滚动窗，就在整整一个窗长里当作一笔巨大的收益参与协方差；跨过它的前向收益同样失真。研究期一个 120 日窗里约七成名字碰到除权日，现金分红只把 β 挪动百分之几，送转能挪 0.1 以上（2019-03..08 那个窗里碰到除权的 2,589 个名字中有 161 个，最大 0.89）。复权价 = 价格 × `adj_factor`；指数点位不需要复权。

## 怎么算

```python
def residual_label(daily, index_bars, horizon=20, beta_days=120, min_days=60):
    """(trade_date x ts_code) label of signal day t on adjusted prices.

    daily: ts_code, trade_date, open, close, adj_factor; index_bars: trade_date, open,
    close of benchmark_index. Label = open(t+1) -> open(t+1+horizon) return minus beta
    x the index's, beta = trailing OLS slope of daily close returns on the index's.
    Cells whose label or beta cannot be computed stay NaN.
    """
    wide = daily.pivot(index="trade_date", columns="ts_code", values=["open", "close", "adj_factor"])
    wide = wide.sort_index()
    opens, closes = wide["open"] * wide["adj_factor"], wide["close"] * wide["adj_factor"]
    index = index_bars.set_index("trade_date").sort_index().reindex(closes.index)
    stock_ret = closes.pct_change(fill_method=None)
    index_ret = index["close"].pct_change(fill_method=None)
    rolling = {"window": beta_days, "min_periods": min_days}
    beta = stock_ret.rolling(**rolling).cov(index_ret).div(index_ret.rolling(**rolling).var(), axis=0)
    forward = opens.shift(-(horizon + 1)) / opens.shift(-1) - 1.0
    index_forward = index["open"].shift(-(horizon + 1)) / index["open"].shift(-1) - 1.0
    return forward - beta.mul(index_forward, axis=0)
```

- 持有期、β 窗长、收缩与截断按包；算不出的格子不进训练，截面排序用 `label.rank(axis=1, pct=True)`。
- 改指数或价格口径之后重跑冒烟：标签、训练和分数会一起变。改之前确认该指数的行情覆盖训练窗和 β 的滚动预热（各年视图里指数行情从哪天起，见 `research-8y-data-surface`）。
- 同一条臂上要对比的候选用同一条基准腿。
