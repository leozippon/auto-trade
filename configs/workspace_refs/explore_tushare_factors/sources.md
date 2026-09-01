# 检索来源与筛选结论

## TuShare 官方资料

- [因子列表，doc 486](https://tushare.pro/document/2?doc_id=486)：`factor_list` 当前列出 202 个股票因子、9 个家族；其中 Alpha101 31、Growth 15、Liquidity 35、Momentum 20、Quality 59、Reversal 3、Risk 25、Size 3、Value 11。该页用于候选名称和公式检索，不证明本地存在因子日值。
- [因子库说明，doc 485](https://tushare.pro/document/2?doc_id=485)：因子库覆盖行情、财务和基本面，多数因子做截面排名；本包因此只借用定义，不读取供应商 rank。
- [因子值，doc 490](https://tushare.pro/document/2?doc_id=490)：`factor_value` 是独立付费日值接口；当前沙箱没有网络和该数据，正式策略不得假设它存在。
- [股票技术因子，doc 296](https://tushare.pro/document/2?doc_id=296)：`stk_factor` 提供 qfq OHLC、MACD、KDJ、RSI、BOLL、CCI，并注明技术指标基于前复权价格；该接口不在沙箱，且其最新日复权锚不替代本仓库 PIT 冻结。
- [股票技术因子专业版，doc 328](https://tushare.pro/document/2?doc_id=328)：`stk_factor_pro` 扩充大量 bfq/qfq/hfq 指标；本包只选能由 OHLCVA、daily_basic 和公告财务直接重算的少数公式。
- [A 股日线，doc 27](https://tushare.pro/document/2?doc_id=27)：日线是未复权 OHLCVA，官方说明 15–16 点入库；本项目采用更保守的 17:30 可见时间。
- [复权因子，doc 28](https://tushare.pro/document/2?doc_id=28)：官方说明当日复权因子约 09:15–09:20 入库；本项目保守盖章 09:30，默认 08:30 只能用 T-1。
- [每日指标，doc 32](https://tushare.pro/document/2?doc_id=32)：提供换手、PE/PB/PS、股息率、股本和市值；官方说明 15–17 点更新，本项目 18:00 可见。

## A 股证据

- Liu, Stambaugh and Yuan, [Size and Value in China](https://www.nber.org/papers/w24458)（NBER w24458；后发表于 JFE）：中国三因子构造排除最小 30% 公司以减弱壳价值影响，价值用 earnings-to-price；这支持优先测试 E/P、规模暴露控制和最小市值尾部敏感性，而不是把小盘收益直接解释成独立 alpha。
- Jacobs and Müller, [Anomalies in the China A-share market](https://pure.eur.nl/en/publications/anomalies-in-the-china-a-share-market)（Pacific-Basin Finance Journal 68, 2021）：2000–2019 的 32 个异常中，价值、风险和交易类证据较强；规模、质量和过去收益整体较弱，但残差动量与反转例外。它支持先分家族证伪，不能把海外动量先验当成 A 股事实。

这些研究是优先级先验，不是当前 Fold 的结论。样本、交易约束和成本不同；只有本环境的完整 PIT Validation 能决定保留、翻转或淘汰。

## 本仓库交叉核对

- `docs/data-documentation.md`：唯一总闸是 `available_at <= inference_at`；本地有 `daily`、`adj_factor`、`daily_basic`、财务报表、`fina_indicator_vip` 与 `index_daily`，但未列出 `factor_value` 或 `stk_factor*` 作为正式输入。
- `docs/data-documentation.md` 还确认：daily/daily_basic 当日收盘后可见；财务按公告时间；事件/宏观/财务保留源单位；daily、daily_basic 与其他逐日表覆盖不能假定完全一致。
- `docs/environment-design.md`：长窗口日线来自滚动 `asof_dir/daily`；fundamentals 是公告可见的多版本事件，macro 含宽基指数；parts 随推断时点刷新。
- `docs/agent-design.md`：正式策略是 `output/` 下以 `main.py` 为入口的包，只依赖当前 context 和只读 snapshot/asof 路径，允许的库以只读 `output/README.md` 为准；读取前确认 schema，缓存按 `asof_version` 失效。
- 已退役的 `tushare_factors` 包：曾正确识别 202 因子菜单、T-1 和供应商 qfq 风险，但其 202 名索引容易诱导注释收集；本包改成 42 个候选、家族级可证伪流程和明确组合淘汰条件。

## 筛选原则

保留条件：当前 PIT parquet 有原料、约 24 个月数据窗可形成有效观测、可在允许的库内从 PIT 原料重算、方向和中性化方式可明确、能进入真实含成本 Validation。
剔除条件：需要网络/供应商因子值、依赖未来或当前快照、供应商预排名、500–1320 日超长窗、复杂但覆盖弱的五年财务构造、与已选项近重复且没有独立假说。
