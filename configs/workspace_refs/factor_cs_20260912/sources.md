# 来源与诚实边界

这些材料只提供机制先验。论文或平台里的收益、IC、分位表不是本环境 Validation 预期；不要粘贴第三方完整源码，只引用并转述机制。

## 文献先验（按家族）

- 流动性与换手：Liu, Stambaugh and Yuan, *Size and Value in China*（JFE 2019，[NBER w24458](https://www.nber.org/papers/w24458)）——CH-3/CH-4 用异常换手代替动量，并排除最小 30% 市值以剥离壳价值；Amihud (2002) 非流动性度量 `|r|/amount`。本包据此把规模中性与 ADV 过滤写成硬约束。
- 彩票需求：Bali, Cakici and Whitelaw, *Maxing Out*（JFE 2011）；Ang, Hodrick, Xing and Zhang 的特质波动之谜（JF 2006）；Jacobs and Müller, *Anomalies in the China A-share market*（[链接](https://pure.eur.nl/en/publications/anomalies-in-the-china-a-share-market)）报告风险类异象在 A 股相对更强。
- 隔夜-日内分解：Lou, Polk and Skouras, *A Tug of War: Overnight versus Intraday Expected Returns*（JFE 2019）；A 股隔夜收益长期为负的现象在多篇中文实证里重复出现，方向以本折输入窗预筛为准。
- 筹码浮盈：Grinblatt and Han, *Prospect Theory, Mental Accounting, and Momentum*（JFE 2005）的资本利得悬垂；An, Wang, Wang and Yu, *Mispricing and the Disposition Effect / V-shaped disposition*（2020 前后）——A 股关系可能非单调，先画十分位再登记方向。
- 资金拥挤：大单资金流与同期收益的机械相关是本包要求正交化的直接原因；两融余额与随后收益负相关的证据来自多篇中文文献，本包不引用具体数字。
- 基本面动量：Bernard and Thomas 的盈利公告后漂移；A 股业绩预告/快报的公告日效应与漂移在中文文献里有一致证据。

## 前几轮已被证伪、本轮不再测

- `factor_cs_20260826` 包的五家族（价值+质量、反转+低波、成长、中期趋势、稀疏 Alpha101 算子）：三轮里胜出的都是小市值暴露，规模因子修正后首折冻结节点的中性化超额由 +0.07% 变为 −8.66%；步进季度的走查读数 +2.0 / −20.4 / −9.0。
- 价值+质量 ridge、`-corr(open, vol, 10)`、资金流 + Alpha101 `#7` 混合：折内数字好看，父本对照在新季度为负。
- LightGBM IC 门 vs ridge 的样本外对照：两者都输给等权。
- ml_numpy 谱系的手工阈值流动性 z-score、闭式 ridge 单轮收工：不是本包的候选形态。

## 本仓库

- 日线、复权、每日指标、财务报表、财务指标、业绩预告/快报、资金流、两融、筹码分布、指数日线已落为 PIT parquet；正式策略从这些原料重算。
- 唯一总闸是 `available_at <= inference_at`；可见时间规则见 `docs/data-documentation.md §3.3`，单位见 `docs/units-reference.md`（沙箱里以 `unit_reference.json` 为准）。
- 正式 ABI、允许的库、`fit(context)`/`context.state_dir`、`models/` 只读挂载见只读 `output/README.md`。
- 离线筛查脚本 `/mnt/tools/screen.py`（只经 `shell` 运行）给 rank IC、ICIR、衰减、规模中性 IC、头部十分位超额、换手与可交易性；它只读 `/mnt/snapshot` 决策视图，不替代 Validation。
