# 来源、沿革与已证伪清单

平台原文只提供机制先验；论坛与研报里的收益数字不是本环境 Validation 预期。不要粘贴第三方完整源码，不要抓取任何站点。

## 沿革与已证伪

- `explore_platform_strategies` 包（前身）登记了九个 playbook：情绪冰点修复篮子、资金流-价格背离低吸、炸板修复低吸、龙头首次强分歧续强、筹码成本带承接低吸、龙虎榜机构净买延续、跨平台热榜升温、行业资金扩散轮动、本地文本催化+关注确认。两轮下来九个机制在日频 08:30 决策下全部证伪；臂随后交出通用因子回退（去膨胀 Sharpe 概率 0.09→0.45 的都是回退产物），步进季度走查读数 −14.0 / −37.8，首折空对照分位 0.65。
- 其中热榜、厂商资金流、龙虎榜/机构席位三类还依赖默认快照不加载或起点过晚的数据集。本包只用默认加载的数据集。
- 涨停延续类（`limitup_board_20260826` 包，09:29 竞价感知）在更早的轮次同样落败，且本轮 08:30 决策看不到竞价，不要复刻。
- 角落状态臂（corner_cases）负责涨跌停动力学、停牌复牌、ST 切换、新股初期、除权除息、解禁日当天的可执行性；本包只在解禁**之后**的漂移上与它相邻，不重做解禁日机制。

## 外部先验（按机制）

- 领涨-跟涨：Hou, *Industry Information Diffusion and the Lead-lag Effect in Stock Returns*（RFS 2007）；A 股题材轮动的「龙头—跟风」描述来自平台口诀，无可引用收益。
- 宽度情绪周期：市场宽度作为择时状态的研究（advance/decline、涨停家数）在中文卖方报告里常见；本包只把它当门控并要求与无门控对照、均线对照同批。
- 跌停反转：价格限制下的过度反应与反转（Chen, Rui and Wang 关于 A 股涨跌停后收益的研究；Kim–Rhee 的价格限制理论）；连续跌停的坏消息续跌是同一文献的另一面。
- 解禁后释放：限售解禁的抛压与事件后收益在中文文献有一致证据；解禁日本身由 corner_cases 覆盖。
- 放量滞涨/地量：Gervais, Kaniel and Mingelgrin, *The High-Volume Return Premium*（JF 2001）在成熟市场为正；A 股注意力驱动下的反向是待检验假说，方向以输入窗普查为准。
- 大宗折溢价：A 股大宗交易折价率与随后收益负相关的实证（多篇中文文献及 block-trade discount 研究）；溢价端的正向证据较弱，事件稀少。

## 本仓库

- 日线、涨跌停/炸板列表、解禁、股东增减持、大宗交易、业绩预告、指数与申万行业日线已落为 PIT parquet；正式策略从这些原料重算。
- 唯一总闸是 `available_at <= inference_at`；可见时间规则见 `docs/data-documentation.md §3.3`，单位见 `docs/units-reference.md`（沙箱里以 `unit_reference.json` 为准）。
- 正式 ABI、允许的库、`fit(context)`/`context.state_dir`、`models/` 只读挂载见只读 `output/README.md`。
- 离线筛查脚本 `/mnt/tools/screen.py`（只经 `shell` 运行）适合截面型机制的 rank IC 与头部超额普查；事件型 cohort 的匹配对照普查用自己的脚本在 `/mnt/snapshot` 上做，不碰验证区间。
