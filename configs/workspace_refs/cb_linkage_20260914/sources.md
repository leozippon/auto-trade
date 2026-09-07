# 来源与诚实边界

这些材料只提供机制先验。论文里的收益、IC、分位表不是本环境 Validation 预期；A 股「转债→正股」的学术证据在本批候选方向里最薄（方向评审把它排在最后），本包的家族更多来自制度事实与转债定价恒等式，胜负只由本折普查与 Validation 裁决。不要粘贴第三方完整源码，只引用并转述机制。

## 文献先验（按家族）

- 转债先于正股定价（家族 1、4）：*Can convertible bond trading predict stock returns? Evidence from China*（Pacific-Basin Finance Journal, 2023）——中国转债的订单不平衡预测正股收益，价格发现发生在转债市场；这是本包唯一直接针对 A 股「转债→正股」方向的学术证据，支持残差动量与成交额异动两个家族的方向 +，但它的信号是订单流，本环境只有日频成交额与收盘价，家族 4 是它的粗代理。
- 转债套利与正股价格压力（家族 2）：Choi, Getmansky and Tookes, *Convertible bond arbitrage, liquidity externalities, and stock prices*（JFE 2009）——转债套利活动通过对正股的交易影响正股流动性与价格。A 股无法便利做空，套利以「买转债—转股—次日卖正股」实现，因此负溢价率在转股期内对应正股次日的供给压力；这是家族 2 剔除 `arb_days_5 ≥ 3` 的依据，方向以普查为准。
- 强赎与促转股（家族 3）：Ingersoll, *An examination of corporate call policies on convertible securities*（JF 1977）与 Asquith and Mullins, *Convertible debt: corporate call policy and voluntary conversion*（JF 1991）——发行人赎回政策、触发赎回与自愿转股的经典分析，支持「发行人有动机让转债被转股」这一机制本身；A 股强赎条款的 15/30 与 130% 标准、强赎公告后的转股卖压在中文实证里有一致描述，本包不引用具体数字。
- 债底与下修（家族 5）：转债「债底 + 期权」的分解是教科书结论；A 股下修条款的发行人行为在中文文献与卖方研究里反复出现，学术证据不足，家族 5 因此把 BP 写成必比的对照并把跌破债底的券排除。
- 正股 5 日反转（家族 6）：A 股短期反转的证据见 `factor_cs_20260912` 包的来源，本包只把它当机制无关对照。

## 制度事实（机制的前提）

- 转债 T+0；2022-08-01 起沪深交易所可转换公司债券交易实施细则：上市首日涨幅 57.3%、跌幅 43.3%，次日起每日 ±20%，此前无涨跌幅限制、只有临时停牌（本地 `cb_daily` 2024 年 `pct_chg` 最大值 57.3，与首日上限一致）。正股 T+1，主板 ±10%、创业板/科创板 ±20%。
- 2022-06 起个人投资者开通转债交易须满足两年交易经验与 10 万元资产的适当性要求，强化了机构与专业投资者主导。
- 转债 1 手 = 10 张 = 1,000 元面值（本地核对 `amount × 1e4 ≈ vol × 10 × close`）。
- 强赎条款常见形式：转股期内连续 30 个交易日中至少 15 个交易日收盘价不低于当期转股价的 130%；下修条款常见形式：连续 30 个交易日中至少 15 个交易日收盘价低于当期转股价的 85% 或 80%。条款参数因券而异，`cb_basic.call_clause`/`reset_clause` 是文本，本包用声明常数。

## 数据接口（TuShare，按名称）

- 可转债基础信息 `cb_basic`：正股映射 `stk_code`、发行规模、上市日、转股起止日、初始转股价、条款文本、发行评级；当前状态字段（`conv_price`、`remain_size`、`newest_rating`、`delist_date`）是每晚刷新值，快照已剔除。
- 可转债行情 `cb_daily`：日频价量，另含转股价值 `cb_value`（= 100 / 转股价 × 正股收盘）、转股溢价率 `cb_over_rate`（= 转债收盘 / 转股价值 − 1）、纯债价值 `bond_value`、纯债溢价率 `bond_over_rate`；两组恒等式已在本地逐行核对。
- 可转债赎回信息 `cb_call`：强赎与到期赎回的分阶段公告（提示、实施、不强赎、已满足条件、到期赎回），公告日、赎回日、登记日、赎回价与数量。

## 本仓库

- 转债三表与日线、复权、每日指标、指数日线已落为 PIT parquet；正式策略从这些原料重算。三表在 `SELECTABLE_DATASETS["macro"]` 里、默认不加载，本臂创建实验时显式选入（参数见宿主实验配置）。
- 唯一总闸是 `available_at <= inference_at`；可见时间规则与转债的历史使用规则（转股价反推、当前状态字段屏蔽）见 `docs/data-documentation.md §3.3`，单位见 `docs/units-reference.md`（沙箱里以 `unit_reference.json` 为准）。
- 正式 ABI、允许的库、`fit(context)`/`context.state_dir`、`models/` 只读挂载见只读 `output/README.md`。
- 离线筛查脚本 `/mnt/tools/screen.py`（只经 `shell` 运行）给截面信号的 rank IC、ICIR、衰减、规模中性 IC、头部十分位超额、换手与可交易性；它只读 `/mnt/snapshot` 决策视图，不替代 Validation。事件型 cohort 与匹配对照普查用自己的脚本在 `/mnt/snapshot` 上做，不碰验证区间。

## 前几轮的相邻臂

- `factor_cs_20260912`（六个截面异象家族）、`platform_20260912`（平台机制）、`corner_cases_20260907`（涨跌停、停牌、ST、新股、除权、解禁日）都在同一只证券上做变换；本包只在「发行人有存续转债」这个池子上与它们相邻，不重做它们的机制。涨跌停当日的可执行性由 corner_cases 负责，本包只汇报拒单。
