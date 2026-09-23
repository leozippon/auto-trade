# 本轮数据与 08:30 字段图

本文是本包可用数据、可见时点、单位与陷阱的**唯一权威表述**。数据集清单、行数、日期覆盖与单位以本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json` 为准：先核对它们，再写阈值或跨域组合；`snapshot` view 与 `valid` view 分别核对，**不得由前者推断后者**。

**开工第一件事**：核对五件事——`macro.datasets` 里有没有 `index_weight`（没有它就没有这个臂的股票池）与 `index_daily`（载体腿的残差标签要它）；`events.datasets` 里**两张前十大股东表 `top10_holders` 与 `top10_floatholders` 是否都在**（只有一张就是一个缺一半公司的信号，见下文）；`unit_reference.json` 对两张表 `hold_float_ratio` 与 `hold_amount` 的登记（百分数 / 股）；`universe` 带不带 `l1_name`（P2）；以及实验事实里的 **`benchmark_index`** 与 `lib/index.py` 的 `INDEX_CODE` 是否一致。**缺任何一块都不得静默回退**到别的池子、别的表或只用一张表再按同一套口径报数——那是另一个登记。起步包的 `lib/holdings.py` 在窗口里读不到任一张表的可见行时直接报错，就是为了让这种情况在第一次 `smoke_backtest` 就暴露。

## 本臂的数据集选择（创建参数，运营者前置条件）

| 域 | 本臂需要 | 状态 |
|---|---|---|
| `events_datasets` | 20260919 轮的 17 个（其中已有 `top10_floatholders`）**再加 `top10_holders`** | `top10_holders` 默认不加载，此前没有任何一条臂选过它 |
| `macro_datasets` | 20260919 轮的 20 个（含 `index_weight`） | 不变 |
| `fundamental_datasets` / `text_datasets` | 20260919 轮的 10 个 / `["report_rc"]` | 不变。本包不读这两个域 |
| `include_intraday` | False | 不挂分钟域 |

这条选择改变了快照契约，所以它有一棵自己的预建视图种子（同一个研究发布，只多这一个数据集）；拿 20260919 那棵种子跑本臂会在创建前检查就被拒。

## 每个回放年自己的输入窗（运行事实 `research_geometry.years[].input_window`）

每个研究年的回放从**它自己的**决策视图起步，那张视图从该年开始前一天往回 60 个月；会话挂的 `/mnt/snapshot` 是研究期末那一张。

| 视图 | 输入窗 | 两张股东表在其中最早的公告 | 对本臂意味着什么 |
|---|---|---|---|
| Y1 回放（2021-07..2022-06） | 20160701..20210630 | 2020-01（数据湖本身从 2020-01 起） | 第一个复核日（2021-07-01）已经能读到 2020Q1 以来的季报，`chg` 的分母有满 4 个季度 |
| Y2 / Y3 / Y4 回放 | 20170701..20220630 / 20180701..20230630 / 20190701..20240630 | 2020-01 | 同上 |
| `/mnt/snapshot`（研究期末） | 20200701..20250630 | 2020 年中（窗口左端） | 离线重建研究期最初几个月时，`chg` 的分母少一两个季度（`references/offline-screen.md`） |

载体腿（`c_base`、`n2`）另有一条**训练窗下限**：成分截面 `index_weight` 从 2020-01 才有，第一张截面之前的日期没有成分、因而没有训练行，所以第一研究年的 `fit` 只有约一年半的带标签训练日，那一年的 `fit` 秒数与读数都不能外推。北向分数不训练，没有这条下限，但它的 `chg` 在 2020Q1 之前没有季度可比。

## 北向持股：两张前十大股东表（`events.top10_holders` / `events.top10_floatholders`）

事件域是**多个数据集的列并集**：先按 `dataset` 过滤，再谈字段与单位，`dataset` 既要进下推过滤也要进列投影。两张表同一个结构，一行是一次公告里的一名股东：

| 列 | 含义 | 单位 |
|---|---|---|
| `ts_code` | 公司 | — |
| `end_date` | 报告期（持股的时点） | YYYYMMDD |
| `ann_date` | 公告日 | YYYYMMDD |
| `holder_name` / `holder_type` | 股东名称 / 类型 | — |
| `hold_amount`、`hold_change` | 持股数、较上期变动 | **股** |
| `hold_ratio` | 占总股本 | **百分数** |
| `hold_float_ratio` | 占流通股 | **百分数**（3.49 = 3.49 %） |

- **可见性**：行级 `available_at` = `ann_date` 当天 23:59:59（规则名 `conservative_date_eod`，逐行成立）。回放按宿主的晚间披露节点放行（每个自然日 23:05 启动、23:15 可读），所以**公告日 A 的行最早被 A + 2 个自然日当天或之后第一个交易日的 08:30 决策读到**；每个回放槽的第一个决策日例外，它读的冻结首片带着锚点当天及之前的全部行。会话能直接核对的只有 `available_at` 列本身，放行节点是宿主规则；离线重建一律按「A + 2」，不按 `available_at <= 决策时刻`（后者早一天）。
- **`trade_date` 为空。** 这两张表的行在事件域里没有交易日，窗口只能按 `ann_date` 取：按 `("trade_date", ">=", …)` 过滤会把它们整个过滤掉，而且不报错。起步包按 `("ann_date", ">=", start)` 下推。
- **只有季末是报告期。** `end_date` 里混着非季末日期（招股书、临时公告等），在研究期末视图里占两张表各 5.9 % / 6.6 % 的行；做季度分期与季度差分前只留 0331 / 0630 / 0930 / 1231，否则会得到错位的「持仓变化」。
- **单表不完整，只能用并集。** 供应商对若干报告期只返回了部分公司：只读流通股东表，2020-12 起有九个报告期沪深 300 成分里带北向行的只有 6–18 只；总股东表另有自己的薄期（2020-09、2022-09 为 0 只）。**两表并集**每个季末覆盖 268–291 只成分（逐季表在 `sources.md`）。同一公司、同一季末两张表都有北向行时，`hold_float_ratio` 逐位一致；起步包优先取流通股东表、缺了用总股东表补。
- **哪一行是北向。** 陆股通 A 股登记在**香港中央结算有限公司**名下；**香港中央结算（代理人）有限公司 / HKSCC NOMINEES 持有的是 A+H 公司的 H 股**，与北向无关。起步包的规则：名字去掉空白后以「香港中央结算」开头，且不含「代理」「H股」「NOMINEE」（不分大小写）。它收进的变体（「…有限公司(A股)」「(陆股通)」「(沪股通)」「香港中央结算公司」）是同一个股东的不同写法；各写法的行数在 `sources.md`。**北向行的 `holder_type` 是「一般企业」**，按股东类型筛外资会一行都找不到。
- **截断**：只有北向排进前十大时才有这一行。一家公司发布了该季报告而其中没有北向行，意思是北向在第十名之下——**是缺失，不是零**。起步包把它记为缺失，分数里取中性秩。
- **晚到**：北向行的公告日在季末之后约 22–31 天（一季报与三季报）、约 45–62 天（半年报）、约 81–120 天（年报）（10–90 分位）；复核日读到的持仓通常已经是一到四个月前的。
- **一个季末只有一次公告**：同一（表、公司、季末）出现多个公告日的情况在研究期末视图里几乎没有；起步包仍按「可见的最新公告日」取值。

## 成分截面 `macro.index_weight` 与基准日线 `macro.index_daily`

```text
sec = pd.read_parquet(
    context.asof_dir + "/macro",
    columns=["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"],
    filters=[("dataset", "=", "index_weight"), ("index_code", "=", INDEX_CODE),
             ("trade_date", ">=", start)],
)
bench = pd.read_parquet(
    context.asof_dir + "/macro",
    columns=["dataset", "available_at", "ts_code", "trade_date", "open", "close"],
    filters=[("dataset", "=", "index_daily"), ("ts_code", "=", INDEX_CODE),
             ("trade_date", ">=", start)],
)
```

| 事项 | 规则 |
|---|---|
| 两张表的指数键**相反** | `index_weight` 的指数在 `index_code`、成分在 `con_code`，`ts_code` 为空；`index_daily` 的指数在 **`ts_code`**，**`index_code` 整列为空**。按 `index_code` 读 `index_daily` 安静地返回 0 行；不按 `ts_code` 过滤则七只指数的日线混在一起 |
| 成分的代码列 | `con_code` 不是 `ts_code`，跨域 join 要自己改名 |
| 可见性 | 两者都是行级 `available_at`，`contract_1730_from:trade_date`——按自己那个交易日 17:30 盖章，08:30 决策看不到当天那张截面 |
| 频率 | 截面每指数每月一张，日期是当月最后一个交易日，2020-01 起无缺口 |
| `weight` 的单位 | 百分数，一张截面求和 100 ± 0.05；本包只用成员资格，不用权重 |
| 基准 | 裁决的中性化、β、跟踪误差与零技能面板的成分换名都对 `benchmark_index` 做；用别的指数建书而仍按它报跟踪误差是 `families.md` 的点名禁止 |

**只有月末截面**：两张之间的真实调整看不见，复核日拿到的是上一张的名单（`references/index-membership.md`）。**每一个决策日都按那一天的截面判定成分**（载体腿的每个训练日也一样）；第一张截面之前的日期没有成分。

## 本轮要读的列

| 域 | 列 | 单位 | 行级 `available_at` | 用途 |
|---|---|---|---|---|
| `events.top10_holders` / `events.top10_floatholders` | `ts_code`、`ann_date`、`end_date`、`holder_name`、`hold_float_ratio` | 百分数（占流通股） | 有，`conservative_date_eod` | `lvl`、`chg` |
| `daily` | `open`/`high`/`low`/`close`、`adj_factor`、`vol`/`amount` | 元/股未复权、累计复权因子、股/元 | 无列，最新一行 = T-1 | 池、买单定价；载体腿的 Alpha158 与标签 |
| `macro.index_weight` | `con_code`、`trade_date`、`weight` | 百分数 | 有，17:30 | 成员资格、覆盖度报告 |
| `macro.index_daily` | `open` / `close` | 指数点 | 有，17:30 | 载体腿的残差标签与 β |
| `universe` | `name` / `l1_name` | — | 无列，整表按回放槽锚点口径 | ST/退过滤、P2 |

本包**不读** `fundamentals`、`text`，也不读事件域的其他数据集（包括 `stk_holdernumber` 与 `stk_holdertrade`）：本臂要量的是北向这一块信息，多带别的列会让「赢了是因为什么」说不清。

## 读法与可见性（通则）

- `asof_dir` 下每个域是 parquet parts 目录（传目录名，不加 `.parquet`），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。
- 带行级 `available_at` 的域，判可见只看它（`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `trade_date` 推；`daily` 与 `universe` 没有 `available_at` 列，按不存在的列过滤会报错。
- 起步包每次读取都带 `columns=`、`filters=` 与日期窗口（股东表按 `ann_date`），不引入跨调用缓存，冷 worker 与热 worker 出单一致。

## 申万一级行业是点内的

`universe.l1_name` 按**决策日**取成分（`in_date ≤ 决策日 < out_date`），2021-12-13 从 SW2014 切到 SW2021，研究期内一级行业的个数与名字会变，跨决策日比较行业名单是错的；为空的名字落在 `未分类`。宿主的行业归因是时间加权的，P2 要逐次读。

## `snapshot` 只有一个时点口径

`/mnt/snapshot` 是研究期末那一天的决策视图：`universe.parquet` 里的名称、ST 状态、`l1_name` 与在册名单都是那一天的；事件域只从 2020 年中起（上面的输入窗表）。**不要拿它原样离线重建历史某一天的决策**——名称、ST、行业会被灌回过去，股东表会被读早一天（`available_at` 与放行节点的差），早年的 `chg` 分母会少季度。离线重建只用来查构造与方向，做法在 `references/offline-screen.md`。

## 单位（最容易错的几处）

| 字段 | 单位 |
|---|---|
| `top10_*.hold_float_ratio` / `hold_ratio` | **百分数**；本包的 `lvl` 就是这个百分数，`chg` 是它对自身均值的比值减一，无量纲 |
| `top10_*.hold_amount` / `hold_change` | **股**（不是手、不是万股） |
| `macro.index_weight.weight` | **百分数**，÷ 100 才是组合权重 |
| `macro.index_daily.pct_chg` | **百分数**；`open`/`close` 是指数点 |
| `daily` 的价格 | 元/股，未复权；`adj_factor` 是累计复权因子 |
| `daily.vol`、`amount` | 股、元（已从原始湖的手、千元归一） |

## 陷阱

- **只读一张股东表**：在薄期里把一半的成分读成「北向不持有」，而这种错误在下游只表现为分数的覆盖变低，不报错。两张表都选中、都读、取并集。
- **把 H 股托管行当北向**：A+H 公司的「香港中央结算（代理人）」持有的是 H 股，常常排第一，水平与变化都与陆股通无关。
- **把「报告里没有北向行」当零**：那是截断，不是零持股；零会让这些名字全体落到分数最底端，是一次静默的规模 / 流动性押注。
- **按 `trade_date` 取窗口**：股东表没有交易日，窗口必须按 `ann_date`。
- **用 `available_at <= 决策时刻` 离线重建**：比回放早一天读到公告；回放是 A + 2。
- **成分的代码列是 `con_code`**；**`index_daily` 按 `ts_code` 过滤**，它的 `index_code` 为空。
- **`is_suspended` 只标盘中临停**，整天停牌的名字在 `daily` 里没有行；用 T-1 有无行判断可交易。
- **一手与整手**：100 万元 50 席上一手买不起的成分约 2–7 %（随年份的股价变，`sources.md`）。买不起只影响买入，**已持有的名字不得因此强卖**；买单按最近整手四舍五入再按现金截断。
- **印花税在研究期内切换**（卖出 10 bp → 5 bp）。
- **复核节奏不得依赖模块级计数器或跨调用缓存**：到期判断拿最新可见交易日与决策日比自然月；`c_shuf` 的打乱只用决策日做种子，冷热 worker 逐字节相同。
