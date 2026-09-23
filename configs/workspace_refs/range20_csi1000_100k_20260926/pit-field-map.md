# 本轮数据与 08:30 字段图

本文是本包可用数据、可见时点、单位与陷阱的**唯一权威表述**。数据集清单、行数、日期覆盖与单位以本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json` 为准：先核对它们，再写阈值；`snapshot` view 与 `valid` view 分别核对，**不得由前者推断后者**。

**开工第一件事**：核对四件事——`macro.datasets` 里有没有 `index_weight`（没有它就没有这个臂的股票池），其中 `index_code = 000852.SH` 的截面在不在、每张是不是 1000 只；`daily` 带不带 `high` / `low` / `pre_close` / `close`；`universe` 带不带 `name` 与 `l1_name`；以及运行事实里的 **`benchmark_index`** 与 `lib/knobs.py` 的 `INDEX` 是否一致（必须是 `000852.SH`）。**缺任何一块都不得静默回退**到别的池子或别的指数——那是另一个登记；起步包在读不到可见截面时直接报错，就是为了让这种情况在第一次 `smoke_backtest` 就暴露。

## 本臂的数据集选择（创建参数）

与 20260925 轮两条新鲜书臂完全相同：20260919 轮的 `fundamental_datasets` / `macro_datasets`（含 `index_weight`）/ `events_datasets` / `text_datasets`，视图种子 `data/pit_views_seed_research_20260919`，不挂分钟域。**本臂只读 `daily`、`macro.index_weight` 与 `universe` 三处**；其余域挂着，但分数里加进任何别的列就是另一条臂（`refs/families.md` 的禁止）。

## 每个回放年自己的输入窗（运行事实 `research_geometry.years[].input_window`）

每个研究年的回放从**它自己的**决策视图起步，那张视图从该年开始前一天往回 60 个月；会话挂的 `/mnt/snapshot` 是研究期末那一张。

| 视图 | 输入窗 | `index_weight` 在其中最早的截面 | 对本臂意味着什么 |
|---|---|---|---|
| Y1 回放（2021-07..2022-06） | 20160701..20210630 | 2020-01（数据湖本身从 2020-01 起，成分标签的下限） | 分数只要 20 根日线，第一个复核日就有完整的分数与截面 |
| Y2 / Y3 / Y4 回放 | 20170701..20220630 / 20180701..20230630 / 20190701..20240630 | 2020-01 | 同上 |
| `/mnt/snapshot`（研究期末） | 20200701..20250630 | 2020-07 | 离线门从 2021-07 起打分，20 根日线的窗不会被左端截断；`r2` 的 60 根也不会 |

本臂不训练，所以没有训练窗下限这个问题；它只有一条：**第一张截面之前的日期没有成分**，别的臂在 2020-01 之前建训练行会碰到它，本臂不会。

## 日线 `daily`

| 列 | 含义 | 单位 |
|---|---|---|
| `high` / `low` / `close` | 当日最高 / 最低 / 收盘 | 元 / 股，**未复权** |
| `pre_close` | 交易所给的昨收，已按当日除权除息调整 | 元 / 股 |
| `up_limit` / `down_limit` | 当日涨跌停价 | 元 / 股 |
| `adj_factor` | 累计复权因子（离线门算收益要用，起步包不用） | — |
| `is_suspended` | **只标盘中临停**；整天停牌的名字当天在 `daily` 里没有行 | — |

- **可见性**：`daily` 没有 `available_at` 列，行按 17:30 盖章；as-of 视图里只有决策时可见的行，所以 08:30 决策读到的最新一根是 **T-1**。离线重建里「日期 t 的分数」就是 t 之后那个交易日的决策读到的分数，`/mnt/tools/screen.py` 从 t+1 的开盘算前向收益，正好对上。
- **振幅用 `pre_close` 做分母**：同一天、同一口径，除权日的跳空不会被读成振幅；`pre_close` 缺失或 ≤ 0 的那一根整根丢掉，不当成安静的一天。
- **涨跌幅限制不同，振幅天然不同**：主板 ±10 %、创业板（300 / 301）与科创板 ±20 %、ST ±5 %。这个分数因此天然低配创业板、偏好 ST——科创板与 ST 在池里被剔掉，创业板没有，书里创业板的占比要逐次汇报（`refs/families.md` 的 P5）。

## 成分截面 `macro.index_weight`

```text
sec = pd.read_parquet(
    context.asof_dir + "/macro",
    columns=["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"],
    filters=[("dataset", "=", "index_weight"), ("index_code", "=", "000852.SH"),
             ("trade_date", ">=", start)],
)
```

| 事项 | 规则 |
|---|---|
| 指数键 | `index_weight` 的指数在 `index_code`、成分在 `con_code`，`ts_code` 为空。**`index_daily` 正相反**：指数在 `ts_code`、`index_code` 整列为空——本臂不读 `index_daily`；会话若为别的目的读它，**必须按 `ts_code` 过滤**，按 `index_code` 读安静地返回 0 行，不过滤则七只指数的日线混在一起 |
| 成分的代码列 | `con_code`，跨域 join 要自己改名 |
| 可见性 | 行级 `available_at`，`contract_1730_from:trade_date`——按自己那个交易日 17:30 盖章，08:30 决策看不到当天那张 |
| 频率 | 每指数每月一张，日期是当月最后一个交易日，2020-01 起无缺口；中证 1000 每张恒为 1000 只 |
| `weight` | 百分数，一张截面求和 100 ± 0.05；本包只用成员资格，权重只用来报书覆盖了多少指数 |
| 基准 | 裁决的中性化、β、跟踪误差与零技能面板的成员侧都对 `benchmark_index` 做；用别的指数建书而仍按它报是 `refs/families.md` 的点名禁止 |

**只有月末截面**：两张之间的真实调整看不见，复核日拿到的是上一张的名单（`refs/references/index-membership.md`）。**每一个决策日都按那一天可见的最新截面判定成分**；离线重建也一样，不能用最新一张去标注历史。

## `universe`

`name`（ST / 退 过滤）与 `l1_name`（申万一级，行业上限与 P2）。回放里它是回放槽锚点那一天的表；2021-12-13 从 SW2014 切到 SW2021，一级行业的个数与名字会变，跨决策日比较行业名单是错的；为空的名字落在 `未分类`（也算一个行业）。宿主的 `top_industry_weight` 按时间加权读，P2 要逐次读。

## 本轮要读的列

| 域 | 列 | 单位 | 行级 `available_at` | 用途 |
|---|---|---|---|---|
| `daily` | `ts_code`、`trade_date`、`high`、`low`、`pre_close`、`close` | 元 / 股未复权 | 无列，最新一行 = T-1 | 分数、池、买单定价、持仓估值 |
| `macro.index_weight` | `con_code`、`trade_date`、`weight` | 百分数 | 有，17:30 | 成员资格、覆盖度报告 |
| `universe` | `name` / `l1_name` | — | 无列，按回放槽锚点口径 | ST / 退过滤、行业上限、P2 |

## 读法与可见性（通则）

- `asof_dir` 下每个域是 parquet parts 目录（传目录名，不加 `.parquet`），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。
- 带行级 `available_at` 的域判可见只看它（`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `trade_date` 推；`daily` 与 `universe` 没有 `available_at` 列，按不存在的列过滤会报错。
- 起步包每次读取都带 `columns=`、`filters=` 与日期窗口，不引入跨调用缓存，冷 worker 与热 worker 出单一致。

## `snapshot` 只有一个时点口径

`/mnt/snapshot` 是研究期末那一天的决策视图：`universe.parquet` 里的名称、ST 状态、`l1_name` 都是那一天的。离线重建历史决策时，名称与行业会被灌回过去——所以离线门不按名称过滤 ST（已知的小偏差，`refs/sources.md` 量过：池里按 5 % 涨跌幅判为 ST 的名字逐年平均 0.2–1.0 只，顶端 50 只里平均 0.1 只），行业读数只作相对参考。

## 陷阱

- **成分的代码列是 `con_code`**；**`index_daily` 按 `ts_code` 过滤**，它的 `index_code` 为空。
- **用最新一张截面标注历史决策日**：中证 1000 在研究期里轮换的名字远多于 1000 只，漏掉的历史成分多半是跌出去的。
- **把停牌日当成振幅为零的一天**：停牌日没有日线；按日历取 20 天的窗会把停牌前后的名字算得更「安静」。起步包按每只股票自己的最近 20 根取窗。
- **一手与整手**：10 万元 12 席（座位约 8,083 元，收盘价 ≤ 80.8 元才买得起）上一手买不起的成分逐年 0.9–3.3 %（`refs/sources.md`）；买不起只影响买入，**已持有的名字不得因此强卖**；买单按最近整手四舍五入再按现金截断。
- **印花税在研究期内切换**（卖出 10 bp → 5 bp，2023-08-28）。
- **复核节奏不得依赖模块级计数器或跨调用缓存**：到期判断拿最新可见交易日与决策日比自然月；`c_shuf` 的打乱只用决策日做种子。
