# 08:30 字段图

先核对本轮 `data_summary.json`、manifest 和 `unit_reference.json`；`events` 的 `datasets` 必须同时列出 `margin_secs` 与 `margin_detail`，否则本包所有家族「不可测」，不要用其他字段冒充。`margin_detail` 默认加载，`margin_secs` **默认不加载**，靠创建实验时的 `events_datasets` 显式选入。下表是仓库常见合同，某一折仍可能缺表或为空。

## 读法

```text
ev = pd.read_parquet(context.asof_dir + "/events",
        columns=["dataset", "trade_date", "ts_code", "exchange",
                 "rzye", "rqye", "rqyl", "rzmre", "rzche", "rzrqye", "available_at"],
        filters=[("trade_date", ">=", start)])
ev = ev[pd.to_datetime(ev["available_at"]) <= pd.Timestamp(context.inference_at)]
roster = ev[ev["dataset"] == "margin_secs"]        # trade_date, ts_code, exchange
detail = ev[ev["dataset"] == "margin_detail"]      # trade_date, ts_code, rzye/rqye/rqyl/...

daily    = pd.read_parquet(context.asof_dir + "/daily", columns=[...], filters=[("trade_date", ">=", start)])
universe = pd.read_parquet(context.asof_dir + "/universe")   # ts_code、name、list_date、l1_code/l1_name
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；不要把 `asof_dir` 读失败回退到 `snapshot_dir`。事件域文件是各数据集列的并集，同名列在不同数据集里含义不同（`name` 在 `margin_secs` 是名册里的证券简称，在 `margin_detail` 是明细行的简称，两者都不能替代 `universe.name` 判 ST），**先按 `dataset` 过滤再谈字段与单位**；每次读都再按 `available_at <= context.inference_at` 过滤一次。两张表都用 `trade_date`（字符串 `YYYYMMDD`）作日期列，都没有 `ann_date`。

## 两融三表

| 数据 | 本包所需字段与单位 | `available_at` 规则与放行节点 | 08:30 的后果 |
| --- | --- | --- | --- |
| `events.margin_secs` | `trade_date`、`ts_code`、`name`、`exchange`（`SSE`/`SZSE`/`BSE`）。**没有任何数值列**——它是一张纯资格名册，单位注册表把它整表登记为「只有标识符」 | `official_preopen_09_from:trade_date`：交易日当天 09:00；节点 `cn_preopen_margin_secs_backfill_0903`（09:03，2 分钟）+ `cn_preopen_margin_secs_retry_0913`（09:13） | T-1 名册在 T 08:30 可见（09:00 盖章 ≤ T-1 09:03 节点完成），当日名册不可见。纳入/剔除事件只能按 `R_{T-1} \ R_{T-2}` 判定，T 日 09:30 是可见后首个可成交价 |
| `events.margin_detail` | `trade_date`、`ts_code`、`name`、`rzye`（融资余额）、`rqye`（融券余额）、`rzmre`（融资买入额）、`rzche`（融资偿还额）、`rzrqye`（融资融券余额）**均为元**；`rqyl`（融券余量）、`rqmcl`（融券卖出量）、`rqchl`（融券偿还量）**均为股** | `official_next_day_09_from:trade_date`：**次日** 09:00；节点 `cn_preopen_margin_backfill_0905`（09:05）+ `cn_preopen_margin_retry_0915`（09:15） | **最新只到 T-2**：交易日 d 的行在 d+1 09:00 才盖章，晚于 T 08:30 的决策时刻。任何写成 `rzye(T-1)` 的公式都是前视 |
| `events.margin`（可选，本包只作核对） | `trade_date`、`exchange_id`、`rzye`/`rzmre`/`rzche`/`rqye`/`rzrqye`（元）、`rqyl`/`rqmcl`（股） | 同 `margin_detail`：次日 09:00 | 每日只有 3 行（SSE/SZSE/BSE），没有截面。只允许用来核对单位与口径，不得做择时闸门 |

单位不是推断的：单位注册表把 `margin`/`margin_detail` 的 `rzye/rzmre/rzche/rqye/rzrqye` 登记为 `CNY`、`rqyl/rqmcl/rqchl` 登记为 `shares`，`margin_secs` 登记在「无数值列」名单里。本地核对（2025-12-31）：`margin_detail.rzye` 求和 2.511e12 元 vs `margin` 汇总 2.524e12 元，比值 0.9948；`rzye / (daily_basic.circ_mv × 1e4)` 的中位数为 3.85%，把 `circ_mv` 误当元读会得到 385 这种不可能的比值。

## 正股日线与宇宙

| 数据 | 本包所需字段 | 08:30 可见边界与单位 |
| --- | --- | --- |
| 合并日线 `daily` | `open/high/low/close`（元/股）、`vol`（股）、`amount`（元）、`pct_chg`（小数）、`adj_factor`、`turnover_rate`（小数）、`circ_mv`（**元**）、`pe_ttm`/`pb`、`up_limit`/`down_limit`、`is_suspended` | 当日行 17:30 才可见，08:30 只有 T-1 及更早；`adj_factor` 当日 09:30 盖章，同样只能用 T-1。**`circ_mv` 在快照里已从源表的万元乘 1e4 归一到元**（`daily.parquet` 的归一化合同），与 `margin_detail` 的元可以直接相除，不要再乘 1e4 |
| `universe` | `ts_code`、`name`（含 ST 标记）、`list_date`、`l1_code`/`l1_name` | 决策日冻结。上市年龄门槛（家族过滤 F3）与 ST 判定只从这里取，不从 `events` 的 `name` 列取，也不回填今天的状态 |

复权：推断时冻结前复权锚 `anchor = adj_factor(T-1)`，`qfq(t) = raw(t) × adj_factor(t) / anchor`；持有期收益必须在同一个锚下算，跨除权日直接用原始收盘会造出假的事件后收益。已知供应商缺陷：`920627.BJ` 2022 年的 `adj_factor` 在 1.0 与 2.0–2.34 之间反复跳变——本包整体剔除 `.BJ`，不受影响，但不要假定 `adj_factor` 逐日非降。

## 名册的覆盖与陷阱

- **分区完整。** 2020-01-02..2025-12-31 共 1,455 个交易日分区，一天不缺；`(trade_date, ts_code)` 无重复键；`available_at_rule` 全部是 `official_preopen_09_from:trade_date`，没有混入其他规则。北交所切片自 2023-02-13（北交所两融开闸）起每天都在，本 PIT 窗口内**没有**缺交易所切片的日子。数据文档 §4 记录的 `.BJ` 缺口（2025-09 起多轮、2026-07-24 起 12 天）是在 2026-08-11 之后回补的：回放看到的是回补后的完整名册，实盘当时看到的是缺口——这是已接受的口径差异，本包整体剔除 `.BJ` 因此不受影响。
- **名册不是股票池。** 2025-12-31 的 4,297 个代码里既有 ETF/LOF（`510/511/512/513/515/516/517/518/520/551/560/561/562/563/588/589` 等前缀），也有 `.BJ`。2022–2025 的 2,705 次原始新增里只有 1,375 次是 A 股非北交所股票，另有 552 次 `.BJ` 与 778 次 ETF/LOF 等非股票代码。**先经 `universe` 求交集，再做差分。**
- **名册不区分三种资格。** 数据文档 §1.7 明写：标的表不区分担保品、融资买入和融券卖出资格，也不代表券商的实际券源。这是本臂最硬的限制，处理方式见 `families.md` 家族 5。
- **2024-04-01..2024-05-31 是供应商侧上交所切片截断，不是名册变更。** 逐日 SSE 计数：20240329 1,983 → 20240401 1,786 → 20240423 1,957 → 20240424 2,012 → 20240425 1,957 → 20240429 1,785 → 20240513 1,957 → 20240514 2,001 → 20240515 1,957 → 20240516 2,001 → 20240517 1,957 → 20240521 1,775，同期 SZSE 从 1,840 平滑走到 1,829、BSE 从 247 走到 251。20240401 剔除的 251 只与 20240423 新增的 248 只重合 226 只；这批「新增」里 50.8% 在事件日前三个交易日仍有 `margin_detail` 行。**整段声明剔除**。
- **新股入册接近自动。** 2022–2025 上市的 748 只 A 股非北交所新股中，613 只（82.0%）在上市首日进名册、87.7% 在 60 个交易日内进。名册差分里 69% 的「常规腿新增」就是这件事。上市年龄门槛（≥ 120 个自然日）是必须的，否则测的是 IPO 而不是资格放松。
- **真实的老股纳入是批次事件。** 剔除上述全部之后剩 524 次，只落在 16 个不同日期上（399 次在 2022-10-24，其余 125 次分布在 15 个日期，集中在每年三季度）。用它构造日频策略会得到一个大部分时间空仓的账户，这是设计事实不是 bug。

## 明细表的覆盖与陷阱

- **分区完整、无重复键。** 2021-12-01..2025-12-31 共 992 个交易日分区，`(trade_date, ts_code)` 无重复，`available_at_rule` 全部是 `official_next_day_09_from:trade_date`。
- **明细表也含 ETF。** 每日约 3,544 行里 88.1% 是 `stock_basic` 里的股票、4.2% 是 `.BJ`；A 股非北交所约 3,101 只/日。同样先经 `universe` 求交集。
- **只有有余额的名字才出现。** 名册成分股当天没有任何融资融券余额时不出现在 `margin_detail` 里，因此「没有行」不等于「不在名册」。这正是家族过滤 F4 能用它证伪名册差分的原因，但反过来不成立：不能用 `margin_detail` 的成员资格代替名册。
- **`rzye` 从不为空也从不为零**（A 股非北交所样本上 null 率 0.0000、零值率 0.0000），分位数为 p05 3,368 万元、中位 2.30 亿元、p95 16.79 亿元。`rqye` 与 `rqyl` 的零值率都是 21.3%——它们同零同非零，可以互相校验。
- **`rqyl > 0` 的覆盖在 2025 年断档。** 名册成分股中 `rqyl > 0` 的月度占比：2022 年 79%–85%、2023 年 77%–80%、2024 年 85%–91%、2025 年 66%–68%。这与 2024-07 融券保证金比例上调、转融券暂停的时间线一致。做任何涉及融券的分层前，先把这条曲线画出来。
- **`rqyl` 是股数不是金额**，`rqye` 才是金额；两者不能混算，也不能与 `rzye` 直接相加（`rzrqye` 已经是和）。

## PIT 自检清单

写完信号后按这四条自查，任何一条不过就不要看收益：

1. 名册差分的两个端点是不是相邻的**可见**名册日，且最新端点是 `T-1` 而不是 `T`。
2. 融资余额的两个端点是不是都 ≤ `T-2`（本包默认 `T-2` 与 `T-22`）。
3. `circ_mv` 有没有被重复乘 1e4；`rqyl` 有没有被当成金额。
4. 上市年龄、ST、行业是不是取自 `universe` 而不是 `events.name` 或今天的 `stock_basic` 状态。
