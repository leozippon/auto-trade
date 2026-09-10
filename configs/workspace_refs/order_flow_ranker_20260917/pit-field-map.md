# 08:30 字段图

先核对本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json`：`events.datasets` 必须列出 **`intraday_flow`** 与 `moneyflow`，两者缺一，对应的特征面整块不可构造。`intraday_flow` 默认不加载，靠创建实验时的 `events_datasets` 显式选入；`moneyflow` 默认加载。缺 `intraday_flow` 时应立刻 `report_issue` 并以 `no_edge` 如实弃权，**不得**用日频量价拼一个「近似 OFI」顶替。

本臂**不挂分钟域**：`include_intraday` 为假，`intraday_1min` 是零行文件，`historical_minutes_available` 为假，因此 09:30 与 15:00 之外的 `execute_at` 一律 `missing_execution_price` 拒单。本包的执行时点只用这两个。

## 读法

```text
flow     = pd.read_parquet(context.asof_dir + "/events",
                           columns=["dataset","available_at","ts_code","trade_date",
                                    "ofi_1d","ofi_amt_1d","nret","zero_share","sealed_limit"],
                           filters=[("dataset", "=", "intraday_flow"), ("trade_date", ">=", start)])
flow     = flow[pd.to_datetime(flow["available_at"], utc=True) <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
mf       = pd.read_parquet(context.asof_dir + "/events",
                           columns=["dataset","available_at","ts_code","trade_date",
                                    "net_mf_amount","buy_sm_amount","sell_sm_amount", ...],
                           filters=[("dataset", "=", "moneyflow"), ("trade_date", ">=", start)])
daily    = pd.read_parquet(context.asof_dir + "/daily",
                           columns=["ts_code","trade_date","high","low","close","pre_close","vol",
                                    "amount","pct_chg","adj_factor","turnover_rate","circ_mv",
                                    "up_limit","is_suspended"],
                           filters=[("trade_date", ">=", start)])
universe = pd.read_parquet(context.asof_dir + "/universe")   # ts_code、name、list_date、l1_code
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名，pandas 自己拼接分片），`snapshot_dir` 下是平铺文件；`context.asof_dir + "/events.parquet"` 不存在，读失败**不得**回退 `snapshot_dir`（那会把停在决策时刻的冻结快照当成滚动视图，是 PIT 违规而不是恢复）。事件域是多个数据集的列并集：同名列在不同数据集里含义不同，**必须先按 `dataset` 过滤再谈字段与单位**，并把 `dataset` 列进 `columns` 以便审计。

## `intraday_flow`：本臂的 A 面数据源

由 `scripts/data/build_intraday_flow.py` 就地把每个分钟按日分区约简成每个股票日一行，聚合逻辑在 `src/autotrade/environment/data/intraday_flow.py`，只用当天自己的分钟 Bar。

| 列 | 含义 | 单位 |
|---|---|---|
| `ts_code` / `trade_date` | 业务键，一天一只一行 | — |
| `ofi_1d` | 成交量加权的 tick-rule 带符号失衡，`Σ sign(close_m − close_{m−1})·vol_m / Σ vol_m` | 无量纲，[−1, 1] |
| `ofi_amt_1d` | 同一构造改用成交额加权 | 无量纲，[−1, 1] |
| `nret` | 当天实际产生的带符号分钟数；完整交易日为 **239** | 计数 |
| `zero_share` | 当天 241 根 Bar 中零成交的比例 | 小数 |
| `sealed_limit` | `zero_share >= 0.5`，当天基本封板 | 布尔 |
| `available_at` / `available_at_rule` | 交易日 **17:30**，规则名 `contract_1730_from:trade_date` | — |

**行级可见规则**：内容完全由当天自己的分钟 Bar 决定，因此与 `daily` 共用收盘合同——交易日 17:30 盖章，晚间节点放行，**08:30 决策能看到的最新一行是 T-1**。判可见永远只看 `available_at`，不要从 `trade_date` 推。

**覆盖**：`20200102` 起，逐交易日一个分区。**当前终点是 `2026-08-18`**，与分钟按日层同步（`data/raw/daily` 已到 2026-09-09，中间 15 个交易日的分钟还没落库）。Held-out 若定在 `20260601..20260930`，这条数据在 8 月 18 日之后就没有行，`ofi_21` 会在 Held-out 后段整段失效——**建种子前必须确认覆盖到 Held-out 终点**，补不齐就把终点收到最后一个有行的交易日并写明这是数据边界。

**两条阈值不在数据层，在策略里**：`nret` 与 `sealed_limit` 只作为诊断随行提供，因为阈值是研究口径而不是数据事实。本包沿用参考研究的口径：保留 `nret >= 200` 且 `zero_share < 0.5` 的股票日，两条剔除率逐折汇报。实测（2026-03-31，5,179 行）：`nret < 200` 剔除 **0.00%**，`sealed_limit` 剔除 **0.04%**；跨年抽样的封板剔除率是 0.3%–3.4%/日。滚动窗口同样不预计算，21/5 日均值与上涨日/下跌日分腿都在策略里形成。

**已经由数据层处理掉、不必再做的**：`.BJ` 已剔除（它有 15:01–15:30 盘后 Bar，会同时破坏「午休是唯一断点」与「241 根分母」两条不变量）；整天零成交的股票日已丢弃；无成交分钟对分子分母的贡献恒为 0，因此停牌分钟天然中性，不需要额外过滤；09:30 集合竞价那根没有前一根，只进分母不进分子；11:30 → 13:01 不产生收益；跨日不产生收益。

## 逐表可见边界

| 数据集 | 行级 `available_at` 规则 | 08:30 能看到的最新行 |
|---|---|---|
| `events.intraday_flow` | `contract_1730_from:trade_date`（17:30） | **T-1** |
| `events.moneyflow` | `official_19_from:trade_date`（19:00），晚间节点 23:35 放行 | **T-1** |
| `daily` | 交易日 17:30 | **T-1** |
| `daily_basic` | 交易日 18:00 | **T-1** |
| `adj_factor` | 交易日 09:30 | **T-1** |
| `stk_limit` / `suspend_d` | 交易日 08:45 | **T-1**（本包只用 T-1 的 `up_limit` 作拒单估计） |
| `universe` | 决策日冻结 | 当日（`list_date <= 决策日`、未退市） |
| `macro.index_daily` | `contract_1730_from:trade_date`（快照格式 v10 起） | **T-1**（不是 T-2；月度/季度宏观与 `cb_call` 仍是 T-2） |

## 单位

| 位置 | 字段 | 单位 |
|---|---|---|
| `events.intraday_flow` | `ofi_1d`、`ofi_amt_1d`、`zero_share` | 无量纲比值；`nret` 是计数、`sealed_limit` 是布尔 |
| `events.moneyflow` | `*_amount`、`net_mf_amount` | **万元**（本包所有列都是对同一 `gross` 的比值，单位约掉） |
| 快照 `daily` | 价 元/股；`vol` 股；`amount`、`circ_mv` 元；`pct_chg`、`turnover_rate` 小数 | 已归一化 |
| 原始湖 `data/raw/daily` | `vol` **手**、`amount` **千元** | 离线普查直接读它会差 100 倍 / 1000 倍 |

## 陷阱

- **`intraday_flow` 是事件域的一个 `dataset`，不是独立的域**：不带 `dataset` 过滤就读会把 `moneyflow` 的行混进来（两者都有 `trade_date`）。
- **不要把 `nret`/`sealed_limit` 当成缺失指示**：它们是诊断列，值恒有效；过滤是研究选择，必须显式做并汇报剔除率。
- **`sealed_limit` 与 `zero_share >= 0.5` 是同一条规则**，用哪一个都行，但不要同时用两个不同的阈值。
- **滚动窗口按股票自己的可用交易日位置计**，不是自然日；被过滤掉的股票日不占位置。因此可见历史越长，窗口越靠近声明口径——回放开局与回放后期的窗口成色不同，第 0 轮普查要报每个调仓日的窗口成色（`min_periods` 触发比例）。
- **整手与价格上限**：买单声明 100 股整数倍；T-1 收盘价 ≤ 30 元使一手 ≤ 3,000 元、不超过单只预算（约 6,300 元）的一半。`.BJ` 在 `intraday_flow` 里本来就没有；`688`/`689` 起买 200 股，本包在股票池里剔除。
- **2026-07-06 起日线量额含 15:05–15:30 盘后定价成交，分钟没有对应 Bar**。这个断点落在本轮 Held-out 区间内。`ofi_1d` 只用分钟自身的量，分子分母同源，**不受影响**；但任何用日线量做分母去比对的检查在那一段会失真。
- **`is_suspended` 只标 `suspend_d` 的 `S` 行（盘中临停），不标复牌**；整天停牌的股票在 `daily` 里根本没有行。真停牌按日线缺行判断。
- **复权锚**：推断时冻结 `anchor = adj_factor(最后一个可见交易日)`，`qfq(t) = raw(t) · adj_factor(t) / anchor`；标签与全部价格类特征在同一个锚下算。`ofi_1d` 与复权无关（它只用同一天内相邻分钟收盘价的**符号**，除权只发生在跨日），但标签必须复权。
- **`moneyflow` 的八列净额恒为 0**：见 `families.md`。用它们建对照时必须带上 `net_mf_amount`。
