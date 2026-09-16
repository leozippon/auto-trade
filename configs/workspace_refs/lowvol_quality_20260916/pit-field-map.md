# 08:30 字段图

先核对本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json`：`fundamentals.datasets` 必须列出 `income_vip`、`cashflow_vip`、`balancesheet_vip`（缺任一，质量筛子与 `c_qual` 对照都不可构造，本臂只剩一个不成形的低波打分，按 `families.md` 诚实终止）；`macro.datasets` 必须列出 `index_daily`（缺它时排名列按登记变体退为总波动，买单元数据 `vol_basis` 写 `total`，结果笔记必须写明——这时本臂排的已经不是残差波动了）。本文是本包可见时点、单位、去重规则与陷阱的**唯一权威表述**。

## 读法

```text
daily = pd.read_parquet(context.asof_dir + "/daily",
                        columns=["ts_code","trade_date","close","pct_chg","amount","adj_factor",
                                 "turnover_rate","circ_mv","pe_ttm","is_suspended"],
                        filters=[("trade_date",">=",start)])                       # 220 个自然日
stmts = pd.read_parquet(context.asof_dir + "/fundamentals",
                        columns=["dataset","available_at","ts_code","end_date","ann_date","report_type",
                                 "n_income_attr_p","n_cashflow_act","total_assets"],
                        filters=[("dataset","in",["income_vip","cashflow_vip","balancesheet_vip"]),
                                 ("ann_date",">=",start)])                          # 700 个自然日
index = pd.read_parquet(context.asof_dir + "/macro",
                        columns=["dataset","available_at","ts_code","trade_date","pct_chg"],
                        filters=[("dataset","=","index_daily"),("ts_code","=","000300.SH"),
                                 ("trade_date",">=",start)])                       # 220 个自然日
universe = pd.read_parquet(context.asof_dir + "/universe")   # ts_code、name、list_date、l1_code
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。`fundamentals` 与 `macro` 都是多个数据集的列并集：同名列在不同数据集里含义与单位不同（`pct_chg` 在 `daily` 是小数、在 `macro.index_daily` 是百分数；`n_income` 在 `income_vip` 含少数股东损益），**必须先按 `dataset` 过滤再谈字段与单位**。两类表要分开：`fundamentals` 与 `macro` 的行带**行级 `available_at` 列**，判可见只看它（字符串，`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `ann_date`/`trade_date` 推；`daily` 与 `universe` **没有 `available_at` 列**（决策视图的 `daily.parquet` 30 列，按它过滤会 KeyError），它们的可见性由 as-of 视图本身给定——模板 `output/README.md`「Reading PIT data」：冻结首片只含决策时刻之前可见的行，后续片只含之后发布的行，所以 08:30 决策时 `daily` 的最新一行就是 T-1，不需要也不能再按时间戳过滤。

## 逐表可见边界

| 数据集 | 本包所需字段与单位 | 行级 `available_at` 规则 | 08:30 能看到的最新行 |
|---|---|---|---|
| `fundamentals.income_vip` | `ts_code`、`end_date`、`ann_date`、`report_type`（只用 `"1"` 合并报表）、`n_income_attr_p`（**元**，年初至今累计） | `source:f_ann_date_or_ann_date` 18:00 | 公告日 ≤ T-1 的报表 |
| `fundamentals.cashflow_vip` | 同上键；`n_cashflow_act`（**元**，经营活动现金流量净额，年初至今累计） | 同上 | T-1 |
| `fundamentals.balancesheet_vip` | 同上键；`total_assets`（**元**，期末时点值） | 同上 | T-1 |
| `macro.index_daily` | `ts_code = 000300.SH`、`trade_date`、`pct_chg`（**百分数**，÷ 100 后才是日收益） | `contract_1730_from:trade_date` 17:30 | T-1 |
| `daily` | `close`（元）、`pct_chg`（小数）、`amount`（元）、`circ_mv`（**元**，已归一化，池底就排它）、`turnover_rate`（小数）、`adj_factor`、`pe_ttm`（倍）、`is_suspended` | **无 `available_at` 列**；可见性由 as-of 视图隐含 | 最新一行 = T-1 |
| `universe` | `name`（ST 筛选）、`list_date`、`l1_code` | **无 `available_at` 列**；决策日冻结 | 当日 |

原始湖 `data/raw/daily` 的 `amount` 是**千元**、`circ_mv` 是**万元**；离线普查直接读原始湖会差 1000 倍 / 1 万倍，快照里已归一化为元。**池底排的就是 `circ_mv`，所以这一处单位错会静默地把池底截反。** 快照决策视图里三张报表只含 `report_type = "1"` 的行，过滤仍要写，别的视图不保证。

## 去重与版本规则

- 同一 `(ts_code, end_date)` 在每张报表里可能有多个版本（首次公告、修订、年报重述）：**每张表各取 `available_at` 最早的一版**，重述版永远不是事件，也永远不替换状态。
- 三张表按 `(ts_code, end_date)` 内连接；事件的可见时点取三者中**最晚**的 `available_at`（三表齐全才算可见；通常同一天）。
- 最新报表 = 每个名字 `available_at` 最新的一份（**按时间戳取，不按年龄并列取**：窗口之前公告的报表年龄全部并列，按行序打破并列会取到最旧的一份），年龄按可见交易日历计：报表公告日（按 `available_at` 换算到上海时间的日期）在日历中的位置（非交易日取其后第一个交易日）到最新可见交易日的距离；`daily` 读 220 个自然日，长于 130 个交易日的年龄上限，窗口之前公告的报表一律判过期。年龄 > 130 个交易日即无信号，名字离开股票池——停止披露的公司退出，不携带过期值。
- 旧报告期的重述不是事件：首版早于读取窗口的重述在窗口里看起来就是首版，所以报告期结束日必须在公告日前 400 个自然日内，否则丢弃。
- 年初至今流量按季度序号年化（`× 4 / qn`）：三月末 ×4、六月末 ×2、九月末 ×4/3、十二月末 ×1。同一决策日上不同名字可能处在不同最新季度（五月初、十月末），截面秩承受这种水平差；「同最新季度内排序」登记为变体。
- 60 日波动窗：最近 60 个可见交易日，个股与沪深 300 都有值的日数 ≥ 40 才有 `ivol60`；整日停牌的名字在 `daily` 里没有行，窗口按可见行计，不补零。

## 陷阱

- **`pct_chg` 两种单位**：`macro.index_daily.pct_chg` 是百分数（`0.68` = 0.68 %），`daily.pct_chg` 是小数；把指数直接当小数用会把 β 压到零、残差波动等于总波动而不报错，本包的**排名列**就整条错掉。
- **`n_income` 不是归母净利润**：`income_vip.n_income` 含少数股东损益，应计只用 `n_income_attr_p`。
- **年初至今不是单季**：`n_cashflow_act` 与 `n_income_attr_p` 都是累计值，两者相减得到年初至今应计，除以期末总资产已是量纲无关；不要再对累计值做季度差分后与时点资产混算。
- **总资产可以是零或空**：新上市或异常行按下限截断后仍参与，但 `accr`/`ocf_ta` 任一为空即无质量列，该名字不入池——筛子需要它，而且池必须与姊妹臂逐字相同。
- **`is_suspended` 只标盘中临停**，整天停牌的股票在 `daily` 里没有行；用最新交易日有无行判断可交易。
- **整手与价格上限**：100 股整手，T-1 收盘价 ≤ 30 元使一手 ≤ 3,000 元；科创板 200 股起，本包剔除。
- **构造的顺序是机制的一部分**：池底 → 质量筛子 → 排名 → 取前 15。先筛质量再截池底、或先取前 15 再筛，都不是本包登记的候选；筛子按 `accr` 降序剔除，剔的是应计最高（盈利最没有现金流支撑）的那一端。
- **对照共用同一个池**：`c_v15`、`c_qual`、`c_pool` 与候选的股票池必须逐字相同，只差成形与分数；池一变，读数就不可比，与姊妹臂的那一对也不可比。
