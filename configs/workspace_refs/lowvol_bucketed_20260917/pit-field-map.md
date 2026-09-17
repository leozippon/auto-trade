# 08:30 字段图

先核对本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json`：`fundamentals.datasets` 必须列出 `income_vip`、`cashflow_vip`、`balancesheet_vip`（缺任一，应计筛子不可构造，股票池也就和姊妹臂不再逐字相同，按 `families.md` 诚实终止）；`macro.datasets` 必须列出 `index_daily`（缺它时排名列按登记变体退为总波动，买单元数据 `vol_basis` 写 `total`，结果笔记必须写明——这时本臂排的已经不是残差波动了）；`universe` 必须带 `l1_name`（缺它，桶就不存在，本臂无从构造）。本文是本包可见时点、单位、去重规则与陷阱的**唯一权威表述**。

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
universe = pd.read_parquet(context.asof_dir + "/universe",
                           columns=["ts_code","name","list_date","l1_name"])
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。`fundamentals` 与 `macro` 都是多个数据集的列并集：同名列在不同数据集里含义与单位不同（`pct_chg` 在 `daily` 是小数、在 `macro.index_daily` 是百分数；`n_income` 在 `income_vip` 含少数股东损益），**必须先按 `dataset` 过滤再谈字段与单位**。两类表要分开：`fundamentals` 与 `macro` 的行带**行级 `available_at` 列**，判可见只看它（字符串，`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `ann_date`/`trade_date` 推；`daily` 与 `universe` **没有 `available_at` 列**（决策视图的 `daily.parquet` 30 列，按它过滤会 KeyError），它们的可见性由 as-of 视图本身给定——模板 `output/README.md`「Reading PIT data」：冻结首片只含决策时刻之前可见的行，后续片只含之后发布的行，所以 08:30 决策时 `daily` 的最新一行就是 T-1，不需要也不能再按时间戳过滤。

## 逐表可见边界

| 数据集 | 本包所需字段与单位 | 行级 `available_at` 规则 | 08:30 能看到的最新行 |
|---|---|---|---|
| `fundamentals.income_vip` | `ts_code`、`end_date`、`ann_date`、`report_type`（只用 `"1"` 合并报表）、`n_income_attr_p`（**元**，年初至今累计） | `source:f_ann_date_or_ann_date` 18:00 | 公告日 ≤ T-1 的报表 |
| `fundamentals.cashflow_vip` | 同上键；`n_cashflow_act`（**元**，经营活动现金流量净额，年初至今累计） | 同上 | T-1 |
| `fundamentals.balancesheet_vip` | 同上键；`total_assets`（**元**，期末时点值） | 同上 | T-1 |
| `macro.index_daily` | `ts_code = 000300.SH`、`trade_date`、`pct_chg`（**百分数**，÷ 100 后才是日收益） | `contract_1730_from:trade_date` 17:30 | T-1 |
| `daily` | `close`（元）、`pct_chg`（小数）、`amount`（元）、`circ_mv`（**元**，已归一化；池底与三分位排的都是它）、`turnover_rate`（小数）、`adj_factor`、`pe_ttm`（倍）、`is_suspended` | **无 `available_at` 列**；可见性由 as-of 视图隐含 | 最新一行 = T-1 |
| `universe` | `name`（ST 筛选）、`list_date`、**`l1_name`（申万一级行业名，桶的键）** | **无 `available_at` 列**；决策日冻结 | 当日 |

原始湖 `data/raw/daily` 的 `amount` 是**千元**、`circ_mv` 是**万元**；离线普查直接读原始湖会差 1000 倍 / 1 万倍，快照里已归一化为元。**池底与规模三分位排的都是 `circ_mv`，所以这一处单位错会静默地把两个桶维度同时截反。** 快照决策视图里三张报表只含 `report_type = "1"` 的行，过滤仍要写，别的视图不保证。

## 申万一级行业是点内的

`universe.l1_name` 不是当前归属回填出来的：宿主按**决策日**取成分（`in_date ≤ 决策日 < out_date`），并且按当天市场实际在用的口径取——申万在 2021-12-13 从 SW2014 切到 SW2021，该日之前的决策日用冻结的 SW2014 成分，之后用 SW2021 成分。研究期跨这个切换点，所以：

- **一级行业的个数与名字在研究期内会变**（SW2014 二十八个、SW2021 三十一个；`电气设备` 之类的名字也重排过）。桶是每个复核日在当天的截面上重新算的，所以这件事不会让构造失效；但**跨决策日比较行业名单是错的**，普查表里要按决策日分别列。
- 切换日附近的那一次季度复核可能因为归属变动而多换几只名字，这是真实的口径变更，不是实现缺陷。
- `l1_name` 为空的名字（上市/退市窗口边界、供应商回填缺口）落在 `未分类` 这一个桶里，和其它桶一样受 2 只上限约束。宿主侧的行业归因用的是同一个名字与同一个 `未分类` 标签，所以篮子里的桶计数与 `stats.benchmark.top_industry_weight` 说的是同一件事。研究期五个年首决策日实测：池底与筛子之后 `未分类` 为 0 只（`sources.md`）。
- 申万成分历史由供应商回填，纳入日期可能异常（有名字的纳入日早于其上市日），所以**纳入日期不能当作可靠的行业变更日**；本包只用「决策日归属」，不用纳入日期做任何判断。

## 去重与版本规则

- 同一 `(ts_code, end_date)` 在每张报表里可能有多个版本（首次公告、修订、年报重述）：**每张表各取 `available_at` 最早的一版**，重述版永远不是事件，也永远不替换状态。
- 三张表按 `(ts_code, end_date)` 内连接；事件的可见时点取三者中**最晚**的 `available_at`（三表齐全才算可见；通常同一天）。
- 最新报表 = 每个名字 `available_at` 最新的一份（**按时间戳取，不按年龄并列取**：窗口之前公告的报表年龄全部并列，按行序打破并列会取到最旧的一份），年龄按可见交易日历计；`daily` 读 220 个自然日，长于 130 个交易日的年龄上限，窗口之前公告的报表一律判过期。年龄 > 130 个交易日即无信号，名字离开股票池。
- 旧报告期的重述不是事件：首版早于读取窗口的重述在窗口里看起来就是首版，所以报告期结束日必须在公告日前 400 个自然日内，否则丢弃。
- 年初至今流量按季度序号年化（`× 4 / qn`）：三月末 ×4、六月末 ×2、九月末 ×4/3、十二月末 ×1。同一决策日上不同名字可能处在不同最新季度，截面秩承受这种水平差。
- 60 日波动窗：最近 60 个可见交易日，个股与沪深 300 都有值的日数 ≥ 40 才有 `ivol60`；整日停牌的名字在 `daily` 里没有行，窗口按可见行计，不补零。

## 陷阱

- **`pct_chg` 两种单位**：`macro.index_daily.pct_chg` 是百分数（`0.68` = 0.68 %），`daily.pct_chg` 是小数；把指数直接当小数用会把 β 压到零、残差波动等于总波动而不报错，本包的**排名列**就整条错掉。
- **`n_income` 不是归母净利润**：`income_vip.n_income` 含少数股东损益，应计只用 `n_income_attr_p`。
- **年初至今不是单季**：`n_cashflow_act` 与 `n_income_attr_p` 都是累计值，两者相减得到年初至今应计，除以期末总资产已是量纲无关。
- **总资产可以是零或空**：新上市或异常行按下限截断后仍参与，但 `accr`/`ocf_ta` 任一为空即无质量列，该名字不入池——筛子需要它，而且池必须与姊妹臂逐字相同。
- **`is_suspended` 只标盘中临停**，整天停牌的股票在 `daily` 里没有行；用最新交易日有无行判断可交易。
- **整手与价格上限**：100 股整手，T-1 收盘价 ≤ 30 元使一手 ≤ 3,000 元；科创板 200 股起，本包剔除。
- **构造的顺序是机制的一部分**：池底 → 应计筛子 → 桶内填位 → 三分位配额。先填桶再截池底、或先取前 15 再按行业砍，都不是本包登记的候选；上限必须计在**本次复核实际要持有的篮子**上（保留带留下的持仓一并计数），计在「新排出来的前 15」上会从保留带漏掉。
- **三分位是筛后池的三分位**：在池底与筛子之后、按位次三等分，不是按 `circ_mv` 的数值分位，也不是全池的三分位。池底已经把小市值的一半切掉了，所以「最小的三分之一」指的是较大一半里最小的三分之一。
- **上限是只数不是权重**：季度之内价格漂移会把两只同业名字的时间加权权重推到 2/15 之上，提名硬门读的 `stats.benchmark.top_industry_weight` 是时间加权量。
- **对照共用同一个池**：`c_uncapped`、`c_rand` 与候选的股票池、池底、筛子必须逐字相同，只差桶与分数；`c_pool` 不成形也不设桶，是底线对照。池一变，读数就不可比，与姊妹臂的那条 `c_uncapped` 也不可比。
