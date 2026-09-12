# 08:30 字段图

先核对本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json`：`fundamentals.datasets` 必须列出 `income_vip` 与 `express_vip`，`events.datasets` 必须列出 `report_rc`；三者缺一，对应的腿整块不可构造（缺 `report_rc` 只剩季度腿，缺 `income_vip` 本臂不可测），`report_issue` 后如实处理。本文是本包可见时点、单位、去重规则与陷阱的**唯一权威表述**。

## 读法

```text
stmts = pd.read_parquet(context.asof_dir + "/fundamentals",
                        columns=["dataset","available_at","ts_code","end_date","ann_date",
                                 "report_type","n_income_attr_p","n_income"],
                        filters=[("dataset","in",["income_vip","express_vip"]), ("ann_date",">=",start)])
rc    = pd.read_parquet(context.asof_dir + "/events",
                        columns=["dataset","available_at","ts_code","quarter","np","org_name","report_date"],
                        filters=[("dataset","=","report_rc"), ("report_date",">=",start)])
daily = pd.read_parquet(context.asof_dir + "/daily",
                        columns=["ts_code","trade_date","close","pct_chg","amount","adj_factor",
                                 "turnover_rate","circ_mv","pe_ttm","is_suspended"],
                        filters=[("trade_date",">=",start)])
universe = pd.read_parquet(context.asof_dir + "/universe")   # ts_code、name、list_date、l1_code
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。`fundamentals` 与 `events` 都是多个数据集的列并集：同名列在不同数据集里含义不同（`n_income` 在 `income_vip` 是含少数股东损益的净利润、在 `express_vip` 是快报净利润），**必须先按 `dataset` 过滤再谈字段与单位**。判可见只看 `available_at`（字符串，`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `ann_date`/`trade_date` 推。

## 逐表可见边界

| 数据集 | 本包所需字段与单位 | 行级 `available_at` 规则 | 08:30 能看到的最新行 |
|---|---|---|---|
| `fundamentals.income_vip` | `ts_code`、`end_date`、`ann_date`、`report_type`（只用 `"1"` 合并报表）、`n_income_attr_p`（**元**） | `source:f_ann_date_or_ann_date` 18:00 | 公告日 ≤ T-1 的报表 |
| `fundamentals.express_vip` | `ts_code`、`end_date`、`ann_date`、`n_income`（**元**）；**`yoy_net_profit` 是去年同期净利润金额，不是增长率** | 每个版本按自身 `ann_date` 18:00 | T-1 |
| `events.report_rc` | `ts_code`、`quarter`（**财年标签**，`2024Q4` = FY2024；1,532 行是裸 `"Q"`，丢弃）、`np`（**万元**，×1e4）、`org_name`、`report_date` | `source:create_time`（与报告日相差 −1..+3 天时可信），否则报告日 22:00 | T-1 |
| `daily` | `close`（元）、`amount`、`circ_mv`（**元**，已归一化）、`turnover_rate`、`pct_chg`（小数）、`adj_factor`、`pe_ttm`（倍）、`is_suspended` | 17:30 / 18:00 / 09:30 合同 | T-1 |
| `universe` | `name`（ST 筛选）、`list_date`、`l1_code` | 决策日冻结 | 当日 |

原始湖 `data/raw/daily` 的 `amount` 是**千元**、`circ_mv` 是**万元**；离线普查直接读原始湖会差 1000 倍 / 1 万倍，快照里已归一化为元。

## 去重与版本规则

- 同一 `(ts_code, end_date)` 在 `income_vip` 里有多个版本（首次公告、修订、年报重述）：**只取 `available_at` 最早的一版**，重述版永远不是事件。
- 年度事件取 `income_vip` 年报与 `express_vip` 快报中**先可见的那份**，另一份忽略；季度腿只用 `income_vip`。
- 一致预期窗口结束于**公告日 00:00**（严格早于公告日），不是 `available_at` 18:00：公告当晚的券商更新是事后信息。每家券商只取窗口内最新一份。
- `report_rc` 同一报告会以不同 `create_time` 重推，按 `(ts_code, quarter, org_name)` 取最新一份即可。

## 信号年龄与新鲜度

年龄 = 最新可见交易日在可见交易日历中的位置 − 公告日（按 `available_at` 换算到上海时间的日期）在同一日历中的位置（公告日为交易日时即该日；否则取其后第一个交易日）。年龄只用可见日线的交易日历计，不用自然日；新鲜度阶梯写在 `families.md`。

## 陷阱

- **`quarter` 是财年不是季度**：把 `2023Q2` 之类的少数季度行混进一致预期，会把半年预测当全年。只取 `\d{4}Q4`。
- **`np` 与 `n_income_attr_p` 差一万倍**：一致预期与实际值必须同一单位后再相减。
- **`express_vip.yoy_net_profit` 是金额**（数据文档 §4 已核实），要增长率用 `fina_indicator_vip.netprofit_yoy` 或自算。
- **单季差分要按同财年累计值**：`Q1 = 累计Q1`，`Qn = 累计Qn − 累计Q(n−1)`；跨财年不能差分。
- **读取窗口要够长**：季度腿的「去年同季」在 120 日龄的报表上可能是 21 个月前公告的，`fundamentals` 至少读 700 个自然日；一致预期只需 150 天。
- **公告日的反应已在价里**：18:00 可见、次日 09:30 入场，公告日的跳空不是本臂的收益来源，也不得作信号。
- **`is_suspended` 只标盘中临停**，整天停牌的股票在 `daily` 里没有行；用最新交易日有无行判断可交易。
- **整手与价格上限**：100 股整手，T-1 收盘价 ≤ 30 元使一手 ≤ 3,000 元；科创板 200 股起，本包剔除。
- **淡季不是缺陷**：`fundamentals` 在 12–1 月与 7 月没有新公告是季节事实，篮子按 `families.md` 的淡季政策持有上一季尾巴。
