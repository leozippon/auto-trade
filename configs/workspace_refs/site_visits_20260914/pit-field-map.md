# 08:30 字段图

先核对本轮 `data_summary.json`、manifest 和单位表；下表是仓库常见合同，某一折仍可能缺表或为空。`data_summary.json` 里 `events` 的 `datasets` 列出本轮实际加载的数据集，只有列在其中的才可读——`stk_surv` 与 `top10_floatholders` **默认不加载**，本臂靠创建实验时的 `events_datasets` 显式选入。

## 读法

```text
daily    = pd.read_parquet(context.asof_dir + "/daily", columns=[...], filters=[("trade_date", ">=", start)])
events   = pd.read_parquet(context.asof_dir + "/events",
                           columns=["dataset", "available_at", "ts_code", "surv_date", "rece_org", "org_type", "rece_mode"])
events   = events[(events["dataset"] == "stk_surv")
                  & (pd.to_datetime(events["available_at"]) <= pd.Timestamp(context.inference_at))]
macro    = pd.read_parquet(context.asof_dir + "/macro", columns=[...])   # dataset == "index_daily"
universe = pd.read_parquet(context.asof_dir + "/universe")               # ts_code、name、list_date、l1_code/l1_name
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；不要把 `asof_dir` 读失败回退到 `snapshot_dir`。

事件域是多个数据集的列并集：`stk_surv` 的行**没有 `trade_date`**（该列来自 `moneyflow` 等按交易日的数据集，对调研行为空），日期只有 `surv_date`，字符串 `YYYYMMDD`。同理 `top10_floatholders` 的行只有 `ann_date` 与 `end_date`。先按 `dataset` 过滤，再取该数据集自己的日期列。

## 本包用到的数据集

| 数据 | 本包所需字段 | 08:30 可见边界与单位 |
| --- | --- | --- |
| 合并日线 `daily` | `close`（复权用）、`amount`（元）、`turnover_rate`（小数）、`circ_mv`（元）、`adj_factor`、`up_limit`/`down_limit`、`is_suspended` | 当日行 17:30 才可见，08:30 只有 T-1 及更早；`adj_factor` 当日 09:30 盖章 |
| `events.stk_surv` | `surv_date`、`ts_code`、`rece_org`、`org_type`、`rece_mode`、`comp_rece`、`rece_place`、`name`、`fund_visitors` | 行级 `available_at = surv_date + 5 自然日 23:59:59`（规则名 `conservative_plus_5d_eod_from:surv_date`）；覆盖自 2022-01-04；主键 `(ts_code, surv_date, rece_org)` |
| `events.top10_floatholders` | `ann_date`、`end_date`、`holder_name`、`holder_type`、`hold_amount`（股）、`hold_ratio`/`hold_float_ratio`（百分数）、`hold_change`（股） | 行级 `available_at` 按 `ann_date` EOD（`conservative_date_eod`）；覆盖自 2020-01；季报滞后见下 |
| `macro.index_daily` | `000300.SH` 的 `close`、`pct_chg`（百分数，先除以 100） | `contract_1730_from:trade_date`（数据日 17:30），T-1 可见、与日线同步（见 README 的可见边界一条）；只用于基准与 β |
| `universe` | `ts_code`、`name`（含 ST 标记）、`list_date`、`l1_code`/`l1_name` | 决策日冻结；行业与 ST 只从这里取，不回填今天的状态 |

## `stk_surv` 的 +5 日规则及其后果

源表没有公告时间列，记录在调研后数日内陆续入库（深交所要求两个交易日内披露），所以可见时间取 `surv_date` 后第 5 个自然日 23:59:59，比强制披露口径更保守。对 08:30 决策的直接后果：

- 一次调研最快在 `surv_date` 之后第 6 个自然日的 08:30 才可见，当日 09:30 才可能成交。实测 2023-01 至 2026-06 的 843 个调研日：6 个自然日 634 天、7 个自然日 161 天，长假尾部最长 15 个自然日；换算成交易日为 4 日 625 天、5 日 151 天，节前样本 1–3 日 67 天。
- 因此**任何 5 日以内的调研事件窗都不可交易**，本包只做 20 日持有的月度漂移。写窗口时要意识到 `[T-5, T-1]` 这一段在 08:30 几乎总是空的，只有上面那 67 个节前样本例外。
- 春节等长假期间迟到披露的记录仍可能提前可见，这是已记录并接受的残余风险（数据文档 §4），不要据此设计跨假期的抢跑逻辑。

## 落库缺口：2025 年的行数下滑不是「调研冷却」

年度厚度（行=一家到访机构）：2022 年 153,297 行 / 2,942 只，2023 年 165,537 / 3,624，2024 年 177,363 / 3,900，2025 年 112,660 / 3,879，2026 年（至 09-02）142,522 / 4,366。2025 年的下滑**不是真实下降**：2025 年 243 个交易日一个分区都不缺，但其中 **56 个分区是零行**——2025-10 整月 17 个交易日全空，2025-09 有 13 天、2025-11 有 10 天、2025-12 有 4 天、3/6/7/8 月零散 12 天。2022 年有 6 个、2024 年 1 个、2026 年 5 个零行分区。

后果与处理：零行日在 60 日与 250 日窗口里表现为「那天全市场没有人调研」，会同时压低 `F_60` 与 `F_250`，并在 2025Q4 造出一个虚假的全市场「调研冷却」。**必须显式识别并剔除零行日**（按 `surv_date` 聚合全市场行数为 0 的交易日），窗口按剩余的有效调研日归一化，而不是按日历交易日数；剔除后的有效日占比要在每个子窗里如实汇报。验证区间覆盖 2025Q3–2025Q4 的折要把这一点写进候选论证，读数与更早的折不可直接比较。

## 季节性与交易所漂移

- 月度季节性（2022–2024 月均行数）：4 月 24,418、8 月 26,636、10 月 24,075，对 6 月 7,492、7 月 8,075、12 月 7,064，约 3 倍摆动，来自业绩说明会日历。
- 60 日可见覆盖随之摆动。实测「过去 60 个交易日有可见调研的股票数 / 其中有基金管理公司到访的股票数」：2024-03-29 为 986 / 761，2024-09-30 为 1,909 / 957，2025-06-30 为 3,424 / 976，2025-12-31 为 960 / 364（受落库缺口污染），2026-06-30 为 4,079 / 1,363。`D_250 >= 3` 的股票数在 1,321–2,005 之间。
- 交易所构成漂移（行占比）：深市 83.2%（2022）→ 77.5% → 74.1% → 75.5% → 60.1%（2026），沪市 16.8% → 36.4%，北交所 2025 年才首次出现、2026 年占 3.5%。60 日可见被调研股票的构成同向漂移（2023-06-30 深 82.1% / 沪 17.9%；2025-12-31 深 56.0% / 沪 40.0% / 北 4.0%）。跨折比较踩在这条非平稳曲线上。

## `top10_floatholders` 的季度披露滞后

行按 `ann_date` EOD 可见，而 `ann_date` 相对报告期末的中位滞后是：一季报与三季报 28–29 个自然日，半年报 57 天，年报 113 天。也就是说 12-31 期的前十大流通股东名单在次年 4 月中旬才有一半可见。逐报告期的覆盖股票数在 3,252–5,779 之间摆动，半年报期系统性偏薄（20220630 3,252、20230630 3,274、20240630 3,412、20250630 3,669，20240930 3,291），因此「相邻两期都覆盖」的股票集合远小于任一单期。`hold_amount`/`hold_change` 是股数，`hold_ratio`/`hold_float_ratio` 是百分数。

## 复权与截面

推断时冻结前复权锚：`anchor = adj_factor(T-1)`，`qfq(t) = raw(t) * adj_factor(t) / anchor`，收益用 qfq 收盘；成交额不复权。家族 4 的 `post10` 必须用同一个锚下的 qfq 价，跨除权日直接用原始收盘会造出假的调研后收益。

截面只含当时已上市、未退市、T-1 未停牌、值有限且有可见源行的股票。**缺 `stk_surv` 行不等于零调研**：这只股票根本没进过调研样本，与「有基线但本期没人来」是两件事，前者不进截面，后者 `F_60 = 0` 参与排序，两者在代码里要能分开。

## 执行

- 最长回看约 253 个真实交易观测（250 日基准窗）；先按股票取尾窗，再留最新截面；模块级缓存只键控 `context.asof_version`，冷启动必须得到相同订单。
- 卖先于买，按 `context.account.cash` 快照本地递减预算，留 3% 费用缓冲，买入向下取整到 100 股，不把未成交卖出当成已到账。
- 涨跌停：买单成交价 ≥ 当日 `up_limit` 拒单、卖单 ≤ `down_limit` 拒单，当日限价 08:30 不可见，只能用 T-1 代理估计；拒单是结果，汇报它。
- 停牌：`is_suspended=True` 的 Bar 一律拒单，且该标志含复牌日；「真停牌」按日线缺行判断。长停牌或退市后没有日线 Bar 的持仓卖不出，入场前把这类尾部锁仓计入仓位上限。

## 常见失败

- 用 `surv_date` 判可见，或把 `surv_date` 当 `trade_date` 去 join 日线。
- 对 `stk_surv` 的行直接计数，让一场几百家机构的说明会主导排序。
- 把 `fund_visitors` 当人数（它是逗号分隔的到访人姓名，54.41% 为 `--`）。
- 零行日与「零调研股票」混为一谈，在 2025Q4 造出全市场调研冷却。
- 把 `top10_floatholders` 的单期水平当变化，或忽略 28/57/113 天的公告滞后；也不要把 `end_date` 当成规整报告期：除 0331/0630/0930/1231 外还混着 0205/0206/0208 这类非报告期日期（种子回放槽实测 223 个不同月日、约 10% 的行不落在四个报告期上），按期分组或做相邻两期差分前必须先过滤。
- 按 `ts_code` 把 `stk_surv` 关联到 `daily`/`universe` 时把北交所整块丢掉：本表的北交所旧代码（83x/43x 开头）带的是 `.SZ` 后缀（种子回放槽实测 4,456 行、210 只、零 `.BJ`），而 `daily`/`universe` 的北交所是 `92xxxx.BJ`，两边逐字符对不上，join 后这些行直接消失——这是代码口径不一致，不是「北交所没有调研」。
- 混单位：`hold_float_ratio` 与 `index_daily.pct_chg` 是百分数，`daily.turnover_rate`/`pct_chg` 是小数，`amount`/`circ_mv` 已归一为元。
- 在 `generate_orders` 里每天全量重读全部事件域并重算滚动量。

## 本轮快照默认没有的东西

- 调研纪要正文。`stk_surv` 只有结构化字段，`irm_qa_sh`/`irm_qa_sz` 是互动易问答、不是调研纪要；文本域只保留标题与截断正文，`report_rc` 的 `eps`/`np`/`rating` 等数值列被 `_build_text` 丢弃，「分析师预测修正」在本环境不可实现。
- `broker_recommend`（券商金股）可见时点是月末后 31 天，7 月名单要到 8 月 31 日才可见，不能当月度信号。
- `top10_holders`（全体股东口径）与 `stk_holdernumber`、`hm_detail`：本包不用，需要时在创建实验时另行选入并重做覆盖普查。
- 分钟线与精确集合竞价（`include_intraday=false`）：成交只有 09:30 与 15:00 两个时点。
