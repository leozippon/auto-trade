# 八年研究期的数据面：表、单位、可见时间与陷阱

研究期 20170701..20250630、`history_floors` 为日线 20100101 与宏观 20140701、宏观只挂 `index_daily`/`index_dailybasic`/`index_weight`/`sw_daily` 的臂共用这份数据面；其中一部分臂另挂财务域，挂没挂看本臂，见「财务域」一节。下面每条都在这份数据上实读或对照代码核对过，不必再派子代理重推一遍；先按运行事实、`data_summary.json` 与 `unit_reference.json` 确认本臂挂的是同一份，几何或数据集不同就先核对再用。

## 视图与窗口

- 会话的 `/mnt/snapshot` 是研究期末（2025-06-30 23:59:59）的决策视图：`daily`、`macro`、`universe`（挂了财务域的臂另有 `fundamentals`）各一个平铺文件，日线、四张宏观表与财务都自 20160701 起。它只供离线探查，正式回放读不到它。
- 一次回放的 `context.snapshot_dir` 是该 span 第一年的决策视图（年初前一天 06-30 23:59:59），整场不变。`context.asof_dir/<域>/` 是 parts 目录：part 0 就是这个决策视图，之后随回放时钟逐日追加。多年 span 一路累加，从 Y1 起跑的整期回放到 Y8 时 `daily` 里是 2010 年起的全部行。
- 每个研究年的决策视图窗口见 `research_geometry.years[].input_window`：年初前 108 个月，截到 `history_floors`。实读：Y1 视图日线 20100104..20170630、宏观自 2014-07；Y3 日线自 20100701；Y8 日线自 20150701、宏观自 2015-07。所以从 Y1、Y2 起跑时指数与申万序列只有 3、4 年历史，需要更长指数窗口的特征（多年 β、按指数算的残差标签、长期指数动量）在前两年取不满：缩短窗口，或在样本说明里写明。
- 挂财务域与不挂财务域的臂都没有事件、文本与分钟；挂公告标题（`text_datasets` 含 `anns_d`）的臂有 `text_index` 域而没有财务域，见「文本域」一节。`auction` 只有 20250116 起的行，不能当全期特征。

## 可见时间：08:30 的决策看到 T-1

- 日线（含每日指标与涨跌停价列）和四张宏观表都按交易日 17:30 盖章，要等交易日晚间的落库任务（23:35 开始）之后才进入 as-of 视图。T 日 08:30 的决策最新只看到 T-1 的行（周一看到上周五），T 日的开盘价在决策时不可见。
- as-of 的 `daily` 没有 `available_at` 列，视图已按落库节点放行。`macro` 保留 `available_at` 与 `available_at_rule`（`contract_1730_from:trade_date`）；`available_at` 是带时区的字符串（`2016-07-01 17:30:00+08:00`），与 `context.inference_at` 比较前用 `pd.to_datetime(..., utc=True)` 解析。

## 日线 daily

- 一行一个 (trade_date, ts_code)，合并了日线、每日指标与涨跌停价：open/high/low/close/pre_close/change/pct_chg/vol/amount、turnover_rate/turnover_rate_f/volume_ratio、pe/pe_ttm/pb/ps/ps_ttm/dv_ratio/dv_ttm、total_share/float_share/free_share、total_mv/circ_mv、up_limit/down_limit、adj_factor、is_suspended。
- 单位已归一：价格 元/股，vol 股，amount 元，pct_chg/turnover_rate/turnover_rate_f/dv_ratio/dv_ttm 是小数（0.05 = 5%），股本 股，total_mv/circ_mv 元，估值是倍数。实读：amount ÷ (vol × close) 中位 0.9998，total_mv = close × total_share，circ_mv = close × float_share。亏损股的 pe/pe_ttm 是空值而不是负数（研究期末前一个月约四分之一为空、零个负值）。
- 价格不复权，pre_close 是除权后的参考价。跨除权日的收益用 `close × adj_factor` 或 `pct_chg`（= close ÷ pre_close − 1），不能用相邻两天的 close：002667.SZ 在 2018-05-02 每股送转 0.7，两天 close 之比给出 −42.6%，复权收益与 pct_chg 都是 −2.3%。个别代码的 adj_factor 会回落，算复权收益时不要假定它逐日不降。
- 回放里的除权由 Broker 结算：除权日把现金红利记入现金，股数变化按 pre_close 结算（运行事实 `broker_replay.ex_date_settlement`），持仓穿过除权日不需要策略自己记账。结算用的除权除息表只给 Broker，不是 as-of 域，策略读不到；每次验证的 result 记录里有逐笔的 `corporate_actions` 结算明细。
- 全日停牌当天没有行，这一天的委托以 `missing_execution_price` 被拒，持仓留在账上；`is_suspended=True` 只是盘中临时停牌，当天量额完整，委托以 `suspended` 被拒。连续的 True 不构成停牌段：判断能否交易，看该股最近一根 bar 是不是视图里最新的交易日。
- 研究期末视图的日线有 5,636 个代码，其中 216 个不在同一视图的 universe 里（在 2025-06-30 前已退市）。

## universe：每个研究年换一个版本

- 列：ts_code、exchange、list_date、market、name、l1_code、l1_name，没有日期列。每个决策视图里是那一天的版本：当天已上市且尚未退市的代码、当天生效的名称（ST 只能从 name 含 "ST" 判断）、当天的申万一级归属。
- 回放中 `asof_dir/universe` 在跨入每个研究年（07-01）时整表换成该年锚点视图的版本，年内不变：年内的上市、改名、戴帽摘帽与行业调整要到下一个 07-01 才看得到。
- 2021-06-30 及以前的版本是申万 2014 口径（28 个一级，含 801020.SI 采掘），2022-06-30 起是申万 2021 口径（31 个一级，新增 801950/801960/801970/801980.SI）。同一代码在 2017 与 2025 两个版本里一级归属不同的有 785/3,077。每个版本都有 l1_code 为空的代码（2017 版 146、2021 版 374、2025 版 3），要显式归入「未分类」，不要直接 groupby。
- `/mnt/snapshot/universe.parquet` 只是 2025-06-30 这一个版本（5,420 行）。拿它给更早的决策日定股票池、行业或 ST，会混进之后的上市、改名与改分类，也会漏掉此前退市的名字（2017 版里有 202 个代码不在 2025 版）。离线复现历史决策只能当近似；策略里读 `context.asof_dir/universe`。

## 宏观 macro：四张表纵向拼成的一张宽表

- 先按 `dataset` 过滤、再取列：每张表只填自己的列。研究期末视图 158 万行，两个 row group 都横跨全期，不能按日期跳过。
- `index_daily`：指数代码在 `ts_code`，`index_code` 整列为空，按 `index_code` 过滤取不到任何行情。7 只指数：000001.SH、000016.SH、000300.SH、000688.SH、000852.SH、000905.SH、399006.SZ。点位；`pct_chg` 是百分数（−0.35 = −0.35%，与日线的小数口径不同）；vol 手、amount 千元，未归一。
- `index_weight`：指数在 `index_code`、成分在 `con_code`（与日线 ts_code 同格式），`ts_code` 整列为空。同样 7 只，每只每月一张截面，日期是当月最后一个交易日；起点不一：000852.SH 自 20141031 起（Y1 视图里 33 张，其余五只自 20140731 起 36 张），000688.SH 自 2020-07 起。按中证 1000 成分回看的训练窗在 Y1 早不过 2014-10。000300/000905/000852 每张恰 300/500/1000 行、三者互不重叠，相邻两张最长隔 36 天。`weight` 是百分数（每张合计约 100），当组合权重先除以 100。
- `index_dailybasic`：只有 6 只（没有 000688.SH）；000852.SH 在这份数据里 20181228 之后没有行，2019 年起取不到中证 1000 的指数换手与估值。total_mv/float_mv 元、股本 股（与日线每日指标的口径不同，不要混算），turnover_rate 百分数。
- `sw_daily`：申万指数行情，研究期末视图 596 个代码，一级之外还有二级、三级与风格指数。正则 `801\d\d0\.SI` 匹配 37 个，比研究期末的 31 个一级多出 801020.SI 与 801250/801260/801270/801280/801300.SI；某一年的一级以那一年 universe 的 l1_code 为准。`pct_change` 是百分数，vol 万股，amount/float_mv/total_mv 万元（单位表标为 inferred）。代码是稳定的键，名称随申万 2021 改版而变：801030.SI 在这里 2021-12-13 之前叫「化工」、之后叫「基础化工」，universe 里 2021 版及以前叫「化工」、2022 版起叫「基础化工」；跨表一律按代码连接。2021-12-13 之前的点位是按申万 2021 口径回算的，与当时按申万 2014 划分的个股归属不是同一套。

## 按时点读成分、行业与指数行情

- T 日某股是否在指数 X 中：`dataset == "index_weight"`、`index_code == X`、`available_at <= inference_at`，取其中 trade_date 最大的那张截面的 con_code。T 日自己的截面 17:30 才盖章，08:30 看不到；月初第一个交易日读到的是上月末那张；两张之间的月中调整与临时剔除都看不到。日期回看 40 个自然日就能命中最新可见的一张。
- 行业：读 `context.asof_dir/universe` 的 l1_code，再按代码连 `sw_daily`。
- 指数行情：`dataset == "index_daily"`、`ts_code == X`。

## 财务域 fundamentals：只在挂了它的臂上

- 先判断：`/mnt/artifacts/data_summary.json` 里 `views.snapshot.domains.fundamentals` 的 `datasets` 列出十个数据集、`rows` 非零才是挂了；为空、为 0 时 `fundamentals.parquet` 是零列空壳，本节不适用，回放里按 `dataset` 过滤读它会报错。
- 十个数据集纵向拼成一张宽表（466 列），先按 `dataset` 过滤、再取列：`income_vip`/`balancesheet_vip`/`cashflow_vip`（三大报表，只有合并报表 `report_type` 1）、`fina_indicator_vip`（财务指标）、`forecast_vip`（业绩预告）、`express_vip`（业绩快报）、`dividend`（分红送转各阶段 `div_proc`）、`fina_audit`（审计意见）、`fina_mainbz_vip`（主营构成，`bz_code` P 产品/D 地区/I 行业）、`disclosure_date`（披露计划）。键是 `ts_code` + `end_date`（报告期，`YYYYMMDD` 字符串）。
- 单位：报表与快报金额是元；`fina_indicator_vip` 的比率与增速是百分数（600519.SH 2023 年 `roe` 36.18 即 36.18%）；`forecast_vip` 的 `net_profit_min/max`、`last_parent_net` 是万元，`p_change_min/max` 是百分数；`express_vip.yoy_net_profit` 是去年同期归母净利润（元），不是增长率。
- 可见时间：每行 `available_at` 是公告日 18:00（三大报表按 `f_ann_date`，否则 `ann_date`；指标、预告、快报、审计按 `ann_date`；分红按 `imp_ann_date`，否则 `ann_date`；主营构成按同一报告期报表最晚一次公告）。回放里财务域随 03:35 的 PIT 落库节点放行，节点周二至周六运行：周二至周五 08:30 看到前一天及以前的公告，周一只看到上周五及以前，周六、周日盖章的行（披露计划之外各数据集约 17–22%）周二才可见。区间第一个决策日例外：冻结首片收到锚点 06-30 23:59:59 为止，锚点落在周末时首日多看到周末的行。
- 历史：各研究年的决策视图（Y1..Y8）里财务都从 2016-01 的公告起。Y1 视图（2017-06-30）看得到 FY2015、2016 各期与 2017Q1；更早的报告期只有 2016 年后重述过的约 6%。所以 Y1 开头 97% 的沪深股票能算最新一期 TTM，能算一年前同期 TTM 的只有 5%：用报表做 TTM 同比要等 FY2017 年报（2018-05 起 97%），`fina_indicator_vip` 的 `netprofit_yoy`、`tr_yoy`、`q_sales_yoy` 等厂商增速从 Y1 第一天就有。`/mnt/snapshot` 不一样：它按 108 个月窗口自 2016-07 起（「视图与窗口」第一条），缺 2016 上半年的公告，FY2015 年报与 2016Q1 大多不在里面，所以在它上面量不出 Y1 看得到什么——同样截到 Y1 锚点的可见行，它只给出约 13% 的沪深股票能算最新一期 TTM。分红「预案」行 2018 年前几乎没有（2016、2017 年分别只有 18、25 行，2018-08 才成批出现，2019 年起每年 7,000 行以上），分红公告类特征的起点落在 Y2 里。
- 版本：每个版本一行、按自己的时间可见。更正版在更正日另起一行，此前只看得到首次公告的数值；按决策时点取最新版本就是同一 (`ts_code`, `end_date`) 按 `available_at` 取最后一行，取首次公告就取最早一行。`update_flag` 不表示是否修订（首个版本里也大量为 1），不要按它过滤。已上市公司约 1.3% 的报告期缺原版，到更正日才第一次出现。
- 三大报表是年初至今累计（Q1、H1、前三季度、全年）：单季 = 本期 − 同年上一期，TTM = 本期 + 上年全年 − 上年同期。最新一期按 `end_date` 取，不按时间取（旧报告期的重述盖的是新日期）。公告事件用每期最早一行的 `available_at`，否则重述会被当成新公告。
- 未来字段：`disclosure_date.actual_date` 是事后回填的实际披露日，97% 的行晚于本行盖章，Y1 视图里 3,055 行晚于决策日；只在它早于决策日时可用，计划日用 `pre_date`。`dividend` 的「预案」「股东大会通过」行偶尔带着事后填上的 `ex_date`/`record_date`/`pay_date`，除权安排只取「实施」行。
- 读取：整表或不投影的读取会撑破策略容器的 16 GiB。研究期末视图 271 万行：整读、只按 `dataset == "fina_mainbz_vip"` 过滤不投影都被杀，`income_vip` 过滤不投影峰值 6.6 GiB，8 列投影读全表 0.84 GiB。一律 `pd.read_parquet(context.asof_dir + "/fundamentals", columns=[...], filters=[("dataset", "==", ...)])`，`available_at` 用 `pd.to_datetime(..., utc=True)` 解析后再与 `context.inference_at` 比较。目录每个交易日多一个分片，整期回放里两张表的投影读取平均每次 3–5 秒：只在调仓日读，不要每个决策日重读。

## 文本域 text_index：只在挂了公告标题的臂上

- 先判断：`/mnt/artifacts/data_summary.json` 里 `views.snapshot.domains` 列出 `text_index` 且 `text_datasets` 含 `anns_d` 才是挂了；这类臂的 `fundamentals.parquet` 是零列空壳，「财务域」一节不适用。
- `context.asof_dir/text_index` 一行一份公告：`dataset`（取 `anns_d`）、`ts_codes`、`title`、`available_at`，自 2016-01 起。一律投影这四列并按 `dataset` 与 `available_at` 过滤读取，`available_at` 用 `pd.to_datetime(..., utc=True)` 解析后再与 `context.inference_at` 比较。
- 可见时间：厂商的接收时间落在公告日前一天到后三天之内就按接收时间盖章，否则按公告日 23:59:59；文本落库节点每天 23:15。研究期的行几乎全部只有日期，所以公告日为 D 的标题最早在 D+2 日 08:30 的决策里可见；2026-08-12 起实时落库的行带接收时间（多在公告日前一晚），Paper 比研究期早一到一天半看到标题。按「首次可见的本地日期之后的第一个交易日」计事件日，两段口径就一致。
- 标题是公告的标题而不是正文；同一公告的摘要、修订、法律意见等伴随文件各是一行，按股票与日期去重后再当事件。

## 读取

- 研究期末视图的日线约 923 万行、9 个按 trade_date 有序的 row group，整期回放的后几年 `asof_dir/daily` 更长。决策期一律 `pd.read_parquet(context.asof_dir + "/daily", columns=[...], filters=[("trade_date", ">=", start)])`，只读命中的行组；宏观同样带 `("dataset", "==", ...)` 过滤与列投影。daily 与 macro 的 `trade_date` 都是 `YYYYMMDD` 字符串，过滤值与比较也用同格式的字符串。universe 只有百余 KB，可以整读。
