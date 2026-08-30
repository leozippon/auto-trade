# 涨停延续 / 竞价感知打板参考包（20260826）

本目录是只读 refs：把社区「打板」话术收成可在本仓库重算的次日/竞价感知假说。
**这不是盘中打板。** 本实验计划使用 `inference_time=09:29`：官方开盘集合竞价（盖章 09:29）可见，**当日 09:30 开盘与之后的分钟都不可见**。策略在 09:29 做决策，订单草图交给 Broker 在随后的 09:30 执行。

**数据覆盖下限**：精确竞价 `auction` 域自 2025-01-16 起才有行（源端 `stk_auction` 起始日），更早窗口的 `auction.parquet` 为空，`data_summary.json` 的 `coverage_start` 会标明。输入或 Validation 落在该起点之前的区间不能依赖竞价字段；机制必须在没有竞价行时按 T-1 涨停/龙虎榜标签退化运行，且不得因此回退到不可见数据。

不要把 T-1 收盘价当成已成交价，也不要假装能排队、看封单变化或在 09:15–09:25 盯盘。

按以下顺序读：

1. `exploration-plan.md`：本折的 cohort 事件研究顺序与验收口径。
2. `playbooks.md`：五个社区机制的 PIT 改写与否证。
3. `pit-field-map.md`：`limit_list_d` / `kpl_list` / `limit_step` / `stk_limit` / `auction` 的可见时间与陷阱。
4. `sources.md`：范围内机制与范围外仓库（盘中打板 / 实时触发 / 东财爬虫一律不要克隆）。

社区帖里的胜率、连板神话和宣传收益一律未核实，本包不转载为事实。

## 硬合同

- 正式策略写在 `output/` 包内：入口固定为 `output/main.py` 的 `generate_orders(context)`，返回严格 JSON 订单数组；辅助模块可放在 `output/` 下并用绝对导入（如 `from lib.features import x`），每个 `.py` 都受同一套静态检查。
- 正式 import 只允许：纯计算标准库（`__future__`、`collections`、`dataclasses`、`datetime`、`decimal`、`functools`、`itertools`、`math`、`statistics`、`typing`）、`numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、`torch`（仅 CPU）及其子模块，以及 `output/` 内自己的模块。qlib / vnpy / joinquant / joblib / pickle 仍不得 import；模型参数用 NumPy 数组或 booster 的 `save_model(context.state_dir + ...)` 持久化。
- 需要拟合的量放在 `fit(context)` 并写入 `context.state_dir`；`models/` 以只读 `context.models_dir` 挂载。不要加载 `.pkl` / `.pt`。
- 沙箱无网络，不要抓网页，也不要去拉聚宽/东财实时榜。
- 每一行必须 `available_at <= context.inference_at`。
- 不要写死 `/mnt/agent/workspace`。先核对本轮 manifest / `data_summary.json`，再经 `context.asof_dir` / `context.snapshot_dir` 读数。
- 不要把 refs 拷进 `output`。
- Broker 负责 T+1、费用、涨跌停与成交；策略只发订单草图。涨停开盘可以拒单，这是真实结果。
- 不要粘贴受版权保护的完整源码（含聚宽策略）。只引用 URL 并转述机制。

开发窗口按年切成常规 Fold：每折的验证区间就是那一年，没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，可用的完整 Validation 次数远多于机制数量（上限以本轮事实为准）。先在输入窗上把多个 cohort 的事件研究做完，再预先登记最强的三条机制，用 `batch_validate` 并列跑完整 Validation；站住的 cohort 再进入后续轮次（门控变体、在 `fit` 里拟合的 cohort 内排序——logistic 或小型树模型——对比等权、持有期与仓位规则），一折跑多轮，并始终按窗口内的子区间判稳健。

## 和 08:30 包的差别

上一轮 `explore_platform_strategies` 默认 08:30，当时明确丢掉「今日竞价弱转强 / 一字板排队」。
本轮 09:29 **可以**用今日竞价价相对昨收的缺口作过滤，仍然 **不能** 把竞价当成已成交，也不能看 09:30 之后的分时。
`kpl_list` 与 `limit_step` 不是当日可交易性证明；交易约束用 `stk_limit`。

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
