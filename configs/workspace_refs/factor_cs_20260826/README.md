# 截面因子探索参考包（20260826）

本目录是只读 refs：只提供可证伪假说、家族公式和 PIT 检查，不是可提交策略，也不是预期收益。
上一轮 `explore_tushare_factors` 已经跑过相近的因子菜单探索。本轮必须写可执行且与父本不同的家族，禁止克隆父策略或把多个家族揉成不透明总分。
开发窗口按年切成常规 Fold：每折的验证区间就是那一年，没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，可用的完整 Validation 次数远多于假说数量（上限以本轮事实为准）。动手前预先登记至少三条互斥假设，用 `batch_validate` 并列验证，并按窗口内的子区间判稳健；一轮胜出后围绕胜者再登记下一轮（`fit` 里拟合的因子权重或树模型打分对比等权、中性化版本、节奏与持仓数变体），一折跑多轮，不要只测一轮就收工。

按以下顺序读：

1. `exploration-plan.md`：本折的可证伪流程与验收口径。
2. `families.md`：五个可分离家族及其代表因子。
3. `pit-rules.md`：T-1、复权、财务版本、截面处理。
4. `sources.md`：公开文献、公开挖掘流程与本仓库交叉核对；含 WorldQuant 101 算子出处。对方收益表不作本折预期。

不要复刻 TuShare 202 因子菜单，也不要把本包当成 `factor_value` 日值接口。

## 硬合同

- 正式策略写在 `output/` 包内：入口固定为 `output/main.py` 的 `generate_orders(context)`，返回严格 JSON 订单数组；辅助模块可放在 `output/` 下并用绝对导入（如 `from lib.features import x`），每个 `.py` 都受同一套静态检查。
- 正式 import 只允许：纯计算标准库（`__future__`、`collections`、`dataclasses`、`datetime`、`decimal`、`functools`、`itertools`、`math`、`statistics`、`typing`）、`numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、`torch`（仅 CPU）及其子模块，以及 `output/` 内自己的模块。qlib / vnpy / joinquant / joblib / pickle 仍不得 import；模型参数用 NumPy 数组或 booster 的 `save_model(context.state_dir + ...)` 持久化。
- 需要拟合的量放在 `fit(context)` 并写入 `context.state_dir`；`models/` 以只读 `context.models_dir` 挂载。不要加载 `.pkl` / `.pt`。
- 沙箱无网络，不要抓网页。
- 每一行必须 `available_at <= context.inference_at`。
- 不要写死 `/mnt/agent/workspace`。先核对本轮 manifest / `data_summary.json`，再经 `context.asof_dir` / `context.snapshot_dir` 读数。
- 不要把 refs 拷进 `output`。
- Broker 负责 T+1、费用、涨跌停与成交；策略只发订单草图。
- 不要粘贴受版权保护的完整源码。只引用 URL 并转述机制。

## 本轮边界

默认 `inference_time=08:30`：当日日线、`daily_basic`、`adj_factor` 都不可见，只能用 T-1 及更早。
财务按公告 `available_at`，不用 `end_date` 偷看。从 PIT parquet 重算分数，不读 TuShare `factor_value` / `stk_factor*`。
WorldQuant 101 只作可选算子控制，不是 101 条公式工厂；先分家族证伪，再考虑等权组合。
判定一律相对同窗对照，并要求年内两个半窗、四个子窗的符号一致；整年合计为正但子窗符号相反的候选不算胜出。

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
