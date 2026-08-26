# 截面因子探索参考包（20260826）

本目录是只读 refs：只提供可证伪假说、家族公式和 PIT 检查，不是可提交策略，也不是预期收益。
上一轮 `explore_tushare_factors` 已经跑过相近的因子菜单探索。本轮必须写可执行且与父本不同的家族，禁止克隆父策略或把多个家族揉成不透明总分。

按以下顺序读：

1. `exploration-plan.md`：10 步可证伪流程。
2. `families.md`：五个可分离家族，每族只留少量代表因子。
3. `pit-rules.md`：T-1、复权、财务版本、截面处理。
4. `sources.md`：公开文献、公开挖掘流程与本仓库交叉核对；含 WorldQuant 101 算子出处。对方收益表不作本折预期。

不要复刻 TuShare 202 因子菜单，也不要把本包当成 `factor_value` 日值接口。

## 硬合同

- 正式策略只能写到 `output/main.py`，入口为 `generate_orders(context)`，返回严格 JSON 订单数组。
- 正式 import 只允许：`__future__`、`collections`、`datetime`、`decimal`、`math`、`numpy`、`pandas`、`statistics`。即使镜像里有 sklearn / lightgbm / qlib / vnpy / joinquant，也不得 import。
- 运行时不挂载 `models/`。不要在推断时加载 `.pkl` / `.pt`。
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
