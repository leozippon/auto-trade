# NumPy 截面排序器参考包（20260826）

本目录是只读 refs：把 Microsoft Qlib 公开工作流里的 **CSI300 + Alpha158 + LightGBM + TopkDropout** 收成可在本 ABI 重写的机制，而不是移植 Qlib。

诚实边界：Qlib 文档和示例里的 CSI300 表是**另一套数据、另一套回测、另一个平台**的结果，不是本环境 Validation 预期。不要抄那些数字，也不要 import qlib。

按以下顺序读：

1. `rewrite-plan.md`：10 步可证伪改写。
2. `features.md`：Alpha158 风格的本地日频组，不要求原列名。
3. `ranking.md`：截面标准化、topk、n_drop 换手控制。
4. `sources.md`：Qlib 工作流 / Alpha158 标签 / TopkDropout 出处与禁止项。对方收益表不作本折预期。

## 硬合同

- 正式策略只能写到 `output/main.py`，入口为 `generate_orders(context)`，返回严格 JSON 订单数组。
- 正式 import 只允许：`__future__`、`collections`、`datetime`、`decimal`、`math`、`numpy`、`pandas`、`statistics`。即使镜像里有 sklearn / lightgbm / qlib / vnpy / joinquant，也不得 import。
- 运行时不挂载 `models/`。不要在推断时加载 `.pkl` / `.pt`。权重必须在 `generate_orders` 内用当时可见行算出来，或写成源码里的显式系数。
- 沙箱无网络，不要 `pip install`，不要抓 GitHub。
- 每一行必须 `available_at <= context.inference_at`。默认 08:30 只用 T-1 及更早日线。
- 不要写死 `/mnt/agent/workspace`。先核对本轮 manifest / `data_summary.json`，再经 `context.asof_dir` / `context.snapshot_dir` 读数。
- 不要把 refs 拷进 `output`。
- Broker 负责 T+1、费用、涨跌停与成交；策略只发订单草图。不要在策略里再写一套回测器。
- 不要粘贴 Qlib / LightGBM 完整源码。只引用 URL 并转述机制。

## 改写目标

用 T-1 日频构造少量 kbar / 滚动收益 / 滚动波动 / 量比特征，截面 z-score，丢掉 NaN，按分数取 top-k，并用 n_drop 类规则限制与当前持仓的换手。
学习器只能是 numpy 里的 OLS/ridge，或秩加权线性组合；可选一棵深度受限的 if/then 规则树。LightGBM 不可 import；sklearn 虽在镜像中也属禁止模块。
