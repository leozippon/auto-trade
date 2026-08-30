# NumPy 截面排序器参考包（20260826）

本目录是只读 refs：把 Microsoft Qlib 公开工作流里的 **CSI300 + Alpha158 + LightGBM + TopkDropout** 收成可在本 ABI 重写的机制，而不是移植 Qlib。

诚实边界：Qlib 文档和示例里的 CSI300 表是**另一套数据、另一套回测、另一个平台**的结果，不是本环境 Validation 预期。不要抄那些数字，也不要 import qlib。

按以下顺序读：

1. `rewrite-plan.md`：本折的可证伪改写流程与验收口径。
2. `features.md`：Alpha158 风格的本地日频组，不要求原列名。
3. `ranking.md`：截面标准化、topk、n_drop 换手控制。
4. `sources.md`：Qlib 工作流 / Alpha158 标签 / TopkDropout 出处与禁止项。对方收益表不作本折预期。

## 硬合同

- 正式策略只能写到 `output/main.py`，入口为 `generate_orders(context)`，返回严格 JSON 订单数组。
- 正式 import 只允许：`__future__`、`collections`、`datetime`、`decimal`、`math`、`numpy`、`pandas`、`statistics`。即使镜像里有 sklearn / lightgbm / qlib / vnpy / joinquant，也不得 import。
- 拟合放在 `fit(context)`（回放开始前调用一次，按 `REFIT_PERIOD` 重训），系数用 `np.save` 写到 `context.state_dir`，`generate_orders` 只读它；`models/` 以只读 `context.models_dir` 挂载。不要加载 `.pkl` / `.pt`（pickle / torch 加载被静态拒绝）。
- 沙箱无网络，不要 `pip install`，不要抓 GitHub。
- 每一行必须 `available_at <= context.inference_at`。默认 08:30 只用 T-1 及更早日线。
- 不要写死 `/mnt/agent/workspace`。先核对本轮 manifest / `data_summary.json`，再经 `context.asof_dir` / `context.snapshot_dir` 读数。
- 不要把 refs 拷进 `output`。
- Broker 负责 T+1、费用、涨跌停与成交；策略只发订单草图。不要在策略里再写一套回测器。
- 不要粘贴 Qlib / LightGBM 完整源码。只引用 URL 并转述机制。

## 改写目标

用 T-1 日频构造少量 kbar / 滚动收益 / 滚动波动 / 量比特征，截面 z-score，丢掉 NaN，按分数取 top-k，并用 n_drop 类规则限制与当前持仓的换手。
**默认候选是拟合出来的排序器**：在推断时可见的 PIT 窗口上，用 numpy 当场解 ridge 闭式解，或对二分类标签做少量梯度步的 logistic。等权与符号加权是它必须比过的对照基线，不是本折的目标产物；可选再加一层深度受限的 if/then 规则。
LightGBM 不可 import；sklearn 虽在镜像中也属禁止模块。系数在 `fit(context)` 内用 numpy 拟合并持久化到 `context.state_dir`，`generate_orders` 只读取它；`fit` 有独立的分钟级预算（`budgets.strategy_fit_timeout_seconds`），`generate_orders` 仍受单日 30 秒上限约束。
开发窗口按年切成常规 Fold：每折的验证区间就是那一年，没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月；预先登记至少三条假设——对照基线、拟合排序器和一个结构性变体——用 `batch_validate` 并列跑完整 Validation，并按窗口内的子区间判稳健。需要按调度重拟合的量放在 `fit(context)`，契约以只读 `output/README.md` 为准。

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
