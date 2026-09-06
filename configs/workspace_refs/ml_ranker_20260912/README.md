# 机器学习截面排序器参考包（20260912）

本目录是只读 refs：把「用机器学习/深度学习做 A 股截面排序」收成能在本 ABI 内、在 `fit` 预算内完成（本臂的策略容器可见 GPU，CPU 路径也必须能跑）、且按反过拟合协议判定的可证伪候选。它不是可提交策略，也不是预期收益。

任务只有一句话：在 `fit(context)` 里先训出 LightGBM 排序器基线（Alpha158 风格特征、10 日开盘到开盘截面 rank 标签、topk + n_drop），再比较 MLP 与序列模型（GRU 或小型 Transformer，读 20–60 日 K 线与特征序列）——每一步只改一个可分离的组件（模型族、标签口径、中性化、集成、换手控制），所有训练都在带禁运的清洗时间序列验证与预登记的小网格下完成，并按本包的「没有边际」读法判定。前几轮 ml_numpy 谱系停在闭式 ridge 与手工阈值的 z-score、从未训练过非线性模型；本轮的第一轮就要把树模型与网络真正跑出来。

按以下顺序读：

1. `exploration-plan.md`：轮次、GPU 的用法与边界、验收口径。
2. `models.md`：特征、标签、三个模型族在合同内的实现要点、持久化、预算实测。
3. `protocol.md`：反过拟合协议——标签禁运、清洗验证、网格、滚动重训、集成、去膨胀 Sharpe。
4. `sources.md`：上游来源、文献、已证伪清单。

## 硬合同

- 正式策略写在 `output/` 包内：入口固定为 `output/main.py` 的 `generate_orders(context)`，返回严格 JSON 订单数组；辅助模块可放在 `output/` 下并用绝对导入，每个 `.py` 都受同一套静态检查，包内文件数与总字节数有上限（`max_strategy_files`/`max_strategy_bytes`，以运行事实为准）。
- 正式 import 只允许：纯计算标准库（`__future__`、`collections`、`dataclasses`、`datetime`、`decimal`、`functools`、`itertools`、`math`、`statistics`、`typing`）、`numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、`torch` 及其子模块（`torch.nn`、`torch.optim`），以及 `output/` 内自己的模块。qlib / joblib / pickle 不得 import。
- **正式回放的策略容器跟随本臂的 `gpu_count`**：大于 0 时每次回放的策略容器（含 fit worker）也挂到 GPU，`fit` 可以真的在 GPU 上训练；为 0 时容器内 `torch.cuda.is_available()` 是 `False`，`fit` 与 `generate_orders` 全在 CPU 上跑。本会话的实际配置见 `runtime_env.json` 的 `sandbox_spec`。两种情况下资源与预算都不变：16 核配额、32 GiB、线程数已按配额设置，一次 `fit` 的墙钟上限是 `budgets.strategy_fit_timeout_seconds`，一次决策是 `budgets.strategy_inference_timeout_seconds`；超时即整场回测失败。因此 `output/` 里的代码一律探测设备（`device = "cuda" if torch.cuda.is_available() else "cpu"`），不得假设 CUDA 存在——同一份策略要能在没有 GPU 的实验里原样回放。GPU 也不是独占的：会话容器、`batch_validate` 三路并发回放的推断容器与 fit worker 共享同一批卡，显存按共享估。
- 持久化只有三种形式：`np.save`/`np.savez`/`np.savez_compressed` 到 `context.state_dir`，`DataFrame.to_parquet` 到 `context.state_dir`，booster 的 `save_model(context.state_dir + "/...")`。`torch.save` 被静态拒绝：网络参数用 `np.savez(context.state_dir + "/net.npz", **{name: tensor.detach().cpu().numpy() ...})` 写出，`generate_orders` 里 `np.load` 后 `torch.from_numpy` 装回。`models/` 以只读 `context.models_dir` 挂载，只放跨折继承的静态资产。
- 沙箱无网络，不要 `pip install`，不要下载预训练权重。
- 每一行必须 `available_at <= context.inference_at`；默认 08:30 只用 T-1 及更早日线。标签只用推断时已经实现的收益。
- 不要写死 `/mnt/agent/workspace`。先核对本轮 `data_summary.json` 与单位表，再经 `context.asof_dir` / `context.snapshot_dir` 读数，每次读取都给 `columns=` 与日期窗口。
- 不要把 refs 拷进 `output`。
- Broker 负责 T+1、费用、涨跌停与成交；策略只发订单草图，不要在策略里再写一套回测器。

## 本轮几何与账户

开发窗口按季度步进切成常规 Fold：每折的验证区间是截至本折季度的连续四个季度（滚动四季窗），相邻两折只相差一个季度，因此只有最后一个季度是父本没见过的新数据段，父本对照也只在这一段上真正是样本外；没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，精确窗口以运行事实为准。首折的第一次 `fit` 大约能看到两年日线（约 480 个交易日 × 约 4,500 只可交易股票，扣掉标签禁运后约 180 万个样本），之后随回放推进而增长。

账户是 10 万元真实资金：佣金万一、最低 5 元/笔，过户费 0.1 bp，卖出印花税按成交日切换（切换前万十，切换后万五），方向滑点 5 bp。单只 7,000–10,000 元时最低佣金就是 5–7 bp/边，一次完整换仓的往返成本约 25–35 bp；每日换仓一年付六成以上。本包的起步声明是 `topk=12, n_drop=2`、每 5 个交易日再平衡、标签持有期 10 日。100 股整手：股价 40 元的一手占单只预算四成以上，要么限制股价（T-1 收盘 ≤ 40 元）要么接受权重偏离；科创板 200 股起、北交所 100 股起，通常买不起一手。

## 「没有边际」的读法

每个候选都按下面的清单读，不看整窗总收益：

- `benchmark.neutralized_excess_return` ≈ 0 或为负：只是规模/β 暴露。排序器最容易学到的就是小市值。
- 相对基线（第一轮的 LightGBM 节点）的中性化超额 ≤ 0：模型族或改动没有信息。
- `vs_parent.beats_parent=false`，或父本对照在本折新季度（`sub_windows` 最后一行）为负而候选也没有转正。
- 四个季度子窗的超额符号不一致（同号少于三个）。
- `cost_sensitivity.excess_at_2x_slippage` ≤ 0：毕业按 2 倍滑点压力裁决。
- `pnl_concentration.top5_share_of_gross_gains` 过高。
- `null_control.excess_percentile` 在 0.5 附近，即与同规模随机组合无法区分；选择之前就能对入围节点用 Fold 专用的 `run_null_control(node_id)` 工具算出来（K=500，约 3.5 分钟一次，每折上限 `max_null_controls_per_fold`，默认 3，只给最终候选用）。`selection_statistics.deflated_sharpe_probability` 接近 0，即 N 次尝试里的最大噪声；每一行验证结果都带它，按 `trials_so_far` 读——每个模型族、每个网格点、每个种子都是一次试验。

没有候选证明边际时，用 `finish_fold(outcome="no_edge", reason=...)` 明确弃权（有父本则保留父本，首折记为 `baseline_missing`）是正当结果；提名任何通过硬规则的节点都会被冻结。

## 禁止退回通用因子

本包的候选形态只有拟合出来的排序器及其预登记变体。手写的价值/反转/动量/低波打分不得作为候选，除非在 `hypothesis` 里被登记为「等权符号打分」这一无学习对照。模型训练失败时退回对照必须在订单 metadata 里标明，不得静默。

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
