# Fold-ready 探索计划

一折跑多轮 `batch_validate`：第一轮并列比较模型族，后续轮次围绕站住的模型族比标签、中性化、集成与换手控制。每一步都要产出可执行候选或明确淘汰。有父本时，必须先完成一次与父本可执行逻辑不同的完整 Validation；父本对照由宿主在会话前跑好，不要自己重跑。

1. 读本轮 `data_summary.json`、`unit_reference.json`、daily schema 与 `runtime_env.json`。确认列名、单位与包版本（`lightgbm`、`torch`）。从运行事实 `budgets.strategy_gpu_count` 读出正式回放的策略容器（含 fit worker）挂几张卡——这是设备口径的权威字段，Fold 与 Meta 都读它。再在会话沙箱 `shell` 里运行 `nvidia-smi` 与 `python -c "import torch; print(torch.cuda.is_available())"`，确认本会话容器自己看得到卡（会话容器的配置在 `runtime_env.json` 的 `sandbox_spec`，逐 Fold 覆盖只改它；Meta 会话没有容器，那里是 `null`）。两者不一致时按 `budgets.strategy_gpu_count` 判断正式回放的设备。
2. 固定不变量并写进工作区笔记：可交易性过滤（T-1 未停牌、ADV20 成交额 ≥ 3,000 万元、T-1 收盘价上限、剔除科创板/北交所或说明为何保留）、`topk=12`、`n_drop=2`、每 5 个交易日再平衡、等额现金、费用缓冲 3%、`REFIT_PERIOD="quarter"`、标签持有期 `h=10`、训练窗 = 输入窗全部可见历史、禁运 `h` 日、网格 ≤ 4 个点。这些是声明值，不是拟合参数。
3. 离线筛查（不占回测预算，只用 `/mnt/snapshot` 决策视图，不碰验证区间）：用 `models.md` 的特征与 `protocol.md` 的清洗验证，在输入窗上把三个模型族各自的验证 rank IC、ICIR、逐月符号一致性与 `fit` 墙钟跑出来。设备一律探测（`device = "cuda" if torch.cuda.is_available() else "cpu"`），会话沙箱只有 8 GiB 内存，特征矩阵用 float32 并按日期分块。墙钟按正式回放实际会用的设备计时：**该设备上的 `fit` 墙钟必须 ≤ `budgets.strategy_fit_timeout_seconds` 的一半**（`batch_validate` 三路并发时三个 fit worker 会争抢宿主 CPU，`budgets.strategy_gpu_count>0` 时还与会话容器共享同一批卡）。同一份代码再以 `CUDA_VISIBLE_DEVICES=""` 跑一次，确认没有 GPU 时仍然正确并记下 CPU 墙钟；CPU 装不进预算的模型族只能作为依赖 GPU 的候选，须在 `hypothesis` 里写明。两条路径都达不到的模型族先缩模型或缩样本，仍达不到的判「本环境不可测」并记录，不能只在卡最闲时跑通就登记为候选。
4. 第一轮 `batch_validate` 三个候选：(a) 等权符号打分对照（无学习）；(b) LightGBM 基线；(c) 同特征同标签的 MLP。三者共用同一过滤、篮子规则、标签与训练窗。`hypothesis` 写清模型族、网格、验证口径、否证条件。
5. 第二轮：序列模型（GRU 或小型 Transformer，二选一并说明理由）对照第一轮胜者；同批可加一个「GRU 只用 Alpha360 原始序列」与「GRU 用 158 特征序列」的变体。第三轮起从 `protocol.md` 与 `models.md` 的家族取一个：标签口径、中性化、跨重训日集成、n_drop 与再平衡节奏。一次只改一个组件。
6. 超参纪律：每个模型族的网格事先写死（≤ 4 个点），在 `fit` 内用清洗验证选点并把选中的点与验证 IC 写进 `context.state_dir` 和订单 metadata；不在 Validation 结果上调参。网格点、种子、模型族都是试验，读 Sharpe 时按 `candidates_evaluated` 打折。
7. 决策时只算最新截面：从 `asof_dir/daily` 读约 100 个交易日（序列模型 60 日 + 预热）的尾窗，算最后一天的特征或序列，加载模型打分，出单；单次决策目标 < 20 s。模块级缓存只键控 `context.asof_version`，冷启动必须得到相同订单。先用 `smoke_backtest` 确认 `fit` 与决策耗时，再跑正式 Validation。
8. 每个候选按 README 的「没有边际」清单读，四个季度子窗分别报相对基线的超额、有效覆盖、换手、拒单、规模倾斜；树模型报特征重要性前十，网络报验证 IC 与训练/验证损失曲线的终点，重要性或梯度集中在市值代理时视为规模暴露的警告。
9. 显式退化：训练失败、验证 IC ≤ 0 或有效覆盖过低时退回等权符号打分，并在订单 metadata 里标明退化——不要静默切换。整个 Validation 大部分决策都在退化路径上时判该候选失败。
10. 用 `step_rollback` 把胜出节点恢复到 `output/` 再 `finish_fold`。不要提交训练脚本、notebook、`.pt`/`.pkl`，也不要提交假定 CUDA 一定存在的代码——设备探测后再用，没有 GPU 时同一份代码必须照样跑完。

## 对照

| 候选 | 必须比过的对照 |
| --- | --- |
| LightGBM 基线 | 等权符号打分 |
| MLP | LightGBM 基线（同特征同标签） |
| GRU / Transformer | 第一轮胜者；`Alpha360` 原始序列 vs 特征序列 |
| 标签口径 / 中性化 / 集成 | 当前胜者的基线口径 |
| n_drop | 每次再平衡全换 topk |

对照相同或更好就保留对照。更复杂的模型要保留，必须在含成本收益与四个季度子窗一致性上都不劣，且至少一项明显更好；难分时看中性化超额与规模倾斜，仍分不出才留更简单的那个。

## Validation 否证

- 任何 `available_at`、T-1、复权锚或标签前视错误：结果无效，不讨论收益。
- 训练与验证之间没有禁运、标签窗越过推断时点、或在 Validation 结果上选网格点：结果无效。
- `fit` 超预算或决策超时：不是策略结果，先把计算压回预算。
- 中性化超额 ≈ 0 或规模倾斜显著为负（小市值）：排序器学到的是规模，淘汰或改用中性化家族。
- 四个季度子窗同号少于三个，或超额只来自单个季度：淘汰。
- 加手续费、滑点、涨跌停与手数约束后优势消失，或 2 倍滑点下超额转负：淘汰。
- 序列模型相对树模型没有可测改善：保留树模型；「深度学习」不是保留理由。
