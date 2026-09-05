# 角落状态（corner cases）参考包（20260907）

本目录是只读 refs：把普通截面策略过滤掉或处理错的 A 股边缘状态——涨跌停板、停牌复牌、ST 状态切换、新股初期、除权除息、解禁事件窗、极端市场日——收成可在本仓库 PIT 重算的可证伪假说。这不是策略产物，也不是预期收益。

本方向不是「打板 2.0」。前代包 `limitup_board_20260826` 只做涨停延续且依赖 `inference_time=09:29` 的竞价过滤；本轮默认 `inference_time=08:30`，当日竞价、当日 `stk_limit`/`suspend_d`、当日日线全部不可见，板动力学只是七个家族之一，且只保留其 T-1 日终改写（见 `sources.md` 的沿革与失效清单）。

角落状态的共同特征是：事件稀疏、执行受限、风险集中。所以本包的三条不可谈判纪律写在最前面：

1. **可执行性必须显式论证。** Broker 对超出当日涨跌停的买/卖、停牌标记、手数与费用一律硬拒：封死的涨停买不进，跌停卖不出，复牌当日常被拒单（见字段图），新股首日没有可见 T-1 行情。每个 playbook 都必须写明真实可达的入场与出场路径（默认只有 09:30 开盘与 15:00 收盘两个成交时点），并把拒单当成结果汇报，不当成要绕过的 bug。
2. **回撤上限。** Held-out 毕业要求最大回撤 ≤ 0.25（绝对值）；Validation 超限只记警告、不阻止冻结，因此它是整条策略链最终要过的线，不是可以推后处理的目标。角落 cohort 天然集中于少数票、少数日，仓位上限、篮子分散、无信号日持币是本方向的一等公民，不是可选项。
3. **PIT 与预登记否证。** 每一行输入必须 `available_at <= context.inference_at`；每个机制在正式 Validation 之前写清预期方向与否证条件，事件数不足如实判「不可测」。诚实的零结果与被证伪的方向都是本方向的合格产出。

按以下顺序读：

1. `exploration-plan.md`：本折的事件数普查、cohort 事件研究顺序与验收口径。
2. `playbooks.md`：七个角落家族的 PIT 改写与否证。
3. `pit-field-map.md`：本轮可用数据集的可见时间、单位与陷阱，以及默认快照里没有的东西。
4. `sources.md`：沿革（前代打板包、姊妹实验的除权种子假说）、范围边界与外部文献。

## 硬合同

- 正式策略写在 `output/` 包内：入口固定为 `output/main.py` 的 `generate_orders(context)`，返回严格 JSON 订单数组；辅助模块可放在 `output/` 下并用绝对导入（如 `from lib.features import x`），每个 `.py` 都受同一套静态检查。
- 正式 import 只允许：纯计算标准库（`__future__`、`collections`、`dataclasses`、`datetime`、`decimal`、`functools`、`itertools`、`math`、`statistics`、`typing`）、`numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、`torch`（仅 CPU）及其子模块，以及 `output/` 内自己的模块。qlib / vnpy / joinquant / joblib / pickle 仍不得 import；模型参数用 NumPy 数组或 booster 的 `save_model(context.state_dir + ...)` 持久化。
- 需要拟合的量放在 `fit(context)` 并写入 `context.state_dir`；`models/` 以只读 `context.models_dir` 挂载。不要加载 `.pkl` / `.pt`。
- 沙箱无网络，不要抓网页。
- 每一行必须 `available_at <= context.inference_at`。
- 不要写死 `/mnt/agent/workspace`。先核对本轮 manifest / `data_summary.json`，再经 `context.asof_dir` / `context.snapshot_dir` 读数。
- 不要把 refs 拷进 `output`。
- Broker 负责 T+1、费用、涨跌停、停牌与成交；策略只发订单草图。涨停开盘拒单、复牌日拒单都是真实结果。
- 不要粘贴受版权保护的完整源码。只引用 URL 并转述机制。

开发窗口按季度步进切成常规 Fold：每折的验证区间是截至本折季度的连续四个季度（滚动四季窗），相邻两折只相差一个季度，因此只有最后一个季度是父本没见过的新数据段，父本对照也只在这一段上真正是样本外；没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，精确窗口以运行事实为准。可用的完整 Validation 次数远多于机制数量（上限以本轮事实为准）。先在输入窗上做各家族的事件数普查与事件研究，再预登记最强的三条互斥机制，用 `batch_validate` 并列跑完整 Validation；站住的机制进入后续轮次（门控变体、`fit` 里拟合的 cohort 内排序或阈值、持有期与仓位规则），一折跑多轮，并始终按年内半窗与子窗判稳健。

## 本轮边界

默认 `inference_time=08:30`：当日日线、`daily_basic`、`adj_factor`（09:30 盖章）、当日 `stk_limit` 与 `suspend_d`（08:45 盖章）、当日竞价全部不可见，一切当日约束只能用 T-1 代理估计，真实约束由 Broker 在成交时执行。前代 09:29 包的竞价缺口过滤在本轮不存在，不要复刻。
股票池未经任何筛选（含 ST、次新、北交所），可交易性由策略自理并说明理由。判定一律相对匹配对照，并要求年内两个半窗、四个子窗的符号一致；整年合计为正但子窗符号相反的候选不算胜出。

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
