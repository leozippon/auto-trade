# 开局不必重新普查合同

每折开局都会派两三个只读子代理去重新导出同一批固定事实：产物入口、`context` 面、只读根、工具预算、单位查法、结果字段。它们每折都得出同样的答案，却落在第一次完整 Validation 之前的关键路径上，还把结论塞进此后每一次 prompt。下面是那批答案。有疑问时读权威件的对应段（只读 `output/README.md`、运行事实、`inputs/` 下的摘要），不要派子代理去重新概括它。

## 产物合同

- `output/` 是以 `main.py` 为入口的 Python 包。`main.py` 必须定义恰好一个同步单参数 `generate_orders(context)`，最多再定义一个同步单参数 `fit(context)`，`REFIT_PERIOD` 若存在必须是模块级的 `day`/`month`/`quarter`/`year` 或 `None`（省略或 `None` = 每次回放只拟合一次）。
- `fit` 在每次回放第一次决策前调用一次，并在落入新 `REFIT_PERIOD` 的第一次决策再调用，拿到的是当天 `generate_orders` 同一个 context，因此看不到决策看不到的行。`fit` 可写 `context.state_dir`，`generate_orders` 只能读它；该目录每次回放都从空开始重建，不进 revision、不进冻结产物。重的计算放 `fit`（独立且宽得多的超时），`generate_orders` 只读系数、算当日特征。
- 包内用绝对导入引用兄弟模块（`from lib.features import momentum` 对应 `output/lib/features.py`），相对导入被拒；每个 `.py` 都是产物，计入文件数、字节上限与指纹。
- 跨 Fold 继承的静态资产放 `models/`，运行时以只读 `context.models_dir` 出现（`np.load`、`pd.read_parquet`、`torch.load`、booster 的 `load_model` 可读；pickle/joblib 不可）。回放期拟合出来的东西只写 `context.state_dir`，两者都不进 `output/`。
- `context.asof_dir` 每个域是一个 **parts 目录**（`pd.read_parquet(context.asof_dir + "/daily")`），`context.snapshot_dir` 每个域是一个**扁平文件**（`.../daily.parquet`）。写错这一处是回放死在第一天最常见的原因，`modification_check` 会静态拦下扁平写法；as-of 读失败不得回退到 snapshot，那是 PIT 违规而不是补救。
- 订单是严格 JSON 数组（无单可交空数组），每单必须有 `symbol`/`action`/`quantity`/`execute_at`；`09:30` 用当日 open、`15:00` 用当日 close，其余时刻要求同一分钟的历史分钟行，缺价整单被拒。

## 只读根与拷贝

- 可写根只有 `workspace`、`output`、`models`；`snapshot`、`asof`、`steps`、`parent_output`、`inputs`、`refs`、`memory` 都是只读。
- 只读产物树里的文件是 0444，`cp` 会照抄这个模式：从 `steps`/`parent_output` 拷进可写树后，必须先在 `shell` 里 `chmod -R u+w <目标>` 才能编辑，否则第一次 `edit_file`/写入就失败。
- `batch_validate` 会把只读模板文件（`README.md`）补进每个缺它的 `candidates/<name>/`。不要自己拷、改或删它：它属于只读基线，一旦与播种时的字节不同，整批在预检就被拒，而错误指向的是那个文件，不是策略逻辑。

## 工具与预算的固定事实

- `smoke_backtest` 走真实回放路径但不占回测名额、不产生可选择节点；`daily_backtest` 与 `batch_validate` 是唯一产生节点的调用。`batch_validate` 一次 2–6 个候选，每个候选各占一次回测与一个 Step，整批在任何东西开跑之前一次性过预检。
- `daily_backtest`、`batch_validate`、`run_null_control`、`ask_user` 期间推理时钟暂停；**等待子代理不暂停**，它照常计入本折的有效推理时间。
- 被拒的批次不消耗名额，但同一签名（错误类型加被拦目标）重复出现会升级提示，再重复开始按次扣回测名额：同一个调用连拒三次时，问题在输入而不在重试。
- 具体数字（推理时限、回测与 Step 上限、单次决策与 `fit` 超时、`run_null_control` 每折次数、文件与字节上限）只以运行事实 `budgets`、`acceptance_rules` 和只读 `output/README.md` 为准，不要凭上一折的记忆。

## 单位

`inputs/` 下的 `data_summary.json` 与 `unit_reference.json` 是本次会话按当前注册表重新生成的、覆盖本视图全部列的权威表。按（文件、`dataset`、列）三元组查，不要按同名字段猜，也不要搬上一折的换算：同名列在不同数据集里单位不同是常态。`data_summary.json` 的 `snapshot` 与 `valid` 两个 view 要分别读，前者推不出后者。

## 读结果

- 每个候选行的 `vs_parent` 是相对本折父本对照的整窗差值；`beats_parent` 要求原始与中性化两个超额差值都为正。没有父本对照时它是 null 并带 `vs_parent_note`；候选逐位复现父本结果时带 `identical_to_parent`，那是覆盖层从未触发，不是比较失败。
- `selection_statistics.deflated_sharpe_probability` 接近 0，表示胜者只是 N 次尝试里的最大噪声。
- `run_null_control(node_id)` 只对决赛候选按需算 `excess_percentile`，0.5 附近表示与同规模随机组合无法区分。父本对照的分位已经在运行事实 `parent_control` 里，不必再花一次名额去算。

## 开局把子代理用在哪里

上面这些不随折变化。真正只有本折才知道、值得开局并行去查的是：本折窗口里各数据域的实际可用性与覆盖、父策略 `output/` 到底实现了什么机制和它已知的失效点、PRIOR 与已挂载 skills 里与本折假设直接相关的结论。合同类问题直接读权威件的那一段，一轮之内自己解决。
