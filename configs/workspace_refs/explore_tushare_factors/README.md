# TuShare 因子探索参考包

本目录是只读 refs：只提供研究假说、公式和执行检查，不是可直接提交的策略。
Fold Agent 必须把正式策略写在 `output/` 包内（入口 `output/main.py`，辅助模块用绝对导入）；不要复制本目录生成替代入口，也不要在策略中写死 `/mnt/agent/workspace`。
正式回放只能使用运行合同允许的库（NumPy、pandas、scipy、sklearn、LightGBM、XGBoost、statsmodels、CPU torch）和当前 `context`；沙箱不能联网、不能安装包，也没有可调用的 TuShare `factor_value`、`stk_factor` 或 `stk_factor_pro`。

目标不是复刻 202 个供应商因子，而是从当前 PIT parquet 重算一个小集合，完成不同因子家族及其组合的完整 Validation。
默认 08:30 推断只能使用 T-1 及更早的日频行，所有输入仍须满足 `available_at <= inference_at`。
开发窗口按年切成常规 Fold：每折的验证区间就是那一年，没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，可用的完整 Validation 次数远多于家族数量（上限以本轮事实为准）。一折要跑多轮 `batch_validate`：候选先在输入窗上离线筛掉，幸存家族预先登记后并列跑完整 Validation；一轮胜出后围绕胜者再登记下一轮——组合方式、在 `fit` 里拟合的因子权重（ridge / 树模型打分）、稳健性与节奏变体——直到预算或假设用尽，并始终按窗口内的子区间判稳健。

按以下顺序使用：

1. `exploration-plan.md`：本折的可证伪探索流程与验收口径。
2. `screened-factor-set.md`：42 个候选及紧凑公式；同义项用于二选一，不应全部堆叠。
3. `compute-and-pit.md`：PIT、复权、财务版本、截面处理和性能纪律。
4. `sources.md`：官方页面、A 股研究证据和本仓库数据交叉核对。

正式策略先读取本次 `data_summary.json`/manifest 所确认的路径、列和单位，再从 `context.asof_dir` 下的只读 parts 计算特征。Refs 不是数据合同；本次运行的 schema 才是。

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
