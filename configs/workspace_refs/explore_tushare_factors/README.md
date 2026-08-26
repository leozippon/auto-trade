# TuShare 因子探索参考包

本目录是只读 refs：只提供研究假说、公式和执行检查，不是可直接提交的策略。
Fold Agent 必须把正式单文件策略写到 `output/main.py`；不要复制本目录生成替代入口，也不要在策略中写死 `/mnt/agent/workspace`。
正式回放只能使用沙箱已有的 NumPy、pandas 和当前 `context`；沙箱不能联网、不能安装包，也没有可调用的 TuShare `factor_value`、`stk_factor` 或 `stk_factor_pro`。

目标不是复刻 202 个供应商因子，而是从当前 PIT parquet 重算一个小集合，完成不同因子家族及其组合的完整 Validation。
默认 08:30 推断只能使用 T-1 及更早的日频行，所有输入仍须满足 `available_at <= inference_at`。

按以下顺序使用：

1. `exploration-plan.md`：10 步可证伪探索流程。
2. `screened-factor-set.md`：42 个候选及紧凑公式；同义项用于二选一，不应全部堆叠。
3. `compute-and-pit.md`：PIT、复权、财务版本、截面处理和性能纪律。
4. `sources.md`：官方页面、A 股研究证据和本仓库数据交叉核对。

正式策略先读取本次 `data_summary.json`/manifest 所确认的路径、列和单位，再从 `context.asof_dir` 下的只读 parts 计算特征。Refs 不是数据合同；本次运行的 schema 才是。
