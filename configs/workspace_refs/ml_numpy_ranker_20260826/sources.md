# 来源与禁止项

只引用公开页面并转述机制。不要 vendor Qlib 树，不要粘贴 yaml/源码全文。

## Qlib（引用，禁止拷贝）

- 项目：[https://github.com/microsoft/qlib](https://github.com/microsoft/qlib)。只引用，不要 vendor 该树，不要粘贴源码或 yaml 全文。
- 工作流文档：[Workflow Management](https://qlib.readthedocs.io/en/latest/component/workflow.html)。公开骨架是 Data → Model 训练/推断 → 信号评估/回测。本环境没有 `qrun`、`DatasetH`、`Recorder`；正式策略只有 `generate_orders`。
- LightGBM + Alpha158 示例配置：[examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml](https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)
- 同文件里的 `TopkDropoutStrategy` kwargs：`topk=50`，`n_drop=5`。这是起步假说，不是要抄的魔法数。公开含义见 `ranking.md`。
- `Alpha158` 处理器在 `qlib.contrib.data.handler`：日频 kbar、滚动统计与量价组。本包只保留四组本地映射，不追求列对齐。
- 默认标签 `Ref($close,-2)/Ref($close,-1)-1`：Qlib 文档写明这是 **T+1 到 T+2** 的收益，因为 A 股当日收盘后最早 T+1 买、T+2 卖。本地改写仍必须遵守本仓库的 `execute_at` 与 Broker T+1，不能按字面用未来收盘当当日标签，也不能再实现一套 Qlib backtest。

这些页面上的 CSI300 IC、超额、年化、回撤是 **Qlib 自己的 cn_data + 他们的 executor** 跑出来的。样本、复权、费用、涨跌停和成交模型都不同。把那些表写进本策略预期属于编造。

## 学习器的边界

正式 `ALLOWED_MODULES` 含 `numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、CPU `torch` 及其子模块，完整清单以只读 `output/README.md` 为准。所以本包不因为库缺失而降级学习器：第一轮仍用自实现的闭式解或极浅规则建立可比基线，后续轮次把库模型排序器作为结构性变体与它同窗比较（见 `README.md` 与 `rewrite-plan.md`）。
真正稀缺的不是库，是可信的标签口径、拟合窗声明和含成本的对照——「机器学习」在本包里指的是这套纪律下的排序器，不是把 Qlib 的梯度提升配置搬过来。

## 运行时模型目录

`models/` 存放跨 Fold 继承的静态资产，正式策略进程以只读 `context.models_dir` 挂载它。回放内的拟合属于 `fit(context)`：它把系数写到 `context.state_dir`，`generate_orders` 只读该目录。pickle / torch 加载仍被静态检查拒绝；允许的是 `np.load`/`np.save`/`np.savez`/parquet 与 booster 自己的 `save_model`/`load_model`，且首个路径参数必须是直接以某个 context 目录为根的字面拼接。

## 不要做的移植

- vn.py、RQAlpha、Backtrader、JoinQuant 回测 API。
- Qlib `DatasetH` / `Trainer` / `Recorder` / 数据库 provider。
- 预训练 `*.pkl` / `*.lgb` / `*.pt`。
- 把对方 README 收益复制进 `generate_orders` 注释当验收标准。
