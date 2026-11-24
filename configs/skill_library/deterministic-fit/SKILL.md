# 可复现的拟合：torch 用严格确定性模式，GPU 上的 XGBoost 从主机数组建 QuantileDMatrix，LightGBM 设 deterministic

合同（Runtime restrictions 一节）要求同一份字节的两次回放下同样的单，而固定种子不足以让一次训练确定。下面是实测过的三个例子、没有测过的范围与核对办法。三处都是代码里的调用或参数，与环境变量无关：策略设不了环境变量，宿主也不替它设。

## 实测：torch 2.13.0+cu130，L20

- 一个只用注意力的模型（`F.scaled_dot_product_attention`，fp32）在 `torch.use_deterministic_algorithms(True, warn_only=True)` 下训练：同一份字节回放 6 次，30 次拟合里每一次的模型都不同（首次拟合后权重最多差约 1e-3）。torch 会打印 "Memory Efficient attention defaults to a non-deterministic algorithm" 的警告：`warn_only=True` 时注意力的反向照旧用非确定的算法。
- 只把这一行改成 `warn_only=False`（严格模式）：4 次回放（其中 1 次设了 `CUBLAS_WORKSPACE_CONFIG`）每次 30 次拟合都逐位相同，没有报错，显存峰值相同。
- 严格模式遇到没有确定性实现的运算会报错：那就是该把这个运算换成确定性写法的信号。这个策略用到的运算（线性层、LayerNorm、注意力）之外没有测过。

## 实测：xgboost 3.2.0，`device="cuda"`，`tree_method="hist"`

- 用带权重的 `xgb.DMatrix` 训练，同一个种子重复拟合得到不同的模型：816,515 行 × 104 列的面板上，预测最大相差约 0.10；关掉子采样（`subsample=1`）也一样。
- 用主机内存里的 numpy 数组建 `xgb.QuantileDMatrix(x, label=y, weight=w, max_bin=...)`，仍在 GPU 上训练：重复拟合的预测与模型字节（`save_raw()`）都完全相同。`XGBRegressor.fit` 在 `hist` 下本来就这样建训练矩阵。这样建的候选，回放之间逐日完全相同（最长 728 个交易日），回放里的拟合时间看不出变化。
- 去掉权重或把特征取整后，预测对上了，模型字节却不总相同：不能依赖。

## 实测：lightgbm 4.7.0，CPU

- 一个没设 `deterministic` 的策略（三个种子，`n_jobs=8`）：同一份字节回放 5 次，首次拟合的三个模型文件每次都不同，20 天的订单没变。回放记录里，这样的策略十二次回放有一次在重训之后换了一个名字。
- 只加一行参数 `deterministic=True, force_row_wise=True`：3 次回放，三个模型文件都逐位相同，订单相同。

## 没有测过的

其他 `tree_method`、别的 GPU 或 xgboost 版本、GPU 上的 LightGBM 都没有测过；LightGBM 单设 `deterministic=True`，或配 `force_col_wise`，没有回放过；torch 的卷积、`torch.compile`、多个 CUDA 流没有测过，严格模式的耗时也没有在同一批里比较过。

## 怎样核对自己的策略

- 在研究工作区用同一份数据与种子拟合两次，比较预测与模型字节。
- 同一份字节跑两次覆盖一次拟合的 `smoke_backtest`：订单计数不同，策略就不确定；计数相同不能反过来证明它确定。
- 同一份字节再 `test` 一次时，返回的 `reproduction`（见 `test` 的描述）为假，策略就不确定。
