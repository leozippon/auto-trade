# 可复现的拟合：GPU 上带权重的 XGBoost 从主机数组建 QuantileDMatrix

合同（Runtime restrictions 一节）要求同一份字节的两次回放下同样的单，而固定种子不足以让一次训练确定。下面是实测过的一个例子、没有测过的范围与核对办法。

## 实测：xgboost 3.2.0，`device="cuda"`，`tree_method="hist"`

- 用带权重的 `xgb.DMatrix` 训练，同一个种子重复拟合得到不同的模型：816,515 行 × 104 列的面板上，预测最大相差约 0.10；关掉子采样（`subsample=1`）也一样。
- 用主机内存里的 numpy 数组建 `xgb.QuantileDMatrix(x, label=y, weight=w, max_bin=...)`，仍在 GPU 上训练：重复拟合的预测与模型字节（`save_raw()`）都完全相同。`XGBRegressor.fit` 在 `hist` 下本来就这样建训练矩阵。这样建的候选，回放之间逐日完全相同（最长 728 个交易日），回放里的拟合时间看不出变化。
- 去掉权重或把特征取整后，预测对上了，模型字节却不总相同：不能依赖。

## 没有测过的

其他 `tree_method`、别的 GPU 或 xgboost 版本、GPU 上的 LightGBM 与 torch 都没有测过。LightGBM 只有一条提醒，不是测量：一个默认设置的 LightGBM 策略在十二次回放里有一次在重训之后换了一个名字，而设了 `deterministic=True` 并配 `force_col_wise` 或 `force_row_wise` 的策略从未分叉。

## 怎样核对自己的策略

- 在研究工作区用同一份数据与种子拟合两次，比较预测与模型字节。
- 同一份字节跑两次覆盖一次拟合的 `smoke_backtest`：订单计数不同，策略就不确定；计数相同不能反过来证明它确定。
- 同一份字节再 `test` 一次时，返回的 `reproduction`（见 `test` 的描述）为假，策略就不确定。
