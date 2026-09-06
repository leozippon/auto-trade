# 特征、标签与三个模型族

记 `C/O/H/L/V/A` 为截至 T-1 的冻结前复权价、成交量（股）与成交额（元），`VWAP = A/V`，`turn` 为换手率（小数）。缺值保持为空，训练时丢行，推断时该股票不进排名。

## 特征

- **Alpha158 风格**：9 个 K 线形状算子（`(C-O)/O`、`(H-L)/O`、`(C-O)/(H-L+eps)`、上下影线相对 `O` 与相对 `H-L`、`(2C-H-L)/O` 及其 `/(H-L+eps)` 版本）、4 个价格位置（`O/C`、`H/C`、`L/C`、`VWAP/C`），以及 29 个滚动算子 × 窗口 `{5,10,20,30,60}`：`C.shift(d)/C`、`mean(C,d)/C`、`std(C,d)/C`、`C` 对时间序号 d 日 OLS 的斜率/C、R²、残差/C、`max(H,d)/C`、`min(L,d)/C`、`quantile(C,d,0.8)/C`、`quantile(C,d,0.2)/C`、`C` 在 d 日内的百分位、`(C-min(L,d))/(max(H,d)-min(L,d)+eps)`、`argmax(H,d)/d`、`argmin(L,d)/d`、两者之差/d、`corr(C, log(V+1), d)`、`corr(C/C.shift(1), log(V/V.shift(1)+1), d)`、上涨日占比、下跌日占比、两者之差、上涨幅度占比 `sum(max(ΔC,0))/sum(|ΔC|)`、下跌幅度占比、两者之差、`mean(V,d)/(V+eps)`、`std(V,d)/(V+eps)`、`std(|r|V,d)/mean(|r|V,d)`、放量占比 `sum(max(ΔV,0))/sum(|ΔV|)`、缩量占比、两者之差。共 158 列。再加 `log(circ_mv)` 与 `turn` 的 20 日均值作为两列显式风格变量（中性化家族会把它们拿掉）。
- **Alpha360 风格（序列模型用）**：过去 60 日的 `C.shift(i)/C`、`O.shift(i)/C`、`H.shift(i)/C`、`L.shift(i)/C`、`VWAP.shift(i)/C`、`V.shift(i)/(V+eps)`，可再加 `turn.shift(i)`，即 `[60, 6或7]` 的张量。
- 标准化：每列按当日可见截面的中位数/MAD（× 1.4826）标准化并截到 ±3，缺失填 0；或按训练区间统计并把参数写进 `context.state_dir`。二选一并保持稳定。

## 标签

对特征日 `T`（决策日前一交易日），持有期 `h`（默认 10）：

```text
label_h(T) = O_{T+1+h} / O_{T+1} - 1                  qfq 开盘价：09:30 成交的真实口径
excess_h(T) = label_h(T) - 截面均值
target(T)   = 截面 rank(excess_h) 映射到 [-0.5, 0.5]
```

拟合日 `D` 只能用 `T + 1 + h <= D - 1` 的样本；训练/验证之间空出 `h` 个交易日禁运（见 `protocol.md`）。

## 模型族一：LightGBM（基线）

- `objective="regression"`（对 rank 标签）；变体 `lambdarank`（按日期分组，`label_gain` 需要整数等级，把 rank 分成 10 档）。
- 声明参数：`num_leaves ∈ {31, 63}` × `learning_rate ∈ {0.05, 0.1}` 为网格；固定 `feature_fraction=0.8`、`bagging_fraction=0.8`、`bagging_freq=1`、`min_data_in_leaf=200`、`lambda_l2=10`、`num_boost_round ≤ 600` 由清洗验证早停决定；`num_threads` 取自环境变量 `OMP_NUM_THREADS`（容器已按 CPU 配额设置，不要自己设更大）。
- 180 万行 × 160 列在 16 线程上训练一个 booster约 2–5 分钟；网格 4 点 + 最终重训在预算内。
- 本族按 CPU 训练：PyPI 的 lightgbm wheel 通常不带 GPU 后端，容器里也不能重编译。想要 GPU 树模型只有 XGBoost 的 `device="cuda"` 一条路，且必须先在会话沙箱里实测确认它在本镜像上真的可用，再登记为候选；跑不起来就如实记录，不要假设。
- 持久化：`booster.save_model(context.state_dir + "/lgb.txt")`；推断：`lgb.Booster(model_file=context.state_dir + "/lgb.txt")`，每次 `generate_orders` 都重新加载（重训会替换文件而不重启 worker）。

## 模型族二：MLP（torch）

- 结构：`160 → 256 → 128 → 1`，ReLU，Dropout 0.2，BatchNorm 可选；损失 MSE（对 rank 标签）或按日期分组的成对 rank 损失（每个日期抽 64 对）；Adam `lr=1e-3`，`weight_decay=1e-5`，batch 4096，最多 10 个 epoch，按清洗验证 rank IC 早停（耐心 2）。网格 ≤ 4 点：`hidden ∈ {(256,128), (128,64)}` × `dropout ∈ {0.1, 0.3}`。
- 可行性：180 万行、batch 4096 每 epoch 约 440 步，16 线程 CPU 下每 epoch 约 1–3 分钟；10 个 epoch 加网格约 30–60 分钟，逼近预算。有 GPU 时同一配置快得多，但卡是共享的，仍要实测。两种设备都先在会话沙箱里计时（CPU 用 `CUDA_VISIBLE_DEVICES=""`），正式回放所用设备上超过预算一半就把网格缩到 2 点或 epoch 缩到 5。
- 持久化：`np.savez(context.state_dir + "/mlp.npz", **{name: p.detach().cpu().numpy() for name, p in model.state_dict().items()})`，另存标准化参数与网格选择；推断时 `np.load` 后 `model.load_state_dict({k: torch.from_numpy(v) ...})`，`model.eval()`，`torch.no_grad()`。
- 线程：容器已设 `OMP_NUM_THREADS`，torch 的 intra-op 线程随之生效；不要在策略里调用 `torch.set_num_threads` 设更大值。
- 设备：`output/` 里也用 `device = "cuda" if torch.cuda.is_available() else "cpu"`，`fit` 与 `generate_orders` 各自探测一次。本臂 `gpu_count>0` 时策略容器有卡，但同一份策略必须在没有卡的实验里原样跑通，所以不要写死 `.cuda()`、不要按 GPU 存在与否改变模型结构或超参。参数一律以 `p.detach().cpu().numpy()` 落盘，装载后再 `.to(device)`。

## 模型族三：序列模型（GRU 或小型 Transformer，torch）

- 输入：`[60, 6或7]` 的 Alpha360 风格序列（或 40 日 × 158 特征序列的降维版，先做 PCA 到 32 维再喂序列，PCA 参数写进 state）。
- GRU：`hidden=64`、1–2 层、末步隐状态接线性头；Transformer：`d_model=64`、4 头、2 层、序列长度 40，均值池化接线性头。损失与优化同 MLP。网格 ≤ 2 点（`hidden ∈ {32, 64}` 或层数 `{1, 2}`）。
- 预算是本族的门槛：180 万个 60 步序列每 epoch 的前后向远重于 MLP，CPU 上尤其重。声明做法：训练样本只取输入窗最后 250 个交易日、每日随机抽 1,500 只（约 37 万个序列），batch 1024，最多 5 个 epoch，先在正式回放会用的设备上实测单 epoch 墙钟；单 epoch > 6 分钟就再缩 `hidden` 或序列长度到 40/20。GPU 上放开样本量之前先想清楚卡是共享的，实测要留余量；仍装不进预算一半的判「本环境不可测」并记录，不得靠一次空载实测就登记候选。
- 持久化与推断同 MLP（`np.savez` 各层参数）；推断只需最新一天的序列，`[4500, 60, 7]` 的前向在 CPU 上是秒级。

## 决策时的计算

- `generate_orders` 只读约 100 个交易日的尾窗（`filters=[("trade_date", ">=", start)]`，只取所需列），算最后一天的特征或序列，加载模型打分，按 `topk/n_drop` 出单；目标 < 20 s，硬上限是 `budgets.strategy_inference_timeout_seconds`。
- 不要在决策时重算全历史特征；不要把 158 列的全历史表留在模块级缓存里（worker 常驻内存有上限，且 `asof_version` 变化就要重算）。
- 前几轮的教训：三个候选在宿主争用下被决策超时误杀过一次；现在容器就绪握手已把启动时间移出决策时钟，但策略自己的计算仍要留 5 倍余量。

## 持仓与订单

- `topk=12`、`n_drop=2`，每 5 个交易日再平衡：在持仓里按最新分数从低到高最多标 2 只为卖出，已不在合法宇宙或没有分数的持仓也卖；从排名里补进未持有的高分股票到 12 只。
- 卖先于买；买入按等额现金、向下取整到 100 股、本地递减现金并留 3% 缓冲；不把未成交卖出当成已到账；非再平衡日返回 `[]`。
- 订单 metadata 写模型族、网格选点、验证 IC、是否退化；不要写未发生的收益。
