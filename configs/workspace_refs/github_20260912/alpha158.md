# Alpha158 的本地转述

Qlib 的 `Alpha158` 处理器（`qlib/contrib/data/handler.py` 里的 `Alpha158` 类，特征表达式由 `qlib/contrib/data/loader.py` 的 `Alpha158DL.get_feature_config` 生成）是 9 个 K 线形状算子、4 个价格位置算子和 29 个滚动算子 × 5 个窗口（5/10/20/30/60）= 158 列。本包把它们转述成本地 pandas 定义；沙箱不挂载上游仓库，下面的公式就是全部。记 `C/O/H/L/V/A` 为截至 T-1 的冻结前复权价、成交量（股）与成交额（元），`VWAP = A / V`，`d` 为窗口长度，`eps = 1e-12`。上游用 `$close` 等原始列并在训练前做标准化，本地先按 T-1 冻结锚前复权再算。

## K 线形状（9）

| 名称 | 定义 |
| --- | --- |
| `KMID` | `(C-O)/O` |
| `KLEN` | `(H-L)/O` |
| `KMID2` | `(C-O)/(H-L+eps)` |
| `KUP` | `(H-max(O,C))/O` |
| `KUP2` | `(H-max(O,C))/(H-L+eps)` |
| `KLOW` | `(min(O,C)-L)/O` |
| `KLOW2` | `(min(O,C)-L)/(H-L+eps)` |
| `KSFT` | `(2C-H-L)/O` |
| `KSFT2` | `(2C-H-L)/(H-L+eps)` |

## 价格位置（4）

`OPEN0 = O/C`、`HIGH0 = H/C`、`LOW0 = L/C`、`VWAP0 = VWAP/C`（上游默认窗口只有 0）。

## 滚动算子（29 × 5）

对每个 `d ∈ {5, 10, 20, 30, 60}`，按股票的时间序列计算：

| 名称 | 定义 | 先验方向（只用于等权对照） |
| --- | --- | --- |
| `ROC` | `C.shift(d)/C` | + |
| `MA` | `mean(C,d)/C` | + |
| `STD` | `std(C,d)/C` | − |
| `BETA` | `C` 对时间序号做 d 日 OLS 的斜率 `/C` | |
| `RSQR` | 同一回归的 R² | |
| `RESI` | 同一回归最后一期残差 `/C` | |
| `MAX` | `max(H,d)/C` | |
| `MIN` | `min(L,d)/C` | |
| `QTLU` | `quantile(C,d,0.8)/C` | |
| `QTLD` | `quantile(C,d,0.2)/C` | |
| `RANK` | 当前 `C` 在过去 d 日里的百分位 | − |
| `RSV` | `(C-min(L,d))/(max(H,d)-min(L,d)+eps)` | − |
| `IMAX` | `argmax(H,d)/d`（距最高点的相对天数） | |
| `IMIN` | `argmin(L,d)/d` | |
| `IMXD` | `(argmax(H,d)-argmin(L,d))/d` | |
| `CORR` | `corr(C, log(V+1), d)` | − |
| `CORD` | `corr(C/C.shift(1), log(V/V.shift(1)+1), d)` | − |
| `CNTP` | `mean(C>C.shift(1), d)` | |
| `CNTN` | `mean(C<C.shift(1), d)` | |
| `CNTD` | `CNTP-CNTN` | |
| `SUMP` | `sum(max(C-C.shift(1),0),d)/(sum(abs(C-C.shift(1)),d)+eps)` | |
| `SUMN` | `sum(max(C.shift(1)-C,0),d)/(sum(abs(C-C.shift(1)),d)+eps)` | |
| `SUMD` | `SUMP-SUMN` | |
| `VMA` | `mean(V,d)/(V+eps)` | + |
| `VSTD` | `std(V,d)/(V+eps)` | |
| `WVMA` | `std(abs(C/C.shift(1)-1)*V, d)/(mean(abs(C/C.shift(1)-1)*V, d)+eps)` | |
| `VSUMP` | `sum(max(V-V.shift(1),0),d)/(sum(abs(V-V.shift(1)),d)+eps)` | |
| `VSUMN` | `sum(max(V.shift(1)-V,0),d)/(sum(abs(V-V.shift(1)),d)+eps)` | |
| `VSUMD` | `VSUMP-VSUMN` | − |

先验方向只用于第一轮的等权符号对照，取标了方向的算子在 `d=20` 的版本各一个（`ROC`、`MA`、`STD`、`RANK`、`RSV`、`CORR`、`CORD`、`VMA`、`VSUMD` 加 `KMID`）rank 后等权；它是对照，不是候选。

## Alpha360（可选变体，家族 D 之外不要先做）

`Alpha360` 是过去 60 日的原始 OHLCV 归一化序列：对 `i = 59..0`，`CLOSE{i} = C.shift(i)/C`、`OPEN{i} = O.shift(i)/C`、`HIGH{i}`、`LOW{i}`、`VWAP{i}` 同理，`VOLUME{i} = V.shift(i)/(V+eps)`，共 360 列。它给序列模型用；树模型上通常不如 Alpha158，本包只在 `families.md` 家族 D 允许时作为一个候选。

## 标签

上游默认标签 `Ref($close,-2)/Ref($close,-1)-1` 是他们平台上对 T+1 的处理（当日收盘后最早 T+1 买、T+2 卖）。本环境 08:30 决策、09:30 成交，所以对特征日 `T`（= 决策日前一交易日）：

```text
label_h(T) = O_{T+1+h} / O_{T+1} - 1            （qfq 开盘价，h 为持有期，默认 10）
excess_h(T) = label_h(T) - 截面均值（或减同期沪深 300）
target(T) = 截面 rank(excess_h)，映射到 [-0.5, 0.5]    （CSRankNorm 的转述）
```

拟合日 `D` 只能用 `T + 1 + h <= D - 1` 的样本（标签已完全实现），训练集与验证集之间空出 `h` 个交易日的禁运。上游 `learn_processors` 是 `DropnaLabel` + `CSRankNorm`，本地等价于丢掉标签缺失的行再截面 rank。

## 特征标准化

上游 `infer_processors` 是 `RobustZScoreNorm(clip_outlier=True)` + `Fillna`：每列用训练区间的中位数与 MAD（× 1.4826）标准化并截到 ±3，缺失填 0。本地两种等价做法二选一并保持稳定：按当日可见截面的中位数/MAD（截面内），或按训练区间的中位数/MAD 并把参数写进 `context.state_dir`（时间序列内）。不要在全历史股票全集上算统计量，不要用未来截面。

## LightGBM 参数（上游示例）

`examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml` 里的 `LGBModel` kwargs（凭公开文件转述，若上游仓库可读请核对）：`loss=mse`、`colsample_bytree=0.8879`、`learning_rate=0.2`、`subsample=0.8789`、`lambda_l1=205.6999`、`lambda_l2=580.9768`、`max_depth=8`、`num_leaves=210`、`num_threads=20`；`qlib/contrib/model/gbdt.py` 默认 `num_boost_round=1000`、`early_stopping_rounds=50`。那是 CSI300 十几年样本的参数。本包首轮声明值：`objective=regression`（对 rank 标签）、`num_leaves=63`、`learning_rate=0.05`、`feature_fraction=0.8`、`bagging_fraction=0.8`、`bagging_freq=1`、`min_data_in_leaf=200`、`lambda_l2=10`、`num_boost_round ≤ 600` 由带禁运的验证早停决定，线程数取自 `OMP_NUM_THREADS`（容器已按 CPU 配额设置）。持久化：`booster.save_model(context.state_dir + "/lgb.txt")`。

## TopkDropout 的转述

`qlib/contrib/strategy/signal_strategy.py` 的 `TopkDropoutStrategy`：大约持有 `topk` 只；每次再平衡卖掉当前持仓里分数最差的 `n_drop` 只，从未持有的高分股票里补进同样数量；`hold_thresh` 决定最短持有天数。本环境没有 Qlib executor：用 `context.account.positions` 当持仓，自己发买卖草图。

1. 在持仓里按最新分数从低到高最多标 `n_drop` 只为卖出；已不在合法宇宙或没有分数的持仓也卖。
2. 从排名里按分数从高到低补进未持有的代码，直到目标达到 `topk`。
3. 卖先于买；买入按等额现金、向下取整到 100 股、本地递减现金并留费用缓冲；不把未成交卖出当成已到账。
4. 非再平衡日返回 `[]`。

对照实验：关掉 n_drop，每次再平衡直接换成最新 topk。全换更好就不要为了「更像 Qlib」保留 n_drop。
