# 所有者的提案：写入新记忆而不扰动无关的键值

所有者在「持续学习」这个总题目下写了几条方向。下面是本臂这一条和三个臂共用的领域框架，原文一字未改。

## Direction 2: Overcoming the Lack of Global Optimization in Continual Learning

- **Updating related memories without disturbing unrelated key-value (KV) representations:** Explore mathematical optimization methods that adjust relevant memories when new KV representations are introduced, while preserving unrelated ones.

## Direction 3: Domain Applications

- **Continual memory adaptation for financial time-series data:** Financial data arrives rapidly and in large volumes, making daily updates challenging. Investigate how memory systems can effectively adapt to the temporal structure and continuous arrival of financial data.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的原话。它写的是一般的模型；本臂把它落到 A 股的截面选股上，每个词的所指如下，`refs/README.md` 的方向与对照都按这张表读。右列只说明所指，不规定做法：机制的实现由会话设计。

| 提案里的词 | 本臂里的所指 |
|---|---|
| key-value (KV) representations | 一个键对应一个值：键是某个对象的表示（一只股票某日的特征或学到的嵌入、一种市场状态、一条标题），值是与它相关、需要记住的量（例如它后来的结局） |
| memories | 模型存着的从键到值的映射：一个线性映射的参数、一张表，或网络里的某一层 |
| new KV representations are introduced | 每天新到期的证据：那天的键连同它已经到期的结局 |
| adjust relevant memories | 新证据写入之后，与新键相关的键读出的值随之改变 |
| preserving unrelated ones | 与新键无关的键读出的值不变；「相关」与「无关」由会话定义 |
| mathematical optimization methods | 写入时求解一个优化问题得到参数的改动（例如带约束的最小二乘），而不只是沿梯度走一步 |
| Global Optimization | 在全部可见数据上重新求解的训练，例如按日程从头重训；本臂里它是对照 |
| continual memory adaptation | 模型带着学到的状态逐日走过一场回放，按每天新到期的证据更新，而不是按日程从头重训 |
| arrives rapidly and in large volumes | 研究年里每个交易日新到约 2,500（第一个研究年）到 4,700（最后一个研究年）只股票的日线行，一年约 54 万到 76 万条公告标题 |
| daily updates are challenging | 逐日更新要在每次 `fit` 的时限里做完，而一场研究年回放约 1,701 个决策日，单日开销按天数放大（`refs/README.md` 的「规模与时间」） |
| temporal structure | 交易日的节奏、结局的期限滞后、研究年里市场状态的切换；切换发生在哪些时期，由会话自己从可见的数据里找 |
| memory systems | 任何把学到的东西从一次决策带到下一次的状态：参数、键值记忆、缓存的样本 |

## 会话查不到的事实（运营者的转述，不是读数）

- 线性联想记忆 W 把键 k 映射成值 `Wk`。已存的键排成矩阵 K0 的各列，新来的键值对是 (K1, V1)。使 `‖(W+Δ)K1 − V1‖² + ‖ΔK0‖²` 最小的写入是 `Δ = R K1ᵀ (C0 + K1 K1ᵀ)⁻¹`，其中 `R = V1 − W K1`、`C0 = K0 K0ᵀ`（要求 `C0 + K1 K1ᵀ` 可逆）；第二项就是旧键读出的值的变化。C0 可以作为外积的累加和逐次维护，旧键本身不必存下。给 C0 乘一个系数（目标的第二项随之加权），就在保留旧值与学会新值之间取舍：系数越大，旧键读出的值动得越少，新键拟合得越差。
- 每次只写一对、此前所有的键都计入 C0，并从 `C0 = δI`、`W = 0` 起，这个写入恰好是递推最小二乘，结果等于在全部键值对上惩罚为 δ 的岭回归；每步先把 C0 乘以遗忘因子 γ < 1，就是它的指数加权形式（第 t 对在第 T 步的权重为 γ^(T−t)）。所以单靠这种软写入，是登记册见过很多次的模型；开放的是键是什么（学出来的表示，而不是原始特征）、哪些键算无关，以及下面的硬保留。
- 硬保留：设 P 是到 C0 零空间的正交投影（由 C0 特征值低于阈值的特征向量张成）。`Δ = R K1ᵀ P (P K1 K1ᵀ P + λI)⁻¹` 在所有满足 `ΔK0 = 0` 的改动里使 `‖ΔK1 − R‖² + λ‖Δ‖²` 最小。阈值只截去为零的特征值时，`W K0` 一点不变；阈值取正数时，旧值的变化 `‖ΔK0‖` 不超过 `‖Δ‖ · √阈值`。
- 容量：键空间是 d 维的。存下的键一旦张满这 d 维（C0 的秩等于 d），零空间为空，硬写入什么也改不了；保留与学习争的是同样的 d 个方向。运营者的核对：d = 8 时存 0、2、4、6、7 个随机键，零空间维数是 8、6、4、2、1，写入还能改动；存 8 个或 12 个键时 C0 的秩为 8、零空间为空，写入为零。

上面的式子运营者都在随机矩阵上用 numpy 核对过（与最小二乘解、加权岭回归与约束参数化的解逐项相符）。会话没有网络；这几段是运营者的转述与核对，不是市场上的测量。
