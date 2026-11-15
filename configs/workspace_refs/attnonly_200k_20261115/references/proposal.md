# 所有者的提案：只用注意力、能整体学习的结构

所有者在「持续学习」这个总题目下写了几条方向。下面是本臂这一条和四个臂共用的领域框架，原文一字未改。

## Direction 2: Overcoming the Lack of Global Optimization in Continual Learning

- **Redesigning the architecture and optimization method from first principles:** Investigate an architecture consisting entirely of Attention layers, removing Feed-Forward Networks (FFNs), to enable global learning.

## Direction 3: Domain Applications

- **Continual memory adaptation for financial time-series data:** Financial data arrives rapidly and in large volumes, making daily updates challenging. Investigate how memory systems can effectively adapt to the temporal structure and continuous arrival of financial data.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的原话。它写的是一般的模型；本臂把它落到 A 股的截面选股上，每个词的所指如下，`refs/README.md` 的方向与对照都按这张表读。右列只说明所指，不规定做法：结构与更新由会话设计。

| 提案里的词 | 本臂里的所指 |
|---|---|
| Attention layers | 每个位置按查询与键的相似度，对一组位置的值加权求和的层；位置可以是同一决策日的各只股票、一只股票的各个历史日，或一份存下的键值条目 |
| Feed-Forward Networks (FFNs) | 每个位置各自经过的前馈块：两层线性映射夹一个逐元素的非线性，通常跟在注意力之后 |
| an architecture consisting entirely of Attention layers | 学到的函数只由注意力和它自身包含的线性映射（查询、键、值与输出投影）构成，没有逐位置的前馈块；归一化与跳连接留不留，提案没有说，由会话定义并写明 |
| global learning | 运营者的读法：一次更新里，模型的每一部分都可以随新证据改变，而不是只改一份记忆或一层。这只是运营者的读法，会话可以提出并写明自己的 |
| optimization method | 怎样更新参数或状态：梯度下降的某种变体、闭式求解、在前向计算里完成的学习，或别的规则 |
| from first principles | 从「模型要存什么、怎样学」出发重新设计结构与更新，而不是在现成的结构上添部件 |
| Global Optimization | 在全部可见数据上重新求解的训练，例如按日程从头重训；本臂里它是对照 |
| continual memory adaptation | 模型带着学到的状态逐日走过一场回放，按每天新到期的证据更新，而不是按日程从头重训 |
| arrives rapidly and in large volumes | 研究年里每个交易日新到约 2,500（第一个研究年）到 4,700（最后一个研究年）只股票的日线行，一年约 54 万到 76 万条公告标题 |
| daily updates are challenging | 逐日更新要在每次 `fit` 的时限里做完，而一场研究年回放约 1,701 个决策日，单日开销按天数放大（`refs/README.md` 的「规模与时间」） |
| temporal structure | 交易日的节奏、结局的期限滞后、研究年里市场状态的切换；切换发生在哪些时期，由会话自己从可见的数据里找 |
| memory systems | 任何把学到的东西从一次决策带到下一次的状态：参数、注意力读取的键值条目、缓存的样本 |

## 会话查不到的事实（运营者的转述，不是读数）

- 一个 softmax 注意力头对每个查询返回值向量的凸组合：权重非负、和为 1，所以每个输出都落在值向量的凸包里；权重固定时它是值的线性函数。没有别的部件时，唯一的非线性在权重里。运营者的核对：50 个随机查询的输出全在凸包内（线性规划可行），对值的线性误差 2e-15。
- 不带跳连接、也不带前馈块的自注意力层层叠加，随深度把所有位置推向同一个向量（位置矩阵趋于秩一）；跳连接抵消这一点。这是文献里已知的结果。运营者的模拟（64 个位置、宽 32、随机权重、20 个种子）显示：位置之间的相对离散度（各位置减去平均位置后的范数比上整体范数）4 层时还接近 1，8 层降到约 10⁻⁵·⁵，16 层约 10⁻¹³；注意力更软时 4 层就降到机器精度；每层带跳连接（层前归一化）时 32 层后仍约为 1。
- 对存下的键值对做注意力就是按键查表：权重足够尖（逆温度大）时，它返回最匹配的键的值，所以一对新的键值追加一行就学会、删掉这一行就忘掉，不需要梯度步。运营者的核对：2,000 个 64 维的随机单位键，逆温度 50 时带噪查询读回对应值的最大误差 5e-10（逆温度 20 时 6e-4，5 时约 3，查表失效）；追加一行后新键读回的误差 5e-13，旧键不受影响；删掉后回到追加之前。
- 线性注意力（不经 softmax）以 (x_i, y_i) 为键值对，在查询 x_q 处返回 `η Σ_i y_i (x_i·x_q)`；它等于从零权重出发、在 `½ Σ_i ‖W x_i − y_i‖²` 上走一步步长为 η 的梯度下降之后的预测 `W₁ x_q`。所以一层线性注意力可以在前向计算里完成一步学习。运营者的核对：两者之差 2e-16。
- 跨 N 只股票的全注意力，每头每层要算 N² 个分数：研究年的截面约 2,500 到 4,700 只，即约 625 万到 2,209 万个分数（float32 下 24 到 84 MiB）。在本机 CPU 上用 numpy、8 个线程量得单头一层的前向：宽 32 或 64 时，2,500 只约 0.12–0.14 s，4,700 只约 0.36 s；沙箱与卡上都没有量过。

上面的式子与模拟运营者都在合成数据上用 numpy 核对过。会话没有网络；这几段是运营者的转述与核对，不是市场上的测量。
