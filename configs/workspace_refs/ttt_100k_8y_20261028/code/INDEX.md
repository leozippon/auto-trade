# 代码索引

这里是七个开源仓库的源代码，2026-10-27 浅克隆，只拷了源文件（Python、配置与说明）和许可文件：没有版本库历史、图片、数据、检查点，也没有与测试时训练无关的大块子目录（下面逐个写明）。它们只作阅读与改写的参考：策略只能导入自己包里的模块和合同允许的库（`output/README.md`），这里的代码要用就搬进 `output/` 改写，清单之外的依赖（专用 CUDA 或 Triton 内核、各种小工具库）要换掉，并保留原来的许可声明。下面的大小是拷进来的部分。

| 目录 | 仓库 | 提交 | 许可 | 文件 / 大小 |
|---|---|---|---|---|
| `ttt-lm-pytorch` | github.com/test-time-training/ttt-lm-pytorch | cd831db10c8c（2024-07-14） | MIT | 3 / 73 KB |
| `LaCT` | github.com/a1600012888/LaCT | a648340f9798（2026-01-04） | MIT | 40 / 280 KB |
| `titans-pytorch` | github.com/lucidrains/titans-pytorch | 1d40c445fa17（2026-07-13） | MIT | 12 / 110 KB |
| `marc` | github.com/ekinakyurek/marc | 95b334872d43（2024-11-12） | MIT | 31 / 226 KB |
| `tttlm` | github.com/socialfoundations/tttlm | 99d4e0aab7fc（2024-04-18） | MIT | 17 / 113 KB |
| `tent` | github.com/DequanWang/tent | e9e926a668d8（2021-05-19） | MIT | 10 / 26 KB |
| `fsnet` | github.com/salesforce/fsnet | c776afc623fa（2026-06-02） | BSD-3-Clause | 36 / 200 KB |

`ttt-lm-pytorch`：《Learning to (Learn at Test Time)》（`../papers/2024_ttt_layers.md`）的官方 PyTorch 模型实现，一个文件 `ttt.py`，以 `transformers` 的模型类写出 TTT-Linear 与 TTT-MLP：隐状态是一组权重，按小批量在序列上走梯度步（`TTTBase` 及其两个子类，`scan` 是沿序列的递推）。作者说明它是纯 PyTorch、没有系统优化，训练慢，适合读懂与小规模使用；可选的 `causal_conv1d` 只用于其中的卷积层。

`LaCT`：《Test-Time Training Done Right》（`../papers/2025_lact.md`）的官方代码。最值得读的是 `minimal_implementations/` 里的两个纯 PyTorch 最小实现：双向的大块 TTT 层（用全部键值更新 SwiGLU 快权重，再作用于全部查询），和带滑窗注意力的因果版本（按块先用后更新）。`lact_llm/` 是语言模型版本，依赖 Triton 内核与 `fla`、`flash_attn`；`lact_nvs/` 是新视角合成。自回归视频扩散的 `lact_ar_video/` 与示例数据没有拷。

`titans-pytorch`：Titans（`../papers/2024_titans.md`）的非官方实现，作者 lucidrains。`titans_pytorch/neural_memory.py` 的 `NeuralMemory` 是测试时更新的神经记忆（按块存入、带动量与衰减），`memory_models.py` 是几种记忆网络，`mac_transformer.py` 是「记忆作上下文」的整体架构。依赖 `tensordict`、`assoc_scan`、`hyper_connections`、`x_transformers` 等小库。训练示例用的 enwik8 数据没有拷。

`marc`：《The Surprising Effectiveness of Test-Time Training for Few-Shot Learning》（`../papers/2024_ttt_few_shot.md`）在 ARC 上的官方代码。`test_time_train.py` 为每个任务用留一与增广构造数据、训练 LoRA 适配器，`arclib/augmenters.py` 是增广，`arclib/voting.py` 是增广视角之间的投票，`predict.py` 用 `vllm` 推断。训练依赖作者分叉的 torchtune（第三方子模块，没有拷）；任务数据没有拷。

`tttlm`：《Test-Time Training on Nearest Neighbors for Large Language Models》（`../papers/2023_ttt_nearest_neighbors.md`）的官方代码。`code/eval_tttlm.py` 对每个输入取近邻、逐个做一步梯度再评估，`build_database.py`、`pile_index.py` 与 `pile_server.py` 是基于 faiss 的分布式近邻索引，`baseline_context.py` 与 `baseline_interpolation.py` 是把近邻放进上下文与插值的对照。拷的只是 `code/` 下的脚本；仓库里附带的 lm-eval 副本、结果与集群提交文件没有拷。

`tent`：Tent（`../papers/2020_tent.md`）的官方示例代码。`tent.py` 一个文件就是全部方法：只收集归一化层的仿射参数、每批最小化预测熵走一步、可以重置回初始状态；其余是 CIFAR-10-C 上的示例脚本与配置（依赖 robustbench）。

`fsnet`：《Learning Fast and Slow for Online Time Series Forecasting》（`../papers/2022_fsnet.md`）的官方代码。`models/ts2vec/fsnet_.py` 是带每层快适配器与联想记忆的扩张卷积（`SamePadConv`），`fsnet.py` 用它组成编码器；`exp/exp_fsnet.py` 的 `test` 是逐步预测、再用到来的真值更新的在线循环（`online_learning` 取 `none`、`full` 或 `regressor`）；`exp/` 下另有经验回放（ER、DER++）、OGD、无记忆消融与朴素在线微调等对照。数据没有拷。

几篇论文的官方代码没有收：TTT-E2E 与 DoubleAdapt 的仓库没有许可声明，TAFAS 的许可只允许非商业使用，PETSA 的是 CC BY-NC-SA，TTT 层的 JAX 训练代码也没有许可声明。
