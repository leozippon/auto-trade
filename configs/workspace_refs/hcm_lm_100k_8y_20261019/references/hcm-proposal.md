# Research Proposal: Hierarchical Continual Memory (HCM)

**Hierarchical Continual Memory Architecture**

## Core Idea

> **A model's knowledge should not be confined to fixed parameters or a finite context window. Instead, it should continuously accumulate and evolve through memory and parameter updates operating at different timescales.**

Explore a unified, simple continual-memory architecture that organizes the model's knowledge into three interconnected levels:

- **Short-term memory:** Use mechanisms such as Linear Attention or RNN-like states to rapidly absorb new information and represent dynamic changes in the current environment.
- **Medium- and long-term memory:** Use the model's existing parameters to store more stable knowledge. Explore updating parameters at different timescales during inference, for example, progressively reducing learning rates and update frequencies from the final Transformer layers toward the earlier layers.
- **External memory:** Preserve long-term experiences that cannot be effectively compressed, and retrieve relevant historical knowledge dynamically so that not all information must be encoded in the model's finite parameters.

These three levels should not merely function as independent storage modules. Instead, explore how knowledge can naturally transfer between them.

For example, new information may first modify the fast-changing state. As evidence accumulates, reliable new patterns may gradually influence more stable parameters. Historical experiences that cannot be effectively compressed may remain in external memory for future retrieval.

**The central idea is not to manually prescribe what short-, medium-, and long-term memory should store, but to let the model gradually develop knowledge at different timescales through a unified learning process.**

The hypothesis that layers closer to the output should update faster must be treated as something to test, rather than an established principle.

## Research Objective

Explore a memory architecture capable of operating continuously over long periods, allowing the model not only to utilize recent information but also to accumulate out-of-distribution knowledge and effectively integrate new experiences with existing knowledge.

Do not impose predefined restrictions on the network architecture, memory representation, parameter update scope, or knowledge transfer mechanism.

Prefer reusing the model's existing Attention, FFN, and optimization processes wherever possible, avoiding unnecessary architectural complexity.

**Key question: Can a model continuously accumulate knowledge through the evolution of its states and parameters across multiple timescales, without repeatedly reinitializing itself, while retaining the ability to adapt to previously unseen environments?**

## References

1. MEMORYLLM: Towards Self-Updatable Large Language Models — ICML 2024, https://arxiv.org/abs/2402.04624. Key insight: separate continuously updatable memory from relatively stable model parameters.
2. M+: Extending MemoryLLM with Scalable Long-Term Memory — ICML 2025, https://arxiv.org/abs/2502.00592. Key insight: external long-term memory and learned retrieval preserve and revisit the distant past.
3. Learning to (Learn at Test Time): RNNs with Expressive Hidden States — ICML 2025, https://arxiv.org/abs/2407.04620. Key insight: updatable parameters as dynamic hidden states.
4. Nested Learning: The Illusion of Deep Learning Architectures — NeurIPS 2025, https://arxiv.org/abs/2512.24695. Key insight: parameter memories at multiple timescales via nested learning with different update frequencies.
5. In-Place Test-Time Training — ICLR 2026, https://arxiv.org/abs/2604.06169. Key insight: test-time learning by updating selected parameters of existing FFN layers in place.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的提案原文，一字未改。它写的是语言模型；本臂第一次把一个真正的预训练语言模型放进选股模型里，每个词的所指如下，`refs/README.md` 的问题与对照都按这张表读。右列只说明所指，不规定做法：每一级记忆的实现由会话设计。

| 提案里的词 | 本臂里的所指 |
|---|---|
| model | 截面选股模型：一个读每只股票匿名化公告标题的预训练语言模型，加十路日线特征，给决策日每只可交易股票打一个分，书买分高的名字 |
| context window | 模型一次读到的标题（起步包是最近五个决策日里新可见的标题）与策略的 PIT 回看窗口 |
| parameters | 起步包里只有输出头在训练；语言模型的参数（整层、某几层、或 peft 适配器）要不要训练、哪些训练，由会话决定 |
| short-term memory（Linear Attention / RNN-like state） | Qwen3.5 二十四层里十八层是 Gated DeltaNet 线性注意力，自带循环状态；它能否跨天、按股票延续，或另加一份每只股票逐日滚动的状态，是开放轴 |
| medium- and long-term memory（parameter updates at different timescales） | 把可训练参数分组（例如输出头、靠近输出的适配器、更深的适配器）、各按自己的学习率与更新周期在每日 `fit` 里更新，是一种可能的实现；「越靠近输出更新越快」是要检验的假设 |
| external memory | 过去「股票-日」片段（嵌入与已实现标签）的存储，按相似度检索后作为一路输入；上一轮的起步包有过一种实现，要不要、怎样做由会话决定 |
| knowledge transfer between levels | 新信息先改状态；状态上的梯度按各自周期进入参数；压缩不了的片段留在存储里被检索 |
| reinitializing | 本平台的常规做法：按 `REFIT_PERIOD` 在尾部窗口上冷重训；起步包的 `cold_quarter` 就是它 |
| previously unseen environments | 研究期里的市场状态切换：熊市、风格轮动的上涨、深度回撤、小盘的崩溃与反弹（`refs/README.md`） |
| continuously over long periods | 一场八年整期回放，模型在第一天预训练一次之后不再重新初始化 |
| test-time learning / in-place updates | 每个交易日的 `fit`（起步包的 `lib/update.daily` 钩子）：在刚到期的标签上做一次小更新，不另开训练流程 |
| Attention / FFN / optimization processes | 预训练模型自己的层与 AdamW；会话可以加结构，但不必为了「像提案」而加 |

提案里的参考文献会话读不到（容器没有网络）；上表与 `refs/README.md` 里对它们的转述是运营者的读法，不是读数。
