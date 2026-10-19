# Research Proposal: Evidence-Guided Continual Updating (EGCU)

The owner's proposal, as relayed to the pack author; the wording below keeps its substance.

## Core Idea

> **Continual learning should not mean continuously modifying parameters. The model should decide, from new evidence in the real environment, whether to learn, how much to change, and when to correct mistaken updates.**

Offline data is a biased sample. After deployment the model meets environments it has not seen, but it also meets random fluctuations that are not new patterns. It should therefore neither absorb everything it sees nor over-protect what it already knows.

One unified adaptive updating mechanism should provide:

- **Identification:** which observations are meaningful change and which are noise.
- **Triggering:** whether to update now or keep accumulating evidence.
- **Regulation:** which states or parameters to change, and by how much.
- **Error correction:** recognising mistaken updates and restoring the model, by correction, rollback or selective reset.

**These should emerge from a single updating principle, not from four separate subsystems.**

## Research Objective

Explore whether the magnitude of an update can be jointly determined by the credibility of the new evidence, the reliability of the existing knowledge, and the conflict between the new and the old patterns.

The focus is whether the model can learn a reliable updating rule, not merely a good initialization.

## References

1. Titans: Learning to Memorize at Test Time — NeurIPS 2025, https://arxiv.org/abs/2501.00663. Key insight: a surprise-gated memory decides how much of an observation to write.
2. End-to-End Test-Time Training for Long Context — 2025. Key insight: online updates whose step sizes are meta-learned end to end.
3. When to Retrain after Drift — ICLR 2026. Key insight: after a drift, retrain only once the post-drift data suffices.
4. When and Where to Reset for Long-Term Test-Time Adaptation — ICLR 2026. Key insight: resets scoped to what went wrong, timed by evidence.
5. Gated Delta Networks — 2024, https://arxiv.org/abs/2412.06464. Key insight: a gated delta rule that both writes and erases memory.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的提案（经协调者转述，保留原意）。它写的是一般的模型；本臂把它落到一个读匿名化公告标题的预训练语言模型加日线特征的选股模型上，每个词的所指如下，`refs/README.md` 的问题与对照都按这张表读。右列只说明所指，不规定做法：更新规则由会话设计。

| 提案里的词 | 本臂里的所指 |
|---|---|
| model | 截面选股模型：预训练语言模型读每只股票的匿名化标题，加十路日线特征，给决策日每只可交易股票打一个分 |
| new evidence in the real environment | 每个交易日新到期的标签（20 日前向残差收益的截面秩）连同它们的特征与标题：决策日 T 的 `fit` 知道 T−21 日及更早各日的标签 |
| random fluctuations that are not new patterns | 一天或几天的标签噪声：八年上零技能的日度主动收益本身就有分数的抽样误差，任何一天的梯度都大部分是噪声 |
| whether to learn / how much to change | 每日 `fit` 里要不要步进、步长多大、改哪一组参数（起步包里只有输出头；适配器、状态都可加） |
| correct mistaken updates | 把参数恢复到一个参考版本、撤回最近几步、或只重置某一组；都是在 `fit` 里做、写进 `context.state_dir` 的事 |
| a single updating principle | 一个从证据到更新量的函数：例如由「新证据的可信度、现有参数的可靠度、新旧模式的冲突」三者共同决定步长与是否回滚；它的形式是会话的设计 |
| learn a reliable updating rule, not merely a good initialization | 对照的意义：同一个初始化下，带规则的更新要胜过「天天无条件更新」与「按季冷重训」，并且胜过「在随机时刻提交」的安慰剂 |
| reliability of existing knowledge | 当前参数在最近到期的样本上的表现、它被多少证据训练过 |
| conflict between new and old patterns | 新梯度与已有参数（或与一个参考版本）的方向与幅度之差 |
| environments it has not seen | 研究期里的市场状态切换（`refs/README.md`） |

提案里的参考文献会话读不到（容器没有网络）；上表与 `refs/README.md` 里对它们的转述是运营者的读法，不是读数。
