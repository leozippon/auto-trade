# 所有者的指示：复现顶会论文的方法并评估它能否赚钱

下面是所有者关于后续实验的指示，原文一字未改。

## 指示原文

For subsequent experiments, please proceed as follows: Search for papers published at ICLR, ICML, and NeurIPS over the past three years on finance, time-series data, and other methods that could potentially be applied to financial markets. For each selected paper, provide the paper itself and its open-source repository (if available) in a research Pack. Let the Agent reproduce and fine-tune the proposed methods, then evaluate whether they have the potential to generate profitable trading strategies.

---

## 术语对照：指示里的词在本臂里指什么

上文是所有者的原话。本臂选的论文是 AlphaQCM（ICML 2025），每个词的所指如下，`refs/README.md` 的三个阶段与对照都按这张表读。右列只说明所指，不规定做法。

| 指示里的词 | 本臂里的所指 |
|---|---|
| the paper itself | `refs/papers/alphaqcm/paper.pdf` 与它的文本抽取 `paper.txt` |
| its open-source repository | 作者的官方仓库在一个固定提交上的快照，`refs/papers/alphaqcm/repo/`；它建在 AlphaGen 的代码之上 |
| research Pack | 本目录：论文、代码、包作者的摘要（`claims.md`）与这份指示 |
| reproduce | 阶段一：论文的数据离线不可得，在本项目研究年数据上按论文的算子、因子池与智能体超参做忠实的小规模替身，并达到 `refs/README.md` 写下的健全性目标 |
| fine-tune | 阶段二：为 A 股调整目标的起点与期限、池的大小、搜索步数、λ 与算子集，每个改动写一行理由；不是指微调预训练权重，本方法没有预训练权重 |
| the proposed methods | 用分布式强化学习（IQN 分位数网络加 DQN 的 Q 网络）逐个 token 生成公式、以 QCM 方差为探索项、把公式收进线性组合的因子池；关键部件是 QCM 方差探索项 |
| formulaic alpha（论文的词） | 由算子、价量字段、常数与时间窗组成的表达式，在每个决策日给每只股票一个打分 |
| alpha pool / meta-alpha（论文的词） | 最多 P 个公式的集合，与它们的线性组合；本臂里组合后的打分决定排序 |
| profitable trading strategies | 在回放里按时点可见的数据、真实的成交规则与成本下，20–60 只左右的多头组合读数胜过同批的对照；只看 IC 不够 |
| evaluate whether they have the potential | 阶段三：与去掉关键部件的版本、同样大小的随机公式池和同样输入的 LightGBM 排序器同批比较，给出三选一的结论 |
