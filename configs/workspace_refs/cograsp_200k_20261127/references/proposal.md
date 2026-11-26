# 所有者的提案：复现近三年论文里可能用于金融市场的方法

所有者的指示，原文一字未改：

For subsequent experiments, please proceed as follows: Search for papers published at ICLR, ICML, and NeurIPS over the past three years on finance, time-series data, and other methods that could potentially be applied to financial markets. For each selected paper, provide the paper itself and its open-source repository (if available) in a research Pack. Let the Agent reproduce and fine-tune the proposed methods, then evaluate whether they have the potential to generate profitable trading strategies.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的原话。本臂选的论文是 COGRASP（IJCAI 2025）：它不在提案点名的三个会议里，选它是因为它让学习器保持不变、只改关系的信息，正好能在同一批里对照「文本里的共现关系」与「行业、收益相关、没有关系」。每个词的所指如下，`refs/README.md` 的阶段、要求与对照都按这张表读。右列只说明所指，不规定做法。

| 提案里的词 | 本臂里的所指 |
|---|---|
| papers … on finance | 本臂的论文：Li、Song、Yuan、Fu，COGRASP: Co-Occurrence Graph Based Stock Price Forecasting，IJCAI 2025（doi 10.24963/ijcai.2025/837） |
| the paper itself | `refs/papers/cograsp/paper.pdf`（会议版）与它的文本抽取 `paper.txt` |
| its open-source repository | `refs/papers/cograsp/repo/`：作者仓库在一个固定提交上的快照，带一张 300 只股票的共现矩阵，图片与检查点已剥离；没有建图代码与价格数据 |
| the proposed methods | 由文本里股票的共同提及建成的加权图，经一层 Chebyshev 图卷积得到关系嵌入，与价量特征一起由三个时间尺度的注意力 LSTM 读出并加权合成 |
| reproduce | 阶段一：让仓库的代码在镜像上跑起来，在研究年数据上用仓库带的图与三张对照图读论文表 4 的相对顺序 |
| fine-tune | 阶段二：用本项目的公告标题、研报与新闻按时点建共现图（代替论文的雪球帖子），在研究年上只用当时可见的数据训练同一个模型 |
| evaluate | 阶段三：同一个模型、同样的训练量与种子，在同一批里对照行业图、收益相关图、恒等图与保持度不变的打乱图，再对照同样特征上的 LightGBM 排序器 |
| profitable trading strategies | 20 万元账户、按本项目成本与持仓规则的回放读数，账户扣费后的收益与基准指数、随机对照组合并列；论文只报预测的相关系数，没有组合 |
