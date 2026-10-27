# DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting

- Authors: Lifan Zhao, Shuming Kong, Yanyan Shen
- Venue: KDD 2023
- arXiv: 2306.09862 (first version 2023-06-16; this text from v3, 2024-04-07)
- Licence: arXiv non-exclusive distribution licence (http://arxiv.org/licenses/nonexclusive-distrib/1.0/)
- Text here: abstract only: the paper's licence does not permit redistributing its full text (arXiv metadata, abstracts included, is CC0)

## Abstract

Stock trend forecasting is a fundamental task of quantitative investment where precise predictions of price trends are indispensable. As an online service, stock data continuously arrive over time. It is practical and efficient to incrementally update the forecast model with the latest data which may reveal some new patterns recurring in the future stock market. However, incremental learning for stock trend forecasting still remains under-explored due to the challenge of distribution shifts (a.k.a. concept drifts). With the stock market dynamically evolving, the distribution of future data can slightly or significantly differ from incremental data, hindering the effectiveness of incremental updates. To address this challenge, we propose DoubleAdapt, an end-to-end framework with two adapters, which can effectively adapt the data and the model to mitigate the effects of distribution shifts. Our key insight is to automatically learn how to adapt stock data into a locally stationary distribution in favor of profitable updates. Complemented by data adaptation, we can confidently adapt the model parameters under mitigated distribution shifts. We cast each incremental learning task as a meta-learning task and automatically optimize the adapters for desirable data adaptation and parameter initialization. Experiments on real-world stock datasets demonstrate that DoubleAdapt achieves state-of-the-art predictive performance and shows considerable efficiency.
