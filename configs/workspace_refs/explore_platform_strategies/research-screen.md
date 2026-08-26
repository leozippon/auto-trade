# Research screen

Retail-platform language is useful for generating hypotheses, not for establishing alpha. The public material is inconsistent, often retrospective, and frequently promotional. This screen keeps only mechanisms that have a distinct local PIT representation and a reproducible negative test.

## Common idea to local hypothesis

| Retail-platform idea | Screen result | Local rebuild |
| --- | --- | --- |
| 情绪周期、赚钱/亏钱效应 | Keep, operationalized | daily counts of closed limit-up, broken-board, limit-down, success ratio, and streak height from `limit_list_d` |
| 龙头、连板、首次强分歧 | Keep, operationalized | prior-day closed board, `limit_times`, `open_times`, seal time, seal amount/turnover, and same-industry breadth |
| 炸板反包、强势股低吸 | Keep, operationalized | prior-day `limit='Z'`, liquidity, close location/return, industry breadth, and PIT money flow |
| 热股、人气榜、飙升榜 | Keep as attention, not sentiment | within-source `ths_hot`/`dc_hot` rank level and rank change, optionally cross-platform confirmation |
| 主力净流入、板块轮动 | Keep with unit control | stock `moneyflow` scaled by daily turnover; aggregate by PIT SW industry; late vendor feeds are robustness checks only |
| 龙虎榜机构净买 | Keep as a sparse event study | aggregate `top_inst.net_buy`, confirm with `top_list.net_amount`, normalize by disclosed/market turnover |
| 游资席位跟随 | Drop from the retained set | `hm_detail` is conservatively delayed at 08:30, `hm_list` has no historical PIT, and account-level research warns of next-day selling |
| 北向资金跟随 | Drop | no dedicated local Northbound holding/change dataset; vendor stock money flow is not a substitute |
| 雪球/股吧评论情绪 | Drop exact reconstruction | no PIT forum posts/comments/likes; use hot-list attention or local published text as different signals, never relabel them forum sentiment |
| 财联社/新闻催化 | Keep only as local text event | bounded `text_index`/library titles and bodies with publication-time filtering; the index does not preserve a guaranteed platform identity |
| 竞价弱转强、一字板排队 | Drop at 08:30 | today's auction appears at 09:29 and the Broker has no queue/depth/partial-fill model |

## Hard rejects

- **Live scraping or inference-time HTTP:** network is disabled and historical web pages are not a PIT data source.
- **Paid Level-2 recipes:** cancellation rate, queue depletion, seal-order additions, tick-by-tick large-order identity, and buy-queue priority are absent.
- **Future labels:** today's high/low/close, today's successful board, later hot rank, or an end-of-day list cannot select a 09:30 trade on the same date.
- **Current-state backfills:** current `hm_list`, current stock names, current concepts, and current Northbound holdings cannot be projected into old folds.
- **Promotional black boxes:** public pages that report extreme annualized returns while withholding exact rules, costs, or code are not rewriteable evidence.
- **Direct board-buy assumptions:** prior-day `fd_amount` is descriptive. The formal Broker does not model queue priority and can reject a limit-price buy.
- **Opaque catch-all factor stacks:** “eight-factor low buy”, generic trend + RSI + ATR stacks, and the old MA20 switch fail the distinct-mechanism requirement.
- **Silent late-feed fallback:** a 2025-only THS/DC confirmation may not disappear silently in earlier folds. Either run a declared common-coverage sample or test the base 2020+ signal separately.

## Web evidence and how it is used

1. [东财股吧转载的“情绪周期各阶段要点”](https://guba.eastmoney.com/news,gssz,758041972.html) documents the retail vocabulary of recovery, divergence, acceleration, decline, and ice point. It is an opinion post with no reproducible backtest; this pack replaces the prose labels with limit-list statistics.
2. [Statistical properties of price limit hits in the Chinese stock markets](https://arxiv.org/abs/1503.03548) studies 2000–2011 A-share high-frequency data and reports next-day continuation more often than reversal after limit hits. It motivates testing continuation, but its old sample and broad hit definition do not validate a current 连板 strategy.
3. [公司新闻、投资者关注与股价运行——来自股吧的证据](https://xbzs.ecnu.edu.cn/CN/10.16382/j.cnki.1000-5579.2017.06.014) reports that forum post attention affected contemporaneous weekly volatility but was not significantly related to current or next-week return in its sample. Attention therefore needs a directional confirmation and a falsification control.
4. [Investor Sentiment and Stock Returns: An Empirical Study from Mobile Internet](https://html.rhhz.net/NFJJ/html/20190303.htm) uses Xueqiu discussions and reports more optimistic mobile sentiment associated with higher next-period returns, especially in poor-information or less-liquid firms. This conflicts with the weaker forum-post result and cannot be recreated without the original posts; the retained hot-list idea is explicitly an attention proxy, not reconstructed Xueqiu sentiment.
5. [Could Fund-dominant Accounts Take Advantage of Investors’ Attention?](https://qks.sufe.edu.cn/mv_html/j00001/202006/44c2c628-9dee-45e0-b8ab-0f112001e2d9_WEB.htm) finds account-level evidence of Dragon-Tiger-list attention and next-day “one-day tour” selling by fund-dominant accounts in its 2015–2019 sample. This supports separating institutional-seat tests from naive hot-money following and requiring next-open execution evidence.
6. [HKEX's April 2024 Stock Connect data-dissemination adjustment](https://www.hkex.com.hk/News/Market-Communications/2024/2404122news?sc_lang=en) states that Northbound real-time buy/sell turnover would cease and individual shareholdings would be available quarterly after quarter end. Even apart from the local data gap, historical daily-follow rules cannot be carried forward unchanged.
7. TuShare's public pages define the fields available in the local lake: [连板天梯](https://tushare.pro/document/2?doc_id=356), [最强板块](https://tushare.pro/document/2?doc_id=357), [THS热榜](https://tushare.pro/document/2?doc_id=320), [东财热榜](https://tushare.pro/document/2?doc_id=321), [龙虎榜](https://tushare.pro/document/2?doc_id=106), and [个股资金流](https://tushare.pro/document/2?doc_id=170). Repository timing, masking, units, and refresh-node contracts still take precedence over those source descriptions.

No source above supplies an expected return for this environment. Any claimed effect must survive the repository's own PIT replay, open-price execution, costs, T+1, and fold-by-fold stability checks.
