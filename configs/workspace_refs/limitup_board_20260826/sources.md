# 来源、范围内与范围外

机制改写见 `playbooks.md`，字段可见性见 `pit-field-map.md`。
社区胜率、连板神话和宣传收益一律未核实，不要写成已验证 alpha。
不要粘贴聚宽策略或任何第三方完整源码。

## 范围内（可 PIT 改写）

一次只测一个。都能改写到当时可见行，不需要 09:30 之后的分钟或实时排队：

- 竞价缺口相对昨收（仅 09:29 `auction` vs `pre_close` / T-1 收盘）
- 首板次日
- 一进二
- 炸板
- 连板高度
- 情绪冰点

## 范围外（不要克隆）

这些需要 09:30+ 分钟、实时盘口或可执行排队，本实验看不见。不要当脚手架，也不要改写成「盘中触发」。

- [123quant/QMT-QuantLimit](https://github.com/123quant/QMT-QuantLimit)：QMT 盘中打板，涨停瞬间买入。
- [freevolunteer/daban](https://github.com/freevolunteer/daban)：实时行情触发，临近封板自动下单。
- 东财爬虫 / 自动下单类仓库：抓实时榜并报单。沙箱无网络，正式策略也不得去拉东财。

## 涨停池标签（机制转述）

公开涨停池写手常用「首板 / 连板高度 / 炸板次数」给池子打标签。本地只能映射到当时可见的 `limit_list_d`（`limit`、`limit_times`、`open_times`）、T-1 `kpl_list` 标签，以及 09:29 `auction` 相对昨收的缺口。
不要把对方池子名单、APP 实时标签或宣传晋级率当输入。
