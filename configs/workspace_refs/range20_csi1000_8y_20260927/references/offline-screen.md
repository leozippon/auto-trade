# 离线门 G-R20 的做法：秩 IC、顶端对同池、登记构造的月度书，分两半

自写笔记。`refs/families.md` 的离线门在整期批次之前、全离线、0 回放年地回答一个问题：**这个分数在本池、八个研究年的决策视图上，有没有登记方向的秩信息，这份信息在它没被挑过的 Y1–Y4 上还在不在，落不落得到一本只做多、50 席等额、带保留带的书上。**判据只在 `refs/families.md`；本页写怎么算、能说明什么。包作者的读数在 `refs/sources.md`，会话要自己重算，不引用包里的数。**命中不改变第一批。**

## 先把分数与池逐日按时点建出来

`/mnt/snapshot` 是研究期末的单一时点视图（日线从 2016-07-01 起），分数只要 20 根日线，研究期从 2017-07 起，所以没有左截断；成分截面要逐日取「日期不晚于 t 的最新一张」，不能用最新一张去标注历史。下面这个 `compute_signal` 就是包作者用的那一份（`r1`；把 `MODE` 改成 `"rand"` 就是同池随机对照，`DAYS` 改成 60 就是 `r2`——算 `r2` 就是筛了它，按 `refs/families.md` 计），写进 `workspace/` 里的一个文件即可：

```python
import numpy as np, pandas as pd

INDEX, MODE, DAYS = "000852.SH", "r1", 20


def own_bar_mean(values, days):                 # 每只股票自己最近 days 根可用日线的均值
    out = np.full(values.shape, np.nan)
    for j in range(values.shape[1]):
        have = np.flatnonzero(np.isfinite(values[:, j]))
        if have.size >= days:
            c = np.concatenate([[0.0], np.cumsum(values[have, j])])
            out[have[days - 1:], j] = (c[days:] - c[:-days]) / days
    return out


def compute_signal(frames):
    high, low, pre = (frames.wide("daily", c) for c in ("high", "low", "pre_close"))
    dates, codes = np.asarray(high.index).astype(str), np.asarray(high.columns).astype(str)
    score = -own_bar_mean(((high - low) / pre.where(pre > 0)).to_numpy(), DAYS)
    macro = frames.load("macro", columns=["dataset", "index_code", "con_code", "trade_date", "weight"])
    sec = macro[(macro.dataset == "index_weight") & (macro.index_code == INDEX) & (macro.weight > 0)]
    sec = sec.assign(trade_date=sec.trade_date.astype(str))
    sect_dates = np.array(sorted(sec.trade_date.unique()))
    pos = {c: i for i, c in enumerate(codes)}
    rows = {d: [pos[c] for c in g.con_code if c in pos] for d, g in sec.groupby("trade_date")}
    member = np.zeros(score.shape, dtype=bool)
    for t, k in enumerate(np.searchsorted(sect_dates, dates, side="right") - 1):
        if k >= 0:
            member[t, rows[sect_dates[k]]] = True     # 日期 <= t 的最新截面：t 之后那个决策读得到它
    board = ~(pd.Index(codes).str.startswith(("688", "689")) | pd.Index(codes).str.endswith(".BJ"))
    ok = member & np.asarray(board)[None, :] & np.isfinite(score)
    if MODE == "rand":                                # 同一批格子上、逐日重抽的同池随机排名
        s = pd.Series(codes)
        score = np.vstack([pd.util.hash_pandas_object(s, index=False, hash_key=f"r20rand{d}x").to_numpy()
                           % 1_000_003 for d in dates]) / 1_000_003.0
    elif MODE != "r1":
        raise ValueError(f"unknown MODE {MODE!r}")
    return pd.DataFrame(np.where(ok, score, np.nan), index=dates, columns=codes)
```

日期 t 的分数只用 t 及以前的日线与 t 那天 17:30 已盖章的截面，正是 t 之后那个交易日 08:30 决策读到的东西；screen 从 t+1 的开盘算前向收益，两边对齐，不需要再移一行。名称是研究期末的，所以离线不按名称过滤 ST（研究期末的 ST 名单与 2017 年的无关，`refs/pit-field-map.md`）。

## 工具一：`/mnt/tools/screen.py`（S2、S3、S4）

```
python /mnt/tools/screen.py --signal <file> --horizons 5,10,20,40 \
    --start 20170701 --end 20250630 --top-fraction 0.053
```

`r1` 与 `rand` 各跑一次整期，再各跑两半（`--start 20170701 --end 20210630`、`--start 20210701 --end 20250630`）与八个研究年（每年的 0701..0630）。`--top-fraction` 是 50 ÷ 每日可打分池的均值（约 942）。读 `ic_mean`、`t_stat`（已按重叠折算）、`positive_month_share`、`ic_size_neutral` 与 `top_excess_mean`。八年一次约一分钟、峰值内存不到 4 GiB。

- **`top_excess` 减的是当日被打分的名字（IC 排序的那个池）的等权均值**——就是本池，不是全市场；更早的包里「对全市场」的写法已过时。所以 S3 直接读 `r1` 的 `top_excess_mean`；`rand` 的应在零附近，是对这条口径的检查。
- h = 20 是本书的复核间隔，40 是持有期量级（中位持有一到三个月）；这个分数慢，IC 随 horizon 上升并不奇怪。

## 工具二：登记构造的月度书（S5 与诊断，自己写的脚本）

**离线书必须跑登记的构造**（登记册 §4：整本重排的前 50 名不是宿主交易的书，只能作标明了的诊断）。在研究期每个自然月的第一个交易日 D（96 个；最后一个没有下一个复核日的开盘，所以 95 个月）：

- **分数与池**：取 T-1 那一行的分数（上面的 `compute_signal`），池 = 有分数的名字，再要求 D 有日线、D 的收盘没有封在涨停价、下一个复核日有开盘价。
- **书**：照 `refs/families.md` 的构造逐月走——持仓排名 ≥ 100 就卖，离开池的卖；每个申万一级最多 14 只，**保留的名字算在内**，超额从排名最差的卖起；座位 = 账面价值 ÷ 50（卖出所得 × 0.98 加现金，× 0.97，再加保留持仓的市值）；只买一手（100 × T-1 收盘）不超过一个座位的名字，从排名最前往下补到 50 只；**保留的名字带着上个月漂移后的市值**，不在复核日拉回等权。
- **持有收益**：D 收盘买入、下一个复核日开盘卖出（或继续持有），按复权价（`close × adj_factor`、`open × adj_factor`）——起步包就是 15:00 买、09:30 卖。
- **主动**：书的月收益 − 同一个池的等权月收益（D 收盘到下一个 D 开盘，买入持有一个月）。池的等权是「同池随机 50 只」的期望，是宿主面板在这个形状上的离线近似。**它没有中性化**：这本书 β 约 0.65–0.8，单边大涨或大跌的年份里未中性化的主动收益主要是 β 差（本池的逐年指数涨跌见 `refs/README.md`）；把每月的主动收益对指数的月收益（`macro` 里 `dataset == "index_daily"` 且 `ts_code == "000852.SH"`）回归，截距才接近宿主的量尺。
- **两半与逐年**：Y1–Y4 与 Y5–Y8 分开报年化主动、主动 IR 与为正的年数；十分位剖面（每个复核日把池按分数分十组，各组收益减池均值）也分两半报。
- 离线的名称与行业是研究期末的（申万 2021 版灌回 2017–2021），不模拟拒单、整手与成本：这是它与宿主回放的已知差别。

## 这道门能说什么、不能说什么

- **它只能否决。** 离线读数只有秩与方向可信（登记册 §4）；离线书读正只说明值得花那 16 个回放年，读负也不省下它（`no_edge` 要一次完整期验证）。
- **八年的离线书是一次有噪声的抽样。** 零技能下主动 IR 的抽样误差每一半约 0.5、八年约 0.35；只有符号与两半、逐年的一致性有意义，**不要拿离线的 %/年 去估宿主的主动 IR**。宿主面板走完整 Broker、按候选自己的成交骨架换名、会抽到本书剔掉的科创板成分，并在基准与规模上中性化，离线近似做不到。
- **这道门筛过的每一个候选配置都计一个 trial**：`r1` 按登记提交则在宿主上计；会话若另筛变体（换窗口、换分数、换构造），每个都在筛过之后第一个记下验证的批次里申报 `offline_trials`。重算 `r1` 本身与跑 `rand` 对照不计。
