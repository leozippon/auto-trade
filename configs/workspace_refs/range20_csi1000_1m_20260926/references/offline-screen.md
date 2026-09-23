# 离线门 G-R20 的做法：秩 IC、顶端对同池随机、月度书与零技能地板

自写笔记。`refs/families.md` 的离线门在整期批次之前、全离线、0 回放年地回答一个问题：**这个分数在本池、本研究期的决策视图上，有没有登记方向的秩信息，而且这份信息落不落得到一本只做多、50 席等额的书上。**判据只在 `refs/families.md`；本页写怎么算、能说明什么。包作者在同一张视图上的读数在 `refs/sources.md`，会话要自己重算，不引用包里的数。

## 先把分数与池逐日按时点建出来

`/mnt/snapshot` 是研究期末的单一时点视图（日线从 2020-07-01 起），分数只要 20 根日线，所以研究期内没有左截断；成分截面要逐日取「日期不晚于 t 的最新一张」，不能用最新一张去标注历史。下面这个 `compute_signal` 就是包作者用的那一份（`r1`；把 `MODE` 改成 `"rand"` 就是同池随机对照），写进 `workspace/` 里的一个文件即可：

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
    return pd.DataFrame(np.where(ok, score, np.nan), index=dates, columns=codes)
```

日期 t 的分数只用 t 及以前的日线与 t 那天 17:30 已盖章的截面，正是 t 之后那个交易日 08:30 决策读到的东西；screen 从 t+1 的开盘算前向收益，两边对齐，不需要再移一行。名称是研究期末的，所以离线不按名称过滤 ST（逐年平均不到一只，`refs/sources.md`）。

## 工具一：`/mnt/tools/screen.py`（S2、S3、S4）

```
python /mnt/tools/screen.py --signal <file> --horizons 5,10,20,40 \
    --start 20210701 --end 20250630 --top-fraction 0.055
```

`r1` 与 `rand` 各跑一次整期，再各跑四个研究年（`--start` / `--end` 换成每年的 0701..0630）。`--top-fraction` 是 50 ÷ 每日可打分池的均值（约 910）。读 `ic_mean`、`t_stat`（已按重叠折算）、`positive_month_share`、`ic_size_neutral` 与 `top_excess_mean`。一次约 25 秒、峰值内存不到 2 GiB。

- **`top_excess` 减的是全市场等权**，不是本池——中证 1000 自己对全市场就有一个风格差。所以 S3 看的是 `r1` 减 `rand` 的差：`rand` 在同一批格子上逐日重抽，它的顶端就是同池随机，差值才是池内的选名。
- h = 20 是本书的复核间隔，40 是持有期（中位持有远长于一个月）；这个分数慢，IC 随 horizon 上升并不奇怪。

## 工具二：月度书与同池零技能地板（S5 与诊断，自己写的脚本）

在研究期每个自然月的第一个交易日 D（48 个；研究期末视图里最后一个复核日没有下一个复核日的开盘，所以 47 个月）：

- **分数与池**：取 T-1 那一行的分数（上面的 `compute_signal`），池 = 有分数的名字，再要求 D 有日线、D 的收盘没有封在涨停价、下一个复核日有开盘价。
- **持有收益**：D 收盘买入、下一个复核日开盘卖出，按复权价（`close × adj_factor`、`open × adj_factor`）——起步包就是 15:00 买、09:30 卖。
- **书**：照登记的构造逐月走——持仓排名 ≥ 100 就卖，离开池的卖，每个申万一级最多 14 只，只买一手（100 × T-1 收盘）≤ 19,400 元的名字，从排名最前往下补到 50 只，等权；主动 = 书的月收益 − 池的等权月收益。池的等权就是「同池随机 50 只」的期望，也就是宿主面板在这个形状上的离线近似。
- **地板**：每月从同一个池（同样的可负担性）里随机抽 50 只、每月重抽，抽 200 本，读它们年化主动与主动 IR 的 p10 / p50 / p90，看 `r1` 的书落在哪个分位。
- **十分位剖面**：每个复核日把池按分数分十组，各组收益减池均值，取全部月份的平均，逐研究年报第一组、第十组与二者之差。

## 这道门能说什么、不能说什么

- **它只能否决。** 离线读数只有秩与方向可信（登记册 §4）；离线书读正只说明值得花那 8 个回放年。
- **47 个月的主动均值误差很大。** 200 本同池随机 50 只书的年化主动 IR 就散在约 −0.6…+0.6（p10…p90）之间；只有符号与逐年一致性有意义，**不要拿离线的 %/年 去估宿主的主动 IR**。宿主面板走完整 Broker、按候选自己的成交骨架换名，并且会抽到本书剔掉的科创板成分，离线近似做不到。
- **这道门筛过的每一个候选配置都计一个 trial**：`r1` 按登记提交则在宿主上计；会话若另筛变体（换窗口、换分数、换构造），每个都在筛过之后第一个记下验证的批次里申报 `offline_trials`。重算 `r1` 本身与跑 `rand` 对照不计。
