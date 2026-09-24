# 离线门 G-V8 的做法：秩 IC、顶端对同池、登记构造的书，分两半

自写笔记。`refs/families.md` 的离线门在整期批次之前、全离线、0 回放年地回答一个问题：**这个分数在本池、八个研究年的决策视图上，有没有登记方向的秩信息，这份信息在 Y1–Y4 这一半上还在不在，落不落得到一本只做多、50 席等额、带保留带的书上。**判据只在 `refs/families.md`；本页写怎么算、能说明什么。包作者的读数在 `refs/sources.md`，会话要自己重算，不引用包里的数。**命中不改变第一批。**

## 先把分数与池逐日按时点建出来

`/mnt/snapshot` 是研究期末的单一时点视图（日线从 2016-07-01 起）；分数只读 T-1 那一行的比率与最近 250 个交易日的换手，所以只有 Y1 头几个复核日的 250 日换手窗被左端截断（`refs/pit-field-map.md`）；成分截面要逐日取「日期不晚于 t 的最新一张」，不能用最新一张去标注历史。下面这个 `compute_signal` 就是包作者用的那一份（四个价值 / 换手包逐字相同，只差第一行的常量；`MODE` 改成 `"rand"` 就是同池随机对照，`NEIGHBOUR = True` 就是登记的邻居——算邻居就是筛了它，按 `refs/families.md` 计），写进 `workspace/` 里的一个文件即可：

```python
import math
import numpy as np, pandas as pd

SCORE, MODE, NEIGHBOUR = "dvy", "cand", False
INDEX = "000300.SH" if SCORE == "dvy" else "000905.SH"
SHORT, LONG = 20, (120 if NEIGHBOUR else 250)


def window_mean(tr, days):                      # 最近 days 个交易日里有效日（换手率 > 0）的均值
    valid = np.isfinite(tr) & (tr > 0)
    v = np.where(valid, tr, 0.0)
    cs = np.vstack([np.zeros((1, v.shape[1])), v.cumsum(axis=0)])
    cn = np.vstack([np.zeros((1, v.shape[1])), valid.cumsum(axis=0)])
    lo = np.maximum(np.arange(1, len(v) + 1) - days, 0)
    total, count = cs[1:] - cs[lo], cn[1:] - cn[lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count >= math.ceil(0.75 * days), total / count, np.nan)


def compute_signal(frames):
    mv = frames.wide("daily", "total_mv")
    dates, codes = np.asarray(mv.index).astype(str), np.asarray(mv.columns).astype(str)
    has_row = mv.to_numpy() > 0                 # 这一行带 daily_basic
    pe = frames.wide("daily", "pe" if NEIGHBOUR else "pe_ttm").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ep = np.where(has_row, np.where(pe > 0, 1.0 / pe, -1.0), np.nan)      # 滚动亏损排最后
    tr = frames.wide("daily", "turnover_rate").to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ab = -window_mean(tr, SHORT) / window_mean(tr, LONG)
    if SCORE == "dvy":
        dv = np.full(mv.shape, np.nan)
        for column in (("dv_ratio",) if NEIGHBOUR else ("dv_ttm", "dv_ratio")):
            x = frames.wide("daily", column).to_numpy()
            dv = np.where(np.isnan(dv) & (x > 0), x, dv)
        parts = [np.where(has_row, np.nan_to_num(dv, nan=0.0), np.nan)]      # 没有现金分红排最后
    else:
        parts = {"ep": [ep], "abturn": [ab], "ch3": [ep, ab]}[SCORE]
    macro = frames.load("macro", columns=["dataset", "index_code", "con_code", "trade_date", "weight"])
    sec = macro[(macro.dataset == "index_weight") & (macro.index_code == INDEX) & (macro.weight > 0)]
    sec = sec.assign(trade_date=sec.trade_date.astype(str))
    sect_dates = np.array(sorted(sec.trade_date.unique()))
    pos = {c: i for i, c in enumerate(codes)}
    rows = {d: [pos[c] for c in g.con_code if c in pos] for d, g in sec.groupby("trade_date")}
    member = np.zeros(mv.shape, dtype=bool)
    for t, k in enumerate(np.searchsorted(sect_dates, dates, side="right") - 1):
        if k >= 0:
            member[t, rows[sect_dates[k]]] = True     # 日期 <= t 的最新截面：t 之后那个决策读得到它
    board = ~(pd.Index(codes).str.startswith(("688", "689")) | pd.Index(codes).str.endswith(".BJ"))
    ok = member & np.asarray(board)[None, :] & np.logical_and.reduce([np.isfinite(p) for p in parts])
    if len(parts) == 1:
        score = parts[0]
    else:                                             # ch3：池内两个百分位秩的均值
        score = np.mean([pd.DataFrame(np.where(ok, p, np.nan)).rank(axis=1, pct=True).to_numpy() for p in parts], axis=0)
    if MODE == "rand":                                # 同一批格子上、逐日重抽的同池随机排名
        s = pd.Series(codes)
        score = np.vstack([pd.util.hash_pandas_object(s, index=False, hash_key=f"v8rand_{d}x").to_numpy()
                           % 1_000_003 for d in dates]) / 1_000_003.0
    elif MODE != "cand":
        raise ValueError(f"unknown MODE {MODE!r}")
    return pd.DataFrame(np.where(ok, score, np.nan), index=dates, columns=codes)
```

日期 t 的分数只用 t 及以前的日线与 t 那天 17:30 已盖章的截面，正是 t 之后那个交易日 08:30 决策读到的东西；screen 从 t+1 的开盘算前向收益，两边对齐，不需要再移一行。筛选工具把非有限值当缺失，所以起步包里「排最后」的 −inf 在这里写成有限的地板：滚动亏损的 `ep` 记 −1（任何正的盈利收益率都在它之上），没有现金分红的 `dvy` 记 0。名称是研究期末的，所以离线不按名称过滤 ST（`refs/pit-field-map.md`）。

## 工具一：`/mnt/tools/screen.py`（S2、S3、S4）

```
python /mnt/tools/screen.py --signal <file> --horizons 5,10,20,40 \
    --start 20170701 --end 20250630 --top-fraction 0.1725
```

`dvy` 与 `rand` 各跑一次整期，再各跑两半（`--start 20170701 --end 20210630`、`--start 20210701 --end 20250630`）与八个研究年（每年的 0701..0630）。`--top-fraction` 是 50 ÷ 每日可打分池的均值（约 290）。读 `ic_mean`、`t_stat`（已按重叠折算）、`positive_month_share`、`ic_size_neutral` 与 `top_excess_mean`。一次整期约两分钟、峰值内存不到 4 GiB。

- **`top_excess` 减的是当日被打分的名字（IC 排序的那个池）的等权均值**——就是本池，不是全市场；更早的包里「对全市场」的写法已过时。所以 S3 直接读 `dvy` 的 `top_excess_mean`；`rand` 的应在零附近，是对这条口径的检查。
- h = 20 是本书的复核间隔，40 是持有期量级。这个分数很慢（相邻复核秩相关约 0.97），IC 随 horizon 上升并不奇怪；池只有约 290 只，顶端一成约 29 只。

## 工具二：登记构造的书（S5 与诊断，自己写的脚本）

**离线书必须跑登记的构造**（登记册 §4：整本重排的前 50 名不是宿主交易的书，只能作标明了的诊断）。在研究期的日频网格上把起步包的规则逐日重放成一本连续的书：

- **分数与池**：每个决策日读 T-1 那一行的分数（上面的 `compute_signal`，但「排最后」要照起步包留在池里），池 = 可交易成分里分数有定义的名字；名称、ST 与行业取各研究年回放槽的表（会话只有研究期末的表时，就用它并写明口径差）。
- **书**：自然月首个决策日复核（空仓时任何一天）；持仓排名 ≥ 100 就卖，离开池的卖；每个申万一级最多 14 只，**保留的名字算在内**，超额从排名最差的卖起；座位 = 账面价值 ÷ 50（卖出所得 × 0.98 加现金，× 0.97，再加保留持仓按 T-1 收盘的市值）；只买一手（100 × T-1 收盘）不超过一个座位的名字，从排名最前往下补到 50 只；两次复核之间只补空座位；**保留的名字带着漂移后的市值**，不在复核日拉回等权。
- **成交与拒单**：卖在当日开盘（没有日线或开在跌停就拒，留到下一个复核日），买在当日收盘（没有日线、收在涨停或超过手上现金就拒）；按复权价（`close × adj_factor`、`open × adj_factor`）算持有收益。
- **量尺**：**不要对池等权量**——月度买入持有的书对逐日再平衡的等权池有偏（`range20` 包作者试过：打乱分数的书在这把尺子上读出约 −0.7 的主动 IR）。照宿主的面板做：保留书里每个回合的进出日与它每天占的权重，把名字换成同侧成分（进场前最近一张截面）里买得起、进出两天都有行情的随机名字，20 份平均当面板；主动 = 书 − 面板，再把日主动收益对指数日收益（`macro` 里 `dataset == "index_daily"` 且 `ts_code == "000300.SH"`）与全市场规模价差（前一日流通市值最小 30 % 等权减最大 30 %）回归，截距 × 244 为中性化主动超额、残差标准差年化为跟踪误差；逐年在那一年自己的日子上重跑。不计费用（面板付同样的成本）。
- **零技能带**：同一本书把分数在每个决策日的池里打乱（`c_shuf` 的构造），各对自己的换名面板，几十次，读主动 IR 的分位，看 `dvy` 落在哪里。
- **两半与逐年**：Y1–Y4 与 Y5–Y8 分开报年化主动、主动 IR 与为正的年数；十分位剖面（每个复核日把可买池按分数分十组，各组复核日收盘到下一个复核日开盘的收益减池均值）也分两半报。
- 离线的名称与行业若取研究期末的表（申万 2021 版灌回 2017–2021），行业读数只作相对参考；拒单是日线规则的近似，不计整手误差之外的成本：这是它与宿主回放的已知差别。

## 这道门能说什么、不能说什么

- **它只能否决。** 离线读数只有秩与方向可信（登记册 §4）；离线书读正只说明值得花那 16 个回放年，读负也不省下它（`no_edge` 要一次完整期验证）。
- **八年的离线书是一次有噪声的抽样。** 零技能下主动 IR 的抽样误差每一半约 0.5、八年约 0.35；只有符号与两半、逐年的一致性有意义，**不要拿离线的 %/年 去估宿主的主动 IR**。宿主面板走完整 Broker、按候选自己的成交骨架换名、会抽到本书剔掉的科创板成分，并在基准与规模上中性化，离线近似做不到全部。
- **这道门筛过的每一个候选配置都计一个 trial**：`dvy` 按登记提交则在宿主上计；会话若另筛变体（换列、换窗口、换构造），每个都在筛过之后第一个记下验证的批次里申报 `offline_trials`。重算 `dvy` 本身与跑 `rand` 对照不计。 只用 `dv_ttm` 的定义算一次也是筛了一个配置。
