# 离线读数：先核对，再只用它能说明的部分

自己写的脚本、离线书与筛选没有回放合同兜底：一处解析或索引错误会安静地给出一个看起来合理的数，之后的取舍全建在它上面。离线读数用来否决想法、给候选排序；量级与提名只看宿主的正式验证。

## 先核对

- 写解析之前先跑一次原始命令、看几行真实输出，不凭印象假定格式。
- 据离线网格、排名或统计量提交批次之前，用一条独立实现的路径重算所选点及其相邻两点。子代理交回的数也一样：核过再写进 `hypothesis` 或 `reason`，还没交回的数不先写。
- 离线引擎先对上一个已验证节点再用：用候选自己的打分与下单函数，在该节点的前几个复核日重算名单与订单，逐笔对照它 `result_ref` 记录里的 `executions`。对不上先修引擎，不读它的收益。
- 在研究期末视图上模拟 08:30 的决策时按下面截断。不要手拼时间字符串去和 `available_at` 比：格式差一个字符，字典序比较就会放进未来的行而不报错。

```python
import pandas as pd


def visible(frame, inference_at):
    """Rows an 08:30 decision at inference_at sees: daily by trade_date, macro by available_at."""
    at = pd.Timestamp(inference_at).tz_convert("Asia/Shanghai")
    if "available_at" in frame.columns:
        return frame[pd.to_datetime(frame["available_at"], utc=True) <= at]
    return frame[frame["trade_date"] < at.strftime("%Y%m%d")]
```

## 能说明什么

- 秩与符号能迁移到宿主；水平只有在离线书跑的就是登记的构造（席位、保留带、行业上限、按书值定的每席资金与整手、ST 过滤、换名上限）、又用宿主的量尺时才可比，而且仍只是一次有噪声的抽样。整本重排的前 N 名只能作标明了的诊断。
- 宿主的量尺是主动序列：节点日收益减去按它自己成交骨架回放的零技能面板，再对 `benchmark_index` 与规模因子回归（构造见 `run_null_control` 的描述与 `neutralized_excess_method`）。由此：
  - `/mnt/tools/screen.py` 的 `top_excess` 减的是当日被打分名字的等权均值，不是面板；书带强规模倾斜时两者可以符号相反。
  - 宽池的 rank IC 是整池的排序，几十席以内的集中书只持有最前面一小截：先看 `top_excess` 的 t 与 hit，IC 显著不等于前几名有边际。
  - 换手、仓位不同的腿（例如每期重新打乱的 `c_shuf`，保留带留不住随机名次）在主动读数上可比，因为面板按各腿自己的骨架抽；在原始收益与 `excess_return` 上不可比。
- 离线门只能否决：命中就不花回放年；没命中不是边际的证据。
- 离线视图不是回放视图（`/mnt/snapshot` 的 `universe` 只有研究期末一版，日线自 2016-07 起，见 `research-8y-data-surface`）：涉及早年、名称、ST 与行业的离线水平都带着这层误差。
