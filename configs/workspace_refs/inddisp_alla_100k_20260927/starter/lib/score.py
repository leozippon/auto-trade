"""Low dispersion inside a Shenwan level-1 industry.

Each name is scored by minus the cross-sectional standard deviation of 21-session
stock returns in its industry, plus a tiny liquidity tie-break so the two seats
prefer names that actually trade. The industry move itself is not the score.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

LOOKBACK_DAYS = 50
UNI_COLUMNS = ["ts_code", "l1_code"]
DAILY_COLUMNS = ["ts_code", "trade_date", "close", "amount"]


def _norm(code):
    text = "" if code is None or (isinstance(code, float) and pd.isna(code)) else str(code).strip()
    if not text or text == "nan":
        return ""
    return text


def score(context, codes):
    start = (context.inference_at - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    bars = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=DAILY_COLUMNS,
        filters=[("trade_date", ">=", start)],
    )
    missing = [name for name in DAILY_COLUMNS if name not in bars.columns]
    if missing:
        raise RuntimeError(f"daily is missing columns {missing}")
    bars = bars.assign(trade_date=bars["trade_date"].astype(str), ts_code=bars["ts_code"].astype(str))
    bars = bars[bars["close"] > 0]
    dates = sorted(bars["trade_date"].unique())
    if len(dates) < 22:
        raise RuntimeError(f"only {len(dates)} sessions in the dispersion window")
    use = dates[-22:]
    window = bars[bars["trade_date"].isin(use)]
    last = window[window["trade_date"] == use[-1]].drop_duplicates("ts_code").set_index("ts_code")
    first = window[window["trade_date"] == use[0]].drop_duplicates("ts_code").set_index("ts_code")
    both = last.join(first[["close"]].rename(columns={"close": "close0"}), how="inner")
    both["ret"] = both["close"] / both["close0"] - 1.0
    universe = pd.read_parquet(context.asof_dir + "/universe", columns=UNI_COLUMNS)
    info = universe.drop_duplicates("ts_code").set_index("ts_code")
    both["l1"] = info["l1_code"].map(_norm).reindex(both.index)
    both = both[both["l1"].astype(str).str.len() > 0]
    if both.empty:
        raise RuntimeError("no names joined to a Shenwan industry")
    disp = both.groupby("l1")["ret"].std(ddof=1)
    disp = disp[disp.notna()]
    both = both[both["l1"].isin(disp.index)]
    both["dispersion"] = both["l1"].map(disp)
    amount = pd.to_numeric(last["amount"], errors="coerce").reindex(both.index).fillna(0.0)
    # Liquidity only breaks ties inside an industry; it does not reorder industries.
    both["value"] = -both["dispersion"] + 1e-8 * np.log1p(amount.to_numpy())
    values = both["value"].reindex(list(codes))
    meta = {"industries": int(disp.shape[0])}
    return values, meta


def shuffle(values, decision_at):
    scored = values.dropna().sort_index()
    day = int(pd.Timestamp(decision_at).strftime("%Y%m%d"))
    order = np.random.default_rng(day).permutation(len(scored))
    return pd.Series(scored.to_numpy()[order], index=scored.index)


def describe(values, book):
    ranks = (-values.reindex(book)).dropna()
    return {"book_score_span": [float(ranks.min()), float(ranks.max())] if len(ranks) else None}
