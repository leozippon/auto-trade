"""Vectorised signal screen over the visible decision view.

Self-contained on purpose: only numpy, pandas and pyarrow, all present in the
sandbox image, so the file is bind-mounted read-only into the Agent session
without an image rebuild. ``--help`` is the single source of the mechanics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT_SNAPSHOT = "/mnt/snapshot"
DEFAULT_HORIZONS = "1,5,10,20"
DEFAULT_TOP_FRACTION = 0.1
DEFAULT_MIN_NAMES = 50
PANEL_COLUMNS = ("open", "adj_factor", "up_limit", "is_suspended", "circ_mv")
SIGNAL_COLUMNS = ("trade_date", "ts_code", "score")
_YYYYMMDD = re.compile(r"^\d{8}$")

HELP = """\
Screen one cross-sectional, long-only signal on the visible decision view in
under a minute, before spending a full Validation replay on it. Measured on a
two-year view (about 490 days x 5,000 names, four horizons): 10-45 s of wall
time depending on host load, peak memory under 1 GB — roughly 100x cheaper
than one replay.

DATA SCOPE
  Reads only the decision view (default /mnt/snapshot): the history visible
  before this Fold's decision time, which ends before the Validation window.
  Any view whose manifest kind is not "decision_input" is refused (a replay
  slot would contain the Validation window itself; the empty
  /mnt/snapshots/train and /mnt/snapshots/valid slots have no manifest), so
  Validation replays remain the selection instrument and Test/Held-out stay
  untouched. Forward returns come from the same visible daily bars: adjusted
  open of t+1 to adjusted open of t+1+h (a score dated t is executable at the
  next open; the Broker fills at the 09:30 open or 15:00 close). The last h+1
  days of the history therefore carry no return for horizon h.

SIGNAL CONTRACT
  --signal is a Python file defining compute_signal(frames) that returns either
      a wide pandas.DataFrame: index trade_date (YYYYMMDD), columns ts_code,
      values score (the fast form; datetime indexes are converted), or
      a long pandas.DataFrame[trade_date, ts_code, score] / a Series indexed by
      (trade_date, ts_code), one row per key.
  Higher score = stronger long candidate; NaN = no view. `frames` gives lazy,
  column-pruned access to the view's tables:
      frames.names                        table names (daily, universe, events, ...)
      frames.columns("daily")             column names of one table
      frames.wide("daily", "close", start=None, end=None)
                                          (trade_date x ts_code) matrix of one column
      frames.load("daily", columns=[...], start=None, end=None)
                                          long table, column-pruned
      frames.daily                        frames.load("daily")
  Build per-name time-series features on wide matrices (e.g. adjusted close
  = frames.wide("daily","close") * frames.wide("daily","adj_factor"), then
  .pct_change(20, fill_method=None)): a wide matrix of the full history builds
  in about a second, whereas long-form sort/groupby over millions of rows takes
  tens of seconds in the sandbox. start/end (YYYYMMDD, inclusive) filter on
  trade_date where the table has it; tables keyed only by available_at
  (fundamentals, macro, text_index) are loaded whole and filtered by you.
  daily.parquet already joins daily_basic, stk_limit, adj_factor and
  is_suspended; columns and units are in /mnt/artifacts/data_summary.json.
  PIT discipline inside compute_signal is yours: a score dated t may use only
  rows with trade_date <= t (or available_at <= t's inference time). The screen
  does not detect look-ahead in your features; the formal replay is the
  enforcement and the final judge.

METRICS (per horizon h, over scored days within --start..--end)
  ic_mean / ic_std / icir  daily Spearman rank IC between score(t) and the
                           h-day forward return, its std, and mean/std
  t_stat                   icir * sqrt(n_days / h): overlap-adjusted, conservative for h > 1
  pos_months               share of calendar months with positive mean IC
  ic_marginal              rank IC against the return from open(t+1+h_prev) to
                           open(t+1+h): decay beyond the previous horizon
  ic_size_neutral          rank IC after residualising rank(score) on log(circ_mv)
                           cross-sectionally each day
  top_excess / top_hit     mean h-day return of the top --top-fraction names by
                           score that are tradable at the t+1 open, minus the
                           equal-weight mean of all names with a return that
                           day; hit = share of days with positive excess
  turnover                 share of the top set that is new versus the previous
                           day (daily rebalance); rank_autocorr =
                           Spearman(score(t), score(t-1))
  coverage                 names scored per day and their share of names with a bar
  tradability              share of scored names dropped from the long side because
                           t+1 opens at the up-limit, is suspended or has no bar
  Days with fewer than --min-names usable names are skipped. Default output is
  a table plus a JSON block; --json prints the JSON document only.
"""


class ScreenError(Exception):
    """A contract or scope violation; reported without a traceback."""


class Grid:
    """Row/column positions of a (trade_date, ts_code)-keyed table on its dates x codes lattice.

    One ``factorize`` per key column, then every column of the table is placed
    by fancy indexing; that is seconds instead of minutes for millions of rows
    of Python-string keys, which ``pivot``, ``sort_values`` or ``duplicated``
    would each rehash.
    """

    def __init__(self, trade_dates: object, ts_codes: object) -> None:
        self.rows, dates = pd.factorize(trade_dates, sort=True)
        self.cols, codes = pd.factorize(ts_codes, sort=True)
        if (self.rows < 0).any() or (self.cols < 0).any():
            raise ScreenError("trade_date and ts_code keys must not be null")
        self.dates = np.asarray(dates)
        self.codes = np.asarray(codes)
        flat = self.rows.astype(np.int64) * len(self.codes) + self.cols
        self.has_duplicates = bool(len(np.unique(flat)) != len(flat))

    def fill(self, values: object, *, dates: np.ndarray | None = None, codes: np.ndarray | None = None) -> pd.DataFrame:
        values = np.asarray(values)
        shape = (len(self.dates), len(self.codes))
        if values.dtype == bool:
            matrix = np.zeros(shape, dtype=bool)
        else:
            matrix = np.full(shape, np.nan)
            if values.dtype != np.float64:
                values = values.astype(float)
        matrix[self.rows, self.cols] = values
        return pd.DataFrame(
            matrix,
            index=pd.Index(self.dates if dates is None else dates, name="trade_date"),
            columns=pd.Index(self.codes if codes is None else codes, name="ts_code"),
        )


class Frames:
    """Lazy, column-pruned access to the parquet tables of one snapshot view."""

    def __init__(self, snapshot_dir: Path) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.names = tuple(sorted(path.stem for path in self.snapshot_dir.glob("*.parquet")))
        self._grids: dict[tuple[str, str | None, str | None], Grid] = {}

    def columns(self, name: str) -> list[str]:
        return list(pq.ParquetFile(self._path(name)).schema_arrow.names)

    def load(
        self,
        name: str,
        columns: list[str] | tuple[str, ...] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        path = self._path(name)
        available = self.columns(name)
        if columns is not None:
            missing = [column for column in columns if column not in available]
            if missing:
                raise ScreenError(f"{name}.parquet has no column(s) {missing}; available: {available}")
        filters = None
        if start or end:
            if "trade_date" not in available:
                raise ScreenError(f"{name}.parquet has no trade_date column; load it whole and filter yourself")
            filters = []
            if start:
                filters.append(("trade_date", ">=", str(start)))
            if end:
                filters.append(("trade_date", "<=", str(end)))
        table = pq.read_table(path, columns=list(columns) if columns is not None else None, filters=filters)
        return table.to_pandas()

    def grid(self, name: str, start: str | None = None, end: str | None = None) -> Grid:
        """The key lattice of one table, cached: re-reads of a parquet file keep its row order."""
        key = (name, start, end)
        if key not in self._grids:
            keys = self.load(name, columns=["trade_date", "ts_code"], start=start, end=end)
            self._grids[key] = Grid(keys["trade_date"], keys["ts_code"])
        return self._grids[key]

    def wide(self, name: str, column: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        if column in ("trade_date", "ts_code"):
            raise ScreenError(f"{column} is a key of the wide matrix, not a value column")
        grid = self.grid(name, start, end)
        if grid.has_duplicates:
            raise ScreenError(f"{name}.parquet is not unique per (trade_date, ts_code); aggregate it before pivoting")
        values = self.load(name, columns=[column], start=start, end=end)[column]
        try:
            return grid.fill(values)
        except (TypeError, ValueError) as exc:
            raise ScreenError(f"{name}.{column} is not numeric or boolean: {exc}") from exc

    def __getattr__(self, name: str) -> pd.DataFrame:
        if name.startswith("_") or name == "names" or name not in self.names:
            raise AttributeError(name)
        return self.load(name)

    def _path(self, name: str) -> Path:
        if name not in self.names:
            raise ScreenError(f"no table {name!r} in {self.snapshot_dir}; available: {list(self.names)}")
        return self.snapshot_dir / f"{name}.parquet"


def open_view(snapshot_dir: Path) -> dict[str, object]:
    """Return the manifest of a decision view; refuse anything else."""
    manifest_path = Path(snapshot_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ScreenError(
            f"{snapshot_dir} has no manifest.json, so it is not a snapshot view (the empty "
            f"/mnt/snapshots/train and /mnt/snapshots/valid slots are not evaluable); "
            f"use the decision view at {DEFAULT_SNAPSHOT}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    kind = manifest.get("kind")
    if kind != "decision_input":
        raise ScreenError(
            f"refusing to evaluate on {snapshot_dir}: manifest kind is {kind!r}, not 'decision_input'. "
            "The screen only runs on the decision view (history visible before the decision time); "
            "a replay slot contains the Validation window, which stays reserved for formal Validation."
        )
    return manifest


def decision_date(manifest: dict[str, object]) -> str | None:
    value = manifest.get("decision_time")
    if not isinstance(value, str) or len(value) < 10:
        return None
    return value[:10].replace("-", "")


def load_signal(path: Path, frames: Frames) -> pd.DataFrame:
    if not path.is_file():
        raise ScreenError(f"signal file not found: {path}")
    spec = importlib.util.spec_from_file_location("agent_signal", path)
    if spec is None or spec.loader is None:
        raise ScreenError(f"cannot import signal file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    compute = getattr(module, "compute_signal", None)
    if not callable(compute):
        raise ScreenError(f"{path} must define compute_signal(frames)")
    return normalise_signal(compute(frames))


def _yyyymmdd(values: object, what: str) -> np.ndarray:
    """Distinct date labels as YYYYMMDD strings; datetimes are formatted, anything else must already match."""
    index = pd.Index(values)
    if pd.api.types.is_datetime64_any_dtype(index):
        index = index.strftime("%Y%m%d")
    text = index.astype(str)
    bad = [item for item in text if not _YYYYMMDD.match(item)]
    if bad:
        raise ScreenError(f"signal {what} must be YYYYMMDD, e.g. {bad[0]!r}")
    return np.asarray(text, dtype=str)


def normalise_signal(result: object) -> pd.DataFrame:
    """Validate a wide, long or Series signal and return it as a dates x codes matrix."""
    if isinstance(result, pd.Series):
        if not isinstance(result.index, pd.MultiIndex) or result.index.nlevels != 2:
            raise ScreenError("a Series signal needs a two-level (trade_date, ts_code) index")
        result = result.rename("score").rename_axis(["trade_date", "ts_code"]).reset_index()
    if not isinstance(result, pd.DataFrame):
        raise ScreenError(f"compute_signal must return a DataFrame or Series, got {type(result).__name__}")
    if set(SIGNAL_COLUMNS) <= set(map(str, result.columns)):
        wide = _long_to_wide(result)
    else:
        wide = _check_wide(result)
    try:
        wide = wide.astype(float)
    except (TypeError, ValueError) as exc:
        raise ScreenError(f"signal score must be numeric: {exc}") from exc
    wide = wide.where(np.isfinite(wide))
    if not wide.notna().any().any():
        raise ScreenError("signal has no finite score")
    return wide


def _long_to_wide(frame: pd.DataFrame) -> pd.DataFrame:
    grid = Grid(frame["trade_date"], frame["ts_code"])
    dates = _yyyymmdd(grid.dates, "trade_date")
    if grid.has_duplicates:
        raise ScreenError("signal has duplicate (trade_date, ts_code) keys")
    try:
        score = pd.to_numeric(frame["score"]).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ScreenError(f"signal score must be numeric: {exc}") from exc
    return grid.fill(score, dates=dates, codes=grid.codes.astype(str))


def _check_wide(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.index.nlevels != 1 or frame.columns.nlevels != 1:
        raise ScreenError(
            f"a wide signal needs a single-level trade_date index and ts_code columns; a long one needs columns {list(SIGNAL_COLUMNS)}"
        )
    index = pd.Index(_yyyymmdd(frame.index, "index (trade_date)"), name="trade_date")
    columns = pd.Index(frame.columns.astype(str), name="ts_code")
    if index.has_duplicates or columns.has_duplicates:
        raise ScreenError("a wide signal must have unique trade_date index labels and unique ts_code columns")
    wide = pd.DataFrame(frame.to_numpy(), index=index, columns=columns)
    return wide.sort_index()


class Panel:
    """Wide (trade_date x ts_code) matrices of the visible daily history."""

    def __init__(self, frames: Frames, decision: str | None) -> None:
        available = frames.columns("daily")
        missing = [column for column in ("trade_date", "ts_code", *PANEL_COLUMNS) if column not in available]
        if missing:
            raise ScreenError(f"daily.parquet lacks column(s) {missing}")
        grid = frames.grid("daily")
        if grid.has_duplicates:
            raise ScreenError("daily.parquet is not unique per (trade_date, ts_code)")
        self.dates = grid.dates.astype(str)
        if decision is not None and self.dates[-1] > decision:
            raise ScreenError(f"daily history ends {self.dates[-1]}, after the decision date {decision}")
        self.codes = pd.Index(grid.codes.astype(str))
        daily = frames.load("daily", columns=PANEL_COLUMNS)
        open_ = daily["open"].to_numpy(dtype=float)
        open_ = np.where(open_ > 0, open_, np.nan)
        circ_mv = daily["circ_mv"].to_numpy(dtype=float)
        self.adj_open = grid.fill(open_ * daily["adj_factor"].to_numpy(dtype=float))
        self.open_at_up_limit = grid.fill(open_ >= daily["up_limit"].to_numpy(dtype=float))
        self.suspended = grid.fill(daily["is_suspended"].to_numpy(dtype=bool))
        self.log_mv = grid.fill(np.log(np.where(circ_mv > 0, circ_mv, np.nan)))
        # What the long side can buy at the next open: a bar exists, the name
        # is not suspended and the open is below the up-limit.
        self.tradable = (
            self.adj_open.shift(-1).notna()
            & ~self.suspended.shift(-1, fill_value=False)
            & ~self.open_at_up_limit.shift(-1, fill_value=False)
        )

    def align(self, signal: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Place a wide signal on the panel grid; unknown dates are an error, unknown codes are counted."""
        unknown_dates = sorted(set(signal.index) - set(self.dates))
        if unknown_dates:
            raise ScreenError(
                f"signal has {len(unknown_dates)} trade_date(s) outside the visible daily history "
                f"(e.g. {unknown_dates[:3]}): dates after the decision time indicate look-ahead, "
                "others are not trading days"
            )
        unknown_codes = int((~signal.columns.isin(self.codes)).sum())
        return signal.reindex(index=self.dates, columns=self.codes), unknown_codes


def forward_returns(adj_open: pd.DataFrame, horizon: int, entry_offset: int = 0) -> pd.DataFrame:
    """Return from the open of t+1+entry_offset to the open of t+1+horizon, indexed by signal date t."""
    return adj_open.shift(-(1 + horizon)) / adj_open.shift(-(1 + entry_offset)) - 1


def row_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) along axis 1 with NaN kept as NaN — what Spearman needs.

    Two argsorts plus a tie-group average in flat index space: about 0.1 s for a
    500 x 5,500 panel where ``DataFrame.rank(axis=1)`` takes half a second, and
    the screen ranks a few dozen such panels per run.
    """
    n_rows, n_cols = values.shape
    order = np.argsort(values, axis=1, kind="stable")  # NaN sorts last
    sorted_values = np.take_along_axis(values, order, axis=1)
    new_group = np.ones((n_rows, n_cols), dtype=bool)
    new_group[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]  # NaN != NaN, so each NaN is its own group
    starts = np.flatnonzero(new_group.ravel())
    ends = np.append(starts[1:], n_rows * n_cols) - 1
    group_rank = (starts + ends) / 2.0
    sorted_ranks = group_rank[np.cumsum(new_group.ravel()) - 1].reshape(n_rows, n_cols)
    sorted_ranks -= (np.arange(n_rows) * n_cols)[:, None] - 1.0
    ranks = np.empty_like(sorted_ranks)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    ranks[np.isnan(values)] = np.nan
    return ranks


def _centered(ranks: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = mask.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(mask, ranks, 0.0).sum(axis=1, keepdims=True) / count
    return np.where(mask, ranks - mean, 0.0)


def rank_ic(a: pd.DataFrame, b: pd.DataFrame, min_names: int) -> pd.Series:
    """Daily Spearman correlation across names; days with fewer than ``min_names`` pairs are NaN."""
    va = a.to_numpy(dtype=float)
    vb = b.to_numpy(dtype=float)
    mask = ~(np.isnan(va) | np.isnan(vb))
    ra = _centered(row_ranks(np.where(mask, va, np.nan)), mask)
    rb = _centered(row_ranks(np.where(mask, vb, np.nan)), mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        ic = (ra * rb).sum(axis=1) / np.sqrt((ra**2).sum(axis=1) * (rb**2).sum(axis=1))
    ic[~np.isfinite(ic) | (mask.sum(axis=1) < min_names)] = np.nan
    return pd.Series(ic, index=a.index)


def size_neutral(score: pd.DataFrame, log_mv: pd.DataFrame) -> pd.DataFrame:
    """Residual of rank(score) on log market cap, cross-sectionally per day."""
    vs = score.to_numpy(dtype=float)
    vx = log_mv.to_numpy(dtype=float)
    mask = ~(np.isnan(vs) | np.isnan(vx))
    s = _centered(row_ranks(np.where(mask, vs, np.nan)), mask)
    x = _centered(vx, mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (s * x).sum(axis=1, keepdims=True) / (x**2).sum(axis=1, keepdims=True)
    residual = s - x * np.nan_to_num(beta)
    return pd.DataFrame(np.where(mask, residual, np.nan), index=score.index, columns=score.columns)


def top_selection(score: pd.DataFrame, tradable: pd.DataFrame, fraction: float) -> pd.DataFrame:
    """Boolean membership of the top ``fraction`` tradable names by score, per day (ties broken by column order)."""
    usable = np.where(tradable.to_numpy(dtype=bool), score.to_numpy(dtype=float), np.nan)
    present = ~np.isnan(usable)
    count = np.ceil(present.sum(axis=1, keepdims=True) * fraction)
    order = np.argsort(-usable, axis=1, kind="stable")  # NaN sorts last
    position = np.empty_like(order)
    np.put_along_axis(position, order, np.arange(1, usable.shape[1] + 1)[None, :].repeat(usable.shape[0], axis=0), axis=1)
    return pd.DataFrame((position <= count) & present, index=score.index, columns=score.columns)


def _summary(series: pd.Series, horizon: int) -> dict[str, object]:
    values = series.dropna()
    n = len(values)
    if n == 0:
        return {"n_days": 0}
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else math.nan
    ratio = mean / std if std and std > 0 else math.nan
    months = values.groupby(values.index.str[:6]).mean()
    return {
        "n_days": n,
        "mean": mean,
        "std": std,
        "ratio": ratio,
        "t_stat": ratio * math.sqrt(n / horizon) if not math.isnan(ratio) else math.nan,
        "positive_share": float((values > 0).mean()),
        "positive_month_share": float((months > 0).mean()),
        "n_months": len(months),
    }


def run_screen(
    frames: Frames,
    manifest: dict[str, object],
    signal: pd.DataFrame,
    *,
    horizons: list[int],
    start: str | None,
    end: str | None,
    top_fraction: float,
    min_names: int,
) -> dict[str, object]:
    panel = Panel(frames, decision_date(manifest))
    score_all, unknown_codes = panel.align(signal)
    first, last = start or panel.dates[0], end or panel.dates[-1]
    if first > last or first > panel.dates[-1] or last < panel.dates[0]:
        raise ScreenError(f"window {first}..{last} is outside the visible history {panel.dates[0]}..{panel.dates[-1]}")
    # Dates are sorted, so the window is one contiguous row slice (a view, not a copy).
    window = slice(int(np.searchsorted(panel.dates, first)), int(np.searchsorted(panel.dates, last, side="right")))

    score = score_all.iloc[window]
    tradable = panel.tradable.iloc[window]
    neutral = size_neutral(score, panel.log_mv.iloc[window])
    top = top_selection(score, tradable, top_fraction)
    top_count = top.sum(axis=1)
    usable = (score.notna() & tradable).sum(axis=1) >= min_names

    scored = score.notna()
    scored_days = scored.sum(axis=1)
    has_bar = panel.adj_open.iloc[window].notna()
    entry = panel.adj_open.shift(-1).iloc[window].notna()
    limit_next = panel.open_at_up_limit.shift(-1, fill_value=False).iloc[window]
    suspended_next = panel.suspended.shift(-1, fill_value=False).iloc[window]
    scored_total = int(scored.sum().sum())
    scored_with_entry = int((scored & entry).sum().sum())
    limit_excluded = int((scored & entry & limit_next).sum().sum())
    unavailable = int((scored & (~entry | suspended_next)).sum().sum())
    new_share = (top & ~top.shift(1, fill_value=False)).sum(axis=1) / top_count.where(top_count > 0)
    autocorr = rank_ic(score, score_all.shift(1).iloc[window], min_names)

    horizon_reports: list[dict[str, object]] = []
    previous = 0
    for horizon in horizons:
        ret = forward_returns(panel.adj_open, horizon).iloc[window]
        ic = _summary(rank_ic(score, ret, min_names), horizon)
        marginal = _summary(rank_ic(score, forward_returns(panel.adj_open, horizon, previous).iloc[window], min_names), horizon)
        neutral_ic = _summary(rank_ic(neutral, ret, min_names), horizon)
        excess = (ret.where(top).mean(axis=1) - ret.mean(axis=1)).where(usable & (top_count > 0))
        top_stats = _summary(excess, horizon)
        horizon_reports.append(
            {
                "horizon": horizon,
                "n_days": ic["n_days"],
                "ic_mean": ic.get("mean"),
                "ic_std": ic.get("std"),
                "icir": ic.get("ratio"),
                "t_stat": ic.get("t_stat"),
                "positive_day_share": ic.get("positive_share"),
                "positive_month_share": ic.get("positive_month_share"),
                "n_months": ic.get("n_months"),
                "ic_marginal": marginal.get("mean"),
                "ic_size_neutral": neutral_ic.get("mean"),
                "icir_size_neutral": neutral_ic.get("ratio"),
                "top_excess_mean": top_stats.get("mean"),
                "top_excess_hit": top_stats.get("positive_share"),
                "top_excess_t_stat": top_stats.get("t_stat"),
                "top_names_mean": float(top_count[usable].mean()) if usable.any() else None,
            }
        )
        previous = horizon

    return {
        "snapshot": {
            "path": str(frames.snapshot_dir),
            "snapshot_id": manifest.get("snapshot_id"),
            "kind": manifest.get("kind"),
            "decision_time": manifest.get("decision_time"),
            "history_start": str(panel.dates[0]),
            "history_end": str(panel.dates[-1]),
            "trade_days": len(panel.dates),
            "codes": len(panel.codes),
        },
        "signal": {
            "finite_scores": int(signal.notna().sum().sum()),
            "codes_not_in_daily": unknown_codes,
        },
        "window": {
            "start": str(first),
            "end": str(last),
            "trade_days": window.stop - window.start,
            "top_fraction": top_fraction,
            "min_names": min_names,
        },
        "coverage": {
            "days_scored": int((scored_days > 0).sum()),
            "names_per_day_mean": float(scored_days[scored_days > 0].mean()) if (scored_days > 0).any() else 0.0,
            "universe_share_mean": float(((scored & has_bar).sum(axis=1) / has_bar.sum(axis=1))[scored_days > 0].mean())
            if (scored_days > 0).any()
            else 0.0,
        },
        "tradability": {
            "up_limit_excluded_share": limit_excluded / scored_with_entry if scored_with_entry else None,
            "suspended_or_no_bar_share": unavailable / scored_total if scored_total else None,
        },
        "turnover": {
            "top_new_share_mean": float(new_share.iloc[1:].dropna().mean()) if new_share.iloc[1:].notna().any() else None,
            "rank_autocorr_mean": float(autocorr.mean()) if autocorr.notna().any() else None,
        },
        "horizons": horizon_reports,
    }


def _fmt(value: object, width: int, digits: int = 4, percent: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-".rjust(width)
    if isinstance(value, float):
        return (f"{value * 100:.{digits - 2}f}%" if percent else f"{value:.{digits}f}").rjust(width)
    return str(value).rjust(width)


def render_table(report: dict[str, object]) -> str:
    snap, sig, win = report["snapshot"], report["signal"], report["window"]
    cov, trade, turn = report["coverage"], report["tradability"], report["turnover"]
    lines = [
        (
            f"view {snap['path']} ({snap['kind']}, decision_time={snap['decision_time']}) "
            f"history {snap['history_start']}..{snap['history_end']} ({snap['trade_days']} days, {snap['codes']} codes)"
        ),
        (
            f"signal finite_scores={sig['finite_scores']} codes_not_in_daily={sig['codes_not_in_daily']} "
            f"window {win['start']}..{win['end']} ({win['trade_days']} days) top_fraction={win['top_fraction']} min_names={win['min_names']}"
        ),
        (
            f"coverage days_scored={cov['days_scored']} names/day={cov['names_per_day_mean']:.0f} "
            f"universe_share={_fmt(cov['universe_share_mean'], 0, percent=True).strip()}"
        ),
        (
            f"tradability up_limit_excluded={_fmt(trade['up_limit_excluded_share'], 0, percent=True).strip()} "
            f"suspended_or_no_bar={_fmt(trade['suspended_or_no_bar_share'], 0, percent=True).strip()}"
        ),
        (
            f"turnover top_new_share/day={_fmt(turn['top_new_share_mean'], 0).strip()} "
            f"rank_autocorr={_fmt(turn['rank_autocorr_mean'], 0).strip()}"
        ),
        "  h  n_days  ic_mean  ic_std   icir  t_stat  pos_days  pos_months  ic_marginal  ic_size_neutral  top_excess  top_hit  top_t",
    ]
    for row in report["horizons"]:
        lines.append(
            f"{_fmt(row['horizon'], 3)}{_fmt(row['n_days'], 8)}{_fmt(row['ic_mean'], 9)}{_fmt(row['ic_std'], 8)}"
            f"{_fmt(row['icir'], 7, 2)}{_fmt(row['t_stat'], 8, 2)}{_fmt(row['positive_day_share'], 10, 2)}"
            f"{_fmt(row['positive_month_share'], 12, 2)}{_fmt(row['ic_marginal'], 13)}{_fmt(row['ic_size_neutral'], 17)}"
            f"{_fmt(row['top_excess_mean'], 12, 4, percent=True)}{_fmt(row['top_excess_hit'], 9, 2)}{_fmt(row['top_excess_t_stat'], 7, 2)}"
        )
    return "\n".join(lines)


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return None if math.isnan(value) else round(value, 6)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    return value


def parse_horizons(text: str) -> list[int]:
    try:
        horizons = sorted({int(item) for item in text.split(",") if item.strip()})
    except ValueError as exc:
        raise ScreenError(f"--horizons must be comma-separated positive integers, got {text!r}") from exc
    if not horizons or horizons[0] <= 0:
        raise ScreenError(f"--horizons must be comma-separated positive integers, got {text!r}")
    return horizons


def _date_arg(value: str | None, flag: str) -> str | None:
    if value is None:
        return None
    if not _YYYYMMDD.match(value):
        raise ScreenError(f"{flag} must be YYYYMMDD, got {value!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="screen.py",
        description=HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--signal", required=True, help="Python file defining compute_signal(frames)")
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help=f"decision view directory (default {DEFAULT_SNAPSHOT})")
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS, help=f"forward horizons in trading days (default {DEFAULT_HORIZONS})")
    parser.add_argument("--start", help="first scored trade_date, YYYYMMDD (default: start of the visible history)")
    parser.add_argument("--end", help="last scored trade_date, YYYYMMDD (default: end of the visible history)")
    parser.add_argument("--top-fraction", type=float, default=DEFAULT_TOP_FRACTION, help=f"long-side selection fraction (default {DEFAULT_TOP_FRACTION})")
    parser.add_argument("--min-names", type=int, default=DEFAULT_MIN_NAMES, help=f"minimum usable names per day (default {DEFAULT_MIN_NAMES})")
    parser.add_argument("--json", action="store_true", help="print the JSON document only")
    return parser


def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args = build_parser().parse_args(argv)
    try:
        if not 0 < args.top_fraction <= 1:
            raise ScreenError(f"--top-fraction must be in (0, 1], got {args.top_fraction}")
        if args.min_names < 2:
            raise ScreenError(f"--min-names must be at least 2, got {args.min_names}")
        horizons = parse_horizons(args.horizons)
        snapshot_dir = Path(args.snapshot)
        manifest = open_view(snapshot_dir)
        frames = Frames(snapshot_dir)
        signal = load_signal(Path(args.signal), frames)
        report = run_screen(
            frames,
            manifest,
            signal,
            horizons=horizons,
            start=_date_arg(args.start, "--start"),
            end=_date_arg(args.end, "--end"),
            top_fraction=args.top_fraction,
            min_names=args.min_names,
        )
    except ScreenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report["signal"]["path"] = str(args.signal)
    report["wall_seconds"] = round(time.perf_counter() - started, 3)
    document = json.dumps(_json_ready(report), ensure_ascii=False, indent=None if args.json else 1)
    if args.json:
        print(document)
    else:
        print(render_table(report))
        print(document)
        print(f"wall time: {report['wall_seconds']:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
