"""Normalized daily market data with point-in-time visibility.

The whole replay window lives in one columnar chunk sorted by
``(available_at, trade_date, symbol)``; a trading day is an index into it and
a bar is materialized only when the Broker or a strategy reads it, so the
memory held per replay is a few hundred bytes per row plus whatever the
current day actually touches.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from datetime import date, datetime, time

import numpy as np
import pandas as pd

from autotrade.environment.strategy import (
    CN_TZ,
    BarTable,
    StrategyContractError,
    _BarPrefix,
    _Chunk,
    _Column,
    _epoch_us,
    _Row,
)


class DailyMarketData:
    REQUIRED = ("trade_date", "open", "close")

    def __init__(self, daily: pd.DataFrame) -> None:
        columns = list(daily.columns)
        symbol_source = "symbol"
        if "symbol" not in columns and "ts_code" in columns:
            symbol_source = "ts_code"
            columns[columns.index("ts_code")] = "symbol"
        missing = [column for column in (*self.REQUIRED, "symbol") if column not in columns]
        if missing:
            raise ValueError(f"daily market data missing columns: {missing}")
        count = len(daily)
        date_codes, dates = _normalized_codes(daily["trade_date"], _date_text)
        symbol_codes, symbols = _normalized_codes(daily[symbol_source], _symbol_text)
        if "" in symbols:
            raise ValueError("daily market data contains an empty symbol")
        keys = date_codes.astype(np.int64) * max(len(symbols), 1) + symbol_codes
        if count and np.unique(keys).size != count:
            ordered = np.sort(keys)
            repeated = np.unique(ordered[1:][ordered[1:] == ordered[:-1]])[:5]
            sample = [
                {"trade_date": dates[int(key) // len(symbols)], "symbol": symbols[int(key) % len(symbols)]}
                for key in repeated
            ]
            raise ValueError(f"daily market data has duplicate business keys: {sample}")
        if "available_at" in columns:
            available_codes, raw = pd.factorize(daily["available_at"], use_na_sentinel=False)
            stamps = [_available_at(value) for value in raw]
        else:
            available_codes = date_codes
            stamps = [_default_available_at(value) for value in dates]
        available_codes = np.asarray(available_codes, dtype=np.int32)
        epoch = np.fromiter((_epoch_us(stamp) for stamp in stamps), dtype=np.int64, count=len(stamps))
        epoch = epoch[available_codes] if count else np.empty(0, dtype=np.int64)
        order = np.lexsort((symbol_codes, date_codes, epoch))

        store: dict[str, _Column] = {}
        for name in columns:
            if name == "trade_date":
                store[name] = _Column("coded", date_codes[order], dates)
            elif name == "symbol":
                store[name] = _Column("coded", symbol_codes[order], symbols)
            elif name == "available_at":
                store[name] = _Column("coded", available_codes[order], [stamp.isoformat() for stamp in stamps])
            else:
                store[name] = _frame_column(daily[name], order)
        if "available_at" not in store:
            store["available_at"] = _Column(
                "coded", available_codes[order], [stamp.isoformat() for stamp in stamps]
            )
        self._chunk = _Chunk(store, count, epoch[order], stamps)
        self._table = BarTable((self._chunk,))
        self.trade_dates: tuple[str, ...] = tuple(dates)
        # Rows of each trading day in store order; a day's rows are contiguous
        # only when its bars share one available_at, so keep explicit indices.
        by_date = date_codes[order]
        day_order = np.argsort(by_date, kind="stable")
        bounds = np.cumsum(np.bincount(by_date, minlength=len(dates)))[:-1]
        self._day_rows: dict[str, np.ndarray] = (
            dict(zip(dates, np.split(day_order, bounds), strict=True)) if count else {}
        )
        self._day_cache: tuple[str, _DayBars] | None = None

    def bars_for_day(self, trade_date: str) -> Mapping[str, Mapping[str, object]]:
        key = str(trade_date)
        cached = self._day_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        rows = self._day_rows.get(key)
        if rows is None:
            return {}
        view = _DayBars(self._chunk, rows)
        self._day_cache = (key, view)
        return view

    def visible_at(self, inference_at: datetime) -> _BarPrefix:
        if inference_at.tzinfo is None or inference_at.utcoffset() is None:
            raise StrategyContractError("inference_at must include a timezone")
        cutoff = _epoch_us(inference_at)
        return self._table.rows(int(np.searchsorted(self._chunk.epoch, cutoff, side="right")))


class _DayBars(Mapping[str, Mapping[str, object]]):
    """Symbol → bar for one trading day; each bar is read from the store on access."""

    __slots__ = ("_chunk", "_index")

    def __init__(self, chunk: _Chunk, rows: np.ndarray) -> None:
        column = chunk.columns["symbol"]
        names = [column.uniques[code] for code in column.values[rows].tolist()]  # type: ignore[index]
        self._chunk = chunk
        self._index: dict[str, int] = dict(zip(names, rows.tolist(), strict=True))

    def __getitem__(self, symbol: str) -> Mapping[str, object]:
        return _Row(self._chunk, self._index[symbol])

    def __iter__(self) -> Iterator[str]:
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)


def _normalized_codes(series: pd.Series, normalize: Callable[[object], str]) -> tuple[np.ndarray, list[str]]:
    """Sorted normalized uniques and each row's code into them."""

    codes, raw = pd.factorize(series, use_na_sentinel=False)
    normalized = np.array([normalize(value) for value in raw], dtype=object)
    if not len(normalized):
        return np.empty(0, dtype=np.int32), []
    uniques, inverse = np.unique(normalized, return_inverse=True)
    return inverse[codes].astype(np.int32), uniques.tolist()


def _frame_column(series: pd.Series, order: np.ndarray) -> _Column:
    """One frame column as a typed store column, decoding to the same strict JSON as before."""

    dtype = series.dtype
    if isinstance(dtype, np.dtype):
        if dtype.kind == "f":
            values = series.to_numpy(dtype=np.float64)[order]
            if np.isinf(values).any():
                raise StrategyContractError("strategy bar must be JSON-compatible")
            return _Column("float", values)
        if dtype.kind == "b":
            return _Column("bool", series.to_numpy(dtype=np.bool_)[order])
        if dtype.kind == "i" or (dtype.kind == "u" and (not len(series) or series.max() <= np.iinfo(np.int64).max)):
            return _Column("int", series.to_numpy().astype(np.int64)[order])
    codes, raw = pd.factorize(series, use_na_sentinel=True)
    uniques = [_json_scalar(value) for value in raw]
    if (codes < 0).any():
        uniques.append(None)
        codes = np.where(codes < 0, len(uniques) - 1, codes)
    return _Column("coded", np.asarray(codes, dtype=np.int32)[order], uniques)


def _json_scalar(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        value = value.astimezone(CN_TZ).isoformat()
    elif hasattr(value, "item"):
        value = _missing_to_none(value.item())
    else:
        value = _missing_to_none(value)
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise StrategyContractError("strategy bar must be JSON-compatible") from exc


def _date_text(value: object) -> str:
    text = str(value).strip()
    normalized = f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) == 8 and text.isdigit() else text
    try:
        return date.fromisoformat(normalized).strftime("%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"invalid trade_date: {value!r}") from exc


def _symbol_text(value: object) -> str:
    return str(value).strip()


def _default_available_at(trade_date: str) -> datetime:
    day = date.fromisoformat(f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:8]}")
    return datetime.combine(day, time(17, 30), tzinfo=CN_TZ)


def _available_at(value: object) -> datetime:
    if isinstance(value, pd.Timestamp):
        parsed = value.to_pydatetime()
    elif isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as exc:
            raise ValueError(f"invalid available_at: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("available_at must include a timezone")
    return parsed.astimezone(CN_TZ)


def _missing_to_none(value: object) -> object:
    """Normalize scalar pandas missing values into strict JSON nulls."""

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return value
    try:
        return None if bool(missing) else value
    except (TypeError, ValueError):
        return value


__all__ = ["DailyMarketData"]
