"""Strategy scheduling, point-in-time context, and JSON order contract.

This module is the single source of truth shared by strategy authors, replay,
and scheduled execution.  Strategies receive an immutable context and return plain
JSON-compatible order objects; they never receive a Broker or an environment
tool surface.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from itertools import chain
from types import MappingProxyType
from typing import Literal, Protocol
from zoneinfo import ZoneInfo

import numpy as np

CN_TZ = ZoneInfo("Asia/Shanghai")
StrategyPeriod = Literal["day", "month", "quarter", "year"]
OrderAction = Literal["buy", "sell"]
_PERIODS = frozenset({"day", "month", "quarter", "year"})
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_ABSENT = object()


class StrategyContractError(ValueError):
    """The schedule, context, or an Agent-produced order is invalid."""


def _epoch_us(value: datetime) -> int:
    """Exact integer microseconds since the epoch of a timezone-aware datetime."""

    delta = value - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


class _Column:
    """One chunk column: a typed array plus the decoding back to strict JSON.

    ``float`` holds float64 with NaN encoding null, ``int`` int64, ``bool``
    bool, and ``coded`` int32 codes into ``uniques`` (any strict JSON value,
    stored once per chunk). ``present`` marks the rows that carry the key at
    all and is None when every row does.
    """

    __slots__ = ("kind", "present", "uniques", "values")

    def __init__(
        self,
        kind: str,
        values: np.ndarray,
        uniques: list[object] | None = None,
        present: np.ndarray | None = None,
    ) -> None:
        self.kind = kind
        self.values = values
        self.uniques = uniques
        self.present = present

    @classmethod
    def from_values(cls, values: list[object], present: np.ndarray | None = None) -> _Column:
        kinds = {type(item) for item in values}
        if kinds == {float}:
            return cls("float", np.array(values, dtype=np.float64), present=present)
        if kinds == {float, type(None)}:
            filled = [math.nan if item is None else item for item in values]
            return cls("float", np.array(filled, dtype=np.float64), present=present)
        if kinds == {int}:
            try:
                return cls("int", np.array(values, dtype=np.int64), present=present)
            except OverflowError:
                pass
        if kinds == {bool}:
            return cls("bool", np.array(values, dtype=np.bool_), present=present)
        uniques: list[object] = []
        lookup: dict[object, int] = {}
        codes = np.empty(len(values), dtype=np.int32)
        for index, item in enumerate(values):
            # Strings dominate and are their own key; everything else is keyed
            # by its strict-JSON identity so 1, 1.0 and True stay distinct.
            key = item if type(item) is str else _strict_json_identity(item)
            code = lookup.get(key)
            if code is None:
                code = lookup[key] = len(uniques)
                uniques.append(item)
            codes[index] = code
        return cls("coded", codes, uniques, present)

    def value(self, offset: int) -> object:
        item = self.values[offset]
        if self.kind == "coded":
            return self.uniques[item]  # type: ignore[index]
        if self.kind == "float":
            return None if math.isnan(item) else float(item)
        return item.item()


class _Chunk:
    """A contiguous block of bars: columns, per-row availability, row cache."""

    __slots__ = ("columns", "epoch", "length", "rows", "sparse", "stamps")

    def __init__(
        self,
        columns: dict[str, _Column],
        length: int,
        epoch: np.ndarray,
        stamps: list[datetime],
    ) -> None:
        self.columns = columns
        self.length = length
        # available_at per row as epoch microseconds, and the parsed datetime
        # per unique of the coded available_at column.
        self.epoch = epoch
        self.stamps = stamps
        self.sparse = any(column.present is not None for column in columns.values())
        self.rows: list[_Row] = []

    def available_at(self, offset: int) -> datetime:
        return self.stamps[self.columns["available_at"].values[offset]]

    def record(self, offset: int) -> dict[str, object]:
        record: dict[str, object] = {}
        for key, column in self.columns.items():
            if column.present is None or column.present[offset]:
                record[key] = column.value(offset)
        return record


class _Row(Mapping[str, object]):
    """One bar read lazily from its chunk; immutable and strict JSON."""

    __slots__ = ("_chunk", "_offset")

    def __init__(self, chunk: _Chunk, offset: int) -> None:
        self._chunk = chunk
        self._offset = offset

    def __getitem__(self, key: str) -> object:
        column = self._chunk.columns[key]
        if column.present is not None and not column.present[self._offset]:
            raise KeyError(key)
        return column.value(self._offset)

    def __iter__(self) -> Iterator[str]:
        offset = self._offset
        for key, column in self._chunk.columns.items():
            if column.present is None or column.present[offset]:
                yield key

    def __len__(self) -> int:
        if not self._chunk.sparse:
            return len(self._chunk.columns)
        return sum(1 for _ in self)

    def __repr__(self) -> str:
        return repr(dict(self))


class _BarPrefix(tuple[Mapping[str, object], ...]):
    """The first ``len(self)`` rows of one table, in order; made only by ``BarTable.rows``."""

    _table: BarTable

    def __new__(cls, table: BarTable, rows: Iterable[_Row]):
        instance = super().__new__(cls, rows)
        instance._table = table
        return instance


class BarTable:
    """Append-only columnar store of strict-JSON bars in arrival order.

    Every bar is one row; rows are materialized as read-only mappings only
    when read, so a four-year cross-section costs a few hundred bytes per row
    instead of a dict tree. A table never changes: ``extended`` returns a new
    table sharing this one's chunks, so a rejected delta leaves the accepted
    prefix untouched and rows already handed out stay valid.
    """

    __slots__ = ("_chunks", "_length", "_monotonic", "_starts")

    def __init__(self, chunks: Sequence[_Chunk] = (), *, monotonic: bool = True) -> None:
        self._chunks = tuple(chunks)
        starts: list[int] = []
        total = 0
        for chunk in self._chunks:
            starts.append(total)
            total += chunk.length
        self._starts = starts
        self._length = total
        self._monotonic = monotonic

    def __len__(self) -> int:
        return self._length

    def extended(
        self,
        records: Iterable[object],
        *,
        inference_at: datetime | None = None,
        decoded_json: bool = False,
        monotonic: bool = False,
    ) -> BarTable:
        """A new table with ``records`` validated and appended after this one's rows.

        ``inference_at`` rejects any bar available later than the decision;
        ``monotonic`` additionally requires ``available_at`` never to decrease,
        continuing from this table's last row.
        """

        if monotonic and not self._monotonic:
            raise StrategyContractError("bar available_at must be monotonic")
        after = self.max_available_at(self._length) if monotonic else None
        chunk, ordered = _chunk_from_records(
            records, inference_at=inference_at, decoded_json=decoded_json, after=after
        )
        if chunk.length == 0:
            return self
        still_monotonic = self._monotonic and ordered
        if still_monotonic and self._length:
            still_monotonic = _epoch_us(self.max_available_at(self._length)) <= int(chunk.epoch[0])  # type: ignore[arg-type]
        return BarTable((*self._chunks, chunk), monotonic=still_monotonic)

    def _locate(self, index: int) -> tuple[_Chunk, int]:
        if not 0 <= index < self._length:
            raise IndexError(index)
        position = bisect_right(self._starts, index) - 1
        return self._chunks[position], index - self._starts[position]

    def _spans(self, count: int) -> Iterator[tuple[_Chunk, int, int]]:
        """(chunk, start row of the chunk, rows of it inside the first ``count``)."""

        if not 0 <= count <= self._length:
            raise IndexError(count)
        for chunk, start in zip(self._chunks, self._starts, strict=True):
            if start >= count:
                return
            yield chunk, start, min(chunk.length, count - start)

    def row(self, index: int) -> _Row:
        chunk, offset = self._locate(index)
        return _Row(chunk, offset)

    def rows(self, count: int) -> _BarPrefix:
        """The first ``count`` rows as a tuple; row objects are shared across calls."""

        parts: list[Sequence[_Row]] = []
        for chunk, _start, used in self._spans(count):
            cached = chunk.rows
            while len(cached) < used:
                cached.append(_Row(chunk, len(cached)))
            parts.append(cached if used == len(cached) else cached[:used])
        return _BarPrefix(self, chain.from_iterable(parts))

    def record(self, index: int) -> dict[str, object]:
        chunk, offset = self._locate(index)
        return chunk.record(offset)

    def available_at(self, index: int) -> datetime:
        chunk, offset = self._locate(index)
        return chunk.available_at(offset)

    def max_available_at(self, count: int) -> datetime | None:
        if count == 0:
            return None
        if self._monotonic:
            return self.available_at(count - 1)
        latest: datetime | None = None
        for chunk, _start, used in self._spans(count):
            stamp = chunk.available_at(int(chunk.epoch[:used].argmax()))
            if latest is None or stamp > latest:
                latest = stamp
        return latest

    def history(self, symbol: object, count: int) -> list[int]:
        """Row indices below ``count`` whose ``symbol`` text equals ``str(symbol)``."""

        target = str(symbol)
        found: list[int] = []
        for chunk, start, used in self._spans(count):
            column = chunk.columns.get("symbol")
            if column is None:
                if target == str(None):
                    found.extend(range(start, start + used))
                continue
            if column.kind == "coded":
                codes = [code for code, item in enumerate(column.uniques) if str(item) == target]  # type: ignore[arg-type]
                mask = np.isin(column.values[:used], codes)
                if column.present is not None:
                    absent = ~column.present[:used]
                    mask &= ~absent
                    if target == str(None):
                        mask |= absent
                found.extend((np.flatnonzero(mask) + start).tolist())
                continue
            for offset in range(used):
                if column.present is not None and not column.present[offset]:
                    item = None
                else:
                    item = column.value(offset)
                if str(item) == target:
                    found.append(start + offset)
        return found

    def prefix_matches(self, other: BarTable, count: int) -> bool:
        """True when both tables hold the same strict-JSON bars in their first ``count`` rows."""

        if count > self._length or count > other._length:
            return False
        if count == 0 or self is other:
            return True
        mine = list(self._spans(count))
        theirs = list(other._spans(count))
        if len(mine) == len(theirs) and all(
            a[0] is b[0] and a[2] == b[2] for a, b in zip(mine, theirs, strict=True)
        ):
            return True
        return all(
            _strict_json_identity(self.record(index)) == _strict_json_identity(other.record(index))
            for index in range(count)
        )


def _chunk_from_records(
    records: Iterable[object],
    *,
    inference_at: datetime | None,
    decoded_json: bool,
    after: datetime | None,
) -> tuple[_Chunk, bool]:
    """Validate bars once and lay them out column by column.

    Returns the chunk and whether its ``available_at`` never decreases. With
    ``after`` set, any decrease (including below ``after``) is an error.
    """

    rows: list[dict[str, object]] = []
    for value in records:
        if decoded_json:
            if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
                raise StrategyContractError("strategy bar must be a JSON object")
            rows.append(value)
        else:
            if not isinstance(value, Mapping):
                raise StrategyContractError("each strategy bar must be a JSON object")
            rows.append(_json_mapping(dict(value), name="strategy bar"))
    count = len(rows)
    parsed: dict[str, tuple[datetime, int, int]] = {}
    stamps: list[datetime] = []
    uniques: list[object] = []
    codes = np.empty(count, dtype=np.int32)
    epoch = np.empty(count, dtype=np.int64)
    ordered = True
    previous = after
    for index, record in enumerate(rows):
        raw = record.get("available_at")
        entry = parsed.get(raw) if isinstance(raw, str) else None
        if entry is None:
            stamp = _parse_datetime(raw, "bar available_at")
            entry = (stamp, _epoch_us(stamp), len(stamps))
            parsed[raw] = entry  # type: ignore[index]
            stamps.append(stamp)
            uniques.append(raw)
        stamp, micro, code = entry
        if inference_at is not None and stamp > inference_at:
            raise StrategyContractError("strategy context contains data not visible at inference time")
        if previous is not None and stamp < previous:
            if after is not None:
                raise StrategyContractError("bar available_at must be monotonic")
            ordered = False
        previous = stamp
        codes[index] = code
        epoch[index] = micro
    columns: dict[str, _Column] = {}
    keys: dict[str, None] = {}
    for record in rows:
        if len(record) != len(keys) or any(key not in keys for key in record):
            keys.update(dict.fromkeys(record))
    for key in keys:
        if key == "available_at":
            columns[key] = _Column("coded", codes, uniques)
            continue
        values = [record.get(key, _ABSENT) for record in rows]
        present: np.ndarray | None = None
        if any(item is _ABSENT for item in values):
            present = np.fromiter((item is not _ABSENT for item in values), dtype=np.bool_, count=count)
            values = [None if item is _ABSENT else item for item in values]
        columns[key] = _Column.from_values(values, present)
    return _Chunk(columns, count, epoch, stamps), ordered


def _parse_time(value: str | time) -> time:
    if isinstance(value, time):
        if value.tzinfo is not None:
            raise StrategyContractError("inference_time must be a local HH:MM without a timezone")
        return value.replace(second=0, microsecond=0)
    text = str(value)
    try:
        parsed = time.fromisoformat(text)
    except ValueError as exc:
        raise StrategyContractError("inference_time must use 24-hour HH:MM") from exc
    if len(text) != 5 or text[2] != ":" or parsed.second or parsed.microsecond:
        raise StrategyContractError("inference_time must use 24-hour HH:MM")
    return parsed


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value)
    normalized = f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) == 8 and text.isdigit() else text
    try:
        return date.fromisoformat(normalized)
    except ValueError as exc:
        raise StrategyContractError(f"invalid trade date: {value!r}") from exc


@dataclass(frozen=True)
class StrategySchedule:
    """User-selected inference cadence and local China-market time."""

    period: StrategyPeriod = "day"
    inference_time: str | time = "08:30"

    def __post_init__(self) -> None:
        if self.period not in _PERIODS:
            raise StrategyContractError(f"period must be one of {sorted(_PERIODS)}")
        parsed = _parse_time(self.inference_time)
        object.__setattr__(self, "inference_time", parsed.strftime("%H:%M"))

    def at(self, trade_date: str | date | datetime) -> datetime:
        return datetime.combine(_as_date(trade_date), _parse_time(self.inference_time), tzinfo=CN_TZ)

    def is_due(
        self,
        trade_date: str | date | datetime,
        previous_trade_date: str | date | datetime | None,
    ) -> bool:
        """Run on each day or the first available trading day of a new period."""

        if previous_trade_date is None:
            return True
        return period_changed(self.period, trade_date, previous_trade_date)

    def to_record(self) -> dict[str, str]:
        return {"period": self.period, "inference_time": str(self.inference_time)}


def period_changed(
    period: StrategyPeriod,
    trade_date: str | date | datetime,
    previous_trade_date: str | date | datetime,
) -> bool:
    """True when ``trade_date`` falls in a later ``period`` than ``previous_trade_date``."""

    current = _as_date(trade_date)
    previous = _as_date(previous_trade_date)
    if period == "day":
        return current != previous
    if period == "month":
        return (current.year, current.month) != (previous.year, previous.month)
    if period == "quarter":
        return (current.year, (current.month - 1) // 3) != (
            previous.year,
            (previous.month - 1) // 3,
        )
    return current.year != previous.year


@dataclass(frozen=True)
class FitSchedule:
    """When a replay invokes the strategy's optional ``fit(context)``.

    ``fit`` always runs at the first decision of a replay window. A declared
    ``refit_period`` (``REFIT_PERIOD`` in ``main.py``) additionally re-runs it at
    the first decision that falls in a new day/month/quarter/year relative to
    the last fit; ``None`` fits once per replay.
    """

    refit_period: StrategyPeriod | None = None

    def __post_init__(self) -> None:
        if self.refit_period is not None and self.refit_period not in _PERIODS:
            raise StrategyContractError(f"refit_period must be one of {sorted(_PERIODS)} or None")

    def is_due(
        self,
        trade_date: str | date | datetime,
        last_fit_date: str | date | datetime | None,
    ) -> bool:
        if last_fit_date is None:
            return True
        if self.refit_period is None:
            return False
        return period_changed(self.refit_period, trade_date, last_fit_date)

    def to_record(self) -> dict[str, object]:
        return {"refit_period": self.refit_period}


@dataclass(frozen=True)
class StrategyOrder:
    symbol: str
    action: OrderAction
    quantity: int
    execute_at: datetime
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_record(cls, value: object, *, inference_at: datetime) -> StrategyOrder:
        if not isinstance(value, Mapping):
            raise StrategyContractError("each order must be a JSON object")
        required = {"symbol", "action", "quantity", "execute_at"}
        missing = sorted(required.difference(value))
        if missing:
            raise StrategyContractError(f"order missing required fields: {missing}")
        symbol = str(value["symbol"]).strip()
        if not symbol:
            raise StrategyContractError("order symbol must be non-empty")
        action = value["action"]
        if action not in ("buy", "sell"):
            raise StrategyContractError("order action must be buy or sell")
        quantity = value["quantity"]
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise StrategyContractError("order quantity must be a positive integer")
        execute_at = _parse_datetime(value["execute_at"])
        normalized_inference = _require_cn_datetime(inference_at, "inference_at")
        if execute_at < normalized_inference:
            raise StrategyContractError("order execute_at cannot be earlier than the PIT inference time")
        metadata = {str(key): item for key, item in value.items() if key not in required}
        return cls(
            symbol=symbol,
            action=action,
            quantity=quantity,
            execute_at=execute_at,
            metadata=MappingProxyType(metadata),
        )

    def to_record(self) -> dict[str, object]:
        return {
            **self.metadata,
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "execute_at": self.execute_at.isoformat(),
        }


def _parse_datetime(value: object, name: str = "order execute_at") -> datetime:
    if not isinstance(value, str):
        raise StrategyContractError(f"{name} must be an ISO-8601 string with timezone")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise StrategyContractError(f"{name} must be an ISO-8601 string with timezone") from exc
    return _require_cn_datetime(parsed, name)


def _require_cn_datetime(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise StrategyContractError(f"{name} must include a timezone")
    normalized = value.astimezone(CN_TZ)
    if normalized.utcoffset() != CN_TZ.utcoffset(normalized):
        raise StrategyContractError(f"{name} must identify a valid point in time")
    return normalized


def validate_order_payload(payload: object, *, inference_at: datetime) -> tuple[StrategyOrder, ...]:
    """Round-trip and validate the complete Agent-to-environment JSON payload."""

    try:
        normalized = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise StrategyContractError("strategy output must be JSON-compatible") from exc
    if not isinstance(normalized, list):
        raise StrategyContractError("strategy output must be a JSON array of orders")
    return tuple(StrategyOrder.from_record(item, inference_at=inference_at) for item in normalized)


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    positions: Mapping[str, int]

    def __post_init__(self) -> None:
        if isinstance(self.cash, bool) or not isinstance(self.cash, (int, float)):
            raise StrategyContractError("account cash must be a non-negative finite number")
        if not math.isfinite(self.cash) or self.cash < 0:
            raise StrategyContractError("account cash must be a non-negative finite number")
        positions: dict[str, int] = {}
        for symbol, quantity in self.positions.items():
            if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
                raise StrategyContractError("position quantities must be non-negative integers")
            normalized_symbol = str(symbol).strip()
            if not normalized_symbol:
                raise StrategyContractError("position symbols must be non-empty strings")
            positions[normalized_symbol] = quantity
        object.__setattr__(self, "cash", float(self.cash))
        object.__setattr__(self, "positions", MappingProxyType(positions))

    def to_record(self) -> dict[str, object]:
        return {"cash": self.cash, "positions": dict(self.positions)}

    @classmethod
    def from_record(cls, value: object) -> AccountSnapshot:
        record = _strict_record(value, required={"cash", "positions"}, name="account")
        positions = record["positions"]
        if not isinstance(positions, Mapping):
            raise StrategyContractError("account positions must be a JSON object")
        return cls(cash=record["cash"], positions=positions)  # type: ignore[arg-type]


class NLQuery(Protocol):
    def __call__(
        self,
        request: Mapping[str, object],
        *,
        inference_at: datetime,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class StrategyContext:
    """Immutable PIT input with one narrow host-mediated NL capability."""

    inference_at: datetime
    bars: tuple[Mapping[str, object], ...]
    account: AccountSnapshot
    snapshot_dir: str = ""
    asof_dir: str = ""
    asof_version: str = "0"
    # Per-replay fitted state written by ``fit`` and read-only afterwards, and
    # the frozen Agent-authored ``models/`` assets; both empty when absent.
    state_dir: str = ""
    models_dir: str = ""
    _nl_query: NLQuery | None = field(default=None, repr=False, compare=False)
    # The columnar store behind ``bars``; ``bars`` is its first len(bars) rows.
    _bars_table: BarTable = field(default_factory=BarTable, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        inference_at = _require_cn_datetime(self.inference_at, "inference_at")
        bars = self.bars
        if isinstance(bars, _BarPrefix):
            table = bars._table
            latest = table.max_available_at(len(bars))
            if latest is not None and latest > inference_at:
                raise StrategyContractError("strategy context contains data not visible at inference time")
        else:
            table = BarTable().extended(bars, inference_at=inference_at)
            bars = table.rows(len(table))
        object.__setattr__(self, "inference_at", inference_at)
        # A plain tuple: the strategy gets rows, never the store behind them.
        object.__setattr__(self, "bars", tuple(bars))
        object.__setattr__(self, "_bars_table", table)
        for name in ("snapshot_dir", "asof_dir", "asof_version", "state_dir", "models_dir"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise StrategyContractError(f"{name} must be a string")
            if "\x00" in value:
                raise StrategyContractError(f"{name} cannot contain NUL")

    def history(self, symbol: str) -> tuple[Mapping[str, object], ...]:
        bars = self.bars
        return tuple(bars[index] for index in self._bars_table.history(symbol, len(bars)))

    def latest(self, symbol: str) -> Mapping[str, object] | None:
        rows = self.history(symbol)
        return rows[-1] if rows else None

    def nl(self, **request: object) -> Mapping[str, object]:
        """Make one JSON-only NL request through the trusted host service."""

        if self._nl_query is None:
            raise StrategyContractError("NL service is not configured for this strategy context")
        normalized = _json_mapping(request, name="NL request")
        _validate_pit_tree(normalized, inference_at=self.inference_at, name="NL request")
        response = self._nl_query(normalized, inference_at=self.inference_at)
        normalized_response = _json_mapping(response, name="NL response")
        _validate_pit_tree(normalized_response, inference_at=self.inference_at, name="NL response")
        return normalized_response

    def to_record(self) -> dict[str, object]:
        record = {
            "inference_at": self.inference_at.isoformat(),
            "bars": [dict(row) for row in self.bars],
            "account": self.account.to_record(),
            "snapshot_dir": self.snapshot_dir,
            "asof_dir": self.asof_dir,
            "asof_version": self.asof_version,
            "state_dir": self.state_dir,
            "models_dir": self.models_dir,
        }
        return _json_mapping(record, name="strategy context")

    @classmethod
    def from_record(
        cls,
        value: object,
        *,
        nl_query: NLQuery | None = None,
    ) -> StrategyContext:
        record = _strict_record(
            value,
            required={
                "inference_at",
                "bars",
                "account",
                "snapshot_dir",
                "asof_dir",
                "asof_version",
                "state_dir",
                "models_dir",
            },
            name="strategy context",
        )
        bars = record["bars"]
        if not isinstance(bars, list):
            raise StrategyContractError("strategy context bars must be a JSON array")
        return cls(
            inference_at=_parse_datetime(record["inference_at"], "inference_at"),
            bars=tuple(bars),  # type: ignore[arg-type]
            account=AccountSnapshot.from_record(record["account"]),
            snapshot_dir=record["snapshot_dir"],  # type: ignore[arg-type]
            asof_dir=record["asof_dir"],  # type: ignore[arg-type]
            asof_version=record["asof_version"],  # type: ignore[arg-type]
            state_dir=record["state_dir"],  # type: ignore[arg-type]
            models_dir=record["models_dir"],  # type: ignore[arg-type]
            _nl_query=nl_query,
        )


def _strict_record(value: object, *, required: set[str], name: str) -> dict[str, object]:
    record = _json_mapping(value, name=name)
    keys = set(record)
    missing = sorted(required - keys)
    extra = sorted(keys - required)
    if missing:
        raise StrategyContractError(f"{name} missing required fields: {missing}")
    if extra:
        raise StrategyContractError(f"{name} contains unsupported fields: {extra}")
    return record


def _json_mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise StrategyContractError(f"{name} must be a JSON object")
    try:
        normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise StrategyContractError(f"{name} must be JSON-compatible") from exc
    if not isinstance(normalized, dict):
        raise StrategyContractError(f"{name} must be a JSON object")
    return normalized


def _strict_json_identity(value: object) -> tuple[object, ...]:
    """Hashable JSON tree identity that preserves bool/int/float distinctions."""

    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, list):
        return ("list", tuple(_strict_json_identity(item) for item in value))
    if isinstance(value, dict):
        return (
            "object",
            tuple((key, _strict_json_identity(item)) for key, item in sorted(value.items())),
        )
    raise StrategyContractError("strategy bar must contain only strict JSON values")


def _validate_pit_tree(value: object, *, inference_at: datetime, name: str) -> None:
    """Reject explicit availability timestamps later than the current PIT cut."""

    if isinstance(value, Mapping):
        available_at = value.get("available_at")
        if (
            available_at is not None
            and _parse_datetime(available_at, f"{name} available_at") > inference_at
        ):
            raise StrategyContractError(f"{name} contains data not visible at inference time")
        for child in value.values():
            _validate_pit_tree(child, inference_at=inference_at, name=name)
    elif isinstance(value, list):
        for child in value:
            _validate_pit_tree(child, inference_at=inference_at, name=name)


StrategyFunction = Callable[[StrategyContext], Sequence[Mapping[str, object]]]


def run_strategy(strategy: StrategyFunction, context: StrategyContext) -> tuple[StrategyOrder, ...]:
    return validate_order_payload(strategy(context), inference_at=context.inference_at)
