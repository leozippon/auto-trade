"""A small synthetic PIT release and the provider that serves it to the real worker.

The worker, the PIT evaluation backend, the Timeview span replay, the Broker
and the verdict all run unchanged; only the snapshot provider is replaced,
because building views from a raw Tushare lake is not what these tests are
about. The provider writes decision views and replay slots in the layout and
under the contracts the evaluation backend checks (one cache root,
``decision/<anchor>`` and ``replay/<phase>/<start>_<end>_<anchor>``, a matching
``provider.json`` and raw generation), and logs every request so a test can
assert what a stage asked for.

The market: 30 names (the size factor needs 30 per day), CSI 300 in the
slot's ``macro.parquet``, and one name, ``000001.SZ``, with a steady daily
edge over the market that ``STRATEGY`` trades twice a month.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from autotrade.environment.runtime import chmod_tree
from autotrade.environment.strategy import CN_TZ
from autotrade.pipelines.config import SNAPSHOT_CACHE_FORMAT_VERSION, SnapshotBundle

GENERATION = "synthetic_generation"
GEOMETRY = {
    "research_start": "20230701",
    "research_end": "20240630",
    "forward_end": "20250630",
    "heldout_end": "20250930",
}
# The release ends inside Held-out.
RELEASE_END = "20250912"
# History starts two months before a two-year research period (20220701) so
# an arm may also research Y1..Y2 of the same market.
FIRST_DAY = date(2022, 5, 2)
SYMBOLS = [f"{number:06d}.SZ" for number in range(1, 31)]
EDGE_SYMBOL = SYMBOLS[0]

STRATEGY = f'''SYMBOL = "{EDGE_SYMBOL}"


def generate_orders(context):
    closes = [bar["close"] for bar in context.bars[-40:] if bar["symbol"] == SYMBOL]
    if not closes:
        return []
    held = int(context.account.positions.get(SYMBOL, 0))
    day = context.inference_at
    at_open = day.replace(hour=9, minute=30).isoformat()
    flat = day.day <= 4 or 16 <= day.day <= 19
    if held and flat:
        return [{{"symbol": SYMBOL, "action": "sell", "quantity": held, "execute_at": at_open}}]
    if not held and not flat:
        quantity = int(context.account.cash * 0.9 / (float(closes[-1]) * 1.12) / 100) * 100
        if quantity > 0:
            return [{{"symbol": SYMBOL, "action": "buy", "quantity": quantity, "execute_at": at_open}}]
    return []
'''


def trading_days() -> list[str]:
    days = []
    current = FIRST_DAY
    end = date.fromisoformat(f"{RELEASE_END[:4]}-{RELEASE_END[4:6]}-{RELEASE_END[6:]}")
    while current <= end:
        if current.weekday() < 5:
            days.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return days


DAYS = trading_days()


def _market() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20230701)
    count = len(DAYS)
    market = rng.normal(0.0003, 0.008, count)
    size = rng.normal(0.0, 0.003, count)
    rows = []
    for index, symbol in enumerate(SYMBOLS):
        exposure = 1.0 if index < 9 else -1.0 if index >= 21 else 0.0
        edge = 0.0025 if symbol == EDGE_SYMBOL else 0.0
        noise = rng.normal(0.0, 0.008, count)
        returns = market + 0.5 * exposure * size + edge + noise
        close = 10.0
        base_cap = 1e5 * (index + 1)
        for day, value in zip(DAYS, returns):
            pre_close = close
            close = round(pre_close * (1.0 + float(value)), 2)
            open_ = round(pre_close * (1.0 + 0.3 * float(value)), 2)
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": symbol,
                    "open": open_,
                    "close": close,
                    "pre_close": pre_close,
                    "pct_chg": round(close / pre_close - 1.0, 6),
                    "circ_mv": base_cap * close / 10.0,
                    "up_limit": round(pre_close * 1.1, 2),
                    "down_limit": round(pre_close * 0.9, 2),
                    "available_at": f"{day[:4]}-{day[4:6]}-{day[6:]}T17:30:00+08:00",
                }
            )
    daily = pd.DataFrame(rows)
    macro = pd.DataFrame(
        {
            "dataset": "index_daily",
            "ts_code": "000300.SH",
            "trade_date": DAYS,
            "pct_chg": market * 100.0,
            "available_at": [f"{day[:4]}-{day[4:6]}-{day[6:]}T17:00:00+08:00" for day in DAYS],
        }
    )
    return daily, macro


DAILY, MACRO = _market()


class SyntheticPITProvider:
    """``ResearchPITSnapshotProvider``'s interface over the synthetic market."""

    requests: ClassVar[list[tuple[str, str, str, datetime]]] = []

    def __init__(self, *, experiment_dir, config, cache_root=None, **_ignored) -> None:
        self.config = config
        self.cache_root = Path(cache_root) if cache_root is not None else Path(experiment_dir) / "pit_views"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.trading_days = list(DAYS)
        (self.cache_root / "provider.json").write_text(
            json.dumps(
                {
                    "schema_version": SNAPSHOT_CACHE_FORMAT_VERSION,
                    "generation_id": GENERATION,
                    "release_raw_dir": "synthetic",
                    "snapshot_config": config.to_record(),
                }
            ),
            encoding="utf-8",
        )

    def prepare(self, *, phase, start, end, decision_time) -> SnapshotBundle:
        type(self).requests.append((phase, start, end, decision_time))
        bundle = self.prepare_decision(decision_time=decision_time, _log=False)
        key = decision_time.strftime("%Y%m%dT%H%M%S%z")
        replay = self.cache_root / "replay" / phase / f"{start}_{end}_{key}"
        if not replay.exists():
            replay.mkdir(parents=True)
            days = DAILY["trade_date"].between(start, end)
            DAILY[days].to_parquet(replay / "daily.parquet", index=False)
            MACRO[MACRO["trade_date"].between(start, end)].to_parquet(replay / "macro.parquet", index=False)
            pd.DataFrame(
                columns=["ts_code", "ex_date", "record_date", "pay_date", "div_listdate", "cash_per_share", "stock_per_share"]
            ).to_parquet(replay / "corporate_actions.parquet", index=False)
            _manifest(
                replay,
                {
                    "snapshot_id": f"replay_{phase}_{start}_{end}",
                    "kind": "replay_slot",
                    "label": phase,
                    "period_start": start,
                    "period_end": end,
                    "available_from": decision_time.isoformat(),
                    "domains": {"corporate_actions": {"rows": 0}},
                },
            )
        return SnapshotBundle(
            snapshot_id=bundle.snapshot_id,
            decision_ref=bundle.decision_ref,
            replay_ref=str(replay),
            generation_id=GENERATION,
        )

    def prepare_decision(self, *, decision_time, _log: bool = True) -> SnapshotBundle:
        if _log:
            type(self).requests.append(("decision", "", "", decision_time))
        key = decision_time.strftime("%Y%m%dT%H%M%S%z")
        decision = self.cache_root / "decision" / key
        if not decision.exists():
            decision.mkdir(parents=True)
            cutoff = decision_time.strftime("%Y%m%d")
            history = DAILY[DAILY["trade_date"] <= cutoff]
            history[history["trade_date"].isin(sorted(set(history["trade_date"]))[-20:])].drop(
                columns="available_at"
            ).to_parquet(decision / "daily.parquet", index=False)
            pd.DataFrame({"ts_code": SYMBOLS}).to_parquet(decision / "universe.parquet", index=False)
            _manifest(
                decision,
                {
                    "snapshot_id": f"decision_{key}",
                    "kind": "decision_input",
                    "decision_time": decision_time.isoformat(),
                },
            )
        return SnapshotBundle(
            snapshot_id=f"decision_{key}",
            decision_ref=str(decision),
            replay_ref="",
            generation_id=GENERATION,
        )


def _manifest(directory: Path, record: dict[str, object]) -> None:
    (directory / "manifest.json").write_text(
        json.dumps({**record, "raw_generation": {"generation_id": GENERATION}}),
        encoding="utf-8",
    )
    chmod_tree(directory, file_mode=0o444, dir_mode=0o555)


def make_arm(tmp_path: Path, **params: object) -> tuple[Path, Path]:
    """A repository with a raw calendar, the strategy and one arm's params."""

    repo = tmp_path / "repo"
    raw = repo / "data" / "raw"
    for dataset in ("daily", "daily_basic", "adj_factor", "stk_limit", "suspend_d"):
        target = raw / dataset / f"trade_date={DAYS[0]}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": [DAYS[0]], "ts_code": [EDGE_SYMBOL]}).to_parquet(target, index=False)
    calendar = raw / "trade_cal" / "exchange=SSE" / "year=2023.parquet"
    calendar.parent.mkdir(parents=True)
    pd.DataFrame({"cal_date": DAYS, "is_open": ["1"] * len(DAYS)}).to_parquet(calendar, index=False)
    (repo / "data" / "pit" / "fundamental_events").mkdir(parents=True)
    status = repo / "results" / "data_quality" / "fundamental_events_status.json"
    status.parent.mkdir(parents=True)
    status.write_text("{}", encoding="utf-8")
    strategy = repo / "strategies" / "main.py"
    strategy.parent.mkdir(parents=True)
    strategy.write_text(STRATEGY, encoding="utf-8")
    experiment = repo / "experiments" / "arm"
    (experiment / "hitl").mkdir(parents=True)
    (experiment / "hitl" / "params.json").write_text(
        json.dumps(
            {
                "experiment_id": "arm",
                "strategy_path": "strategies/main.py",
                "data_backend": "pit",
                "execution_mode": "trusted",
                "developer_mode": "baseline",
                "initial_cash": 1_000_000,
                # A one-name book cannot track CSI 300, and this capital would
                # default to the tracking mandate: the arm switches it off.
                "tracking_error_cap": 0,
                "include_fundamentals": False,
                "include_macro": False,
                "include_events": False,
                "include_text": False,
                "include_intraday": False,
                **GEOMETRY,
                **params,
            }
        ),
        encoding="utf-8",
    )
    (experiment / "hitl" / "control.json").write_text(
        json.dumps({"schema_version": 1, "mode": "auto"}), encoding="utf-8"
    )
    return repo, experiment


def decision_anchor(day: str) -> datetime:
    return datetime(int(day[:4]), int(day[4:6]), int(day[6:]), 23, 59, 59, tzinfo=CN_TZ)
