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

The scripted developer that drives the worker's LLM sessions lives here too:
``NominatingLLM`` replays a script whose ``test`` names the last working-copy
row the session read, ``agent_then`` launches children before a call, and
``NoShellRunner`` refuses every shell command. ``daily_experiment`` is the
smaller trusted-daily experiment the worker-option tests load.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from autotrade.environment.llm import (
    LOCAL_QWEN_MODEL,
    ProviderResponse,
    ScriptedLLM,
    ToolCall,
)
from autotrade.environment.replay.null_control import run_null_control
from autotrade.environment.runtime import chmod_tree
from autotrade.environment.strategy import CN_TZ, StrategySchedule
from autotrade.environment.tools import CommandResult
from autotrade.pipelines.config import SNAPSHOT_CACHE_FORMAT_VERSION, SnapshotBundle
from autotrade.pipelines.pit_backend import prebuild_asof_stash
from tests.unit.acceptance_fixture import REQUIRED_RULES

GENERATION = "synthetic_generation"
# Random-baseline draws per null control in the end-to-end tests, which assert
# nothing about the count; test_null_control pins the production default.
E2E_BASELINE_DRAWS = 8


def few_baseline_draws(*args, **kwargs):
    """``run_null_control`` at ``E2E_BASELINE_DRAWS`` draws whatever count the
    caller names: patched over ``pit_backend.run_null_control``."""

    return run_null_control(*args, **{**kwargs, "k": E2E_BASELINE_DRAWS})


# Two research years: the session researches on Y1 and Y2 is the validation year.
GEOMETRY = {
    "research_start": "20220701",
    "research_end": "20240630",
    "forward_end": "20250630",
    "heldout_end": "20250930",
}
# The release ends inside Held-out.
RELEASE_END = "20250912"
# History starts two months before the research period (20220701).
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
                    # Book-to-price and turnover spreads over the names,
                    # independent of size and of each other, so the slot's
                    # value and turnover legs are real spreads.
                    "pb": (1.0 + (index * 7) % len(SYMBOLS) / 10.0) * close / 10.0,
                    "turnover_rate": 0.01 * (1.0 + (index * 11) % len(SYMBOLS) / 10.0),
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
                columns=[
                    "ts_code", "ex_date", "record_date", "pay_date", "div_listdate",
                    "cash_per_share", "stock_per_share", "bonus_per_share",
                ]
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
            if phase == "valid":
                # The prebuilt as-of state the seed prebuilds for a research slot
                # (scripts/data/prebuild_pit_views_seed.py): a test's replay
                # opens on it, and the backend refuses a slot without one.
                prebuild_asof_stash(
                    snapshot_dir=bundle.decision_ref,
                    replay_dir=replay,
                    schedule=StrategySchedule(),
                    phase=phase,
                    generation_id=GENERATION,
                    start=start,
                    end=end,
                    host_dir=self.cache_root / ".prebuild_host",
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


def make_experiment(tmp_path: Path, **params: object) -> tuple[Path, Path]:
    """A repository with a raw calendar, the strategy and one experiment's params."""

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
    experiment = repo / "experiments" / "experiment"
    (experiment / "hitl").mkdir(parents=True)
    (experiment / "hitl" / "params.json").write_text(
        json.dumps(
            {
                "experiment_id": "experiment",
                "strategy_path": "strategies/main.py",
                "data_backend": "pit",
                "execution_mode": "trusted",
                "developer_mode": "baseline",
                # The local model the tests' VLLM credential serves; a file
                # that leaves the roles out runs on the hosted default.
                "model": LOCAL_QWEN_MODEL,
                "subagent_model": LOCAL_QWEN_MODEL,
                "compact_model": LOCAL_QWEN_MODEL,
                "initial_cash": 1_000_000,
                # The boards creation stamps on this capital
                # (``broker.default_permitted_boards``).
                "permitted_boards": ["main", "gem", "star", "bj"],
                # No tracking_error_cap: a one-name book cannot track CSI 300,
                # and the capital buys no mandate of its own.
                "include_fundamentals": False,
                "include_macro": False,
                "include_events": False,
                "include_text": False,
                "include_intraday": False,
                # No prebuilt view cache: every view is built from the raw
                # calendar, reaching two years back.
                "pit_views_seed": "",
                "window_months": 24,
                **REQUIRED_RULES,
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


def daily_experiment(
    tmp_path: Path, *, developer_mode: str = "baseline"
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    experiment = repo / "experiments" / "smoke"
    hitl = experiment / "hitl"
    hitl.mkdir(parents=True)
    strategy = repo / "strategies" / "main.py"
    strategy.parent.mkdir(parents=True)
    strategy.write_text(
        "def generate_orders(context):\n    return []\n", encoding="utf-8"
    )
    daily = repo / "data" / "daily.parquet"
    daily.parent.mkdir(parents=True)
    # Reaches into the default Held-out, which is all the loader checks.
    days = pd.bdate_range("2026-06-01", "2026-09-30")
    pd.DataFrame(
        {
            "trade_date": [stamp.strftime("%Y%m%d") for stamp in days],
            "symbol": ["000001.SZ"] * len(days),
            "open": [10.0] * len(days),
            "close": [10.0] * len(days),
            "pre_close": [10.0] * len(days),
        }
    ).to_parquet(daily, index=False)
    (hitl / "params.json").write_text(
        json.dumps(
            {
                "experiment_id": "smoke",
                "strategy_path": "strategies/main.py",
                "daily_path": "data/daily.parquet",
                "execution_mode": "trusted",
                "developer_mode": developer_mode,
                # The local model the tests' VLLM credential serves; a file
                # that leaves the roles out runs on the hosted default.
                "model": LOCAL_QWEN_MODEL,
                "subagent_model": LOCAL_QWEN_MODEL,
                "compact_model": LOCAL_QWEN_MODEL,
                "data_backend": "daily",
                "initial_control_mode": "auto",
                "strategy_period": "day",
                "inference_time": "08:30",
                "initial_cash": 100_000,
                **REQUIRED_RULES,
            }
        ),
        encoding="utf-8",
    )
    (hitl / "control.json").write_text(
        json.dumps({"schema_version": 1, "mode": "auto"}), encoding="utf-8"
    )
    return repo, experiment


class NoShellRunner:
    def run(self, argv, *, cwd, timeout_seconds, max_output_chars, input_text=None):
        del argv, cwd, timeout_seconds, max_output_chars, input_text
        return CommandResult(126, stderr="shell is disabled in this test")


# The working copy as a one-candidate validate round: the scripted
# sessions write output/ through a child and validate it as it stands.
VALIDATE_WORKING_COPY = ToolCall(
    "valid",
    "validate",
    {
        "candidates": [
            {
                "name": "working_copy",
                "hypothesis": "the implemented working copy beats its baseline",
                "path": "output",
                "control": False,
            }
        ],
    },
)

# The working copy over the validation steps tested so far, under a name of
# its own: the scripted test still names the full-span working-copy node.
VALIDATE_REVEALED = ToolCall(
    "revealed",
    "validate",
    {
        "span": "revealed",
        "candidates": [
            {
                "name": "revealed_copy",
                "hypothesis": "the working copy read over the steps tested so far",
                "path": "output",
            }
        ],
    },
)


# A scripted test that names the working-copy row it has just read: node ids
# carry the session's opaque run ref, which a script cannot know in advance.
LAST_WORKING_COPY_NODE = "<working copy node>"

TEST_WORKING_COPY = ToolCall("test", "test", {"candidate": LAST_WORKING_COPY_NODE})

# The arguments a scripted call may name the last working-copy node in.
_NODE_ARGUMENTS = ("candidate", "node_id")


class NominatingLLM(ScriptedLLM):
    def complete(self, messages, **kwargs):
        response = super().complete(messages, **kwargs)
        if not any(
            call.arguments.get(key) == LAST_WORKING_COPY_NODE
            for call in response.tool_calls
            for key in _NODE_ARGUMENTS
        ):
            return response
        rows = re.findall(
            # A completed row names its node by handle (node_id where shared).
            r'"name":\s*"working_copy".*?"(?:handle|node_id)":\s*"([^"]+)"',
            "\n".join(message.content or "" for message in messages),
            flags=re.DOTALL,
        )
        return ProviderResponse(
            tool_calls=tuple(
                ToolCall(
                    call.id,
                    call.name,
                    {
                        key: rows[-1] if value == LAST_WORKING_COPY_NODE and key in _NODE_ARGUMENTS else value
                        for key, value in call.arguments.items()
                    },
                )
                for call in response.tool_calls
            )
        )


def agent_then(
    *tool_calls: ToolCall,
    roles: tuple[str, ...] = ("Explore",),
    summary: str = "委托完成",
    implement: dict[str, object] | None = None,
) -> tuple[ProviderResponse, ...]:
    launches = tuple(
        ToolCall(f"ex_{role}", "agent", {"agent": role, "task": f"review {role}"})
        for role in roles
    )
    responses: list[ProviderResponse] = [
        ProviderResponse(tool_calls=(*launches, *tool_calls))
    ]
    for role in roles:
        if implement is not None and role == "general-purpose":
            responses.append(
                ProviderResponse(
                    tool_calls=(
                        ToolCall("w", "write_file", dict(implement)),
                    )
                )
            )
        responses.append(
            ProviderResponse(
                content=summary,
                usage={"prompt_tokens": 50, "completion_tokens": 7, "total_tokens": 57},
            )
        )
    return tuple(responses)
