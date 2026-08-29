#!/usr/bin/env python3
"""Advance the local ADM-Cube Paper account by one committed trading day."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.llm import MODEL_CHOICES, build_model_gateway
from autotrade.environment.nl import NLConfig
from autotrade.environment.sandbox import DEFAULT_IMAGE, SandboxConfig, SandboxLimits
from autotrade.environment.strategy import StrategySchedule
from autotrade.paper import DailyPaperEngine
from autotrade.pipelines import PaperPITData, ResearchPITSnapshotProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument(
        "--models-dir",
        type=Path,
        help="The activated revision's models/ tree, if it has one.",
    )
    parser.add_argument(
        "--strategy-revision",
        required=True,
        help="Operator-assigned immutable revision ID.",
    )
    parser.add_argument("--data-backend", choices=("pit", "daily"), default="pit")
    parser.add_argument(
        "--daily-path",
        type=Path,
        help="Required only for the compatibility daily backend.",
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--fundamental-events-root",
        type=Path,
        default=Path("data/pit/fundamental_events"),
    )
    parser.add_argument(
        "--fundamental-events-status",
        type=Path,
        default=Path("results/data_quality/fundamental_events_status.json"),
    )
    parser.add_argument("--pit-cache-root", type=Path)
    parser.add_argument("--disable-historical-intraday", action="store_true")
    parser.add_argument("--max-intraday-row-group-rows", type=int, default=2_000_000)
    parser.add_argument(
        "--trade-date",
        required=True,
        help="One local YYYYMMDD trading day; no realtime polling.",
    )
    parser.add_argument("--state-root", type=Path, default=Path("data/trading/paper"))
    parser.add_argument(
        "--strategy-period", choices=("day", "month", "quarter", "year"), default="day"
    )
    parser.add_argument("--inference-time", default="08:30")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--sandbox-image", default=DEFAULT_IMAGE)
    parser.add_argument("--sandbox-cpus", type=float, default=SandboxLimits().cpus)
    parser.add_argument("--sandbox-memory", default=SandboxLimits().memory)
    parser.add_argument("--sandbox-pids", type=int, default=64)
    parser.add_argument("--decision-timeout-seconds", type=float, default=30.0)
    parser.add_argument(
        "--nl-model",
        choices=("", *MODEL_CHOICES),
        default=MODEL_CHOICES[0],
        help="Set to enable evidence-grounded NL answers; empty keeps local search only.",
    )
    parser.add_argument(
        "--nl-api-key-env",
        default=None,
        help="Optional override of the DeepSeek credential variable; unset resolves the key from the NL model's profile.",
    )
    parser.add_argument("--nl-env-file", type=Path, default=Path(".env"))
    parser.add_argument("--nl-max-results", type=int, default=8)
    parser.add_argument("--nl-max-calls-per-decision", type=int, default=10)
    parser.add_argument(
        "--nl-max-total-calls",
        type=int,
        default=None,
        help="Hard NL ceiling for the day; unset derives it from the replay length.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pit: PaperPITData | None = None
    if args.data_backend == "daily":
        if args.daily_path is None:
            raise ValueError("--daily-path is required with --data-backend daily")
        daily = args.daily_path
        nl_query = None
        context_data = None
        execution_price = None
    else:
        nl_llm = None
        if args.nl_model:
            nl_llm = build_model_gateway(
                args.nl_model,
                env_file=args.nl_env_file,
                deepseek_api_key_env=args.nl_api_key_env or "DEEPSEEK_API_KEY",
                thinking_enabled=False,
            )
        provider = ResearchPITSnapshotProvider(
            experiment_dir=args.state_root / "research",
            raw_dir=args.raw_dir,
            fundamental_events_root=args.fundamental_events_root,
            fundamental_events_status=args.fundamental_events_status,
            config=SnapshotConfig(
                include_intraday=not args.disable_historical_intraday,
                replay_include_minutes=not args.disable_historical_intraday,
            ),
            cache_root=args.pit_cache_root,
        )
        pit = PaperPITData(
            provider,
            trade_date=args.trade_date,
            runtime_root=args.state_root / ".pit_runtime",
            nl_llm=nl_llm,
            nl_config=NLConfig(
                max_results=args.nl_max_results,
                max_calls_per_decision=args.nl_max_calls_per_decision,
                max_total_calls=args.nl_max_total_calls,
            ),
            max_intraday_row_group_rows=args.max_intraday_row_group_rows,
        )
        daily = pit.daily
        nl_query = pit.nl_service.query
        context_data = pit.context_data
        execution_price = pit.execution_price
    engine = DailyPaperEngine(
        strategy_path=args.strategy,
        strategy_revision=args.strategy_revision,
        daily=daily,
        state_root=args.state_root,
        models_dir=args.models_dir,
        schedule=StrategySchedule(
            period=args.strategy_period,
            inference_time=args.inference_time,
        ),
        profile=BrokerProfile(initial_cash=args.initial_cash),
        sandbox=SandboxConfig(
            image=args.sandbox_image,
            limits=SandboxLimits(
                cpus=args.sandbox_cpus,
                memory=args.sandbox_memory,
                pids=args.sandbox_pids,
                timeout_seconds=args.decision_timeout_seconds,
            ),
        ),
        nl_query=nl_query,
        context_data=context_data,
        execution_price=execution_price,
    )
    try:
        result = engine.run_day(args.trade_date)
    finally:
        engine.close()
        if pit is not None:
            pit.close()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
