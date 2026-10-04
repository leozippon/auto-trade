#!/usr/bin/env python3
"""Experiment pipeline entrypoint (docs/pipeline-design.md).

Runs one scheduled daily strategy experiment against a daily parquet input and
prints its JSON result. The docs do not prescribe a CLI; this thin wrapper only
wires documented components.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.strategy import StrategySchedule
from autotrade.pipelines import DailyStrategyPipeline, StrategyExperimentConfig
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", type=Path, required=True, help="Python file defining generate_orders(context)")
    parser.add_argument("--daily", type=Path, required=True, help="daily parquet input")
    parser.add_argument("--strategy-period", choices=("day", "month", "quarter", "year"), default="day")
    parser.add_argument("--inference-time", default="08:30", metavar="HH:MM")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument(
        "--execution-mode",
        choices=("sandbox", "trusted"),
        default="sandbox",
        help="Docker isolation by default; trusted runs reviewed code on the host",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = StrategyExperimentConfig(
        strategy_path=args.strategy,
        schedule=StrategySchedule(args.strategy_period, args.inference_time),
        # The creation default, as for a new arm. A single daily file carries
        # no ex-date table, so this runner credits no dividend and the tax
        # never has anything to charge; the profile still states the rule.
        broker_profile=BrokerProfile(
            initial_cash=args.initial_cash,
            dividend_tax=bool(WEB_CREATE_DEFAULTS["dividend_tax"]),
        ),
        execution_mode=args.execution_mode,
    )
    result = DailyStrategyPipeline(config).run(args.daily)
    print(json.dumps(result.to_record(), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
