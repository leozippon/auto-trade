"""Shared CLI plumbing for the rolling-experiment entrypoints (docs/pipeline-design.md).

Single-sources the argparse argument groups and ``--help`` wording so a thin
wrapper cannot drift away from the parameters the worker actually accepts. The
provider and session wiring itself lives in ``autotrade.pipelines.worker``, also
used by the interactive HITL worker; this module only renders the validated
parameter file ``worker.load_worker_options`` consumes, so a CLI run and a
console run are configured through exactly the same validation.

``run_experiment.py`` does not use these groups: it drives a single
``DailyStrategyPipeline`` strategy replay, not a rolling Fold/Epoch experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.runtime import write_json_atomic
from autotrade.pipelines.hitl_state import MODEL_CHOICES, WEB_CREATE_DEFAULTS
from autotrade.pipelines.worker import (
    NON_PERSISTABLE_PARAMS,
    InteractiveWorkerOptions,
    load_worker_options,
)

DEFAULT_AGENT_MODEL = MODEL_CHOICES[0]
DEFAULT_META_MODEL = DEFAULT_AGENT_MODEL
DEFAULT_NL_MODEL = DEFAULT_AGENT_MODEL
DEFAULT_COMPACT_MODEL = DEFAULT_AGENT_MODEL

# The four period labels the console form is seeded with; only this cadence has
# standing defaults (see resolve_period_args).
DEFAULT_FOLD_PERIOD = str(WEB_CREATE_DEFAULTS["fold_period"])
PERIOD_ARGS = (
    "development_first_period",
    "development_last_period",
    "heldout_first_period",
    "heldout_last_period",
)


def _opt_help(text: str, verbose_help: bool) -> str | None:
    """Return ``text`` when the caller renders full help, else ``None``.

    ``run_experiment`` documents every flag; ``run_audit_session`` is terse and
    leaves most shared flags help-less. Gating keeps each ``--help`` identical.
    """
    return text if verbose_help else None


def resolve_meta_learning_directive(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> str:
    if args.meta_learning_directive and args.meta_learning_directive_file:
        parser.error(
            "pass only one of --meta-learning-directive or --meta-learning-directive-file"
        )
    if args.meta_learning_directive_file:
        return args.meta_learning_directive_file.read_text(encoding="utf-8")
    return args.meta_learning_directive


def resolve_fold_exploration_directive(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> str:
    if args.fold_exploration_directive and args.fold_exploration_directive_file:
        parser.error(
            "pass only one of --fold-exploration-directive or --fold-exploration-directive-file"
        )
    if args.fold_exploration_directive_file:
        return args.fold_exploration_directive_file.read_text(encoding="utf-8")
    return args.fold_exploration_directive


def resolve_period_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Fill the four period labels from the console defaults, or demand them.

    Only the default cadence has standing labels. A label written for one
    cadence silently mis-parses under another (``2022Q1`` is not a year, ``2023``
    is not a quarter), so any other ``--fold-period`` must carry all four
    explicitly and fail fast here rather than deep inside schedule building.
    """
    if args.fold_period == DEFAULT_FOLD_PERIOD:
        for name in PERIOD_ARGS:
            if not getattr(args, name):
                setattr(args, name, str(WEB_CREATE_DEFAULTS[name]))
        return
    missing = [
        f"--{name.replace('_', '-')}" for name in PERIOD_ARGS if not getattr(args, name)
    ]
    if missing:
        parser.error(
            f"--fold-period {args.fold_period} requires explicit period args: {', '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# argparse argument groups shared by both entrypoints
# ---------------------------------------------------------------------------
def add_path_arguments(parser: argparse.ArgumentParser, repo_root: Path) -> None:
    parser.add_argument("--raw-dir", type=Path, default=repo_root / "data/raw")
    parser.add_argument(
        "--fundamental-events-root",
        type=Path,
        default=repo_root / "data/pit/fundamental_events",
    )
    parser.add_argument(
        "--fundamental-events-status",
        type=Path,
        default=repo_root / "results/data_quality/fundamental_events_status.json",
    )
    parser.add_argument(
        "--experiments-root", type=Path, default=repo_root / "experiments"
    )
    parser.add_argument(
        "--work-root", type=Path, default=repo_root / ".runtime/sandboxes"
    )
    parser.add_argument(
        "--strategy-path",
        type=Path,
        default=repo_root / "configs/agent_output_template/main.py",
        help="Baseline strategy seeded into the first Fold's working copy.",
    )


def add_calendar_arguments(parser: argparse.ArgumentParser) -> None:
    """Development window, optional Test stage and Held-out (worker-validated)."""
    parser.add_argument(
        "--fold-period",
        choices=("week", "month", "quarter", "year"),
        default=DEFAULT_FOLD_PERIOD,
        help="Cadence unit the development and held-out labels are written in.",
    )
    period_help = (
        f"console default at --fold-period {DEFAULT_FOLD_PERIOD}; required for any other cadence"
    )
    parser.add_argument("--development-first-period", help=period_help)
    parser.add_argument("--development-last-period", help=period_help)
    parser.add_argument("--heldout-first-period", help=period_help)
    parser.add_argument("--heldout-last-period", help=period_help)
    parser.set_defaults(test_stage=bool(WEB_CREATE_DEFAULTS["test_stage"]))
    parser.add_argument(
        "--test-stage",
        dest="test_stage",
        action="store_true",
        help=(
            "Roll Folds inside the development window (first period validation only, "
            "each later period a frozen Test) instead of one regular Fold per period "
            "judged by Held-out alone."
        ),
    )


def add_schedule_arguments(
    parser: argparse.ArgumentParser, *, verbose_help: bool
) -> None:
    """The item-6 fixed-cycle strategy schedule the user picks per experiment."""
    parser.add_argument(
        "--strategy-period",
        choices=("day", "month", "quarter", "year"),
        default="day",
        help=_opt_help(
            "Strategy invocation cadence: every trading day, or the first available "
            "trading day of each new month/quarter/year.",
            verbose_help,
        ),
    )
    parser.add_argument(
        "--inference-time",
        default="08:30",
        metavar="HH:MM",
        help=_opt_help(
            "Fixed Asia/Shanghai inference time-of-day; any valid 24-hour HH:MM.",
            verbose_help,
        ),
    )


def add_snapshot_window_arguments(
    parser: argparse.ArgumentParser, *, verbose_help: bool
) -> None:
    parser.add_argument(
        "--window-months",
        type=int,
        default=int(WEB_CREATE_DEFAULTS["window_months"]),
        help=_opt_help(
            "Default PIT history window in months for decision-input snapshots and Fold input windows.",
            verbose_help,
        ),
    )
    parser.add_argument(
        "--daily-window-months",
        type=int,
        help="Override daily decision-input window in months.",
    )
    parser.add_argument(
        "--fundamentals-window-months",
        type=int,
        help="Override fundamentals decision-input window in months.",
    )
    parser.add_argument(
        "--events-window-months",
        type=int,
        help="Override events decision-input window in months.",
    )
    parser.add_argument(
        "--macro-window-months",
        type=int,
        help="Override macro decision-input window in months.",
    )
    parser.add_argument(
        "--text-window-months",
        type=int,
        help="Override text decision-input window in months.",
    )
    parser.add_argument(
        "--intraday-trade-days",
        type=int,
        default=SnapshotConfig().intraday_trade_days,
        help=_opt_help(
            "Number of recent visible trading days included in historical intraday_1min decision snapshots.",
            verbose_help,
        ),
    )
    for domain in ("events", "macro", "text", "fundamentals", "intraday"):
        parser.add_argument(
            f"--no-include-{domain}",
            dest=f"include_{domain}",
            action="store_false",
            help=_opt_help(
                f"Exclude the {domain} domain from decision snapshots and evaluation slots.",
                verbose_help,
            ),
        )
    parser.set_defaults(
        screen_exclude_st=bool(WEB_CREATE_DEFAULTS["screen_exclude_st"])
    )
    parser.add_argument(
        "--no-screen-exclude-st",
        dest="screen_exclude_st",
        action="store_false",
        help="Universe screen: keep ST names at the decision anchor.",
    )
    parser.add_argument(
        "--screen-exclude-st",
        dest="screen_exclude_st",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--screen-exclude-new-listed-days",
        type=int,
        default=int(WEB_CREATE_DEFAULTS["screen_exclude_new_listed_days"]),
        help="Universe screen: exclude stocks listed within N days of the anchor (0=off).",
    )
    parser.add_argument(
        "--screen-min-circ-mv-yi",
        type=float,
        default=None,
        help="Universe screen: minimum circulating market cap (亿元).",
    )
    parser.add_argument(
        "--screen-max-circ-mv-yi",
        type=float,
        default=None,
        help="Universe screen: maximum circulating market cap (亿元).",
    )
    parser.add_argument(
        "--screen-min-price",
        type=float,
        default=None,
        help="Universe screen: minimum close price at the anchor.",
    )
    parser.add_argument(
        "--screen-max-price",
        type=float,
        default=None,
        help="Universe screen: maximum close price at the anchor.",
    )
    parser.add_argument(
        "--screen-boards",
        nargs="+",
        choices=["main", "gem", "star", "bj"],
        default=list(WEB_CREATE_DEFAULTS["screen_boards"]),
        help="Universe screen: restrict to these boards (empty = all).",
    )


def add_model_arguments(parser: argparse.ArgumentParser, *, verbose_help: bool) -> None:
    parser.add_argument(
        "--model",
        default=DEFAULT_AGENT_MODEL,
        choices=MODEL_CHOICES,
        help=_opt_help("Ordinary Fold Agent main-conversation model.", verbose_help),
    )
    parser.add_argument(
        "--meta-model",
        default=DEFAULT_META_MODEL,
        choices=MODEL_CHOICES,
        help=_opt_help("Meta-learning Agent main-conversation model.", verbose_help),
    )
    parser.add_argument(
        "--nl-model",
        default=DEFAULT_NL_MODEL,
        choices=MODEL_CHOICES,
        help=_opt_help(
            f"NL Sub Agent model; defaults to {DEFAULT_NL_MODEL} (independent interface).",
            verbose_help,
        ),
    )
    parser.add_argument(
        "--compact-model",
        default=DEFAULT_COMPACT_MODEL,
        choices=MODEL_CHOICES,
        help=_opt_help(
            f"Context compaction model; defaults to {DEFAULT_COMPACT_MODEL} with thinking disabled.",
            verbose_help,
        ),
    )
    parser.add_argument(
        "--disable-context-compact",
        action="store_true",
        help=_opt_help("Disable semantic context compaction.", verbose_help),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "xhigh"),
        default="xhigh",
        help=_opt_help(
            "Reasoning effort for the Agent conversation and its sub-agents when "
            "thinking is enabled; default xhigh. These are the levels the local Qwen "
            "template distinguishes on the wire (legacy high/max in params.json "
            "resolve to xhigh).",
            verbose_help,
        ),
    )
    parser.add_argument(
        "--compact-token-threshold",
        type=int,
        default=200_000,
        help=_opt_help(
            "Estimated context tokens that trigger semantic compaction; default 200000.",
            verbose_help,
        ),
    )
    parser.add_argument(
        "--compact-keep-recent-messages",
        type=int,
        default=int(WEB_CREATE_DEFAULTS["compact_keep_recent_messages"]),
        help=_opt_help(
            "Raw non-summary messages preserved after semantic compaction.",
            verbose_help,
        ),
    )
    parser.add_argument(
        "--compact-max-tokens",
        type=int,
        default=int(WEB_CREATE_DEFAULTS["compact_max_tokens"]),
        help=_opt_help(
            "Maximum output tokens for one compaction summary.", verbose_help
        ),
    )
    parser.add_argument(
        "--compact-max-calls",
        type=int,
        default=int(WEB_CREATE_DEFAULTS["compact_max_calls"]),
        help=_opt_help(
            "Maximum semantic compaction provider calls per Agent session.",
            verbose_help,
        ),
    )


def add_meta_directive_arguments(
    parser: argparse.ArgumentParser, *, verbose_help: bool
) -> None:
    parser.add_argument(
        "--meta-learning-directive",
        default="",
        help=_opt_help(
            "Optional experiment-level research direction injected into each meta-learning prompt.",
            verbose_help,
        ),
    )
    parser.add_argument(
        "--meta-learning-directive-file",
        type=Path,
        help=_opt_help(
            "Optional UTF-8 text file whose content is injected as the meta-learning research direction.",
            verbose_help,
        ),
    )


def add_fold_exploration_directive_arguments(
    parser: argparse.ArgumentParser, *, verbose_help: bool
) -> None:
    parser.add_argument(
        "--fold-exploration-directive",
        default="",
        help=_opt_help(
            "Optional experiment-level exploration direction injected into every ordinary Fold prompt.",
            verbose_help,
        ),
    )
    parser.add_argument(
        "--fold-exploration-directive-file",
        type=Path,
        help=_opt_help(
            "Optional UTF-8 text file whose content is injected into every ordinary Fold prompt.",
            verbose_help,
        ),
    )


def add_acceptance_arguments(
    parser: argparse.ArgumentParser, *, verbose_help: bool
) -> None:
    parser.add_argument(
        "--min-return",
        type=float,
        default=0.0,
        help=_opt_help("Minimum validation total return.", verbose_help),
    )
    parser.add_argument(
        "--min-sharpe",
        type=float,
        default=0.0,
        help=_opt_help("Minimum validation Sharpe.", verbose_help),
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.25,
        help=_opt_help("Maximum validation drawdown.", verbose_help),
    )


# ---------------------------------------------------------------------------
# worker parameter rendering
# ---------------------------------------------------------------------------
def _relative(repo_root: Path, value: Path | None) -> str | None:
    """Worker paths are repo-relative; keep absolute ones outside the repo out."""
    if value is None:
        return None
    path = Path(value).resolve()
    return (
        str(path.relative_to(repo_root))
        if path.is_relative_to(repo_root)
        else str(path)
    )


def build_worker_params(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    meta_learning_directive: str = "",
    fold_exploration_directive: str = "",
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Render the argparse namespace as the worker's validated parameter file.

    The worker owns every default and every validation rule, so the CLI only
    supplies the keys the operator actually chose. Anything absent here falls
    back to the same default a console-created experiment gets.
    """
    forbidden = sorted(set(overrides or {}) & NON_PERSISTABLE_PARAMS)
    if forbidden:
        raise ValueError(
            "provider endpoint parameters cannot be persisted: " + ", ".join(forbidden)
        )
    params: dict[str, object] = {
        "experiment_id": args.experiment_id,
        "experiments_root": _relative(repo_root, args.experiments_root),
        "work_root": _relative(repo_root, args.work_root),
        "strategy_path": _relative(repo_root, args.strategy_path),
        "data_backend": "pit",
        "execution_mode": "trusted" if getattr(args, "local_dev", False) else "sandbox",
        "developer_mode": "llm",
        "raw_dir": _relative(repo_root, args.raw_dir),
        "fundamental_events_root": _relative(repo_root, args.fundamental_events_root),
        "fundamental_events_status": _relative(
            repo_root, args.fundamental_events_status
        ),
        "fold_period": args.fold_period,
        "development_first_period": args.development_first_period,
        "development_last_period": args.development_last_period,
        "test_stage": bool(args.test_stage),
        "heldout_first_period": args.heldout_first_period,
        "heldout_last_period": args.heldout_last_period,
        "strategy_period": args.strategy_period,
        "inference_time": args.inference_time,
        "max_fold_minutes": args.max_fold_minutes,
        "window_months": args.window_months,
        "intraday_trade_days": args.intraday_trade_days,
        "include_events": args.include_events,
        "include_macro": args.include_macro,
        "include_text": args.include_text,
        "include_fundamentals": args.include_fundamentals,
        "include_intraday": args.include_intraday,
        "screen_exclude_st": args.screen_exclude_st,
        "screen_exclude_new_listed_days": args.screen_exclude_new_listed_days,
        "screen_min_circ_mv_yi": args.screen_min_circ_mv_yi,
        "screen_max_circ_mv_yi": args.screen_max_circ_mv_yi,
        "screen_min_price": args.screen_min_price,
        "screen_max_price": args.screen_max_price,
        "screen_boards": list(args.screen_boards),
        "model": args.model,
        "meta_model": args.meta_model,
        "nl_model": args.nl_model,
        "compact_model": args.compact_model,
        "reasoning_effort": args.reasoning_effort,
        "no_thinking": bool(getattr(args, "no_thinking", False)),
        "disable_context_compact": bool(args.disable_context_compact),
        "compact_token_threshold": args.compact_token_threshold,
        "compact_keep_recent_messages": args.compact_keep_recent_messages,
        "compact_max_tokens": args.compact_max_tokens,
        "compact_max_calls": args.compact_max_calls,
        "min_return": args.min_return,
        "min_sharpe": args.min_sharpe,
        "max_drawdown": args.max_drawdown,
        "meta_learning_directive": meta_learning_directive,
        "fold_exploration_directive": fold_exploration_directive,
    }
    for window in ("daily", "fundamentals", "events", "macro", "text"):
        value = getattr(args, f"{window}_window_months", None)
        if value is not None:
            params[f"{window}_window_months"] = value
    params.update(overrides or {})
    return {key: value for key, value in params.items() if value is not None}


def build_worker_options(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    meta_learning_directive: str = "",
    fold_exploration_directive: str = "",
    overrides: dict[str, object] | None = None,
) -> InteractiveWorkerOptions:
    """Persist the CLI parameters and load them back through the worker.

    Writing ``hitl/params.json`` first is deliberate: the worker's loader is the
    single place that validates parameter names, path containment, period
    labels and the release pin, and a resumed or console-inspected run must see
    exactly the configuration this invocation used.
    """
    experiment_dir = Path(args.experiments_root).resolve() / args.experiment_id
    params = build_worker_params(
        args,
        repo_root=repo_root,
        meta_learning_directive=meta_learning_directive,
        fold_exploration_directive=fold_exploration_directive,
        overrides=overrides,
    )
    (experiment_dir / "hitl").mkdir(parents=True, exist_ok=True)
    write_json_atomic(experiment_dir / "hitl" / "params.json", params)
    return load_worker_options(experiment_dir, repo_root=repo_root)
