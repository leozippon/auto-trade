"""Agent-visible projections of host-computed metric blocks.

``metrics`` is the host-side compact metric block written into ledger records;
``agent_visible_metrics`` keeps only its numeric fields and the whitelisted
benchmark and exposure fields. No forward or Held-out evidence passes through
these projections.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from autotrade.environment.data.summary import HOST_PATH_RE


def metrics(summary: dict[str, object] | None) -> dict[str, object] | None:
    if not summary:
        return None
    keys = (
        "total_return",
        "long_return",
        "sharpe",
        "max_drawdown",
        "order_count",
        "trade_count",
        "turnover",
        # Compact Barra-lite block (benchmark/excess return, beta, size tilt)
        # from the backtest tool — descriptive attribution per step.
        "benchmark",
    )
    compact = {key: summary.get(key) for key in keys if key in summary}
    exposure = summary.get("exposure")
    if isinstance(exposure, dict):
        compact["exposure"] = {
            key: exposure.get(key)
            for key in ("avg_gross", "max_gross", "zero_position_days", "replay_days")
            if key in exposure
        }
    # One scalar each from the cost-sensitivity and concentration blocks: this
    # block rides in every ledger record, so it keeps the two
    # numbers that change a judgement — whether the excess survives twice the
    # modelled slippage, and how much of the gain one name produced — and
    # leaves the full blocks in the backtest summary.
    for source, field in (
        ("cost_sensitivity", "excess_at_2x_slippage"),
        ("pnl_concentration", "top_name_share_of_gross_gains"),
    ):
        block = summary.get(source)
        if isinstance(block, dict) and field in block:
            compact[field] = block.get(field)
    return compact


# The benchmark fields one compact metric block keeps. Raw excess alone cannot
# separate real edge from a small-cap or high-beta tilt, so the size/beta
# neutralized excess -- the tie-breaker the research guidance names
# -- and the caliber it was computed under ride beside it. Descriptive
# attribution only: nothing here is forward or Held-out evidence.
_BENCHMARK_TEXT_KEYS = frozenset({"label", "neutralized_excess_method"})
_BENCHMARK_KEYS = (
    "label",
    "benchmark_return",
    "excess_return",
    "beta",
    "n_days",
    "size_tilt",
    "neutralized_excess_return",
    "neutralized_excess_method",
)


def _visible_metrics(value: object) -> dict[str, object] | None:
    return agent_visible_metrics(value if isinstance(value, dict) else None)


def agent_visible_metrics(summary: dict[str, object] | None) -> dict[str, object] | None:
    """Compact metric projection safe for Agent-visible history."""

    compact = metrics(summary)
    if compact is None:
        return None
    compact = {
        key: value
        for key, value in compact.items()
        if key in {"benchmark", "exposure"}
        or (isinstance(value, (int, float)) and not isinstance(value, bool))
    }
    benchmark = compact.get("benchmark")
    if isinstance(benchmark, dict):
        compact["benchmark"] = {
            key: benchmark.get(key)
            for key in _BENCHMARK_KEYS
            if key in benchmark
            and (
                isinstance(benchmark.get(key), str)
                if key in _BENCHMARK_TEXT_KEYS
                else isinstance(benchmark.get(key), (int, float))
                and not isinstance(benchmark.get(key), bool)
            )
        }
    else:
        compact.pop("benchmark", None)
    exposure = compact.get("exposure")
    if isinstance(exposure, dict):
        compact_exposure = {
            key: exposure.get(key)
            for key in ("avg_gross", "max_gross", "zero_position_days", "replay_days")
            if key in exposure
            and isinstance(exposure.get(key), (int, float))
            and not isinstance(exposure.get(key), bool)
        }
        if compact_exposure:
            compact["exposure"] = compact_exposure
        else:
            compact.pop("exposure", None)
    else:
        compact.pop("exposure", None)
    return compact


# The null-control fields a tool result carries. ``rejects_mean`` rides along because a null whose orders are mostly rejected
# is a weaker comparison, ``status``/``reason`` because a failed or unavailable
# null (a result with no filled trade) must not read as a missing one, and the
# null's own centre and spread over its ``k`` draws because a percentile alone
# does not say how far the observed excess sits from them. Informational:
# nothing in the pipeline gates on it.
NULL_CONTROL_KEYS = (
    "status",
    "reason",
    "observed_excess",
    "excess_percentile",
    "null_excess_mean",
    "null_excess_p05",
    "null_excess_p95",
    "k",
    "rejects_mean",
    "dropped_trips_mean",
    "step",
)


def allowed_keys(block: object, keys: Sequence[str]) -> dict[str, object] | None:
    """Whitelisted projection of one host-computed block; None when absent."""

    if not isinstance(block, Mapping):
        return None
    return {key: block.get(key) for key in keys if key in block}


# One line of a failed control's reason is enough to decide what to do about
# it, and the ledger keeps the full string either way.
PARENT_CONTROL_ERROR_MAX_CHARS = 400


def parent_control_error_text(value: object) -> str | None:
    """A Fold-era parent control's failure reason as the console shows it.

    Host paths are redacted and the text is bounded.
    """

    if not isinstance(value, str):
        return None
    text = HOST_PATH_RE.sub("[host_path]", value).strip()
    if not text:
        return None
    if len(text) > PARENT_CONTROL_ERROR_MAX_CHARS:
        text = text[: PARENT_CONTROL_ERROR_MAX_CHARS - 1] + "…"
    return text
