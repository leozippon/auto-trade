"""Experiment result visualization from experiment_ledger.jsonl.

Produces, per experiment:

- ``epoch_comparison_returns.png``: every Epoch's per-Fold return, CSI 300
  benchmark return, and compounded equity curves on Fold-aligned charts, with
  held-out appended to the final Epoch and a metrics table underneath;
- ``epoch_returns/<epoch_id>_returns.png``: one detailed Fold-return chart per
  Epoch, including the scored return, CSI 300 benchmark, cumulative equity,
  drawdown, and a compact metrics table. The final Epoch appends held-out
  points.

A Fold is scored on the result its run actually produced: its own Validation
replay under the default single-window design, or the frozen Test when the
experiment runs a Test stage. Charts and tables name which of the two they are
showing. The genuine out-of-sample record of the default design is
``summary.walk_forward`` (each Fold's inherited parent replayed on that Fold's
window) and the Held-out points.

Labels are English to stay font-safe on headless hosts.
"""

from __future__ import annotations
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .ledger import (
    ExperimentLedger,
    experiment_verdict,
    latest_fold_records,
    latest_heldout_records,
)

DEV_COLOR = "#1f77b4"
TEST_COLOR = "#d62728"
HELDOUT_COLOR = "#9467bd"
LONG_COLOR = "#2ca02c"
BENCHMARK_COLOR = "#4d5566"
GRID_COLOR = "#d9dde3"
DEFAULT_BENCHMARK_CODE = "000300.SH"
DEFAULT_BENCHMARK_LABEL = "CSI 300"


def build_experiment_report(ledger_path: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Charts + summary from the ledger only. Benchmark returns come from each
    record's frozen ``benchmark`` block (computed at replay time from slot
    data), so the report never reads the mutable raw lake and always matches
    what the Agent and the console saw."""
    ledger = ExperimentLedger(ledger_path)
    # Latest record per (epoch, fold) / held-out period: after a fold re-run
    # the formal report counts only the superseding attempt.
    folds = list(latest_fold_records(ledger.read("fold")).values())
    heldout = latest_heldout_records(ledger.read("heldout"))
    if not folds:
        raise ValueError(f"no fold records in {ledger_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [_fold_row(record) for record in folds]
    rows += [_heldout_row(record) for record in heldout]
    benchmark_info = _benchmark_summary(rows)

    epoch_files = _plot_epoch_returns(rows, output_dir / "epoch_returns")
    epoch_comparison = output_dir / "epoch_comparison_returns.png"
    _plot_epoch_comparison(rows, epoch_comparison)
    summary = _summarize(rows)
    summary["benchmark"] = benchmark_info
    # Walk-forward record: each Fold's inherited strategy replayed unchanged on
    # that Fold's window by the host before the session (docs/pipeline-design.md
    # §2.2). Genuine out-of-sample evidence in Epoch 1; later Epochs revisit
    # windows the lineage has already seen.
    summary["walk_forward"] = _walk_forward(folds)
    # Graduation verdict of the frozen strategy on Held-out; None until recorded.
    summary["verdict"] = experiment_verdict(heldout)
    # Flag the whole report when frozen benchmark blocks are missing for scored
    # periods so downstream gates can react (docs/pipeline-design.md §4.2).
    summary["status"] = "ok" if str(benchmark_info.get("status")) == "ok" else "warning"
    summary["epoch_return_charts"] = [str(path) for path in epoch_files]
    summary["epoch_comparison_chart"] = str(epoch_comparison)
    return summary


def _fold_row(record: dict[str, object]) -> dict[str, object]:
    """One development Fold, scored on the result its run actually produced.

    A regular Fold has no Test region (the default): it is scored on its own
    Validation replay and its row spans the validation window. With a Test
    stage the frozen Test is the scored result and the row spans the test
    window; a Test that failed scores nothing, it does not fall back.
    """
    validation = record.get("validation_result") or {}
    test = record.get("test_result") or {}
    scored = test or validation
    source = "frozen_test" if test else "validation"
    period = record.get("test_period") if test else record.get("validation_period")
    benchmark_return, benchmark_label = _frozen_benchmark(scored)
    scored_return = _num(scored.get("total_return"))
    return {
        "epoch_id": str(record.get("epoch_id", "")),
        "label": str(record.get("fold_id", "")).replace("fold_", ""),
        "kind": "development",
        "source": source,
        "fold_status": record.get("fold_status"),
        "return": scored_return,
        "long_return": _num(scored.get("long_return")),
        "sharpe": _num(scored.get("sharpe")),
        "drawdown": _num(scored.get("max_drawdown")),
        "orders": scored.get("order_count"),
        # Secondary series, charted only where it is not the scored result.
        "valid_return": _num(validation.get("total_return")),
        "selected_step": record.get("selected_step_id"),
        "finish_reason": record.get("finish_reason"),
        "period_start": _period_part(period, "start"),
        "period_end": _period_part(period, "end"),
        "benchmark_return": benchmark_return,
        "benchmark_label": benchmark_label,
        "active_return": (
            scored_return - benchmark_return
            if scored_return is not None and benchmark_return is not None
            else None
        ),
    }


def _walk_forward(folds: list[dict[str, object]]) -> list[dict[str, object]]:
    """Per-Epoch transitions previous Fold -> this Fold from ``parent_control``."""
    by_epoch: dict[str, list[dict[str, object]]] = {}
    ordered = sorted(
        folds,
        key=lambda row: (str(row.get("epoch_id", "")), str(row.get("validation_period", ""))),
    )
    for record in ordered:
        control = record.get("parent_control")
        if not isinstance(control, dict):
            continue
        result = control.get("validation_result") or {}
        benchmark_return, _label = _frozen_benchmark(result)
        total = _num(result.get("total_return"))
        period = record.get("validation_period")
        by_epoch.setdefault(str(record.get("epoch_id", "")), []).append(
            {
                "fold": str(record.get("fold_id", "")).replace("fold_", ""),
                "period_start": _period_part(period, "start"),
                "period_end": _period_part(period, "end"),
                "parent_strategy_artifact_id": control.get("parent_strategy_artifact_id"),
                "status": control.get("status"),
                "error": control.get("error"),
                "return": total,
                "benchmark_return": benchmark_return,
                "excess_return": (
                    total - benchmark_return
                    if total is not None and benchmark_return is not None
                    else None
                ),
                "sharpe": _num(result.get("sharpe")),
                "max_drawdown": _num(result.get("max_drawdown")),
            }
        )
    return [
        {
            "epoch_id": epoch_id,
            "transitions": transitions,
            "mean_excess_return": _mean(
                [row["excess_return"] for row in transitions if row["excess_return"] is not None]
            ),
            "mean_sharpe": _mean(
                [row["sharpe"] for row in transitions if row["sharpe"] is not None]
            ),
        }
        for epoch_id, transitions in by_epoch.items()
    ]


def _heldout_row(record: dict[str, object]) -> dict[str, object]:
    result = record.get("result") or {}
    benchmark_return, benchmark_label = _frozen_benchmark(result)
    heldout_return = _num(result.get("total_return"))
    return {
        "epoch_id": str(record.get("epoch_id", "")),
        "label": str(record.get("fold_id", "")).replace("heldout_", "HO "),
        "kind": "heldout",
        "source": "heldout",
        "fold_status": "heldout",
        "return": heldout_return,
        "long_return": _num(result.get("long_return")),
        "sharpe": _num(result.get("sharpe")),
        "drawdown": _num(result.get("max_drawdown")),
        "orders": result.get("order_count"),
        "selected_step": None,
        "finish_reason": "heldout",
        "period_start": _period_part(record.get("period"), "start"),
        "period_end": _period_part(record.get("period"), "end"),
        "benchmark_return": benchmark_return,
        "benchmark_label": benchmark_label,
        "active_return": (
            heldout_return - benchmark_return
            if heldout_return is not None and benchmark_return is not None
            else None
        ),
    }


def _period_part(value: object, key: str) -> str | None:
    if isinstance(value, dict):
        parsed = value.get(key)
        return str(parsed) if parsed else None
    if isinstance(value, str) and ".." in value:
        start, end = value.split("..", 1)
        return start if key == "start" else end
    return None


def _frozen_benchmark(result: dict[str, object]) -> tuple[float | None, str | None]:
    """(benchmark_return, label) from a result's frozen ``benchmark`` block."""
    block = result.get("benchmark")
    if not isinstance(block, dict):
        return None, None
    label = block.get("label")
    return _num(block.get("benchmark_return")), (str(label) if label else None)


def _benchmark_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Coverage of the frozen benchmark blocks across scored periods."""
    scored = [row for row in rows if row.get("return") is not None]
    covered = sum(1 for row in scored if row.get("benchmark_return") is not None)
    if covered == len(scored) and scored:
        status = "ok"
    elif covered:
        status = "partial_coverage"
    else:
        status = "missing_frozen_benchmark"
    return {
        "source": "ledger_frozen_style",
        "code": DEFAULT_BENCHMARK_CODE,
        "label": _benchmark_label(rows),
        "status": status,
        "covered_periods": covered,
        "total_periods": len(scored),
    }


def _plot_epoch_returns(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
    dev_epochs = sorted({str(row["epoch_id"]) for row in rows if row["kind"] == "development"})
    if not dev_epochs:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    last_epoch = dev_epochs[-1]
    paths: list[Path] = []
    heldout_rows = [row for row in rows if row["kind"] == "heldout"]
    for epoch_id in dev_epochs:
        epoch_rows = [row for row in rows if row["kind"] == "development" and row["epoch_id"] == epoch_id]
        if epoch_id == last_epoch:
            epoch_rows = epoch_rows + heldout_rows
        path = output_dir / f"{epoch_id}_returns.png"
        _plot_single_epoch(epoch_rows, epoch_id, path)
        paths.append(path)
    return paths


def _plot_epoch_comparison(rows: list[dict[str, object]], path: Path) -> None:
    dev_rows = [row for row in rows if row["kind"] == "development"]
    epochs = sorted({str(row["epoch_id"]) for row in dev_rows})
    if not epochs:
        return

    fold_labels = _ordered_labels(dev_rows)
    heldout_rows = [row for row in rows if row["kind"] == "heldout" and row["return"] is not None]
    heldout_labels = [str(row["label"]) for row in heldout_rows]
    labels = fold_labels + heldout_labels
    x = list(range(len(labels)))
    heldout_start = len(fold_labels)
    benchmark_by_label = _benchmark_by_label(rows)
    benchmark_values = [_plot_num(benchmark_by_label.get(label)) for label in labels]
    benchmark_label = _benchmark_label(rows)
    result_label = _result_label(dev_rows)

    fig = plt.figure(figsize=(max(13, 0.9 * len(labels)), 12.0), constrained_layout=True)
    fig.patch.set_facecolor("#fbfbfd")
    grid = fig.add_gridspec(3, 1, height_ratios=(2.45, 1.9, 1.75))
    ax_ret = fig.add_subplot(grid[0])
    ax_eq = fig.add_subplot(grid[1], sharex=ax_ret)
    _style_axis(ax_ret)
    _style_axis(ax_eq)
    colors = plt.get_cmap("tab10")
    last_epoch = epochs[-1]
    plotted: list[tuple[str, list[float]]] = []
    for index, epoch_id in enumerate(epochs):
        epoch_rows = {
            str(row["label"]): row
            for row in dev_rows
            if str(row["epoch_id"]) == epoch_id and row["return"] is not None
        }
        values = [_plot_num(epoch_rows[label]["return"]) if label in epoch_rows else float("nan") for label in fold_labels]
        if epoch_id == last_epoch:
            values += [_plot_num(row["return"]) for row in heldout_rows]
        else:
            values += [float("nan")] * len(heldout_rows)
        color = colors(index % 10)
        ax_ret.plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            color=color,
            label=f"{epoch_id} {result_label.lower()}",
        )
        ax_eq.plot(
            x,
            _compound_curve(values),
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            color=color,
            label=f"{epoch_id} compounded",
        )
        plotted.append((epoch_id, values))

    if any(not _is_nan(value) for value in benchmark_values):
        ax_ret.plot(
            x,
            benchmark_values,
            color=BENCHMARK_COLOR,
            linestyle=":",
            marker="x",
            linewidth=1.9,
            markersize=5.2,
            label=f"{benchmark_label} benchmark",
        )
        ax_eq.plot(
            x,
            _compound_curve(benchmark_values),
            color=BENCHMARK_COLOR,
            linestyle=":",
            marker="x",
            linewidth=1.9,
            markersize=5.0,
            label=f"{benchmark_label} compounded",
        )

    if heldout_rows:
        for axis in (ax_ret, ax_eq):
            axis.axvspan(heldout_start - 0.5, len(labels) - 0.5, color=HELDOUT_COLOR, alpha=0.08)
        ax_ret.scatter(
            list(range(heldout_start, len(labels))),
            [_plot_num(row["return"]) for row in heldout_rows],
            color=HELDOUT_COLOR,
            marker="D",
            s=80,
            zorder=5,
            label="Held-out",
        )
    ax_ret.axhline(0.0, color="#6e6e6e", linewidth=0.9)
    ax_eq.axhline(1.0, color="#6e6e6e", linewidth=0.9)
    ax_ret.set_title("Epoch comparison: fold return and compounded equity", pad=14)
    ax_ret.set_ylabel("Fold return")
    ax_eq.set_ylabel("Compounded equity")
    ax_eq.set_xticks(x)
    ax_eq.set_xticklabels(labels, rotation=45, ha="right")
    ax_ret.tick_params(labelbottom=False)
    ax_ret.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}x"))
    ax_ret.legend(loc="upper left", ncols=min(3, len(epochs) + 1), fontsize=8.5)
    ax_eq.legend(loc="upper left", ncols=min(3, len(epochs)), fontsize=8.5)
    _annotate_last_points(ax_ret, x, plotted)

    table_ax = fig.add_subplot(grid[2])
    table_ax.axis("off")
    columns = [
        "Epoch",
        "Folds",
        "Mean ret",
        "Median ret",
        "Cum ret",
        "Positive",
        "Mean Sharpe",
        "Cum active",
        "Worst max loss",
        "Worst fold",
        "Held-out ret",
    ]
    cells = [_epoch_metric_row(epoch_id, dev_rows, heldout_rows if epoch_id == last_epoch else []) for epoch_id in epochs]
    table = table_ax.table(
        cellText=cells,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        bbox=[0.0, 0.08, 1.0, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    _style_table(table)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _result_label(rows: list[dict[str, object]]) -> str:
    """Which stage the development rows are scored on, for chart and table text."""
    sources = {str(row["source"]) for row in rows if row["kind"] == "development"}
    if sources == {"frozen_test"}:
        return "Frozen test"
    if sources == {"validation"}:
        return "Validation"
    return "Fold result"


def _ordered_labels(rows: list[dict[str, object]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        label = str(row["label"])
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def _benchmark_by_label(rows: list[dict[str, object]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for row in rows:
        label = str(row["label"])
        if label not in out and row.get("benchmark_return") is not None:
            out[label] = _num(row.get("benchmark_return"))
    return out


def _benchmark_label(rows: list[dict[str, object]]) -> str:
    """Chart label. Non-ASCII frozen labels (沪深300) map to the English default
    so headless-host fonts render them (module docstring)."""
    for row in rows:
        label = row.get("benchmark_label")
        if label and str(label).isascii():
            return str(label)
    return DEFAULT_BENCHMARK_LABEL


def _epoch_metric_row(
    epoch_id: str,
    dev_rows: list[dict[str, object]],
    heldout_rows: list[dict[str, object]],
) -> list[str]:
    epoch_rows = [row for row in dev_rows if str(row["epoch_id"]) == epoch_id]
    returns = [_num(row["return"]) for row in epoch_rows if _num(row["return"]) is not None]
    sharpes = [_num(row["sharpe"]) for row in epoch_rows if _num(row["sharpe"]) is not None]
    drawdowns = [_num(row["drawdown"]) for row in epoch_rows if _num(row["drawdown"]) is not None]
    worst = min(epoch_rows, key=lambda row: _plot_num(row["return"])) if returns else None
    heldout_returns = [_num(row["return"]) for row in heldout_rows if _num(row["return"]) is not None]
    return [
        epoch_id,
        str(len(returns)),
        _fmt_pct(_mean(returns)),
        _fmt_pct(_median(returns)),
        _fmt_pct(_compound_return(returns)),
        _fmt_pct(_mean([1.0 if value > 0 else 0.0 for value in returns])),
        _fmt(_mean(sharpes)),
        _fmt_pct(_compound_active_return(epoch_rows)),
        _fmt_pct(max(drawdowns) if drawdowns else None),
        str(worst["label"]) if worst else "-",
        _fmt_pct(_mean(heldout_returns)),
    ]


def _plot_single_epoch(rows: list[dict[str, object]], epoch_id: str, path: Path) -> None:
    labels = [str(row["label"]) for row in rows]
    x = list(range(len(rows)))
    valid = [_plot_num(row.get("valid_return")) for row in rows]
    scored = [_plot_num(row["return"]) for row in rows]
    benchmark = [_plot_num(row["benchmark_return"]) for row in rows]
    benchmark_label = _benchmark_label(rows)
    result_label = _result_label(rows)
    long_values = [_plot_num(row["long_return"]) for row in rows]
    equity: list[float] = []
    drawdown: list[float] = []
    current = 1.0
    peak = 1.0
    for value in scored:
        if not _is_nan(value):
            current *= 1.0 + float(value)
        peak = max(peak, current)
        equity.append(current)
        drawdown.append(current / peak - 1.0)

    fig = plt.figure(figsize=(max(13, 0.9 * len(rows)), 10.8), constrained_layout=True)
    fig.patch.set_facecolor("#fbfbfd")
    grid = fig.add_gridspec(3, 1, height_ratios=(2.5, 1.55, 1.85))
    ax_ret = fig.add_subplot(grid[0])
    ax_eq = fig.add_subplot(grid[1], sharex=ax_ret)
    table_ax = fig.add_subplot(grid[2])
    _style_axis(ax_ret)
    _style_axis(ax_eq)

    dev_idx = [i for i, row in enumerate(rows) if row["kind"] == "development"]
    ho_idx = [i for i, row in enumerate(rows) if row["kind"] == "heldout"]
    bar_width = 0.32
    ax_ret.bar(
        dev_idx,
        [long_values[i] for i in dev_idx],
        width=bar_width,
        color=LONG_COLOR,
        alpha=0.22,
        label="Long contribution",
    )
    if dev_idx:
        ax_ret.plot(
            dev_idx,
            [scored[i] for i in dev_idx],
            color=TEST_COLOR,
            marker="o",
            linewidth=2.0,
            label=f"{result_label} return",
        )
        # Only a second series where it is not the scored one already.
        if any(rows[i]["source"] == "frozen_test" for i in dev_idx):
            ax_ret.plot(
                dev_idx,
                [valid[i] for i in dev_idx],
                color=DEV_COLOR,
                marker="s",
                linewidth=1.7,
                linestyle="--",
                alpha=0.85,
                label="Validation return",
            )
        if any(not _is_nan(benchmark[i]) for i in dev_idx):
            ax_ret.plot(
                dev_idx,
                [benchmark[i] for i in dev_idx],
                color=BENCHMARK_COLOR,
                marker="x",
                linewidth=1.7,
                linestyle=":",
                alpha=0.9,
                label=f"{benchmark_label} return",
            )
    if ho_idx:
        ax_ret.plot(
            ho_idx,
            [scored[i] for i in ho_idx],
            color=HELDOUT_COLOR,
            marker="D",
            linestyle="none",
            markersize=9,
            label="Held-out return",
        )
        if any(not _is_nan(benchmark[i]) for i in ho_idx):
            ax_ret.scatter(
                ho_idx,
                [benchmark[i] for i in ho_idx],
                color=BENCHMARK_COLOR,
                marker="x",
                s=78,
                zorder=5,
                label=f"{benchmark_label} held-out",
            )
        ax_ret.axvspan(min(ho_idx) - 0.5, max(ho_idx) + 0.5, color=HELDOUT_COLOR, alpha=0.08)
    ax_ret.axhline(0.0, color="#6e6e6e", linewidth=0.9)
    ax_ret.set_ylabel("Fold return")
    ax_ret.set_title(f"{epoch_id}: {result_label.lower()} return, long contribution, and held-out", pad=14)
    ax_ret.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax_ret.legend(loc="upper left", ncols=3, fontsize=8.5)
    _add_best_worst_note(ax_ret, labels, scored, dev_idx)

    ax_eq.plot(x, equity, color="#234f9b", marker="o", linewidth=2.0, label=f"Compounded {result_label.lower()} equity")
    if any(not _is_nan(value) for value in benchmark):
        ax_eq.plot(
            x,
            _compound_curve(benchmark),
            color=BENCHMARK_COLOR,
            marker="x",
            linestyle=":",
            linewidth=1.8,
            label=f"{benchmark_label} compounded",
        )
        active_equity = _active_curve(scored, benchmark)
        ax_eq.plot(
            x,
            active_equity,
            color="#0f766e",
            marker=".",
            linestyle="--",
            linewidth=1.6,
            label="Relative equity vs benchmark",
        )
    ax_dd = ax_eq.twinx()
    ax_dd.fill_between(x, drawdown, 0.0, color=TEST_COLOR, alpha=0.14, label="Peak-to-current loss")
    ax_eq.axhline(1.0, color="#6e6e6e", linewidth=0.9)
    ax_eq.set_ylabel("Equity multiple")
    ax_dd.set_ylabel("Peak-to-current loss")
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f}%"))
    ax_eq.set_xticks(x)
    ax_eq.set_xticklabels(labels, rotation=45, ha="right")
    ax_eq.legend(loc="upper left", fontsize=8.5)
    ax_dd.legend(loc="upper right", fontsize=8.5)

    table_ax.axis("off")
    columns = [
        "Fold",
        result_label,
        "CSI300",
        "Active",
        "Long",
        "Sharpe",
        "Max loss",
        "Orders",
        "Step",
    ]
    cells = [
        [
            str(row["label"]),
            _fmt_pct(row["return"]),
            _fmt_pct(row["benchmark_return"]),
            _fmt_pct(row["active_return"]),
            _fmt_pct(row["long_return"]),
            _fmt(row["sharpe"]),
            _fmt_pct(row["drawdown"]),
            _fmt_int(row["orders"]),
            str(row["selected_step"] or "-"),
        ]
        for row in rows
    ]
    table = table_ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.22)
    _style_table(table)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _style_axis(ax) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#b8bec8")
    ax.spines["bottom"].set_color("#b8bec8")


def _style_table(table) -> None:
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d3d7df")
        cell.set_linewidth(0.45)
        if row == 0:
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(weight="bold", color="#233044")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f8fafc")


def _annotate_last_points(ax, x: list[int], plotted: list[tuple[str, list[float]]]) -> None:
    for epoch_id, values in plotted:
        valid_points = [(i, value) for i, value in zip(x, values) if not _is_nan(value)]
        if not valid_points:
            continue
        idx, value = valid_points[-1]
        ax.annotate(
            f"{epoch_id} {value * 100:.1f}%",
            xy=(idx, value),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            color="#233044",
            clip_on=True,
        )


def _add_best_worst_note(ax, labels: list[str], values: list[float], indices: list[int]) -> None:
    points = [(idx, values[idx]) for idx in indices if not _is_nan(values[idx])]
    if not points:
        return
    best_idx, best_value = max(points, key=lambda item: item[1])
    worst_idx, worst_value = min(points, key=lambda item: item[1])
    note = (
        f"Best fold: {labels[best_idx]}  {best_value * 100:.1f}%\n"
        f"Worst fold: {labels[worst_idx]}  {worst_value * 100:.1f}%"
    )
    ax.text(
        0.985,
        0.94,
        note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        color="#233044",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ffffff", "edgecolor": "#d3d7df", "alpha": 0.92},
    )


def _compound_curve(values: list[float]) -> list[float]:
    curve: list[float] = []
    current = 1.0
    for value in values:
        if _is_nan(value):
            curve.append(float("nan"))
            continue
        current *= 1.0 + value
        curve.append(current)
    return curve


def _active_curve(strategy_returns: list[float], benchmark_returns: list[float]) -> list[float]:
    curve: list[float] = []
    strategy = 1.0
    benchmark = 1.0
    for strategy_return, benchmark_return in zip(strategy_returns, benchmark_returns):
        if _is_nan(strategy_return) or _is_nan(benchmark_return):
            curve.append(float("nan"))
            continue
        strategy *= 1.0 + strategy_return
        benchmark *= 1.0 + benchmark_return
        curve.append(strategy / benchmark if benchmark else float("nan"))
    return curve


def _is_nan(value: float) -> bool:
    return value != value


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    dev = [row for row in rows if row["kind"] == "development"]
    heldout = [row for row in rows if row["kind"] == "heldout"]
    dev_returns = [row["return"] for row in dev if row["return"] is not None]
    dev_benchmarks = [row["benchmark_return"] for row in dev if row["benchmark_return"] is not None]
    dev_active = [row["active_return"] for row in dev if row["active_return"] is not None]
    return {
        "folds": len(dev),
        "heldout_periods": len(heldout),
        "development": {
            # Which stage produced these numbers, per Fold: "validation" under
            # the default single-window design, "frozen_test" with a Test stage.
            "result_sources": _counts([str(row["source"]) for row in dev]),
            "mean_return": _mean(dev_returns),
            "median_return": _median(dev_returns),
            "std_return": _std(dev_returns),
            "positive_rate": _mean([1.0 if value > 0 else 0.0 for value in dev_returns]),
            "worst_return": min(dev_returns) if dev_returns else None,
            "mean_sharpe": _mean([row["sharpe"] for row in dev if row["sharpe"] is not None]),
            "mean_benchmark_return": _mean(dev_benchmarks),
            "mean_active_return": _mean(dev_active),
            "std_active_return": _std(dev_active),
            # Ratio-standard compounded active return ∏(1+r)/∏(1+b)−1, matching the
            # "Relative equity vs benchmark" chart and the "Cum active" table column.
            "compound_active_return": _compound_active_return(dev),
            # One-sample t-stat of per-fold active return vs zero; null when n<2 or std==0.
            "active_return_tstat": _tstat(dev_active),
            "fold_status_counts": _counts([str(row["fold_status"]) for row in dev]),
        },
        "heldout": {
            "returns": {row["label"]: row["return"] for row in heldout},
            "benchmark_returns": {row["label"]: row["benchmark_return"] for row in heldout},
            "active_returns": {row["label"]: row["active_return"] for row in heldout},
            "mean_return": _mean([row["return"] for row in heldout if row["return"] is not None]),
            "mean_active_return": _mean([row["active_return"] for row in heldout if row["active_return"] is not None]),
        },
    }


def _num(value: object) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _plot_num(value: object) -> float:
    parsed = _num(value)
    return float("nan") if parsed is None else parsed


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _std(values: list[float]) -> float | None:
    """Sample standard deviation (ddof=1); None when fewer than two points."""
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _tstat(values: list[float]) -> float | None:
    """One-sample t-statistic of the mean against zero: mean / (std/√n).
    None when fewer than two points or the sample has zero dispersion."""
    std = _std(values)
    if std is None or std == 0:
        return None
    mean = sum(values) / len(values)
    return mean / (std / math.sqrt(len(values)))


def _compound_return(values: list[float]) -> float | None:
    if not values:
        return None
    current = 1.0
    for value in values:
        current *= 1.0 + value
    return current - 1.0


def _compound_active_return(rows: list[dict[str, object]]) -> float | None:
    """Compounded active return as the strategy/benchmark equity ratio
    ∏(1+rᵢ)/∏(1+bᵢ)−1, matching ``_active_curve``. Only folds carrying both a
    strategy and a benchmark return contribute."""
    strategy = 1.0
    benchmark = 1.0
    count = 0
    for row in rows:
        scored_return = _num(row.get("return"))
        bench = _num(row.get("benchmark_return"))
        if scored_return is None or bench is None:
            continue
        strategy *= 1.0 + scored_return
        benchmark *= 1.0 + bench
        count += 1
    if count == 0 or benchmark == 0:
        return None
    return strategy / benchmark - 1.0


def _counts(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def _fmt_int(value: object) -> str:
    return "-" if value is None else str(value)
