"""The signal screen: metrics on a synthetic panel, alignment, scope refusal, contract errors."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.data.snapshot import finalize_snapshot_dir
from autotrade.environment.sandbox import SCREENING_TOOL_SOURCE
from autotrade.environment.screening import screen


def _write_view(out_dir: Path, daily: pd.DataFrame, *, kind: str = "decision_input", **fields: object) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out_dir / "daily.parquet", index=False)
    if kind == "decision_input":
        fields.setdefault("decision_time", f"{pd.Timestamp(daily['trade_date'].max()).date().isoformat()}T23:59:59+08:00")
    finalize_snapshot_dir(out_dir, kind=kind, **fields)


def _panel_frame(opens: np.ndarray, dates: list[str], codes: list[str], *, adj: np.ndarray | None = None) -> pd.DataFrame:
    days, names = opens.shape
    adj = np.ones_like(opens) if adj is None else adj
    rng = np.random.default_rng(1)
    circ_mv = np.exp(rng.normal(12.0, 1.0, size=names))
    return pd.DataFrame(
        {
            "trade_date": np.repeat(dates, names),
            "ts_code": np.tile(codes, days),
            "open": opens.ravel(),
            "close": (opens * 1.001).ravel(),
            "adj_factor": adj.ravel(),
            "up_limit": (opens * 1.1).ravel(),
            "down_limit": (opens * 0.9).ravel(),
            "circ_mv": np.tile(circ_mv, days),
            "is_suspended": False,
        }
    )


def _synthetic_view(root: Path, *, days: int = 90, names: int = 120, seed: int = 0) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """A panel whose open(t+1)->open(t+2) return loads on a hidden factor known at t."""
    rng = np.random.default_rng(seed)
    dates = list(pd.bdate_range("2021-01-04", periods=days).strftime("%Y%m%d"))
    codes = [f"{i:06d}.SZ" for i in range(1, names + 1)]
    factor = rng.standard_normal((days, names))
    returns = rng.standard_normal((days, names)) * 0.02
    returns[1:] += 0.03 * factor[:-1]
    opens = 10.0 * np.cumprod(1.0 + np.vstack([np.zeros((1, names)), returns[:-1]]), axis=0)
    view = root / "view"
    _write_view(view, _panel_frame(opens, dates, codes))
    keys = {"trade_date": np.repeat(dates, names), "ts_code": np.tile(codes, days)}
    planted = pd.DataFrame({**keys, "score": factor.ravel()})
    noise = pd.DataFrame({**keys, "score": rng.standard_normal(days * names)})
    return view, planted, noise


def _signal_file(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _reads_parquet(path: Path, parquet: Path) -> Path:
    return _signal_file(
        path,
        "import pandas as pd\n\n\ndef compute_signal(frames):\n"
        f"    return pd.read_parquet({str(parquet)!r})\n",
    )


def _run(view: Path, signal: Path, **kwargs: object) -> dict[str, object]:
    frames = screen.Frames(view)
    manifest = screen.open_view(view)
    scores = screen.load_signal(signal, frames)
    options = {"horizons": [1, 5], "start": None, "end": None, "top_fraction": 0.1, "min_names": 20}
    options.update(kwargs)
    return screen.run_screen(frames, manifest, scores, **options)


def test_row_ranks_and_rank_ic_match_pandas_with_ties_and_nans() -> None:
    rng = np.random.default_rng(3)
    values = rng.integers(0, 6, size=(7, 40)).astype(float)  # many ties
    values[rng.random(values.shape) < 0.2] = np.nan
    values[3] = np.nan  # an all-NaN day
    frame = pd.DataFrame(values)
    np.testing.assert_allclose(screen.row_ranks(values), frame.rank(axis=1).to_numpy())

    other = pd.DataFrame(rng.normal(size=values.shape) + values / 3)
    other[other > 1.5] = np.nan
    ic = screen.rank_ic(frame, other, min_names=5)
    for day in range(len(frame)):
        pair = pd.concat([frame.iloc[day], other.iloc[day]], axis=1).dropna()
        expected = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman") if len(pair) >= 5 else np.nan
        assert (np.isnan(ic.iloc[day]) and np.isnan(expected)) or ic.iloc[day] == pytest.approx(expected)

    tradable = pd.DataFrame(True, index=frame.index, columns=frame.columns)
    tradable.iloc[0, :20] = False
    top = screen.top_selection(frame, tradable, 0.25)
    usable = int(frame.iloc[0, 20:].notna().sum())
    assert top.iloc[0].sum() == math.ceil(usable * 0.25) and not top.iloc[0, :20].any()
    assert top.iloc[3].sum() == 0
    assert frame.iloc[0].where(top.iloc[0]).min() >= frame.iloc[0].where(tradable.iloc[0] & ~top.iloc[0]).max()


def test_planted_signal_is_recovered_and_noise_is_not(tmp_path: Path) -> None:
    view, planted, noise = _synthetic_view(tmp_path)
    planted.to_parquet(tmp_path / "planted.parquet", index=False)
    noise.to_parquet(tmp_path / "noise.parquet", index=False)

    report = _run(view, _reads_parquet(tmp_path / "planted.py", tmp_path / "planted.parquet"))
    one, five = report["horizons"]
    assert one["horizon"] == 1 and one["n_days"] == 88  # the last two days have no t+2 open
    assert one["ic_mean"] > 0.5 and one["t_stat"] > 10
    assert one["positive_month_share"] == 1.0 and one["n_months"] == 5
    assert one["ic_size_neutral"] > 0.5
    assert one["top_excess_mean"] > 0.01 and one["top_excess_hit"] > 0.9
    # Only the first day of the five-day window loads on the factor, so the
    # marginal IC beyond day one is noise and the horizon profile decays.
    assert five["ic_mean"] < one["ic_mean"]
    assert abs(five["ic_marginal"]) < 0.1
    assert report["signal"]["finite_scores"] == 90 * 120
    assert report["coverage"]["universe_share_mean"] == 1.0
    assert report["tradability"]["up_limit_excluded_share"] == 0.0
    assert 0.8 < report["turnover"]["top_new_share_mean"] <= 1.0
    assert abs(report["turnover"]["rank_autocorr_mean"]) < 0.1

    report = _run(view, _reads_parquet(tmp_path / "noise.py", tmp_path / "noise.parquet"))
    one = report["horizons"][0]
    assert abs(one["ic_mean"]) < 0.05 and abs(one["t_stat"]) < 2.5


def test_forward_returns_use_next_open_adjusted_and_skip_untradable_entries(tmp_path: Path) -> None:
    dates = ["20210104", "20210105", "20210106", "20210107", "20210108"]
    codes = ["A", "B", "C"]
    # B splits 2:1 on the third day: the raw open halves and adj_factor doubles.
    opens = np.array(
        [
            [10.0, 20.0, 5.0],
            [11.0, 22.0, 5.5],
            [12.0, 12.0, 6.3],
            [12.5, 12.5, 6.6],
            [13.0, 13.0, 7.0],
        ]
    )
    adj = np.ones_like(opens)
    adj[2:, 1] = 2.0
    daily = _panel_frame(opens, dates, codes, adj=adj)
    # C opens at its up-limit on day 2 and is suspended on day 3.
    daily.loc[(daily.trade_date == "20210105") & (daily.ts_code == "C"), "up_limit"] = 5.5
    daily.loc[(daily.trade_date == "20210106") & (daily.ts_code == "C"), "is_suspended"] = True
    view = tmp_path / "view"
    _write_view(view, daily)
    panel = screen.Panel(screen.Frames(view), "20210108")

    fwd = screen.forward_returns(panel.adj_open, 1)
    assert fwd.loc["20210104", "A"] == pytest.approx(12.0 / 11.0 - 1)
    assert fwd.loc["20210104", "B"] == pytest.approx(24.0 / 22.0 - 1)  # split-adjusted, not 12/22
    assert fwd.loc["20210106", "A"] == pytest.approx(13.0 / 12.5 - 1)
    assert np.isnan(fwd.loc["20210107", "A"]) and np.isnan(fwd.loc["20210108", "A"])
    assert screen.forward_returns(panel.adj_open, 2).loc["20210104", "A"] == pytest.approx(12.5 / 11.0 - 1)
    assert screen.forward_returns(panel.adj_open, 2, 1).loc["20210104", "A"] == pytest.approx(12.5 / 12.0 - 1)

    assert bool(panel.tradable.loc["20210104", "C"]) is False  # t+1 open at the up-limit
    assert bool(panel.tradable.loc["20210105", "C"]) is False  # suspended at t+1
    assert bool(panel.tradable.loc["20210106", "C"]) is True
    assert bool(panel.tradable.loc["20210108", "A"]) is False  # no t+1 bar
    top = screen.top_selection(pd.DataFrame(1.0, index=panel.dates, columns=panel.codes), panel.tradable, 1.0)
    assert list(top.loc["20210104"]) == [True, True, False]

    # A wide score equal to the realised one-day forward return is a perfect ranking.
    fwd.to_parquet(tmp_path / "foresight.parquet")
    report = _run(view, _reads_parquet(tmp_path / "sig.py", tmp_path / "foresight.parquet"), horizons=[1], min_names=2)
    assert report["horizons"][0]["ic_mean"] == pytest.approx(1.0)
    assert report["horizons"][0]["n_days"] == 3
    assert report["tradability"]["up_limit_excluded_share"] == pytest.approx(1 / 9)

    wide = screen.Frames(view).wide("daily", "open", start="20210105", end="20210106")
    assert wide.shape == (2, 3) and wide.loc["20210106", "B"] == 12.0
    assert screen.Frames(view).wide("daily", "is_suspended").loc["20210106", "C"]


def test_only_decision_views_are_evaluated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dates = list(pd.bdate_range("2021-01-04", periods=4).strftime("%Y%m%d"))
    daily = _panel_frame(np.full((4, 2), 10.0), dates, ["A", "B"])
    replay = tmp_path / "replay"
    _write_view(replay, daily, kind="replay_slot", label="valid", period_start=dates[0], period_end=dates[-1])
    empty = tmp_path / "valid"
    empty.mkdir()
    signal = _signal_file(tmp_path / "sig.py", "def compute_signal(frames):\n    return frames.daily\n")

    assert screen.main(["--snapshot", str(replay), "--signal", str(signal)]) == 2
    assert "not 'decision_input'" in capsys.readouterr().err
    assert screen.main(["--snapshot", str(empty), "--signal", str(signal)]) == 2
    assert "no manifest.json" in capsys.readouterr().err

    future = tmp_path / "future"
    _write_view(future, daily, decision_time="2021-01-05T23:59:59+08:00")
    with pytest.raises(screen.ScreenError, match="after the decision date"):
        screen.Panel(screen.Frames(future), screen.decision_date(screen.open_view(future)))


def test_signal_contract_violations_fail_fast(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    view, planted, _ = _synthetic_view(tmp_path, days=30, names=40)
    frames = screen.Frames(view)
    manifest = screen.open_view(view)
    wide = planted.pivot(index="trade_date", columns="ts_code", values="score")

    with pytest.raises(screen.ScreenError, match="duplicate"):
        screen.normalise_signal(pd.concat([planted, planted.head(1)]))
    with pytest.raises(screen.ScreenError, match="no finite score"):
        screen.normalise_signal(planted.assign(score=np.nan))
    with pytest.raises(screen.ScreenError, match="DataFrame or Series"):
        screen.normalise_signal([1, 2, 3])
    with pytest.raises(screen.ScreenError, match="YYYYMMDD"):
        screen.normalise_signal(planted.assign(trade_date="2021-01-04"))
    with pytest.raises(screen.ScreenError, match="must be numeric"):
        screen.normalise_signal(planted.assign(score="high"))
    with pytest.raises(screen.ScreenError, match="single-level"):
        screen.normalise_signal(planted.set_index(["trade_date", "ts_code"]))
    with pytest.raises(screen.ScreenError, match="unique trade_date index"):
        screen.normalise_signal(pd.concat([wide, wide.head(1)]))
    with pytest.raises(screen.ScreenError, match="YYYYMMDD"):
        screen.normalise_signal(wide.rename(index=lambda d: f"{d[:4]}-{d[4:6]}-{d[6:]}"))
    with pytest.raises(screen.ScreenError, match="two-level"):
        screen.normalise_signal(planted["score"])
    leaked = planted.assign(trade_date=planted["trade_date"].mask(planted.index == 0, "20300101"))
    with pytest.raises(screen.ScreenError, match="outside the visible daily history"):
        screen.run_screen(
            frames, manifest, screen.normalise_signal(leaked),
            horizons=[1], start=None, end=None, top_fraction=0.1, min_names=2,
        )
    with pytest.raises(screen.ScreenError, match="outside the visible history"):
        screen.run_screen(
            frames, manifest, screen.normalise_signal(planted),
            horizons=[1], start="20300101", end=None, top_fraction=0.1, min_names=2,
        )
    with pytest.raises(screen.ScreenError, match="no column"):
        frames.load("daily", columns=["nope"])
    with pytest.raises(screen.ScreenError, match="no table"):
        frames.load("events")
    with pytest.raises(screen.ScreenError, match="key of the wide matrix"):
        frames.wide("daily", "ts_code")

    # Long, Series and wide forms with datetime dates all normalise to the same matrix.
    from_long = screen.normalise_signal(planted)
    as_series = planted.assign(trade_date=pd.to_datetime(planted["trade_date"])).set_index(["trade_date", "ts_code"])["score"]
    as_wide = wide.set_index(pd.to_datetime(wide.index))
    assert from_long.shape == (30, 40) and from_long.index[0] == "20210104"
    pd.testing.assert_frame_equal(screen.normalise_signal(as_series), from_long)
    pd.testing.assert_frame_equal(screen.normalise_signal(as_wide), from_long)

    no_entry = _signal_file(tmp_path / "no_entry.py", "x = 1\n")
    assert screen.main(["--snapshot", str(view), "--signal", str(no_entry)]) == 2
    assert "must define compute_signal" in capsys.readouterr().err
    assert screen.main(["--snapshot", str(view), "--signal", str(tmp_path / "missing.py")]) == 2
    assert "not found" in capsys.readouterr().err
    assert screen.main(["--snapshot", str(view), "--signal", str(no_entry), "--horizons", "0"]) == 2
    assert "positive integers" in capsys.readouterr().err


def test_cli_runs_standalone_and_prints_json(tmp_path: Path) -> None:
    source = SCREENING_TOOL_SOURCE.read_text(encoding="utf-8")
    imported = set(re.findall(r"^(?:from|import)\s+([A-Za-z_]+)", source, re.MULTILINE))
    assert imported <= {"__future__", "argparse", "importlib", "json", "math", "re", "sys", "time", "pathlib", "numpy", "pandas", "pyarrow"}

    view, planted, _ = _synthetic_view(tmp_path, days=40, names=60)
    planted.to_parquet(tmp_path / "planted.parquet", index=False)
    signal = _signal_file(
        tmp_path / "sig.py",
        "import pandas as pd\n\n\ndef compute_signal(frames):\n"
        "    assert 'daily' in frames.names and 'open' in frames.columns('daily')\n"
        "    window = frames.wide('daily', 'close', start='20210201', end='20210205')\n"
        "    assert list(window.index) == ['20210201', '20210202', '20210203', '20210204', '20210205']\n"
        "    assert window.shape == (5, 60)\n"
        f"    planted = pd.read_parquet({str(tmp_path / 'planted.parquet')!r})\n"
        "    return planted.pivot(index='trade_date', columns='ts_code', values='score')\n",
    )
    completed = subprocess.run(
        [sys.executable, str(SCREENING_TOOL_SOURCE), "--snapshot", str(view), "--signal", str(signal), "--json", "--horizons", "3,1", "--min-names", "10"],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert [row["horizon"] for row in report["horizons"]] == [1, 3]
    assert report["snapshot"]["kind"] == "decision_input" and report["snapshot"]["trade_days"] == 40
    assert report["window"] == {"start": "20210104", "end": "20210226", "trade_days": 40, "top_fraction": 0.1, "min_names": 10}
    assert report["horizons"][0]["ic_mean"] > 0.5
    assert report["wall_seconds"] > 0

    table = subprocess.run(
        [sys.executable, str(SCREENING_TOOL_SOURCE), "--snapshot", str(view), "--signal", str(signal), "--min-names", "10"],
        capture_output=True, text=True, check=False, cwd=tmp_path,
    )
    assert table.returncode == 0, table.stderr
    assert "ic_size_neutral" in table.stdout and "wall time:" in table.stdout
