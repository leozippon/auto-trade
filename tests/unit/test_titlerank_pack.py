"""The titlerank_alla_100k_8y_20261012 starter: a ranker on announcement titles in an event book.

What must hold: the control c_rule is the incdraft lane's book at the same
seats, order for order, so the lane's comparison with the one rule we trust
is like for like; the ridge twin, trained from the view alone, finds a word
that moves prices and the book buys exactly the names whose fresh titles
carry it; a holding keeps the event it was bought on through later refits,
because each review is scored once and recorded; a review the fit did not
record stops the book instead of trading nothing; and the net reads a unit
as the maximum over its texts, whatever their order or padding. The starter
runs as a session runs it, ``lib.*`` imported from the pack's own directory,
over point-in-time views written to disk, one per decision.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.strategy import CN_TZ

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKS = REPO_ROOT / "configs" / "workspace_refs"
STARTER = PACKS / "titlerank_alla_100k_8y_20261012" / "starter"
INCDRAFT = PACKS / "incdraft_alla_8y_20261012" / "starter"
DRAFT = "关于公司限制性股票激励计划（草案）的公告"


def _drop_lib() -> None:
    for name in [module for module in sys.modules if module == "lib" or module.startswith("lib.")]:
        sys.modules.pop(name)


def _load(monkeypatch: pytest.MonkeyPatch, starter: Path) -> SimpleNamespace:
    """The starter's ``lib`` modules, imported from its own directory."""
    _drop_lib()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(starter))
    import lib.book
    import lib.knobs

    modules = SimpleNamespace(book=lib.book, knobs=lib.knobs)
    if starter == STARTER:
        import lib.events
        import lib.label
        import lib.models
        import lib.titles

        modules.events, modules.label, modules.models, modules.titles = lib.events, lib.label, lib.models, lib.titles
    sys.path.remove(str(starter))
    _drop_lib()
    return modules


@pytest.fixture(autouse=True)
def _clean_lib():
    yield
    _drop_lib()


def _views(root: Path, daily: pd.DataFrame, universe: pd.DataFrame, text: pd.DataFrame, day: str) -> Path:
    """The as-of view a decision on `day` (YYYYMMDD) reads: bars before it, titles stamped before it."""
    view = root / f"view_{day}"
    stamp = pd.to_datetime(text["available_at"], utc=True)
    frames = {
        "daily": daily[daily["trade_date"] < day],
        "universe": universe,
        "text_index": text[stamp < pd.Timestamp(day, tz=CN_TZ)],
    }
    for name, frame in frames.items():
        (view / name).mkdir(parents=True)
        frame.to_parquet(view / name / "part-0.parquet", index=False)
    return view


def _context(view: Path, state: Path, day: str, cash: float, positions: dict[str, int]) -> SimpleNamespace:
    return SimpleNamespace(
        inference_at=datetime.strptime(day, "%Y%m%d").replace(hour=8, minute=30, tzinfo=CN_TZ),
        asof_dir=str(view),
        state_dir=str(state),
        account=SimpleNamespace(cash=cash, positions=positions),
    )


def _orders(orders: list[dict[str, object]]) -> list[tuple[object, ...]]:
    return [(order["symbol"], order["action"], order["quantity"], order["execute_at"]) for order in orders]


def test_c_rule_trades_as_the_incdraft_book_at_the_same_seats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Drafts fresh at one review are bought, and sold at the review HOLD trading days on, with the
    orders the incdraft 100k book sends at 20 seats: c_rule is that rule in this book."""
    codes = {"300001.SZ": (33.0, "电子"), "600001.SH": (30.0, "机械设备"), "000001.SZ": (12.0, "银行"),
             "002001.SZ": (25.0, "医药生物"), "688001.SH": (50.0, "计算机")}
    days = pd.bdate_range("2023-10-02", "2024-05-10").strftime("%Y%m%d")
    daily = pd.DataFrame([
        {"ts_code": code, "trade_date": day, "close": close, "amount": 5.0e7, "is_suspended": False}
        for code, (close, _) in codes.items() for day in days
    ])
    universe = pd.DataFrame([{"ts_code": code, "name": f"样本{i}", "l1_name": industry}
                             for i, (code, (_, industry)) in enumerate(codes.items())])
    text = pd.DataFrame(
        [{"dataset": "anns_d", "ts_codes": code, "title": DRAFT, "available_at": "2024-03-05 23:59:59+08:00"}
         for code in ("300001.SZ", "600001.SH", "688001.SH")]
        + [{"dataset": "anns_d", "ts_codes": code, "title": DRAFT, "available_at": "2024-04-30 23:59:59+08:00"}
           for code in ("000001.SZ", "002001.SZ")]
    )
    mine = _load(monkeypatch, STARTER)
    theirs = _load(monkeypatch, INCDRAFT)
    monkeypatch.setattr(mine.knobs, "MODEL", "rule")
    monkeypatch.setattr(theirs.knobs, "SEATS", 20)
    state = tmp_path / "state"
    state.mkdir()
    cash, positions = 100_000.0, {}
    # 2024-05-06 is the first review at least 40 trading days after 2024-03-11.
    for day, expect_sells in (("20240311", 0), ("20240506", 2)):
        view = _views(tmp_path, daily, universe, text, day)
        context = _context(view, state, day, cash, dict(positions))
        mine.events.update(context)
        ours = mine.book.run(context)
        assert _orders(ours) == _orders(theirs.book.run(context))
        assert sum(order["action"] == "sell" for order in ours) == expect_sells
        for order in ours:
            quantity = int(order["quantity"])
            if order["action"] == "buy":
                positions[str(order["symbol"])] = quantity
                cash -= quantity * codes[str(order["symbol"])][0]
            else:
                positions.pop(str(order["symbol"]))
                cash += quantity * codes[str(order["symbol"])][0]
    assert sorted(positions) == ["000001.SZ", "002001.SZ"]


def _planted_world(seed: int = 7):
    """60 names over 15 months, one title each a week. A name whose title says 吉祥 drifts up
    by 1 % a day for 40 trading days from the close of the review it is fresh at."""
    rng = np.random.default_rng(seed)
    codes = [f"{600000 + i}.SH" for i in range(30)] + [f"{300000 + i}.SZ" for i in range(1, 31)]
    days = pd.bdate_range("2016-01-04", "2017-04-28")
    neutral = ["关于召开股东大会的通知", "关于签订日常经营合同的公告", "董事会决议公告", "关于对外投资的公告", "关于公司章程修订的公告"]
    returns = rng.normal(0.0, 0.01, (len(days), len(codes)))
    rows = []
    mondays = [i for i in range(1, len(days)) if days[i].isocalendar()[1] != days[i - 1].isocalendar()[1]]
    planted_last = []
    for review in mondays:
        # Titles stamped on the Tuesday before: fresh at this review.
        stamp = (days[review] - pd.Timedelta(days=6)).strftime("%Y-%m-%d") + " 23:59:59+08:00"
        planted = set(rng.choice(len(codes), 3, replace=False).tolist())
        for j, code in enumerate(codes):
            title = "关于吉祥事项的公告" if j in planted else neutral[int(rng.integers(len(neutral)))]
            rows.append({"dataset": "anns_d", "ts_codes": code, "title": title, "available_at": stamp})
            if j in planted:
                returns[review + 1: review + 41, j] += 0.01
        planted_last = sorted(codes[j] for j in planted)
    close = 10.0 * np.exp(np.cumsum(returns, axis=0))
    daily = pd.DataFrame({
        "ts_code": np.tile(codes, len(days)),
        "trade_date": np.repeat(days.strftime("%Y%m%d"), len(codes)),
        "close": close.ravel(),
        "up_limit": close.ravel() * 1.1,
        "adj_factor": 1.0,
        "circ_mv": np.tile(rng.uniform(1e5, 1e6, len(codes)), len(days)),
        "is_suspended": False,
    })
    universe = pd.DataFrame({"ts_code": codes, "name": [f"样本{i}" for i in range(len(codes))],
                             "l1_name": [f"行业{i % 12}" for i in range(len(codes))]})
    return daily, universe, pd.DataFrame(rows), days, mondays, planted_last


def test_the_ridge_twin_learns_a_planted_word_and_the_book_buys_its_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    daily, universe, text, days, mondays, planted = _planted_world()
    lib = _load(monkeypatch, STARTER)
    monkeypatch.setattr(lib.knobs, "MODEL", "ridge")
    state = tmp_path / "state"
    state.mkdir()
    day = days[mondays[-1]].strftime("%Y%m%d")
    view = _views(tmp_path, daily, universe, text, day)
    context = _context(view, state, day, 100_000.0, {})
    lib.events.update(context)
    recorded = pd.read_parquet(state / f"events_{day}.parquet")
    assert sorted(recorded["ts_code"]) == planted
    assert np.load(state / "events_index.npy").tolist() == [[int(day), 60, 3]]
    assert sorted(order["symbol"] for order in lib.book.run(context)) == planted


def test_a_holding_keeps_its_recorded_event_and_a_missing_record_stops_the_book(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The book reads what the fit recorded at each review, never a rescoring under a later model;
    a review the fit did not record is an error, not an empty book."""
    lib = _load(monkeypatch, STARTER)
    days = pd.bdate_range("2024-01-02", "2024-03-29").strftime("%Y%m%d")
    daily = pd.DataFrame([{"ts_code": code, "trade_date": day, "close": 10.0, "is_suspended": False}
                          for code in ("600001.SH", "600002.SH") for day in days])
    universe = pd.DataFrame({"ts_code": ["600001.SH", "600002.SH"], "name": ["甲", "乙"], "l1_name": ["银行", "电子"]})
    view = _views(tmp_path, daily, universe, pd.DataFrame(columns=["dataset", "ts_codes", "title", "available_at"]), "20240325")
    state = tmp_path / "state"
    state.mkdir()
    # Recorded at the review of 2024-02-26 (20 trading days ago); today's review recorded nothing new.
    pd.DataFrame({"ts_code": ["600001.SH"], "score": [0.3]}).to_parquet(state / "events_20240226.parquet")
    pd.DataFrame({"ts_code": pd.Series(dtype=object), "score": pd.Series(dtype=float)}).to_parquet(
        state / "events_20240325.parquet")
    context = _context(view, state, "20240325", 50_000.0, {"600001.SH": 4_000, "600002.SH": 1_000})
    np.save(state / "events_index.npy", np.array([[20240226, 900, 1]], dtype=np.int64))
    with pytest.raises(RuntimeError, match="recorded no events for the review of 20240325"):
        lib.book.run(context)
    np.save(state / "events_index.npy", np.array([[20240226, 900, 1], [20240325, 950, 0]], dtype=np.int64))
    orders = lib.book.run(context)
    # 600001.SH is inside its hold; 600002.SH has no recorded event in the window and is sold.
    assert _orders(orders) == [("600002.SH", "sell", 1_000, "2024-03-25T09:30:00+08:00")]


def test_the_net_reads_a_unit_as_its_strongest_text(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    lib = _load(monkeypatch, STARTER)
    models = lib.models
    vocab = models.vocabulary([["关于签订合同的公告", "限制性股票激励计划草案"]] * models.MIN_CHARS)
    torch.manual_seed(0)
    net = models.CharCNN(len(vocab)).eval()
    texts = ["关于签订合同的公告", "限制性股票激励计划草案", "董事会决议"]
    units = [texts, texts[::-1], [texts[0]], [texts[1]], [texts[2]], texts + ["董事会决议"]]
    ids, start, count = models.encode(units, vocab)
    with torch.no_grad():
        scores = net(models._tokens(torch.from_numpy(ids), start, count, np.arange(len(units)), "cpu")).numpy()
        alone = [net(models._tokens(torch.from_numpy(ids), start, count, np.array([i]), "cpu")).item()
                 for i in range(len(units))]
    # Order, padding to a wider batch and a repeated text change nothing.
    np.testing.assert_allclose(scores, alone, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(scores[0], scores[1], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(scores[0], scores[5], rtol=1e-5, atol=1e-6)


def test_the_label_is_size_matched_and_a_locked_close_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each name's 40-day return less its float-cap quintile's mean at the entry close; a name
    locked at the upper limit that close cannot be bought, so it has no label but stays in the pool."""
    lib = _load(monkeypatch, STARTER)
    names, t = 10, 1
    forward = np.linspace(-0.2, 0.25, names)
    adj = np.ones((t + lib.knobs.HOLD + 1, names))
    adj[t + lib.knobs.HOLD] = 1.0 + forward
    locked = np.zeros(adj.shape, dtype=bool)
    locked[t, 9] = True
    data = {"adj": adj, "bar": np.ones(adj.shape, dtype=bool), "locked": locked,
            "mv": np.tile(np.arange(1.0, names + 1), (adj.shape[0], 1))}
    out = lib.label.abnormal(data, t)
    # Float-cap percentile ranks 0.1 .. 1.0 fall in quintiles {0}, {1, 2}, {3, 4}, {5, 6}, {7, 8, 9}
    # (the designer's bucketing); the locked name 9 is in its quintile's mean but has no label.
    expected = np.empty(names)
    for group in ([0], [1, 2], [3, 4], [5, 6], [7, 8, 9]):
        expected[group] = forward[group] - forward[group].mean()
    expected[9] = np.nan
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_titles_lose_the_company_and_the_digits(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = _load(monkeypatch, STARTER)
    titles = pd.Series([
        "开尔新材：第三届董事会第十一次（临时）会议决议公告",
        "山东步长制药股份有限公司关于拟对外投资设立控股子公司的公告",
        "北京金自天正智能控制股份有限公司2022年度内部控制审计报告",
        "关于持股5%以上股东减持计划的公告",
    ])
    assert lib.titles.normalize(titles).tolist() == [
        "第三届董事会第十一次（临时）会议决议公告",
        "关于拟对外投资设立控股子公司的公告",
        "0000年度内部控制审计报告",
        "关于持股0%以上股东减持计划的公告",
    ]


def test_every_leg_is_named_and_two_controls_at_once_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    lib = _load(monkeypatch, STARTER)
    knobs = lib.knobs
    cases = [
        ({}, "nn"), ({"SEED": 2000}, "nn_s2"), ({"MODEL": "ridge"}, "ridge"),
        ({"LABELS": "shuffled"}, "c_shuf"), ({"TEXT": "blind", "SEED": 3000}, "c_blind_s3"),
        ({"MODEL": "ridge", "LABELS": "shuffled"}, "c_shuf_r"), ({"MODEL": "rule"}, "c_rule"),
    ]
    for changes, name in cases:
        with monkeypatch.context() as patch:
            for key, value in changes.items():
                patch.setattr(knobs, key, value)
            assert knobs.leg() == name
    for changes in ({"LABELS": "shuffled", "TEXT": "blind"}, {"MODEL": "rule", "SEED": 2000}, {"SEED": 1234}):
        with monkeypatch.context() as patch:
            for key, value in changes.items():
                patch.setattr(knobs, key, value)
            with pytest.raises(ValueError):
                knobs.leg()
