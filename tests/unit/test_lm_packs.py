"""The 20261019 packs: one language-model baseline under two continual-learning proposals.

What must hold. The two packs share the starter, the families record and the
rules file byte for byte, since a pack is mounted alone and cannot point at
another. Each starter is a valid strategy package and no pack cites a reading
after the research period. The text helper dates a title by the first
decision that can see it under the nightly node rule, refuses a title the
decision could not see, collapses the vendor's prefixed duplicate onto the
plain title, strips the issuer from a title, and keeps the newest titles per
name within the text window. A name without a bar on a day is left out of
that day's standardisation, so a tradable name's features do not depend on how
far back the panel reads. The daily fit routes to the hook in both modes and
refits from scratch only when the cold-quarter mode meets a new quarter.
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
from autotrade.environment.strategy_loader import validate_strategy_package

REPO_ROOT = Path(__file__).resolve().parents[2]
REFS = REPO_ROOT / "configs" / "workspace_refs"
PACKS = [REFS / "hcm_lm_100k_8y_20261019", REFS / "egcu_lm_100k_8y_20261019"]
SHARED = ("families.md", "references/open-rules.md")
# A decision on Tuesday 2024-10-08 08:30, the first trading day after the
# National Day holiday; the Monday before it, 09-30, was the previous one.
CALENDAR = ["20240924", "20240925", "20240926", "20240927", "20240930", "20241008", "20241009"]
DECISION = datetime(2024, 10, 8, 8, 30, tzinfo=CN_TZ)


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    }


def _drop_packages() -> None:
    for name in [module for module in sys.modules if module.split(".", 1)[0] == "lib"]:
        sys.modules.pop(name)


@pytest.fixture
def lib(monkeypatch: pytest.MonkeyPatch):
    _drop_packages()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(PACKS[0] / "starter"))
    from lib import features, knobs, text

    yield SimpleNamespace(features=features, knobs=knobs, text=text)
    _drop_packages()


def test_the_two_packs_share_the_starter_and_the_contract_files() -> None:
    assert _files(PACKS[1] / "starter") == _files(PACKS[0] / "starter")
    for name in SHARED:
        assert (PACKS[1] / name).read_bytes() == (PACKS[0] / name).read_bytes(), name


@pytest.mark.parametrize("pack", PACKS, ids=lambda path: path.name)
def test_the_starter_is_a_valid_package(pack: Path) -> None:
    assert validate_strategy_package(pack / "starter" / "main.py").refit_period == "day"


@pytest.mark.parametrize("pack", PACKS, ids=lambda path: path.name)
def test_no_pack_cites_a_reading_after_the_research_period(pack: Path) -> None:
    for path in pack.rglob("*"):
        if path.is_file() and path.suffix in (".md", ".py"):
            text = path.read_text(encoding="utf-8")
            for word in ("前推", "Held-out", "held-out", "forward IR", "F2 下界", "样本外读数"):
                assert word not in text, (pack.name, path.name, word)


def test_a_title_is_dated_by_the_first_decision_after_the_node_that_publishes_it(lib) -> None:
    stamps = [
        "2024-09-26 23:59:59+08:00",  # date-only Thursday: Saturday's node, Monday's decision
        "2024-09-29 10:00:00+08:00",  # receipt on a Sunday morning: that night's node, Monday
        "2024-10-06 23:59:59+08:00",  # date-only inside the holiday: the first day after it
        "2024-10-07 23:15:00+08:00",  # stamped at the node's cutoff: still that night's node
        "2024-10-07 23:15:01+08:00",  # one second later: the next night's node
        "2024-10-08 23:59:59+08:00",  # date-only Tuesday: Thursday, past this calendar
        None,
    ]
    assert lib.text.decision_days(stamps, CALENDAR) == [
        "20240930", "20240930", "20241008", "20241008", "20241009", None, None,
    ]


def _context(root: Path, rows, when=DECISION) -> SimpleNamespace:
    (root / "text_index").mkdir(exist_ok=True)
    pd.DataFrame(rows, columns=["dataset", "ts_codes", "title", "available_at"]).to_parquet(
        root / "text_index" / "part-0.parquet", index=False
    )
    (root / "universe").mkdir(exist_ok=True)
    pd.DataFrame({"ts_code": ["000001.SZ", "000002.SZ"], "name": ["平安银行", "*ST万科"]}).to_parquet(
        root / "universe" / "part_0000.parquet", index=False
    )
    return SimpleNamespace(asof_dir=str(root), inference_at=when)


def test_read_dedupes_the_prefixed_copy_anonymises_and_dates_each_row(lib, tmp_path: Path) -> None:
    rows = [
        ("anns_d", "000001.SZ", "关于回购股份实施完毕的公告", "2024-10-06 23:59:59+08:00"),
        ("anns_d", "000001.SZ", "平安银行：关于回购股份实施完毕的公告", "2024-10-06 23:59:59+08:00"),  # the vendor's copy
        ("anns_d", "000002.SZ", "万科：000002万科", "2024-10-06 23:59:59+08:00"),  # nothing but the issuer
        ("anns_d", "000002.SZ", "关于对深圳万科股份有限公司的问询函", "2024-09-27 23:59:59+08:00"),  # Monday's
        ("anns_d", "000002.SZ", "2024年第三季度报告", "2024-09-20 23:59:59+08:00"),  # before `since`
        ("other", "000001.SZ", "关于回购股份实施完毕的公告", "2024-10-06 23:59:59+08:00"),
    ]
    out = lib.text.read(_context(tmp_path, rows), CALENDAR[:-1], "20240930")
    assert out.to_dict("records") == [
        {"ts_code": "000001.SZ", "day": "20241008", "title": "关于回购股份实施完毕的公告"},
        {"ts_code": "000002.SZ", "day": "20240930", "title": "关于对公司的问询函"},
    ]
    with pytest.raises(ValueError, match="after its first day"):
        lib.text.read(_context(tmp_path, rows), CALENDAR[:-1], CALENDAR[0])


def test_read_dates_an_early_anchor_title_by_the_decision_and_refuses_a_later_stamp(lib, tmp_path: Path) -> None:
    """A research year's first view is cut at the anchor instant, so a title
    dated the anchor day is there a node before the rule would publish it:
    the view decides, and the row is dated by the decision that holds it."""
    anchor_day = [("anns_d", "000001.SZ", "股票交易异常波动公告", "2024-10-07 23:59:59+08:00")]
    out = lib.text.read(_context(tmp_path, anchor_day), CALENDAR[:-1], "20240930")
    assert out["day"].tolist() == ["20241008"]
    with pytest.raises(ValueError, match="must end with the decision day"):
        lib.text.read(_context(tmp_path, anchor_day), CALENDAR, "20240930")
    stamped_later = [("anns_d", "000001.SZ", "股票交易异常波动公告", "2024-10-08 09:00:00+08:00")]
    with pytest.raises(RuntimeError, match="stamped after"):
        lib.text.read(_context(tmp_path, stamped_later), CALENDAR[:-1], "20240930")


def test_anonymise_strips_the_issuer_and_keeps_the_event(lib) -> None:
    cases = [
        ("博天环境：关于子公司出售资产的公告", "博天环境", "关于子公司出售资产的公告"),
        ("正邦科技：002157正邦科技业绩说明会、路演活动信息20210620", "正邦科技", "业绩说明会、路演活动信息20210620"),
        ("关于对中电电机股份有限公司重大资产置换预案的二次问询函", "中电电机", "关于对公司重大资产置换预案的二次问询函"),
        ("[临时公告]贝特瑞:关于延期披露2018年年度报告的提示性公告", "贝特瑞", "关于延期披露2018年年度报告的提示性公告"),
        ("沃特股份：沃特股份：2021年6月18日投资者关系活动记录表", "沃特股份", "2021年6月18日投资者关系活动记录表"),
        ("*ST凯迪：关于持有限售股股东减持的公告", "*ST凯迪", "关于持有限售股股东减持的公告"),
        ("300848美瑞新材20200620关注函", "美瑞新材", "20200620关注函"),
        ("股票交易异常波动公告", "", "股票交易异常波动公告"),
    ]
    assert lib.text.anonymise([c[0] for c in cases], [c[1] for c in cases]) == [c[2] for c in cases]


def test_table_keeps_the_newest_titles_within_the_text_window(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lib.knobs, "TEXT_DAYS", 2)
    monkeypatch.setattr(lib.knobs, "MAX_TITLES", 2)
    rows = pd.DataFrame({
        "ts_code": ["A", "A", "A", "A", "B", "Z"],
        "day": ["d1", "d1", "d2", "d3", "d3", "d1"],
        "title_id": [0, 1, 2, 3, 4, 5],
    })
    out = lib.text.table(rows, ["A", "B"], ["d1", "d2", "d3"])
    assert out.tolist() == [
        [[0, 1], [-1, -1]],  # d1: A's two titles of the day
        [[2, 0], [-1, -1]],  # d2: the new title first, then one of d1's
        [[3, 2], [4, -1]],  # d3: d1 has left the window
    ]


def test_a_long_suspended_name_leaves_the_others_features_alone(lib, tmp_path: Path) -> None:
    """The last name trades only on the first 15 of 100 days: a 100-day panel
    holds its forward-filled close, a 70-day one never sees it, and the five
    names trading throughout must read the same features in both."""
    rng = np.random.default_rng(7)
    days = [day.strftime("%Y%m%d") for day in pd.bdate_range("2024-01-02", periods=100)]
    codes = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ", "000009.SZ"]
    frames = []
    for i, code in enumerate(codes):
        n = 15 if code == "000009.SZ" else len(days)
        close = 10.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02 * (i + 1), n)))
        frames.append(pd.DataFrame({
            "ts_code": code, "trade_date": days[:n], "open": close * 0.99, "close": close, "adj_factor": 1.0,
            "amount": rng.uniform(1e4, 1e5, n), "turnover_rate": rng.uniform(0.5, 5.0, n),
            "circ_mv": rng.uniform(1e5, 1e6, n), "pb": rng.uniform(0.5, 5.0, n),
        }))
    (tmp_path / "daily").mkdir()
    pd.concat(frames).to_parquet(tmp_path / "daily" / "part-0.parquet", index=False)
    (tmp_path / "macro").mkdir()
    level = 5000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(days))))
    pd.DataFrame({
        "dataset": "index_daily", "ts_code": lib.knobs.INDEX, "trade_date": days, "open": level, "close": level,
        "available_at": [f"{day[:4]}-{day[4:6]}-{day[6:]} 17:00:00+08:00" for day in days],
    }).to_parquet(tmp_path / "macro" / "part-0.parquet", index=False)
    context = SimpleNamespace(asof_dir=str(tmp_path), inference_at=datetime(2024, 5, 21, 8, 30, tzinfo=CN_TZ))

    long, short = lib.features.panel(context, 100), lib.features.panel(context, 70)
    assert long.dates[-1] == short.dates[-1] == days[-1] and short.codes == codes[:-1]
    # The last ten days are past the 60-day history of the short panel too.
    np.testing.assert_allclose(long.x[-10:, :-1], short.x[-10:], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(long.valid[-10:, :-1], short.valid[-10:])
    assert not long.x[15:, -1].any() and not long.valid[15:, -1].any()


@pytest.fixture
def learn(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    _drop_packages()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(PACKS[0] / "starter"))
    from lib import knobs, learn, model, update

    yield SimpleNamespace(knobs=knobs, learn=learn, model=model, update=update)
    _drop_packages()


def test_gather_averages_the_present_titles_and_flags_names_with_text(learn) -> None:
    import torch

    table = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 8.0]], dtype=torch.float16)
    e, has = learn.model.gather(table, torch.tensor([[0, 1], [2, -1], [-1, -1]]))
    assert e.tolist() == [[2.0, 0.0], [0.0, 8.0], [0.0, 0.0]]
    assert has.tolist() == [1.0, 1.0, 0.0]


def test_the_daily_fit_refits_cold_at_a_new_quarter_and_otherwise_reaches_the_hook(
    learn, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def pretrain(context, dev):
        calls.append("pretrain")
        return {"model": learn.model.Head(4), "dim": 4, "quarter": "", "last_day": "20240905", "steps": 0, "extra": {}}

    monkeypatch.setattr(learn.learn, "pretrain", pretrain)
    monkeypatch.setattr(learn.update, "daily", lambda context, state: calls.append(f"hook:{state['quarter']}"))
    monkeypatch.setattr(learn.knobs, "TEXT", False)

    def context(day: str) -> SimpleNamespace:
        return SimpleNamespace(
            inference_at=datetime(int(day[:4]), int(day[4:6]), int(day[6:]), 8, 30, tzinfo=CN_TZ),
            state_dir=str(tmp_path),
        )

    monkeypatch.setattr(learn.knobs, "UPDATE", "cold_quarter")
    learn.learn.fit(context("20240927"))
    learn.learn.fit(context("20240930"))
    learn.learn.fit(context("20241008"))
    assert calls == ["pretrain", "hook:2024Q3", "pretrain"]
    monkeypatch.setattr(learn.knobs, "UPDATE", "frozen")
    learn.learn.fit(context("20250102"))
    assert calls[-1] == "hook:2024Q4"
    monkeypatch.setattr(learn.knobs, "TEXT", True)
    with pytest.raises(RuntimeError, match="encoder"):
        learn.learn.fit(context("20250103"))
    monkeypatch.setattr(learn.knobs, "UPDATE", "daily")
    with pytest.raises(ValueError, match=r"knobs\.UPDATE"):
        learn.learn.fit(context("20250103"))
