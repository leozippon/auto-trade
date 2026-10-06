"""The incdraft2_alla_100k_8y_20261013 pack: batch 1 replays the predecessor's bytes, stage 2 splits the same events.

What must hold. Batch 1 re-validates the predecessor's b20, b30, c_late20 and
c_late30 under a board-matched panel, and only byte-identical legs count once
in the lineage's trial family: the starter with its seat and leg lines edited
as the predecessor arms edited them, next to the strategy contract, must carry
the fingerprints those arms recorded. Stage 2 partitions the very events the
unsplit book trades by whether a title of the event's day names a stock
option, the delay control enters later than first visibility, and no stage-2
setting can stand in for a batch-1 leg. The pack never cites a reading after
the research period.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from autotrade.environment.artifacts import artifact_fingerprint
from autotrade.environment.strategy import CN_TZ

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK = REPO_ROOT / "configs" / "workspace_refs" / "incdraft2_alla_100k_8y_20261013"
PREDECESSOR = REPO_ROOT / "configs" / "workspace_refs" / "incdraft_alla_8y_20261012"
CONTRACT = REPO_ROOT / "configs" / "agent_output_template" / "README.md"

# Fingerprints the predecessor's 100k arms recorded (both arms, identical).
RECORDED = {
    ("SEATS = 20", 'CANDIDATE = "event"'): "c54c7c96eaca0211411e541d9c634342e0619f9a6573505393f0252dd9f247ec",
    ("SEATS = 30", 'CANDIDATE = "event"'): "718cbf17a0b8898b1dd7f39a9e4d27adc17afc11a8f4dd3a3064df4eb331f878",
    ("SEATS = 20", 'CANDIDATE = "late"'): "b7dfc8e7c2baf2b6e66d91aa38e8b0e67db6f5d74c89225c98e24c138ed27038",
    ("SEATS = 30", 'CANDIDATE = "late"'): "a1e2a20b1d61aa0c80e94afa869b5d6440739154a490be036e87428b84a3e151",
}

# Code: (T-1 close, Shenwan L1 industry, title, stamp date). A decision on Monday 2024-03-11.
NAMES = {
    "000001.SZ": (10.0, "银行", "2024年限制性股票激励计划（草案）", "2024-03-05"),
    "000002.SZ": (10.0, "房地产", "2024年股票期权激励计划（草案）", "2024-03-05"),
    "000003.SZ": (10.0, "电子", "2024年股票期权与限制性股票激励计划（草案）", "2024-03-05"),
    "000004.SZ": (10.0, "机械设备", "第二期股权激励计划（草案）", "2024-03-05"),
    # Visible Friday 2024-03-01: bought at the 2024-03-04 review, so not fresh
    # on 03-11 -- unless the delay control moves it three trading days later.
    "000005.SZ": (10.0, "计算机", "2024年限制性股票激励计划（草案）", "2024-02-29"),
}


def test_the_starter_is_the_predecessor_starter_byte_for_byte() -> None:
    ours = sorted(p.relative_to(PACK / "starter") for p in (PACK / "starter").rglob("*") if p.is_file())
    theirs = sorted(p.relative_to(PREDECESSOR / "starter") for p in (PREDECESSOR / "starter").rglob("*") if p.is_file())
    assert ours == theirs
    for relative in ours:
        assert (PACK / "starter" / relative).read_bytes() == (PREDECESSOR / "starter" / relative).read_bytes(), relative


@pytest.mark.parametrize(("lines", "fingerprint"), list(RECORDED.items()))
def test_batch_one_legs_carry_the_predecessor_fingerprints(tmp_path: Path, lines: tuple[str, str], fingerprint: str) -> None:
    output = tmp_path / "output"
    shutil.copytree(PACK / "starter", output)
    shutil.copy2(CONTRACT, output / "README.md")
    knobs = output / "lib" / "knobs.py"
    text = knobs.read_text(encoding="utf-8")
    for line in lines:
        key = line.split(" = ")[0]
        [old] = [row for row in text.splitlines() if row.startswith(key + " = ")]
        text = text.replace(old, line)
    knobs.write_text(text, encoding="utf-8")
    assert artifact_fingerprint(output) == fingerprint


def _drop_lib() -> None:
    for name in [module for module in sys.modules if module == "lib" or module.startswith("lib.")]:
        sys.modules.pop(name)


@pytest.fixture
def lib(monkeypatch: pytest.MonkeyPatch):
    _drop_lib()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(PACK / "stage2"))
    from lib import book, knobs

    yield SimpleNamespace(book=book, knobs=knobs)
    _drop_lib()


def _view(root: Path) -> Path:
    days = pd.bdate_range("2023-10-02", "2024-03-08").strftime("%Y%m%d")
    daily = pd.DataFrame([
        {"ts_code": code, "trade_date": day, "close": close, "amount": 5.0e7, "is_suspended": False}
        for code, (close, *_) in NAMES.items() for day in days
    ])
    universe = pd.DataFrame([{"ts_code": code, "name": f"样本{i}", "l1_name": row[1]} for i, (code, row) in enumerate(NAMES.items())])
    text = pd.DataFrame([
        {"dataset": "anns_d", "ts_codes": code, "title": title, "available_at": f"{stamp} 23:59:59+08:00"}
        for code, (_, _, title, stamp) in NAMES.items()
    ])
    for name, frame in (("daily", daily), ("universe", universe), ("text_index", text)):
        (root / name).mkdir()
        frame.to_parquet(root / name / "part-0.parquet", index=False)
    return root


def _bought(lib, root: Path, monkeypatch: pytest.MonkeyPatch, **switches: object) -> set[str]:
    for key, value in switches.items():
        monkeypatch.setattr(lib.knobs, key, value)
    context = SimpleNamespace(
        inference_at=datetime(2024, 3, 11, 8, 30, tzinfo=CN_TZ), asof_dir=str(root),
        account=SimpleNamespace(cash=100_000.0, positions={}),
    )
    return {order["symbol"] for order in lib.book.run(context) if order["action"] == "buy"}


def test_the_sides_partition_the_unsplit_events(lib, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _view(tmp_path)
    assert lib.knobs.leg() == "rs20"
    rs = _bought(lib, root, monkeypatch)
    op = _bought(lib, root, monkeypatch, SIDE="opt", SEATS=12)
    assert rs == {"000001.SZ", "000004.SZ"}  # restricted stock, and a plan naming neither instrument
    assert op == {"000002.SZ", "000003.SZ"}  # an option plan, and a mixed one
    monkeypatch.setattr(lib.knobs, "leg", lambda: "all20")  # the unsplit book, which stage 2 never runs
    assert _bought(lib, root, monkeypatch, SIDE="all", SEATS=20) == rs | op


def test_the_delay_control_enters_three_trading_days_later(lib, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bought = _bought(lib, _view(tmp_path), monkeypatch, SIDE="all", DELAY=3)
    # Visible Wednesday 03-06 + 3 = Monday 03-11: not yet fresh. Visible Friday 03-01 + 3 = 03-06: fresh.
    assert bought == {"000005.SZ"}


@pytest.mark.parametrize("switches", [
    {"SIDE": "all"},  # b20 runs from refs/starter/, never from stage 2
    {"SIDE": "all", "SEATS": 30},
    {"SIDE": "all", "CANDIDATE": "late"},
    {"SIDE": "opt", "SEATS": 20},
    {"SIDE": "rs", "DELAY": 3},
])
def test_stage_two_refuses_any_unregistered_leg(lib, monkeypatch: pytest.MonkeyPatch, switches: dict[str, object]) -> None:
    for key, value in switches.items():
        monkeypatch.setattr(lib.knobs, key, value)
    with pytest.raises(ValueError):
        lib.knobs.leg()


def test_the_pack_cites_no_reading_after_the_research_period() -> None:
    for path in PACK.rglob("*"):
        if path.is_file() and path.suffix in (".md", ".py"):
            text = path.read_text(encoding="utf-8")
            for word in ("前推", "Held-out", "held-out", "forward IR", "F2 下界"):
                assert word not in text, (path.name, word)
