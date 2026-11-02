"""The open-exploration packs of round 20261014 that the Paper books started from: one common contract, one direction per pack.

What must hold. The packs give a direction each and share the rules file, the
record of occupied and closed directions and the starter byte for byte, since
a pack is mounted alone and cannot point at another. The starter is a valid
strategy package. Its read helpers return only visible rows, matched as asked,
and its device helper falls back to the CPU. No pack cites a reading after the
research period.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from autotrade.environment.strategy import CN_TZ
from autotrade.environment.strategy_loader import validate_strategy_package
from tests.unit.reference_pack_fixture import AFTER_RESEARCH

REPO_ROOT = Path(__file__).resolve().parents[2]
REFS = REPO_ROOT / "configs" / "workspace_refs"
PACKS_14 = [REFS / f"open_{name}_100k_8y_20261014" for name in ("free", "commit")]
COMMON = ("families.md", "references/open-rules.md")


def _files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_the_round_shares_one_contract_and_one_starter() -> None:
    for pack in PACKS_14[1:]:
        for name in COMMON:
            assert (pack / name).read_bytes() == (PACKS_14[0] / name).read_bytes(), (pack.name, name)
        assert _files(pack / "starter") == _files(PACKS_14[0] / "starter"), pack.name


@pytest.mark.parametrize("pack", PACKS_14, ids=lambda path: path.name)
def test_the_starter_is_a_valid_package(pack: Path) -> None:
    validate_strategy_package(pack / "starter" / "main.py")


def test_no_pack_cites_a_reading_after_the_research_period() -> None:
    for pack in PACKS_14:
        for path in pack.rglob("*"):
            if path.is_file() and path.suffix in (".md", ".py"):
                text = path.read_text(encoding="utf-8")
                for word in AFTER_RESEARCH:
                    assert word not in text, (pack.name, path.name, word)


def _drop_packages() -> None:
    for name in [module for module in sys.modules if module.split(".", 1)[0] == "lib"]:
        sys.modules.pop(name)


def _starter(monkeypatch: pytest.MonkeyPatch, pack: Path) -> None:
    _drop_packages()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(pack / "starter"))


@pytest.fixture
def lib(monkeypatch: pytest.MonkeyPatch):
    _starter(monkeypatch, PACKS_14[0])
    from lib import device, titles

    yield SimpleNamespace(device=device, titles=titles)
    _drop_packages()


# (dataset, code, title, available_at). A decision on Monday 2024-03-11 08:30.
TITLES = [
    ("anns_d", "000001.SZ", "关于回购股份实施完毕暨注销的公告", "2024-03-05 23:59:59+08:00"),
    ("anns_d", "000001.SZ", "关于回购股份实施完毕暨注销的公告", "2024-03-05 23:59:59+08:00"),  # a duplicate row
    ("anns_d", "000001.SZ", "关于回购股份实施完毕暨注销的公告", "2023-12-01 23:59:59+08:00"),  # 95 days earlier
    ("anns_d", "000002.SZ", "关于回购注销部分限制性股票的公告", "2024-03-06 23:59:59+08:00"),
    ("anns_d", "000003.SZ", "关于回购股份方案的公告", "2024-03-07 23:59:59+08:00"),
    # A receipt-time stamp written in UTC: 2024-03-07 18:00 local, so its day is 03-07.
    ("anns_d", "000004.SZ", "关于回购股份实施完毕的公告", "2024-03-07 10:00:00+00:00"),
    ("anns_d", "000005.SZ", "关于回购股份实施完毕的公告", "2024-02-20 23:59:59+08:00"),  # 15 days before 000005's next
    ("anns_d", "000005.SZ", "关于回购股份实施完毕的公告", "2024-03-06 23:59:59+08:00"),
    ("other", "000006.SZ", "关于回购股份实施完毕的公告", "2024-03-06 23:59:59+08:00"),
]
DECISION = datetime(2024, 3, 11, 8, 30, tzinfo=CN_TZ)


def _context(root: Path, rows: list[tuple[str, str, str, str]]) -> SimpleNamespace:
    (root / "text_index").mkdir()
    pd.DataFrame(rows, columns=["dataset", "ts_codes", "title", "available_at"]).to_parquet(
        root / "text_index" / "part-0.parquet", index=False
    )
    return SimpleNamespace(asof_dir=str(root), inference_at=DECISION)


def test_titles_are_read_by_local_day_pattern_and_dataset(lib, tmp_path: Path) -> None:
    context = _context(tmp_path, TITLES)
    rows = lib.titles.read(context, "20240301", "回购", "完毕", exclude="限制性股票")
    assert rows[["ts_code", "day"]].values.tolist() == [
        ["000001.SZ", "20240305"],
        ["000004.SZ", "20240307"],
        ["000005.SZ", "20240306"],
    ]
    assert lib.titles.read(context, "20240301", "限制性股票")["ts_code"].tolist() == ["000002.SZ"]


def test_an_event_needs_its_quiet_period_read_not_assumed(lib, tmp_path: Path) -> None:
    rows = lib.titles.read(_context(tmp_path, TITLES), "20231101", "回购", "完毕", exclude="限制性股票")
    events = lib.titles.first_after_quiet(rows, 90, "20240301")
    # 000001 was quiet for 95 days; 000005 repeated after 15.
    assert events.values.tolist() == [["000001.SZ", "20240305"], ["000004.SZ", "20240307"]]


def test_a_title_stamped_after_the_decision_fails_the_read(lib, tmp_path: Path) -> None:
    late = [("anns_d", "000007.SZ", "关于回购股份实施完毕的公告", "2024-03-11 09:00:00+08:00")]
    with pytest.raises(RuntimeError, match="stamped after this decision"):
        lib.titles.read(_context(tmp_path, TITLES + late), "20240301", "回购")


def test_the_device_falls_back_to_the_cpu(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lib.device.torch.cuda, "is_available", lambda: False)
    assert lib.device.device().type == "cpu"
