"""The open-exploration packs of rounds 20261014 and 20261008b: one common contract per round, one direction per pack.

What must hold. Within a round the packs give a direction each and share the
rules file and the record of occupied and closed directions byte for byte,
since a pack is mounted alone and cannot point at another. Each starter is a
valid strategy package; a round whose arms claim a card ships the device
helper and one that claims none ships no CUDA path. Round 20261008b's starters
share every common module, carry the events helper exactly where the arm
mounts the events domain, and their built-in shuffled control keeps a name's
stand-in from one review to the next. The ensemble pack's fixed sleeves are
the frozen Paper strategies, changed only in their import lines and their seat
counts, and its legs run the candidate and the two required controls on the
same account. The read helpers return only visible rows, matched as asked. The
combined book gives each sleeve its own holdings and cash and never lets one
sleeve trade another's names. A pack names the lineage its arm is created
with, and cites no reading after the research period.
"""

from __future__ import annotations

import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from autotrade.environment.strategy import CN_TZ
from autotrade.environment.strategy_loader import validate_strategy_package

REPO_ROOT = Path(__file__).resolve().parents[2]
REFS = REPO_ROOT / "configs" / "workspace_refs"
PACKS_14 = [REFS / f"open_{name}_100k_8y_20261014" for name in ("free", "commit", "avoid", "earnings", "adapt")]
FIVE_YEAR_08B = [REFS / f"open_{name}_100k_5y_20261008b" for name in ("tables", "exclude")]
EIGHT_YEAR_08B = [REFS / f"open_{name}_100k_8y_20261008b" for name in ("guidance", "capital")]
ENSEMBLE = REFS / "open_ensemble_100k_8y_20261008b"
PACKS_08B = [*FIVE_YEAR_08B, *EIGHT_YEAR_08B, ENSEMBLE]
ROUNDS = {
    "create_round_20261014": (PACKS_14, 1),
    "create_round_20261008b": (PACKS_08B, 0),
}
COMMON = ("families.md", "references/open-rules.md")


def _files(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.mark.parametrize("packs", [PACKS_14, PACKS_08B], ids=["20261014", "20261008b"])
def test_a_round_shares_one_contract(packs: list[Path]) -> None:
    for pack in packs[1:]:
        for name in COMMON:
            assert (pack / name).read_bytes() == (packs[0] / name).read_bytes(), (pack.name, name)


@pytest.mark.parametrize("pack", PACKS_14[1:], ids=lambda path: path.name)
def test_round_20261014_ships_one_starter(pack: Path) -> None:
    assert _files(pack / "starter") == _files(PACKS_14[0] / "starter"), pack.name


def test_round_20261008b_starters_share_their_common_modules() -> None:
    """The five starters agree on every module they share, the four non-ensemble
    ones on their entry module too; only the five-year packs read the events
    domain, and no starter keeps a device helper, since no arm has a card."""
    reference = _files(EIGHT_YEAR_08B[0] / "starter")
    for pack in PACKS_08B:
        files = _files(pack / "starter")
        shared = {name for name in files if name.startswith("lib/") and name in reference}
        assert shared >= {"lib/controls.py", "lib/trade.py", "lib/score.py", "lib/titles.py", "lib/fund.py"}, pack.name
        assert {name: files[name] for name in shared} == {name: reference[name] for name in shared}, pack.name
        assert ("lib/events.py" in files) is (pack in FIVE_YEAR_08B), pack.name
        assert "lib/device.py" not in files, pack.name
        if pack != ENSEMBLE:
            assert files["main.py"] == reference["main.py"], pack.name
    events = {(pack / "starter" / "lib" / "events.py").read_bytes() for pack in FIVE_YEAR_08B}
    assert len(events) == 1


def test_the_starter_control_keeps_each_stand_in_from_review_to_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """`knobs.SHUFFLE` runs the held shuffle over the day's tradable names: the
    same scores give the same stand-ins on any day, so a keep-band book keeps
    the candidate's holding period."""
    _starter(monkeypatch, EIGHT_YEAR_08B[0])
    from lib import controls, score

    codes = [f"{i:06d}.SZ" for i in range(1, 31)]
    values = pd.Series([float(i) if i % 4 else float("nan") for i in range(30)], index=codes)
    first = score.shuffle(values, datetime(2024, 3, 11, 8, 30, tzinfo=CN_TZ))
    assert first.equals(score.shuffle(values, datetime(2024, 4, 1, 8, 30, tzinfo=CN_TZ)))
    assert first.equals(controls.held_shuffle(values, values.index))
    assert sorted(first) == sorted(values.dropna())
    _drop_packages()


@pytest.mark.parametrize("pack", [*PACKS_14, *PACKS_08B], ids=lambda path: path.name)
def test_the_starter_is_a_valid_package(pack: Path) -> None:
    validate_strategy_package(pack / "starter" / "main.py")


@pytest.mark.parametrize("round_name", sorted(ROUNDS))
def test_every_arm_of_a_round_mounts_its_direction_and_names_its_lineage(round_name: str) -> None:
    import importlib

    packs, gpus = ROUNDS[round_name]
    rnd = importlib.import_module(f"scripts.experiments.{round_name}").ROUND
    mounted = set()
    for arm in rnd.arms:
        params = rnd.request_params(arm)
        pack = REPO_ROOT / str(params["workspace_reference"])
        mounted.add(pack)
        readme = (pack / "README.md").read_text(encoding="utf-8")
        section = readme.split("## 血缘", 1)[1].split("\n## ", 1)[0]
        lineage = set(params.get("lineage_arms") or ())
        assert lineage <= set(re.findall(r"`([a-z0-9_]+_\d{8})`", section)), arm
        assert section.lstrip().startswith("没有") is not bool(lineage), arm
        assert int(params["gpu_count"]) == gpus, arm
    assert mounted == set(packs)


def test_round_20261008b_mounts_the_events_domain_on_the_five_year_arms_only() -> None:
    from scripts.experiments.create_round_20261008b import ROUND

    for arm in ROUND.arms:
        params = ROUND.request_params(arm)
        five = "_5y_" in arm
        assert params["research_start"] == ("20200701" if five else "20170701"), arm
        assert bool(params["include_events"]) is five, arm
        assert ("5y" in str(params["pit_views_seed"])) is five, arm
        assert params["permitted_boards"] == ["main", "gem"], arm


def test_no_pack_cites_a_reading_after_the_research_period() -> None:
    for pack in [*PACKS_14, *PACKS_08B]:
        for path in pack.rglob("*"):
            if path.is_file() and path.suffix in (".md", ".py"):
                text = path.read_text(encoding="utf-8")
                for word in ("前推", "Held-out", "held-out", "forward IR", "F2 下界", "样本外读数"):
                    assert word not in text, (pack.name, path.name, word)


# The frozen artifacts' files by their SHA-256 prefix, as the ensemble README
# records them; the sleeves differ from them in their import lines and in the
# seat count of their knobs (16, where both artifacts hold 30).
SLEEVE_HASH_ROW = re.compile(r"^\| `([a-z/_.]+)`(?:（`b30`）/ `([a-z/_.]+)`（`esop60`）)? \| `([0-9a-f]{16}|—)` \| `([0-9a-f]{16})` \|$")


def _recorded_hashes() -> dict[tuple[str, str], str]:
    recorded: dict[tuple[str, str], str] = {}
    for line in (ENSEMBLE / "README.md").read_text(encoding="utf-8").splitlines():
        line = line.replace("| — |", "| `—` |")
        match = SLEEVE_HASH_ROW.match(line)
        if match:
            b30_file, esop_file, b30_hash, esop_hash = match.groups()
            if b30_hash != "—":
                recorded[("b30", b30_file)] = b30_hash
            recorded[("esop60", esop_file or b30_file)] = esop_hash
    return recorded


def test_the_ensemble_sleeves_are_the_frozen_strategies_but_for_their_imports() -> None:
    recorded = _recorded_hashes()
    assert len(recorded) == 11
    for (sleeve, name), prefix in recorded.items():
        copy = ENSEMBLE / "starter" / "sleeves" / sleeve / Path(name).name
        original = re.sub(rf"^from sleeves\.{sleeve} import ", "from lib import ", copy.read_text(encoding="utf-8"), flags=re.M)
        if copy.name == "knobs.py":
            assert re.search(r"^SEATS = 16$", original, flags=re.M), (sleeve, name)
            original = re.sub(r"^SEATS = 16$", "SEATS = 30", original, flags=re.M)
        assert hashlib.sha256(original.encode("utf-8")).hexdigest()[:16] == prefix, (sleeve, name)
    for sleeve in ("b30", "esop60"):
        files = {path.name for path in (ENSEMBLE / "starter" / "sleeves" / sleeve).glob("*.py")}
        assert files == {Path(name).name for (owner, name) in recorded if owner == sleeve} | {"__init__.py"}


def _drop_packages() -> None:
    for name in [module for module in sys.modules if module.split(".", 1)[0] in ("lib", "sleeves")]:
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


# (dataset, code, available_at, change_ratio) in the events domain.
EVENTS = [
    ("stk_holdertrade", "000001.SZ", "2024-03-01 19:00:00+08:00", 0.5),
    ("stk_holdertrade", "000002.SZ", "2024-03-08 19:00:00+08:00", 1.5),
    # 23:30 on 03-07 UTC is 07:30 on 03-08 local: inside the window by local day.
    ("stk_holdertrade", "000003.SZ", "2024-03-07 23:30:00+00:00", 2.5),
    ("block_trade", "000004.SZ", "2024-03-08 21:00:00+08:00", None),
]


def test_the_events_helper_reads_one_dataset_by_local_day_and_refuses_the_future(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _starter(monkeypatch, FIVE_YEAR_08B[0])
    from lib import events

    (tmp_path / "events").mkdir()
    frame = pd.DataFrame(EVENTS, columns=["dataset", "ts_code", "available_at", "change_ratio"])
    frame.to_parquet(tmp_path / "events" / "part-0.parquet", index=False)
    context = SimpleNamespace(asof_dir=str(tmp_path), inference_at=DECISION)
    rows = events.read(context, "stk_holdertrade", ["change_ratio"], since="20240308")
    assert rows[["ts_code", "change_ratio"]].values.tolist() == [["000002.SZ", 1.5], ["000003.SZ", 2.5]]
    assert len(events.read(context, "stk_holdertrade", ["change_ratio"])) == 3
    early = SimpleNamespace(asof_dir=str(tmp_path), inference_at=datetime(2024, 3, 8, 8, 30, tzinfo=CN_TZ))
    with pytest.raises(RuntimeError, match="stamped after this decision"):
        events.read(early, "stk_holdertrade", ["change_ratio"])
    _drop_packages()


def test_the_combined_book_gives_each_sleeve_its_own_holdings_and_cash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ownership is the first claim, the rest goes to the last sleeve; cash is
    each share of the account's value less the sleeve's holdings; a buy of a
    name another sleeve holds or already bought is dropped, and a sleeve that
    sells a name it does not hold stops the book."""
    _starter(monkeypatch, ENSEMBLE)
    from lib import combine

    (tmp_path / "daily").mkdir()
    pd.DataFrame(
        {"ts_code": ["A", "B", "C", "D"], "trade_date": "20240308", "close": [10.0, 20.0, 5.0, 8.0]}
    ).to_parquet(tmp_path / "daily" / "part-0.parquet", index=False)
    seen = {}

    def sleeve(name, buys, sells=()):
        def run(context):
            seen[name] = (context.account.cash, dict(context.account.positions))
            return [{"symbol": code, "action": "sell", "quantity": 100} for code in sells] + [
                {"symbol": code, "action": "buy", "quantity": 100} for code in buys
            ]

        return run

    first = combine.Sleeve("first", sleeve("first", ["D", "B"], sells=["A"]), lambda context, codes: {"A", "C"} & set(codes), 0.6)
    last = combine.Sleeve("last", sleeve("last", ["D", "C"]), lambda context, codes: pytest.fail("never asked"), 0.4)
    context = SimpleNamespace(
        asof_dir=str(tmp_path), inference_at=DECISION, account=combine.Account(cash=4_000.0, positions={"A": 100, "B": 100})
    )
    orders = combine.combine(context, [first, last])
    # A is claimed by the first sleeve, B by nobody: it goes to the last.
    # The account is worth 4,000 + 1,000 + 2,000 = 7,000: the first asks
    # 4,200 - 1,000, the last 2,800 - 2,000, together 4,000 = the cash.
    assert seen == {"first": (3_200.0, {"A": 100}), "last": (800.0, {"B": 100})}
    assert [(o["sleeve"], o["action"], o["symbol"]) for o in orders] == [
        ("first", "sell", "A"),
        ("first", "buy", "D"),
        ("last", "buy", "C"),
    ]
    rogue = combine.Sleeve("last", sleeve("last", [], sells=["A"]), None, 0.4)
    with pytest.raises(RuntimeError, match="does not hold"):
        combine.combine(context, [first, rogue])
    with pytest.raises(ValueError, match="sum to at most 1"):
        combine.combine(context, [first, combine.Sleeve("last", last.run, None, 0.5)])
    _drop_packages()


def test_the_ensemble_legs_keep_the_seats_and_scale_the_shares(monkeypatch: pytest.MonkeyPatch) -> None:
    """The candidate is every sleeve; "without" spends the merged book's share
    on the given sleeves alone, in their proportions; a given sleeve alone
    takes the whole of it. The shipped package is the given sleeves at 16
    seats each, the 36-seat ceiling leaving room for a sleeve of the arm's."""
    _starter(monkeypatch, ENSEMBLE)
    import main
    from lib import combine
    from sleeves.b30 import knobs as b30_knobs
    from sleeves.esop60 import knobs as esop60_knobs

    def run(context: object) -> list[dict[str, object]]:
        return []

    sleeves = [
        combine.Sleeve("esop60", run, None, 0.35),
        combine.Sleeve("own", run, None, 0.3),
        combine.Sleeve("b30", run, None, 0.35),
    ]
    assert combine.leg(sleeves, "merged") == sleeves
    without = combine.leg(sleeves, "without")
    assert [sleeve.name for sleeve in without] == ["esop60", "b30"]
    assert [sleeve.share for sleeve in without] == pytest.approx([0.5, 0.5])
    assert [(sleeve.name, sleeve.share) for sleeve in combine.leg(sleeves, "b30")] == [("b30", pytest.approx(1.0))]
    with pytest.raises(ValueError, match="leg must be"):
        combine.leg(sleeves, "own")
    assert main.LEG == "merged" and [sleeve.name for sleeve in main.SLEEVES] == ["esop60", "b30"]
    assert b30_knobs.SEATS == esop60_knobs.SEATS == 16
    sys.modules.pop("main")
    _drop_packages()
