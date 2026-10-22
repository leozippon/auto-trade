"""The 20261022 retrieval pack: a case memory kept inside each replay, and the in-context example beside it.

What must hold. The starter is a valid strategy package that ships no module
it does not use and never needs vLLM; the in-context example lives outside it
and works once copied into `lib/`. A name's pattern depends only on its own
window and history, never on how far back the panel reads or which other
names it holds. A case's key is filed on its own decision day and its outcome
only once the label's forward window has closed, and the same case reads the
same whichever fit files it; a frozen memory never grows. Retrieval returns
the exact nearest cases across chunk boundaries, and the placebo swaps
outcomes within a day without leaving any in place. The carrier refits only at
a new quarter. A decision reads the morning's scores and nothing else. A
prompt carries no code, date or title, the reply is parsed or left out, and
redaction strips what could name an issuer. No pack file cites a reading after
the research period.
"""

from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.strategy import CN_TZ
from autotrade.environment.strategy_loader import validate_strategy_package

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK = REPO_ROOT / "configs" / "workspace_refs" / "retrieval_lm_100k_8y_20261022"
STARTER = PACK / "starter"
EXAMPLE = PACK / "examples" / "lib" / "llm_cases.py"
INDEX = "000852.SH"
NAMES = [f"{i:06d}.SZ" for i in range(1, 61)]
DAYS = [day.strftime("%Y%m%d") for day in pd.bdate_range("2021-01-04", periods=702)]


def _drop_packages() -> None:
    for name in [module for module in sys.modules if module.split(".", 1)[0] in ("lib", "vllm")]:
        sys.modules.pop(name)


@pytest.fixture
def lib(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("torch")
    _drop_packages()
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(STARTER))
    from lib import features, knobs, learn, memory, model, score, text

    yield SimpleNamespace(
        features=features, knobs=knobs, learn=learn, memory=memory, model=model, score=score, text=text
    )
    _drop_packages()


def _market(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i, code in enumerate(NAMES):
        close = 10.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01 + 0.0005 * i, len(DAYS))))
        frame = pd.DataFrame({
            "ts_code": code, "trade_date": DAYS, "open": close * np.exp(rng.normal(0.0, 0.003, len(DAYS))),
            "close": close, "adj_factor": 1.0, "turnover_rate": rng.uniform(0.5, 5.0, len(DAYS)), "is_suspended": False,
        })
        if code == NAMES[-1]:
            frame = frame.iloc[:40]  # listed briefly at the start, then gone
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _view(root: Path, market: pd.DataFrame, bars: int) -> SimpleNamespace:
    """An as-of directory holding the first `bars` days, and the context of
    the decision on the next day."""

    root.mkdir(parents=True)
    visible = set(DAYS[:bars])
    (root / "daily").mkdir()
    market[market["trade_date"].isin(visible)].to_parquet(root / "daily" / "part_0000.parquet", index=False)
    rng = np.random.default_rng(11)
    level = 5000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.008, len(DAYS))))
    index = pd.DataFrame({
        "dataset": "index_daily", "ts_code": INDEX, "trade_date": DAYS, "open": level, "close": level,
        "available_at": [f"{d[:4]}-{d[4:6]}-{d[6:]} 17:00:00+08:00" for d in DAYS],
    })
    (root / "macro").mkdir()
    index[index["trade_date"].isin(visible)].to_parquet(root / "macro" / "part_0000.parquet", index=False)
    decision = DAYS[bars]
    return SimpleNamespace(
        asof_dir=str(root), inference_at=datetime(int(decision[:4]), int(decision[4:6]), int(decision[6:]), 8, 30, tzinfo=CN_TZ),
        pretrained_dir="", account=SimpleNamespace(cash=100_000.0, positions={}),
    )


def test_the_starter_is_a_valid_package_that_ships_only_what_it_uses() -> None:
    assert validate_strategy_package(STARTER / "main.py").refit_period == "day"
    sources = {path: path.read_text(encoding="utf-8") for path in STARTER.rglob("*.py")}
    for path in (STARTER / "lib").glob("*.py"):
        if path.stem == "__init__":
            continue
        users = [other for other, text in sources.items() if other != path and re.search(rf"\b{path.stem}\b", text)]
        assert users, f"lib/{path.name} is imported by nothing"
    assert not any("vllm" in text for text in sources.values())
    assert "from vllm import" in EXAMPLE.read_text(encoding="utf-8")


def test_no_pack_file_cites_a_reading_after_the_research_period() -> None:
    for path in PACK.rglob("*"):
        if path.is_file() and path.suffix in (".md", ".py"):
            text = path.read_text(encoding="utf-8")
            for word in ("前推", "Held-out", "held-out", "forward IR", "F2 下界", "样本外读数"):
                assert word not in text, (path.relative_to(PACK), word)


def test_a_pattern_depends_only_on_its_own_window(lib, tmp_path: Path) -> None:
    """A long and a short panel give every name the same pattern and
    validity on their common late days, and a name dropped from the market
    leaves the others' patterns alone."""

    market = _market()
    context = _view(tmp_path / "a", market, 300)
    long, short = lib.features.panel(context, 300), lib.features.panel(context, 120)
    assert long.dates[-1] == short.dates[-1]
    shared = [code for code in short.codes if code in long.codes]
    for back in (1, 5):
        a = lib.features.patterns(long, len(long.dates) - back)[[long.codes.index(c) for c in shared]]
        b = lib.features.patterns(short, len(short.dates) - back)[[short.codes.index(c) for c in shared]]
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-5)
    assert NAMES[-1] not in short.codes and not long.valid[-1, long.codes.index(NAMES[-1])]
    valid = lib.features.patterns(long, len(long.dates) - 1)[long.valid[-1]]
    assert valid.shape[1] == 2 * lib.knobs.WINDOW and np.abs(valid[:, : lib.knobs.WINDOW]).mean() < 2.0


def _fit(lib, context, state: Path) -> None:
    context.state_dir = str(state)
    lib.learn.fit(context)


def test_a_case_is_filed_only_once_its_outcome_has_matured_and_reads_the_same_whoever_files_it(
    lib, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lib.knobs, "TEXT", False)
    market = _market()
    state = tmp_path / "state"
    state.mkdir()
    first = _view(tmp_path / "a", market, 700)
    _fit(lib, first, state)
    filed, pending = lib.memory.index(first)
    horizon = lib.knobs.HORIZON + 1
    assert len(pending) == horizon and pending[-1] == DAYS[700] and pending[0] == DAYS[700 - horizon + 1]
    assert len(filed) == lib.knobs.MEMORY_DAYS + 1 - horizon and filed[-1] == DAYS[700 - horizon]
    second = _view(tmp_path / "b", market, 701)
    _fit(lib, second, state)
    filed2, pending2 = lib.memory.index(second)
    assert filed2 == filed + [pending[0]] and pending2 == pending[1:] + [DAYS[701]]
    # The same day filed by a first fit on the later view reads the same.
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    _fit(lib, _view(tmp_path / "c", market, 701), fresh)
    day = pending[0]
    for name in (f"keys_{day}.npz", f"outcome_{day}.npz"):
        a, b = np.load(state / name), np.load(fresh / name)
        for field in a.files:
            if a[field].dtype.kind == "S":
                assert (a[field] == b[field]).all(), (name, field)
            else:
                np.testing.assert_allclose(a[field], b[field], rtol=0, atol=1e-6, err_msg=f"{name}:{field}")
    outcome = np.load(state / f"outcome_{day}.npz")["y"]
    assert np.isfinite(outcome).mean() > 0.9 and np.nanmax(np.abs(outcome)) <= 0.5


def test_a_frozen_memory_never_grows(lib, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lib.knobs, "TEXT", False)
    monkeypatch.setattr(lib.knobs, "MEMORY", "frozen")
    market = _market()
    state = tmp_path / "state"
    state.mkdir()
    context = _view(tmp_path / "a", market, 700)
    _fit(lib, context, state)
    before = lib.memory.index(context)
    later = _view(tmp_path / "b", market, 701)
    _fit(lib, later, state)
    assert lib.memory.index(later) == before
    assert not (state / f"keys_{DAYS[701]}.npz").exists() and (state / f"scores_{DAYS[701]}.npz").exists()


def test_the_carrier_refits_only_at_a_new_quarter(lib, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lib.knobs, "TEXT", False)
    monkeypatch.setattr(lib.knobs, "SCORER", "cold_quarter")
    monkeypatch.setattr(lib.knobs, "TRAIN_CASES", 4096)
    calls = []
    fit = lib.model.fit
    monkeypatch.setattr(lib.model, "fit", lambda memory, day, dev: calls.append(day) or fit(memory, day, dev))
    market = _market()
    state = tmp_path / "state"
    state.mkdir()
    # DAYS[700] is 2023-09-07, DAYS[701] the next day, both in 2023Q3.
    for bars in (700, 701):
        _fit(lib, _view(tmp_path / str(bars), market, bars), state)
    assert calls == [DAYS[700]]
    scores = np.load(state / f"scores_{DAYS[701]}.npz")
    assert np.isfinite(scores["values"]).all() and len(scores["codes"]) > 50


def test_the_decision_reads_the_mornings_scores(lib, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lib.knobs, "TEXT", False)
    market = _market()
    state = tmp_path / "state"
    state.mkdir()
    context = _view(tmp_path / "a", market, 700)
    _fit(lib, context, state)
    values, meta = lib.score.score(context, [NAMES[0], NAMES[-1], "999999.SZ"])
    assert np.isfinite(values.iloc[0]) and values.iloc[1:].isna().all()
    assert meta["scorer"] == "retrieval" and int(meta["memory_cases"]) > 20_000
    later = _view(tmp_path / "b", market, 701)
    later.state_dir = str(state)
    with pytest.raises(RuntimeError, match="fit has not run"):
        lib.score.score(later, NAMES)


def test_retrieval_finds_the_exact_nearest_cases_across_chunks(lib, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(lib.knobs, "TEXT", False)
    monkeypatch.setattr(lib.memory, "CHUNK", 37)
    rng = np.random.default_rng(3)
    mem = lib.memory._empty(torch.device("cpu"))
    days = [f"2024{m:02d}01" for m in range(1, 6)]
    keys = [rng.normal(size=(50, 2 * lib.knobs.WINDOW)).astype("float16") for _ in days]
    ys = [rng.uniform(-0.5, 0.5, 50).astype("float32") for _ in days]
    codes = [np.array([f"{i:06d}.SZ" for i in range(50)], dtype="S12") for _ in days]
    lib.memory._append(mem, days[:2], codes[:2], keys[:2], ys[:2])
    lib.memory._append(mem, days[2:], codes[2:], keys[2:], ys[2:])
    query = rng.normal(size=(9, 2 * lib.knobs.WINDOW)).astype("float32")
    idx, d2 = lib.memory.retrieve(mem, query, 10)
    stored = np.concatenate(keys).astype("float32")
    brute = ((stored[None, :, :] - query[:, None, :]) ** 2).sum(axis=2)
    np.testing.assert_array_equal(idx, np.argsort(brute, axis=1, kind="stable")[:, :10])
    np.testing.assert_allclose(d2, np.sort(brute, axis=1)[:, :10], rtol=1e-5)
    weights = np.exp(-(d2 - d2[:, :1]) / (lib.knobs.BANDWIDTH * np.median(d2, axis=1, keepdims=True)))
    expected = (weights * np.concatenate(ys)[idx]).sum(axis=1) / weights.sum(axis=1)
    np.testing.assert_allclose(lib.memory.predict(mem, idx, d2), expected, rtol=1e-5)


def test_the_placebo_swaps_outcomes_within_a_day_and_keeps_none(lib) -> None:
    codes = np.array([f"{i:06d}.SZ" for i in range(40)], dtype="S12")
    y = np.linspace(-0.5, 0.5, 40).astype("float32")
    swapped = lib.memory.swapped(codes, y, "20240102")
    assert sorted(swapped) == sorted(y) and not (swapped == y).any()
    order = np.random.default_rng(1).permutation(40)
    np.testing.assert_array_equal(lib.memory.swapped(codes[order], y[order], "20240102"), swapped[order])
    assert not np.array_equal(lib.memory.swapped(codes, y, "20240103"), swapped)


def test_a_names_text_is_the_mean_over_the_window(lib) -> None:
    code = np.array([0, 0, 1, 1, -1])
    day = np.array([3, 7, 2, 8, 8])
    vectors = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0], [9.0, 9.0]], dtype="float32")
    mean, count = lib.text.means(code, day, vectors, 3, 7)
    window = lib.knobs.TEXT_DAYS  # days 3..7 count at 7
    assert window == 5
    np.testing.assert_allclose(mean, [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert count.tolist() == [2, 0, 0]


@pytest.fixture
def example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The example copied into a starter's `lib/`, as the README says, over a
    stand-in for vLLM that records what it is sent and answers by a rule."""

    pytest.importorskip("torch")
    _drop_packages()
    root = tmp_path / "output"
    shutil.copytree(STARTER, root)
    shutil.copy(EXAMPLE, root / "lib" / "llm_cases.py")
    sent: list = []

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def chat(self, chats, params, **kwargs):
            sent.append((chats, params.kwargs, kwargs))
            replies = ["73", "the answer is 12.", "none"] * len(chats)
            return [SimpleNamespace(outputs=[SimpleNamespace(text=replies[i])]) for i in range(len(chats))]

    vllm = ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(root))
    from lib import llm_cases, memory

    yield SimpleNamespace(llm_cases=llm_cases, memory=memory, sent=sent)
    _drop_packages()


def test_the_example_prompts_carry_no_identifier_and_its_replies_rerank_the_shortlist(example) -> None:
    import torch

    rng = np.random.default_rng(5)
    width = 2 * 20 + 64
    mem = example.memory._empty(torch.device("cpu"))
    codes = [np.array([f"{i:06d}.SZ" for i in range(200)], dtype="S12")]
    key = rng.normal(size=(200, width)).astype("float16")
    key[:100, 40:] = 0  # cases without titles
    example.memory._append(mem, ["20240102"], codes, [key], [rng.uniform(-0.5, 0.5, 200).astype("float32")])
    query = rng.normal(size=(10, width)).astype("float32")
    values = np.linspace(-0.4, 0.4, 10)
    idx, _ = example.memory.retrieve(mem, query, 8)
    context = SimpleNamespace(pretrained_dir="/store", inference_at=datetime(2024, 3, 1, 8, 30, tzinfo=CN_TZ))
    names = [f"{600000 + i}.SH" for i in range(10)]
    out = example.llm_cases.rerank(context, mem, names, query, values, idx, shortlist=3, cases=4)
    chats, params, options = example.sent[-1]
    assert params["temperature"] == 0.0 and options["chat_template_kwargs"] == {"enable_thinking": False}
    assert len(chats) == 3
    for chat in chats:
        content = " ".join(message["content"] for message in chat)
        assert not re.search(r"\d{4}|\.S[HZ]|20\d\d", content), content
        assert content.count("-> outcome percentile") == 4
    # Shortlist = names 9, 8, 7 (highest first): 73, 12, unparsed.
    assert out[9] == pytest.approx(1.73) and out[8] == pytest.approx(1.12) and out[7] == values[7]
    np.testing.assert_array_equal(out[:7], values[:7])
    placebo = example.llm_cases.random_cases(mem, names[9], "20240301", 4)
    assert len(set(placebo)) == 4 and (placebo == example.llm_cases.random_cases(mem, names[9], "20240301", 4)).all()


def test_redaction_keeps_the_event_type_only(example) -> None:
    redacted = example.llm_cases.redact([
        "关于公司获得《XY-123型设备》注册证的公告", "2023年年度报告", "关于收到中国证监会立案告知书的公告",
        "关于“ABC项目”中标的提示性公告",
    ])
    assert redacted == ["关于公司获得注册证的公告", "年年度报告", "关于收到中国证监会立案告知书的公告", "关于中标的提示性公告"]
    assert all(len(title) <= 24 for title in redacted)
    parsed = [example.llm_cases.parse(text) for text in ("73", "about 101", "x 7 y", "")]
    assert parsed[0] == 73.0 and np.isnan(parsed[1]) and parsed[2] == 7.0 and np.isnan(parsed[3])
