"""The two novel-method lanes of the seqnovel_alla_100k_8y_20261012 starter.

The pack runs c_bag4 (the 100k sequence bag, four seeds per head) at its
defaults and opens one of two lanes on it, so what must hold is: the switches
run one lane at a time with registered values only, and a monthly cadence
exactly when the update lane is on; every leg keeps c_bag4's members and seeds
and only adds or swaps one seed slot; the state-space head's closed form is
the recurrence's last state and never mixes names; quarterly refits fall on
c_bag4's dates with c_bag4's cold resets, whatever the cadence, and the
monthly fits between them are updates; an update trains on the dates
labelled since the previous fit, replays as many older ones, continues each
member's checkpoint, and its placebo only permutes the new dates' labels. The
modules run as a starter runs them, imported from the pack's own ``starter``
directory.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from autotrade.environment.strategy import CN_TZ

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER = REPO_ROOT / "configs" / "workspace_refs" / "seqnovel_alla_100k_8y_20261012" / "starter"


def _drop_starter() -> None:
    for name in [module for module in sys.modules if module in ("lib", "main") or module.startswith("lib.")]:
        sys.modules.pop(name)


@pytest.fixture
def pack(monkeypatch: pytest.MonkeyPatch):
    _drop_starter()
    # The pack is copied into every session's output/: import it without
    # writing bytecode next to it.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(STARTER))
    import main
    from lib import knobs, model, state

    yield SimpleNamespace(main=main, knobs=knobs, model=model, state=state)
    _drop_starter()


def test_the_switches_run_one_lane_at_a_time_with_registered_values(pack, monkeypatch: pytest.MonkeyPatch) -> None:
    knobs = pack.knobs
    assert knobs.leg() == "c_bag4"
    knobs.check_refit("quarter")
    with pytest.raises(ValueError, match="REFIT_PERIOD = 'quarter'"):
        knobs.check_refit("month")
    monkeypatch.setattr(knobs, "PLACEBO", True)
    with pytest.raises(ValueError, match="belongs to a lane"):
        knobs.leg()
    monkeypatch.setattr(knobs, "SSM", "add")
    assert knobs.leg() == "plc_add"
    monkeypatch.setattr(knobs, "UPDATE", "replay")
    with pytest.raises(ValueError, match="one lane at a time"):
        knobs.leg()
    monkeypatch.setattr(knobs, "SSM", "off")
    assert knobs.leg() == "plc_replay"
    with pytest.raises(ValueError, match="REFIT_PERIOD = 'month'"):
        knobs.check_refit("quarter")
    knobs.check_refit("month")
    monkeypatch.setattr(knobs, "PLACEBO", False)
    assert knobs.leg() == "cl_replay"
    monkeypatch.setattr(knobs, "UPDATE", "warm")
    assert knobs.leg() == "cl_warm"
    monkeypatch.setattr(knobs, "UPDATE", "off")
    monkeypatch.setattr(knobs, "SSM", "replace")
    assert knobs.leg() == "ssm_rep"
    for name, value in (("SEED_BASE", 1000.0), ("SEED_BASE", 5000), ("PLACEBO", 1), ("SSM", "swap")):
        monkeypatch.setattr(knobs, name, value)
        with pytest.raises(ValueError, match=name):
            knobs.leg()
        monkeypatch.undo()


def test_every_leg_keeps_c_bag4s_members_and_moves_one_seed_slot(pack, monkeypatch: pytest.MonkeyPatch) -> None:
    knobs, model = pack.knobs, pack.model
    monkeypatch.setattr(knobs, "SEED_BASE", 2000)
    bag = [(head, 2000 + 100 * k + j) for k, head in enumerate(("mlp", "lstm", "tcn", "attn")) for j in range(4)]
    assert model.members() == bag
    slot = [2400, 2401, 2402, 2403]
    monkeypatch.setattr(knobs, "SSM", "add")
    assert model.members() == bag + [("ssm", seed) for seed in slot]
    monkeypatch.setattr(knobs, "SSM", "replace")
    assert model.members() == bag[4:] + [("ssm", seed) for seed in slot]
    monkeypatch.setattr(knobs, "PLACEBO", True)
    # A re-seeded copy of the LSTM: its own seeds, so its own checkpoints.
    assert model.members() == bag[4:] + [("lstm", seed) for seed in slot]
    assert len(set(model.members())) == 16
    monkeypatch.setattr(knobs, "SSM", "off")
    monkeypatch.setattr(knobs, "UPDATE", "warm")
    assert model.members() == bag


def _scan(net, x):
    """The selective scan step by step, for comparison with the head's closed form."""
    model = sys.modules["lib.model"]
    u = net.proj(x)
    steps = u.shape[1]
    inner = net.out_proj.in_features
    z, streams, dt = net.in_proj(u).split([inner, inner + 2 * model.SSM_STATE, model.SSM_HEADS], dim=-1)
    streams = torch.nn.functional.silu(net.conv(streams.transpose(1, 2))[..., :steps].transpose(1, 2))
    xs, b, c = streams.split([inner, model.SSM_STATE, model.SSM_STATE], dim=-1)
    dt = torch.nn.functional.softplus(dt + net.dt_bias)
    a = -torch.exp(net.a_log)
    xs = xs.reshape(len(x), steps, model.SSM_HEADS, -1)
    state = torch.zeros(len(x), model.SSM_HEADS, xs.shape[-1], model.SSM_STATE, dtype=x.dtype)
    for t in range(steps):
        decay = torch.exp(dt[:, t] * a)[:, :, None, None]
        state = decay * state + dt[:, t][:, :, None, None] * xs[:, t, :, :, None] * b[:, t][:, None, None, :]
    y = torch.einsum("nhpk,nk->nhp", state, c[:, -1]) + net.skip[:, None] * xs[:, -1]
    y = net.gate_norm(y.reshape(len(x), -1) * torch.nn.functional.silu(z[:, -1]))
    return state, net.head(net.norm(net.out_proj(y) + u[:, -1])).squeeze(-1)


def test_the_state_space_head_computes_the_recurrences_last_state_name_by_name(pack) -> None:
    model = pack.model
    torch.manual_seed(1400)
    net = model.SSMHead().double()
    x = torch.randn(6, 60, 12, dtype=torch.float64)
    state, scores = _scan(net, x)
    with torch.no_grad():
        np.testing.assert_allclose(net.last_state(x)[1].numpy(), state.numpy(), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(net(x).numpy(), scores.numpy(), rtol=1e-9, atol=1e-12)
        # Padding rows, as the training batches add them, change no name's score.
        padded = torch.cat([x, torch.zeros(250, 60, 12, dtype=torch.float64)])
        np.testing.assert_allclose(net(padded)[:6].numpy(), scores.numpy(), rtol=1e-9, atol=1e-12)


def _at(day: pd.Timestamp) -> datetime:
    return datetime(day.year, day.month, day.day, 8, 30, tzinfo=CN_TZ)


def _eve(day: datetime) -> int:
    """The day before a decision, as the fake newest labelled date."""
    return int((pd.Timestamp(day.date()) - pd.Timedelta(days=1)).strftime("%Y%m%d"))


def _drive(pack, monkeypatch, tmp_path, decisions):
    """main.fit at each decision with the panel and the two training paths faked; the kinds in order."""
    calls: list[str] = []
    monkeypatch.setattr(pack.main.panel, "build", lambda context, days, labels: {"dates": np.array(["0"])})
    monkeypatch.setattr(pack.model, "fit", lambda context, data, cold: calls.append("cold" if cold else "warm") or 0.1)
    monkeypatch.setattr(pack.model, "update", lambda context, data, since: calls.append(f"update>{since}") or 20.0)
    monkeypatch.setattr(pack.model, "newest_labelled", lambda context, data: _eve(context.inference_at))
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    for day in decisions:
        pack.main.fit(SimpleNamespace(inference_at=_at(day), state_dir=str(state_dir)))
    return calls


# The first day of every quarter and month of the eight research years; the
# state reads only the calendar.
QUARTERS = pd.date_range("2017-07-01", "2025-06-30", freq="QS")
MONTHS = pd.date_range("2017-07-01", "2025-06-30", freq="MS")


def test_quarterly_refits_fall_on_c_bag4s_dates_and_the_months_between_are_updates(
    pack, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    quarterly = _drive(pack, monkeypatch, tmp_path / "q", QUARTERS)
    assert len(quarterly) == 32
    assert [k for k, kind in enumerate(quarterly) if kind == "cold"] == list(range(0, 32, 4))
    assert set(quarterly[1:]) - {"cold"} == {"warm"}

    monkeypatch.setattr(pack.knobs, "UPDATE", "replay")
    monkeypatch.setattr(pack.main, "REFIT_PERIOD", "month")
    monthly = _drive(pack, monkeypatch, tmp_path / "m", MONTHS)
    assert [kind for kind in monthly if not kind.startswith("update")] == quarterly
    assert [day.month % 3 == 1 for day in MONTHS] == [not kind.startswith("update") for kind in monthly]
    # Each update starts after the newest date the fit before it could see.
    assert monthly[1:3] == ["update>20170630", "update>20170731"]

    # A replay opening mid-quarter (a smoke) is cold there, updates until the
    # quarter turns, and resets cold four quarterly refits later, as c_bag4 does.
    opening = pd.Timestamp("2024-05-27")
    late = _drive(pack, monkeypatch, tmp_path / "late", [opening, *pd.date_range(opening, "2025-06-30", freq="MS")])
    assert late[:4] == ["cold", "update>20240526", "warm", "update>20240630"]
    assert [kind for kind in late if not kind.startswith("update")] == ["cold", "warm", "warm", "warm", "cold"]


def _panel(rows: int = 830, names: int = 40) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range(end="2025-05-23", periods=rows).strftime("%Y%m%d").to_numpy()
    return {
        "dates": dates,
        "codes": np.array([f"N{k:02d}" for k in range(names)]),
        "seq": rng.normal(1.0, 0.1, (rows, names, 12)).astype(np.float32),
        "has_bar": np.ones((rows, names), dtype=bool),
        "nbars": np.cumsum(np.ones((rows, names)), axis=0),
        "not_st": np.ones(names, dtype=bool),
        "yrank": (rng.random((rows, names)) - 0.5).astype(np.float32),
    }


def test_a_monthly_update_trains_on_the_new_dates_and_its_placebo_only_permutes_their_labels(
    pack, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    knobs, model = pack.knobs, pack.model
    data = _panel()
    context = SimpleNamespace(inference_at=_at(pd.Timestamp("2025-05-26")), state_dir=str(tmp_path))
    since = 20250425
    labelled = model.labelled_dates(context, data)
    monkeypatch.setattr(knobs, "UPDATE", "warm")
    new, replay = model.update_dates(context, data, since)
    assert new == [t for t in labelled if int(data["dates"][t]) > since] and len(new) == 9 and replay == []
    monkeypatch.setattr(knobs, "UPDATE", "replay")
    new, replay = model.update_dates(context, data, since)
    train = model.fit_dates(context, data)[0]
    assert len(replay) == len(new) and set(replay) <= set(train) and max(data["dates"][replay].astype(int)) <= since
    assert model.update_dates(context, data, since) == (new, replay)
    with pytest.raises(RuntimeError, match="no date labelled after"):
        model.update_dates(context, data, int(data["dates"][labelled[-1]]))

    cache = model.batches(data, new + replay, torch.device("cpu"))
    shuffled = model.placebo_labels(cache, data, new)
    for t in new:
        assert not torch.equal(shuffled[t][1], cache[t][1])
        assert torch.equal(torch.sort(shuffled[t][1]).values, torch.sort(cache[t][1]).values)
        assert shuffled[t][0] is cache[t][0]
    assert all(shuffled[t] is cache[t] for t in replay)
    again = model.placebo_labels(cache, data, new)
    assert all(torch.equal(again[t][1], shuffled[t][1]) for t in new)

    # The update continues every member's own checkpoint and saves it; the
    # placebo's steps on permuted labels end somewhere else.
    monkeypatch.setattr(model, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(model, "members", lambda: [("mlp", 1000), ("ssm", 1400)])
    with pytest.raises(RuntimeError, match="without a checkpoint for mlp seed 1000"):
        model.update(context, data, since)
    start = {}
    for head, seed in model.members():
        torch.manual_seed(seed)
        start[head] = model.HEADS[head]().state_dict()
    ends = {}
    for placebo in (False, True):
        for head, seed in model.members():
            torch.save(start[head], model._path(context, head, seed))
        monkeypatch.setattr(knobs, "PLACEBO", placebo)
        assert model.update(context, data, since) == float(len(new))
        ends[placebo] = {head: torch.load(model._path(context, head, seed), weights_only=True) for head, seed in model.members()}
    for head in start:
        changed = [key for key in start[head] if not torch.equal(start[head][key], ends[False][head][key])]
        assert changed, head
        assert any(not torch.equal(ends[False][head][key], ends[True][head][key]) for key in start[head]), head
