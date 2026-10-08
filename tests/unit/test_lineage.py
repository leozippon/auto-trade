"""A lineage: earlier arms whose trials join this arm's freeze-gate family.

Arms are synthetic but shaped as the pipeline writes them -- ``hitl/params.json``,
a ledger whose research session lists its Validations, and a style sidecar per
Validation -- so extraction, the recorded lineage and the gate read real files.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autotrade.environment.artifacts import REVISION_MANIFEST_FILE
from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR
from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME
from autotrade.environment.runtime import write_json_atomic
from autotrade.pipelines import verdict
from autotrade.pipelines.config import AcceptanceRules
from autotrade.pipelines.experiment import (
    freeze_gate_for,
    lineage_ledger_record,
    lineage_summary,
)
from autotrade.pipelines.ledger import ExperimentLedger, lineage_record
from autotrade.pipelines.lineage import (
    extract_lineage,
    lineage_arm_ids,
    write_lineage,
)
from autotrade.pipelines.research_session import LINEAGE_NOTE, arm_record
from autotrade.pipelines.session_resume import REVISIONS_DIR, STEP_SIDECAR_DIR

RESEARCH_START, RESEARCH_END = "20210701", "20250630"
DAYS = [day.strftime("%Y%m%d") for day in pd.bdate_range(RESEARCH_START, RESEARCH_END)]
FIRST_YEAR = [day for day in DAYS if day <= "20220630"]
SCALE = math.sqrt(TRADING_DAYS_PER_YEAR)
_MARKET = np.random.default_rng(7)
BENCHMARK = _MARKET.normal(0.0, 0.20 / SCALE, len(DAYS))
SIZE = _MARKET.normal(0.0, 0.10 / SCALE, len(DAYS))
# The component the trials of one family share: two trials loading ``w`` and
# ``v`` on it correlate at about ``w·v``.
COMMON = _MARKET.normal(0.0, 0.10 / SCALE, len(DAYS))


def _payload(*, seed: int, loading: float, days=DAYS) -> dict[str, object]:
    """One Validation's style sidecar over ``days``, with a zero-skill panel,
    so the graded series is the active one, as on every Validation now."""

    rng = np.random.default_rng(seed)
    keep = [index for index, day in enumerate(DAYS) if day in set(days)]
    noise = rng.normal(0.0, 0.10 / SCALE, len(DAYS))
    strategy = (
        0.08 / TRADING_DAYS_PER_YEAR
        + 0.8 * BENCHMARK
        + 0.3 * SIZE
        + loading * COMMON
        + math.sqrt(1.0 - loading**2) * noise
    )
    panel = 0.7 * BENCHMARK + rng.normal(0.0, 0.02 / SCALE, len(DAYS))

    def pairs(values) -> list[list[object]]:
        return [[DAYS[index], float(values[index])] for index in keep]

    return {
        "strategy_daily": pairs(strategy),
        "benchmark_daily": pairs(BENCHMARK),
        "size_factor_daily": pairs(SIZE),
        "panel_daily": pairs(panel),
    }


def _row(directory: Path, index: int, *, seed: int, loading: float, span: str = "full",
         control: bool = False, batch: str = "b1", offline: int | None = 0, days=DAYS,
         bytes_of: int | None = None, fingerprint: str | None = None) -> dict[str, object]:
    """One ledger ``steps[]`` row, its result and style sidecar on disk, and
    the strategy its revision holds, which the gate reads for a nominee.

    Each row replays bytes of its own unless ``bytes_of`` names the row index
    whose bytes it validates again (another revision of the same strategy), or
    ``fingerprint`` names bytes another arm validates too.
    """

    result = directory / "artifacts/results" / f"valid_{index:03d}"
    payload = _payload(seed=seed, loading=loading, days=days)
    write_json_atomic(result / "result.json", {"initial_cash": 1_000_000.0})
    write_json_atomic(result / STYLE_ARTIFACT_NAME, payload)
    revision = f"revision_{directory.name}_{index}"
    fingerprint = fingerprint or f"bytes_{directory.name}_{index if bytes_of is None else bytes_of}"
    output = directory / REVISIONS_DIR / revision / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "main.py").write_text("def generate_orders(context):\n    return []\n", encoding="utf-8")
    write_json_atomic(
        directory / REVISIONS_DIR / revision / REVISION_MANIFEST_FILE,
        {"revision_id": revision, "fingerprint": fingerprint},
    )
    return {
        "step_id": f"research__{directory.name}__valid_{index:03d}",
        "revision_id": revision,
        "fingerprint": fingerprint,
        "control": control,
        "batch_id": batch,
        "offline_trials": offline,
        "span": span,
        "summary": {"total_return": 0.1, "max_drawdown": 0.1},
        "validation_result_ref": str(result / "result.json"),
        "neutralized": verdict.neutralized_statistics(payload),
    }


def _arm(root: Path, experiment_id: str, rows, *, research=(RESEARCH_START, RESEARCH_END),
         forward: bool = False, pack: str = "", arm_end: str = "no_deliverable") -> Path:
    """An arm whose one research session recorded ``rows`` (``_row`` keywords).

    ``forward`` adds what a frozen arm goes on to write: a ``forward`` record
    naming a replay that runs past research end. ``pack`` is the reference
    pack it mounted; ``arm_end`` the status an unfrozen arm ended with.
    """

    directory = root / experiment_id
    write_json_atomic(
        directory / "hitl/params.json",
        {
            "experiment_id": experiment_id,
            "research_start": research[0],
            "research_end": research[1],
            **({"workspace_reference": pack} if pack else {}),
        },
    )
    ledger = ExperimentLedger(directory / "ledgers/experiment_ledger.jsonl")
    ledger.append(
        {
            "record_type": "research_session",
            "experiment_id": experiment_id,
            "epoch_id": "research",
            "fold_id": "research",
            "run_id": f"run_{experiment_id}",
            "session_key": "research",
            "steps": [_row(directory, index, **kwargs) for index, kwargs in enumerate(rows)],
            "arm_end": None if forward else {"status": arm_end, "reason": "no_edge"},
            "frozen": {"artifact_id": "strategy_research_x"} if forward else None,
        }
    )
    if forward:
        replay = directory / "artifacts/results/heldout_000"
        days = [day.strftime("%Y%m%d") for day in pd.bdate_range("20250701", "20260630")]
        write_json_atomic(
            replay / STYLE_ARTIFACT_NAME,
            {name: [[day, 0.001] for day in days] for name in ("strategy_daily", "benchmark_daily")},
        )
        ledger.append(
            {
                "record_type": "forward",
                "experiment_id": experiment_id,
                "epoch_id": "forward",
                "fold_id": "forward",
                "run_id": f"run_{experiment_id}_forward",
                "result_ref": str(replay / "result.json"),
                "verdict": {"status": "graduated", "reasons": []},
            }
        )
    return directory


def _analysis(row) -> dict[str, object]:
    return json.loads(
        (Path(str(row["validation_result_ref"])).parent / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8")
    )


def _record(arm: Path, extraction, *, pack: str = "") -> dict[str, object]:
    """What creation and the start of the research session write: the
    console's series file, then the pipeline's ledger record read from it,
    naming the reference ``pack`` the arm mounts."""

    write_lineage(arm, extraction)
    record = lineage_ledger_record(arm, workspace_reference=pack)
    ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl").append(record)
    return record


def _own_rows(directory: Path) -> list[dict[str, object]]:
    """The new arm's own session: a control, the nominee and a sub-span trial,
    whose first batch declared one configuration screened offline."""

    return [
        _row(directory, 0, seed=100, loading=0.0, control=True, batch="own1", offline=1),
        _row(directory, 1, seed=101, loading=0.6, batch="own1", offline=1),
        _row(directory, 2, seed=102, loading=0.6, span="Y1", batch="own2", offline=0, days=FIRST_YEAR),
    ]


def test_an_arm_without_a_lineage_deflates_over_its_own_family(tmp_path: Path) -> None:
    """Its two host trials at their measured correlation and its one declared
    screen as one more trial; a control is no trial."""

    rows = _own_rows(tmp_path / "new_arm")
    dsr = freeze_gate_for(
        [], rows, rows[1], experiment_dir=tmp_path / "new_arm", acceptance=AcceptanceRules()
    )["deflated_sharpe"]
    assert {
        key: dsr[key]
        for key in ("trials", "host_trials", "offline_trials", "controls", "trial_correlation_pairs")
    } == {"trials": 3, "host_trials": 2, "offline_trials": 1, "controls": 1, "trial_correlation_pairs": 1}
    correlation = dsr["trial_correlation"]
    assert correlation == pytest.approx(0.25222736470355794, rel=1e-12)
    assert dsr["effective_trials"] == pytest.approx(correlation + (1 - correlation) * 2 + 1)
    assert (dsr["lineage_trials"], dsr["lineage_arms"]) == (0, [])
    assert "lineage" not in arm_record(())


def test_a_correlated_lineage_adds_almost_nothing_and_an_independent_one_its_count() -> None:
    """N_eff = ρ̄ + (1 − ρ̄)·M over the union: lineage trials that are the
    nominee again (correlation 1) leave one effective trial; independent ones
    count one each."""

    nominee = _payload(seed=1, loading=0.0)
    own = verdict.neutral_daily(nominee)

    def gate(series) -> dict[str, object]:
        return verdict.freeze_gate(
            nominee,
            rules=AcceptanceRules(),
            trials=1,
            trial_analyses=[nominee],
            lineage_trials=len(series),
            lineage_series=series,
            full_span_validations=2,
            summary={},
        )["deflated_sharpe"]

    alone = gate([])
    copies = gate([{day: scale * value for day, value in own.items()} for scale in (0.5, 2.0, 3.0, 4.0, 5.0)])
    assert (alone["trials"], copies["trials"]) == (1, 6)
    assert copies["trial_correlation"] == pytest.approx(1.0)
    assert copies["effective_trials"] == pytest.approx(alone["effective_trials"]) == pytest.approx(1.0)
    assert copies["information_ratio_bar"] == pytest.approx(alone["information_ratio_bar"])

    independent = gate(
        [verdict.neutral_daily(_payload(seed=10 + index, loading=0.0)) for index in range(8)]
    )
    assert independent["trials"] == 9
    assert independent["effective_trials"] == pytest.approx(9.0, abs=0.3)
    assert independent["information_ratio_bar"] > alone["information_ratio_bar"] + 0.5


def test_the_recorded_lineage_joins_the_family_and_outlives_its_arms(tmp_path: Path) -> None:
    """Lineage arms' non-control revisions and declared offline screens join
    the family, their controls and their forward replay do not, and once the
    lineage is recorded the gate reads only the arm's own files."""

    root = tmp_path / "experiments"
    first = _arm(
        root,
        "first",
        [
            {"seed": 1, "loading": 0.6, "control": True, "offline": 2},
            {"seed": 2, "loading": 0.6, "offline": 2},
            {"seed": 3, "loading": 0.6, "span": "Y1", "days": FIRST_YEAR, "batch": "b2", "offline": 1},
        ],
        forward=True,
    )
    second = _arm(
        root,
        "second",
        [{"seed": 4, "loading": 0.6, "control": True}, {"seed": 5, "loading": 0.6}],
    )
    lineage_rows = [
        row
        for arm in (first, second)
        for row in ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl").read("research_session")[0]["steps"]
        if not row["control"]
    ]
    lineage_analyses = [_analysis(row) for row in lineage_rows]

    arm = root / "new_arm"
    own = _own_rows(arm)
    extraction = extract_lineage(
        root, ["first", "second"], research_start=RESEARCH_START, research_end=RESEARCH_END
    )
    assert [item["revision_id"] for item in extraction["series"]] == [
        "revision_first_1",
        "revision_first_2",
        "revision_second_1",
    ]
    assert max(day for item in extraction["series"] for day, _ in item["daily"]) <= RESEARCH_END
    record = _record(arm, extraction)
    # 2 + 1 revisions and 2 + 1 declared screens; the two controls are not trials.
    assert (record["trials"], record["host_trials"], record["offline_trials"], record["controls"]) == (6, 3, 3, 2)
    correlation, _pairs = verdict.trial_correlation(lineage_analyses)
    assert record["effective_trials"] == pytest.approx(correlation + (1 - correlation) * 3 + 3)

    shutil.rmtree(first)
    shutil.rmtree(second)
    records = ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl").read()
    dsr = freeze_gate_for(records, own, own[1], experiment_dir=arm, acceptance=AcceptanceRules())["deflated_sharpe"]
    assert dsr["lineage_arms"] == ["first", "second"]
    assert (dsr["trials"], dsr["host_trials"], dsr["offline_trials"], dsr["lineage_trials"]) == (9, 2, 1, 6)
    assert dsr["controls"] == 1
    # The same formula and series as if every trial were the arm's own.
    direct = verdict.freeze_gate(
        _analysis(own[1]),
        rules=AcceptanceRules(),
        trials=2 + 3,
        offline_trials=1 + 3,
        trial_analyses=[_analysis(own[1]), _analysis(own[2]), *lineage_analyses],
        full_span_validations=2,
        summary={},
    )["deflated_sharpe"]
    assert dsr["trial_correlation_pairs"] == direct["trial_correlation_pairs"] == 10
    for key in ("trial_correlation", "effective_trials", "information_ratio_bar", "deflated_sharpe_probability"):
        assert dsr[key] == pytest.approx(direct[key], rel=1e-12), key
    # The lineage's three declared screens count one each like the arm's own
    # one; the validated five keep the union's ρ̄.
    union = dsr["trial_correlation"]
    assert dsr["effective_trials"] == pytest.approx(union + (1 - union) * 5 + 4)
    assert arm_record((), lineage_record(records))["lineage"] == {
        "arms": ["first", "second"],
        "trials": 6,
        "effective_trials": record["effective_trials"],
        "note": LINEAGE_NOTE,
    }
    with pytest.raises(ValueError, match="already records its lineage"):
        ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl").append(
            lineage_ledger_record(arm, workspace_reference="")
        )


def test_a_lineage_counts_one_strategy_validated_twice_as_one_trial(tmp_path: Path) -> None:
    """The lineage's trials are the gate's: a strategy probed on a year and
    then run on the full span is one trial whose series is the full span's,
    and a control later submitted as a candidate is one trial. An arm whose
    rows name no bytes reads them off its revision manifests, and one whose
    revisions were not kept cannot be a lineage."""

    root = tmp_path / "experiments"
    probed = _arm(
        root,
        "probed",
        [
            {"seed": 1, "loading": 0.6, "span": "Y1", "days": FIRST_YEAR},
            {"seed": 2, "loading": 0.6, "span": "Y1", "days": FIRST_YEAR},
            {"seed": 3, "loading": 0.6, "control": True},
            {"seed": 4, "loading": 0.6, "bytes_of": 0, "batch": "b2"},
            {"seed": 5, "loading": 0.6, "bytes_of": 2, "batch": "b2"},
        ],
    )
    extraction = extract_lineage(root, ["probed"], research_start=RESEARCH_START, research_end=RESEARCH_END)
    assert [(item["revision_id"], item["span"]) for item in extraction["series"]] == [
        ("revision_probed_1", "Y1"),
        ("revision_probed_2", "full"),
        ("revision_probed_3", "full"),
    ]
    assert {key: extraction["arms"][0][key] for key in ("host_trials", "controls")} == {
        "host_trials": 3,
        "controls": 0,
    }

    # A ledger written before rows named their bytes: the manifests do.
    ledger = ExperimentLedger(probed / "ledgers/experiment_ledger.jsonl")
    records = ledger.read()
    for row in records[0]["steps"]:
        fingerprint = row.pop("fingerprint")
        write_json_atomic(
            probed / REVISIONS_DIR / str(row["revision_id"]) / REVISION_MANIFEST_FILE,
            {"revision_id": row["revision_id"], "fingerprint": fingerprint},
        )
    ledger.rewrite(records)
    assert extract_lineage(root, ["probed"], research_start=RESEARCH_START, research_end=RESEARCH_END) == extraction
    shutil.rmtree(probed / REVISIONS_DIR / "revision_probed_4")
    with pytest.raises(ValueError, match="lineage arm probed: revision revision_probed_4 has no manifest"):
        extract_lineage(root, ["probed"], research_start=RESEARCH_START, research_end=RESEARCH_END)


def test_bytes_validated_in_two_arms_are_one_trial(tmp_path: Path) -> None:
    """Sibling arms on a pack that prescribes its batches validate the same
    bytes. The arm and its lineage are one family: those bytes are one trial,
    represented by their longest series -- the lineage's full span, not the
    arm's own probe of them on a year."""

    root = tmp_path / "experiments"
    for name, seed in (("qwen", 2), ("mimo", 3)):
        _arm(root, name, [{"seed": 1, "loading": 0.6, "fingerprint": "prescribed"}, {"seed": seed, "loading": 0.6}])
    extraction = extract_lineage(root, ["qwen", "mimo"], research_start=RESEARCH_START, research_end=RESEARCH_END)
    assert [arm["fingerprints"] for arm in extraction["arms"]] == [
        ["prescribed", "bytes_qwen_1"],
        ["prescribed", "bytes_mimo_1"],
    ]
    assert lineage_summary(extraction)["host_trials"] == 3

    arm = root / "heir"
    own = [
        _row(arm, 0, seed=1, loading=0.6, span="Y1", days=FIRST_YEAR, fingerprint="prescribed"),
        _row(arm, 1, seed=101, loading=0.6),
    ]
    _record(arm, extraction)
    records = ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl").read()
    joined = freeze_gate_for(records, own, own[1], experiment_dir=arm, acceptance=AcceptanceRules())[
        "deflated_sharpe"
    ]
    assert (joined["trials"], joined["host_trials"], joined["lineage_trials"]) == (4, 2, 2)
    qwen, mimo = (
        ExperimentLedger(root / name / "ledgers/experiment_ledger.jsonl").read()[0]["steps"] for name in ("qwen", "mimo")
    )
    direct = verdict.freeze_gate(
        _analysis(own[1]),
        rules=AcceptanceRules(),
        trials=4,
        trial_analyses=[_analysis(own[1]), _analysis(qwen[0]), _analysis(qwen[1]), _analysis(mimo[1])],
        full_span_validations=1,
        summary={},
    )["deflated_sharpe"]
    assert joined["trial_correlation_pairs"] == direct["trial_correlation_pairs"] == 6
    for key in ("trial_correlation", "effective_trials", "information_ratio_bar", "deflated_sharpe_probability"):
        assert joined[key] == pytest.approx(direct[key], rel=1e-12), key


def test_a_reference_pack_declares_its_screens_once(tmp_path: Path) -> None:
    """Arms mounting one reference pack declare its screens once, at the most
    any of them declared: 17 and 20 add 20, however the pack's path is spelt.
    Arms on other packs and an arm mounting none add theirs, and the arm's own
    declarations join its pack's. A lineage extracted before its trials
    carried their bytes cannot be joined."""

    root = tmp_path / "experiments"
    _arm(root, "qwen", [{"seed": 1, "loading": 0.5, "offline": 17}], pack="configs/workspace_refs/book")
    _arm(root, "mimo", [{"seed": 2, "loading": 0.5, "offline": 20}], pack="configs/workspace_refs/book/")
    _arm(root, "star", [{"seed": 3, "loading": 0.5, "offline": 6}], pack="configs/workspace_refs/star")
    _arm(root, "bare", [{"seed": 4, "loading": 0.5, "offline": 2}])
    names = ["qwen", "mimo", "star", "bare"]
    extraction = extract_lineage(root, names, research_start=RESEARCH_START, research_end=RESEARCH_END)
    assert lineage_summary(extraction)["offline_trials"] == 20 + 6 + 2

    arm = root / "heir"
    own = [_row(arm, 0, seed=101, loading=0.5, offline=8)]
    _record(arm, extraction, pack="configs/workspace_refs/star")
    records = ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl").read()
    rules = AcceptanceRules()
    dsr = freeze_gate_for(records, own, own[0], experiment_dir=arm, acceptance=rules)["deflated_sharpe"]
    # The arm's own 8 on the star pack stand for that pack's 6: the lineage
    # adds its four strategies and 20 + 2 screens.
    assert (dsr["host_trials"], dsr["offline_trials"], dsr["lineage_trials"], dsr["trials"]) == (1, 8, 4 + 22, 35)
    assert dsr["effective_trials"] == pytest.approx(
        dsr["trial_correlation"] + (1 - dsr["trial_correlation"]) * 5 + 30
    )
    assert arm_record((), lineage_record(records))["lineage"]["note"] == LINEAGE_NOTE

    stale = json.loads((arm / "ledgers/lineage_series.json").read_text(encoding="utf-8"))
    for item in (*stale["arms"], *stale["series"]):
        item.pop("fingerprints", None)
        item.pop("fingerprint", None)
        item.pop("workspace_reference", None)
    write_lineage(arm, stale)
    with pytest.raises(ValueError, match="the lineage of qwen, mimo, star, bare was extracted before"):
        freeze_gate_for(records, own, own[0], experiment_dir=arm, acceptance=rules)


def test_a_lineage_arm_that_cannot_be_one_is_refused_by_name(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    _arm(root, "ok", [{"seed": 1, "loading": 0.5}])
    _arm(root, "controls_only", [{"seed": 2, "loading": 0.5, "control": True}])
    _arm(root, "four_years_earlier", [{"seed": 3, "loading": 0.5}], research=("20200701", "20240630"))
    # Its trials are real Validations, but the host stopped the search they
    # belong to: no lineage counts them.
    _arm(root, "blocked", [{"seed": 5, "loading": 0.5}], arm_end="environment_blocked")
    leaky = _arm(root, "leaky", [{"seed": 4, "loading": 0.5}])
    sidecar = leaky / "artifacts/results/valid_000" / STYLE_ARTIFACT_NAME
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    for name in ("strategy_daily", "benchmark_daily", "size_factor_daily", "panel_daily"):
        payload[name].append(["20250701", float(np.random.default_rng(len(name)).normal(0.0, 0.01))])
    write_json_atomic(sidecar, payload)
    for arms, message in (
        (["ok", "missing"], "lineage arm missing does not exist"),
        (
            ["four_years_earlier"],
            "lineage arm four_years_earlier researched 20200701..20240630, not this arm's 20210701..20250630",
        ),
        (["controls_only"], "lineage arm controls_only has no recorded non-control trial"),
        (["ok", "blocked"], "lineage arm blocked ended environment_blocked"),
        (["leaky"], "lineage arm leaky revision revision_leaky_0 has days outside the research period"),
    ):
        with pytest.raises(ValueError, match=re.escape(message)):
            extract_lineage(root, arms, research_start=RESEARCH_START, research_end=RESEARCH_END)
    for value, message in (
        (["new_arm"], "lineage_arms lists the arm itself (new_arm)"),
        (["ok", "ok"], "lineage_arms lists ok more than once"),
        (["../ok"], "not experiment ids"),
        ("ok", "must be a list of experiment ids"),
    ):
        with pytest.raises(ValueError, match=re.escape(message)):
            lineage_arm_ids(value, "new_arm")
    assert lineage_arm_ids([], "new_arm") == ()


def test_the_research_session_gates_on_the_lineage_its_ledger_records(tmp_path: Path) -> None:
    """End to end through the pipeline: an arm whose params name a lineage its
    creation never wrote refuses to start; one created with it starts from an
    empty ledger, records the lineage itself and its freeze gate counts it."""

    from tests.unit.test_rolling_pipeline import GEOMETRY, _freezing

    pipeline, *_rest, ledger = _freezing(tmp_path)
    pipeline.config = replace(pipeline.config, lineage_arms=("earlier",))
    with pytest.raises(RuntimeError, match="its creation wrote no lineage series"):
        pipeline.run_research_session()
    assert ledger.read() == []
    research = (GEOMETRY.research_start, GEOMETRY.research_end)
    span = [day for day in DAYS if research[0] <= day <= research[1]]
    _arm(
        tmp_path / "experiments",
        "earlier",
        [
            {"seed": 6, "loading": 0.3, "days": span, "offline": 4},
            {"seed": 7, "loading": 0.3, "days": span, "offline": 4},
        ],
        research=research,
    )
    write_lineage(
        pipeline.config.experiment_dir,
        extract_lineage(tmp_path / "experiments", ["earlier"], research_start=research[0], research_end=research[1]),
    )
    assert ledger.read() == []
    dsr = pipeline.run_research_session()["freeze_gate"]["deflated_sharpe"]
    assert (dsr["trials"], dsr["host_trials"], dsr["lineage_trials"]) == (8, 2, 6)
    assert dsr["lineage_arms"] == ["earlier"]
    assert [record["record_type"] for record in ledger.read()] == ["lineage", "research_session"]


def test_the_console_listing_counts_the_lineage_as_the_gate_does(tmp_path: Path) -> None:
    """A running session's best node, read off its live Validations, is
    deflated against the same trials, N_eff and DSR ``freeze_gate_for``
    gives, lineage included."""

    from autotrade.webui.registry import _research_best

    root = tmp_path / "experiments"
    _arm(root, "first", [{"seed": 2, "loading": 0.6}, {"seed": 3, "loading": 0.6}])
    arm = root / "new_arm"
    write_json_atomic(arm / "hitl/params.json", {"experiment_id": "new_arm"})
    own = _own_rows(arm)
    _record(arm, extract_lineage(root, ["first"], research_start=RESEARCH_START, research_end=RESEARCH_END))
    for row in own:
        write_json_atomic(
            arm / STEP_SIDECAR_DIR / f"{row['step_id']}.json",
            {
                **{key: row[key] for key in ("step_id", "revision_id", "span", "summary", "control", "batch_id", "offline_trials")},
                "result_ref": row["validation_result_ref"],
            },
        )
    ledger = ExperimentLedger(arm / "ledgers/experiment_ledger.jsonl")
    best = _research_best(arm, ledger.read(), running=True)
    dsr = freeze_gate_for(ledger.read(), own, own[1], experiment_dir=arm, acceptance=AcceptanceRules())["deflated_sharpe"]
    assert dsr["lineage_trials"] == 2
    assert best["step_id"] == own[1]["step_id"]
    assert best["trials"] == dsr["trials"] == 5
    assert best["deflated_sharpe_probability"] == pytest.approx(dsr["deflated_sharpe_probability"])
