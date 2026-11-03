"""The ablation: a nominee frozen with its one-line control, read once on the
sealed years beside it and reported, never judged.

The arm's own pipeline runs end to end over the scripted Validations of
``test_seed_replicates``: a candidate whose ``MECHANISM = False`` earns less
than the same bytes with the mechanism on, so the increment the forward
stage reads is known to be positive before it is read.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from autotrade.environment.runtime import utc_now_iso
from autotrade.pipelines.experiment import replay_and_analysis
from autotrade.pipelines.verdict import ablation_increment
from scripts.dev.check_verdicts import check_arm
from tests.unit.acceptance_fixture import REQUIRED_RULES
from tests.unit.test_seed_replicates import _arm, _source

NOMINEE = _source(1000, fit=False)
ABLATION = _source(1000, fit=False, mechanism=False)


def test_an_ablation_is_frozen_and_its_increment_is_reported_never_judged(tmp_path: Path):
    pipeline, evaluator = _arm(
        tmp_path, [(NOMINEE, "full"), (ABLATION, "full")], ablation=1, ablation_required=True
    )
    record = pipeline.run_research_session()
    gate = record["freeze_gate"]
    assert gate["passed"] is True and gate["reasons"] == []
    assert gate["ablation"] == {
        "required": True,
        "step_id": "research_step_1",
        "line": "main.py:4: MECHANISM = True -> MECHANISM = False",
    }
    frozen = record["frozen"]["ablation"]
    assert frozen["source_step_id"] == "research_step_1"
    assert frozen["line"] == gate["ablation"]["line"]
    assert Path(frozen["output_path"], "main.py").read_text(encoding="utf-8") == ABLATION

    research = len(evaluator.requests)
    forward = pipeline.run_forward()
    replays = evaluator.requests[research:]
    # The nominee and then its ablation on the sealed years, as one book
    # each; then the nominee alone forward.
    assert [request.revision.revision_id for request in replays] == [
        record["frozen"]["artifact_id"],
        frozen["artifact_id"],
        record["frozen"]["artifact_id"],
    ]
    assert (replays[0].start, replays[0].end) == (replays[1].start, replays[1].end)
    sealed = forward["sealed"]
    reading = sealed["ablation"]
    assert reading["source_step_id"] == "research_step_1"
    increment = reading["increment"]
    # The sealed gate's statistic on the book's active series minus the
    # ablation's, with its t-statistic and no bar.
    expected = ablation_increment(
        replay_and_analysis(sealed["result_ref"])[1],
        replay_and_analysis(reading["result_ref"])[1],
        start=sealed["start"],
        end=sealed["end"],
        alpha=0.05,
    )
    assert increment == pytest.approx(expected)
    assert "bar" not in increment
    assert increment["residual_information_ratio"] > 0
    assert increment["t_statistic"] == pytest.approx(
        increment["residual_information_ratio"] * math.sqrt(increment["years"])
    )
    # Never a reason, whatever it reads.
    assert not any("ablation" in reason for reason in forward["verdict"]["reasons"])
    assert "ablation" not in sealed["slice"]


@pytest.mark.parametrize(
    ("candidates", "ablation", "reason", "problem"),
    [
        ([(NOMINEE, "full")], None, "freeze_ablation_missing", None),
        (
            [(NOMINEE, "full"), (_source(1000, fit=False, mechanism=False, swaps=3), "full")],
            1,
            "freeze_ablation_invalid",
            "more than one line of output/main.py (2 lines)",
        ),
        (
            [(NOMINEE, "full"), (NOMINEE.replace("MECHANISM = True\n", ""), "full")],
            1,
            "freeze_ablation_invalid",
            "changes the number of lines",
        ),
        ([(NOMINEE, "full"), (NOMINEE, "full")], 1, "freeze_ablation_invalid", "same bytes as the nominee"),
        ([(NOMINEE, "full")], 0, "freeze_ablation_invalid", "is the nominee itself"),
    ],
    ids=["missing", "two_lines", "a_dropped_line", "same_bytes", "the_nominee"],
)
def test_a_required_ablation_is_the_nominee_with_exactly_one_line_changed(
    tmp_path: Path, candidates, ablation, reason, problem
):
    pipeline, _evaluator = _arm(tmp_path, candidates, ablation=ablation, ablation_required=True)
    record = pipeline.run_research_session()
    gate = record["freeze_gate"]
    assert gate["reasons"] == [reason]
    assert record["frozen"] is None and record["arm_end"]["status"] == "no_deliverable"
    if problem is not None:
        assert problem in gate["ablation"]["problem"]


def test_an_arm_that_asks_for_no_ablation_refuses_one(tmp_path: Path):
    pipeline, _evaluator = _arm(tmp_path, [(NOMINEE, "full"), (ABLATION, "full")], ablation=1)
    gate = pipeline.run_research_session()["freeze_gate"]
    assert gate["reasons"] == ["freeze_ablation_invalid"]
    assert "ask for no ablation" in gate["ablation"]["problem"]


def test_the_consistency_check_judges_the_ablation_again(tmp_path: Path):
    """``check_verdicts`` reads the registration with its ablation and the
    increment from the two stored replays, and both reproduce."""

    pipeline, _evaluator = _arm(
        tmp_path, [(NOMINEE, "full"), (ABLATION, "full")], ablation=1, ablation_required=True
    )
    pipeline.run_research_session()
    pipeline.run_forward()
    arm = pipeline.config.experiment_dir
    (arm / "hitl" / "params.json").write_text(
        json.dumps({**REQUIRED_RULES, "ablation_required": True, "_created_at": utc_now_iso()}),
        encoding="utf-8",
    )
    judgements = check_arm(arm)
    assert [item.reading for item in judgements] == ["freeze_registration", "sealed", "ablation", "forward"]
    assert [item.differences for item in judgements] == [[], [], [], []]
