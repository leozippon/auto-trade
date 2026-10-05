"""Seed replicates: a strategy that trains a model is judged on its seeds together.

The arm's own pipeline runs end to end -- the real artifact store, ledger,
freeze gate and verdict -- over scripted Validations whose daily active return
is set per training seed (``SEED_BASE`` in the candidate's ``main.py``) and per
stage, so the readings each seed gets are known before the gate reads them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME
from autotrade.pipelines.config import (
    ArtifactRevision,
    EvaluationRequest,
    EvaluationResult,
    ResearchSessionRequest,
    ResearchSessionResult,
    RollingExperimentConfig,
    StepResult,
    acceptance_for,
    research_span,
)
from autotrade.pipelines.experiment import RollingExperimentPipeline, freeze_gate_for
from autotrade.pipelines.ledger import (
    ExperimentLedger,
    experiment_verdict,
    forward_record,
)
from tests.unit.test_rolling_pipeline import (
    CONFIG_PROFILE,
    CONFIG_SCHEDULE,
    DAYS,
    GEOMETRY,
    RELEASE_END,
    Snapshots,
    _write_result,
)

FIT = '''REFIT_PERIOD = "quarter"
SEED_BASE = {seed}
SEEDS_PER_HEAD = {heads}
MAX_SWAPS = {swaps}


def fit(context):
    return None


def generate_orders(context):
    return []
'''
NO_FIT = '''SEED_BASE = {seed}
SEEDS_PER_HEAD = {heads}
MAX_SWAPS = {swaps}


def generate_orders(context):
    return []
'''
SEEDED = acceptance_for(
    {"require_seed_replicates": True, "require_forward_plain_selection": True}
)
# Daily active return by stage and training seed: research Validations replay
# in mode ``valid``, the forward stage in ``heldout``. An active return of
# 0.0009 a day or more reads an IR above 3.5 over the two research years, far
# above any bar the few trials here set; -0.0010 reads near -3.9. Forward,
# seed 1000 beats its panel well clear of its noise, seed 2000 barely and seed
# 3000 loses nearly as much as the other two win: their mean stays above zero
# and inside its noise.
ALPHAS = {
    ("valid", 1000): 0.0012,
    ("valid", 2000): 0.0010,
    ("valid", 3000): 0.0009,
    ("valid", 4000): -0.0010,
    ("heldout", 1000): 0.0015,
    ("heldout", 2000): 0.0002,
    ("heldout", 3000): -0.0014,
}


def _source(seed: int, *, swaps: int = 4, heads: int = 2, fit: bool = True) -> str:
    return (FIT if fit else NO_FIT).format(seed=seed, swaps=swaps, heads=heads)


class SeedEvaluator:
    """A result whose zero-skill panel is the replay's market and size
    exposure, so the active series is the seed's daily ``alpha`` plus noise."""

    def __init__(
        self, root: Path, *, fail_seed: int | None = None, bare_seed: int | None = None
    ) -> None:
        self.root = root
        self.requests: list[EvaluationRequest] = []
        # Forward, this seed's replay raises; that one's stores no panel.
        self.fail_seed = fail_seed
        self.bare_seed = bare_seed

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.requests.append(request)
        source = (Path(request.revision.output_path) / "main.py").read_text(encoding="utf-8")
        seed = int(re.search(r"SEED_BASE = (\d+)", source)[1])  # type: ignore[index]
        if seed == self.fail_seed and request.mode == "heldout":
            raise TimeoutError("strategy inference exceeded 360s")
        result = _write_result(
            self.root / f"{request.mode}_{len(self.requests):03d}",
            [day for day in DAYS if request.start <= day <= min(request.end, RELEASE_END)],
            alpha=ALPHAS[(request.mode, seed)],
            seed=len(self.requests),
        )
        if seed == self.bare_seed and request.mode == "heldout":
            return result
        sidecar = Path(result.result_ref).parent / STYLE_ARTIFACT_NAME
        analysis = json.loads(sidecar.read_text(encoding="utf-8"))
        analysis["panel_daily"] = [
            [day, 0.9 * market + 0.2 * size]
            for (day, market), (_day, size) in zip(
                analysis["benchmark_daily"], analysis["size_factor_daily"], strict=True
            )
        ]
        sidecar.write_text(json.dumps(analysis), encoding="utf-8")
        return result


class SeedDeveloper:
    """Validates each scripted ``(source, span)`` and freezes ``nominee`` with
    the candidates ``replicates`` names as its seed replicates."""

    def __init__(self, store, evaluator, candidates, *, nominee: int, replicates):
        self.store = store
        self.evaluator = evaluator
        self.candidates = candidates
        self.nominee = nominee
        self.replicates = tuple(replicates)

    def __call__(self, request: ResearchSessionRequest) -> ResearchSessionResult:
        steps = []
        for number, (source, span) in enumerate(self.candidates):
            directory = self.store.root.parent / "candidates" / f"candidate_{number}"
            directory.mkdir(parents=True)
            (directory / "main.py").write_text(source, encoding="utf-8")
            revision = self.store.create_revision(directory)
            typed = ArtifactRevision(str(revision.revision_id), Path(revision.output_path))
            replayed = research_span(request.research_years, span)
            validation = self.evaluator.evaluate(
                replayed.request(typed, schedule=CONFIG_SCHEDULE, broker_profile=CONFIG_PROFILE)
            )
            steps.append(
                StepResult(f"research_step_{number}", typed.revision_id, validation, span=replayed.label)
            )
        return ResearchSessionResult(
            f"conversation_{request.run_id}",
            tuple(steps),
            "freeze",
            node_id=steps[self.nominee].step_id,
            seed_replicates=tuple(steps[index].step_id for index in self.replicates),
        )


def _arm(tmp_path: Path, candidates, *, nominee=0, replicates=(), rules=SEEDED, **replays):
    config = RollingExperimentConfig(
        experiment_id="arm",
        experiments_root=tmp_path / "experiments",
        geometry=GEOMETRY,
        acceptance=rules,
    )
    store = FilesystemArtifactStore(config.experiment_dir / "artifacts" / "strategy")
    evaluator = SeedEvaluator(config.experiment_dir / "artifacts" / "results", **replays)
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=store,
        evaluator=evaluator,
        developer=SeedDeveloper(store, evaluator, candidates, nominee=nominee, replicates=replicates),
        trading_days=DAYS,
        ledger=ExperimentLedger(config.ledger_path),
    )
    return pipeline, evaluator


def test_a_model_nominee_freezes_with_its_seed_replicate_and_both_replay_forward(tmp_path: Path):
    """The registered replicates are checked, counted in the seed mean, frozen
    beside the nominee, and replayed exactly like it; the forward record
    carries their readings and the series the seeds are judged on together,
    whose mean above zero does not graduate a book while its bound is not."""

    pipeline, evaluator = _arm(
        tmp_path,
        [(_source(1000), "full"), (_source(2000), "full"), (_source(3000), "full")],
        replicates=(1, 2),
    )
    record = pipeline.run_research_session()
    gate = record["freeze_gate"]
    seeds = gate["seed_replicates"]
    assert gate["passed"] is True and gate["reasons"] == []
    assert seeds["trains_a_model"] is True
    assert [entry["seed_line"] for entry in seeds["replicates"]] == [
        "main.py: SEED_BASE = 2000",
        "main.py: SEED_BASE = 3000",
    ]
    ratios = [gate["information_ratio"], *(entry["information_ratio"] for entry in seeds["replicates"])]
    assert seeds["mean_information_ratio"] == pytest.approx(sum(ratios) / 3)
    assert seeds["mean_information_ratio"] >= seeds["information_ratio_bar"]
    assert gate["thresholds"]["require_seed_replicates"] is True
    frozen = record["frozen"]
    assert [item["source_step_id"] for item in frozen["seed_replicates"]] == [
        "research_step_1",
        "research_step_2",
    ]
    assert all(Path(item["output_path"], "main.py").is_file() for item in frozen["seed_replicates"])

    research = len(evaluator.requests)
    forward = pipeline.run_forward()
    replays = evaluator.requests[research:]

    # Three replays of one span, under one Broker, the nominee first.
    assert [request.revision.revision_id for request in replays] == [
        frozen["artifact_id"],
        *(item["artifact_id"] for item in frozen["seed_replicates"]),
    ]
    shapes = [(r.mode, r.start, r.end, r.continuation, r.broker_profile) for r in replays]
    assert shapes == [shapes[0]] * 3
    assert [row["source_step_id"] for row in forward["seed_replicates"]] == [
        "research_step_1",
        "research_step_2",
    ]
    assert forward["seed_replicates"][0]["refits_executed"] == forward["refits_executed"]
    for name in ("forward", "heldout"):
        block = forward["slices"][name]
        members = [block, *block["seed_replicates"]]
        assert block["seed_mean"]["members"] == 3
        for key in ("plain_selection", "strategy_return", "panel_return"):
            assert block["seed_mean"]["raw_readings"][key] == pytest.approx(
                sum(member["raw_readings"][key] for member in members) / 3
            )
    block = forward["slices"]["forward"]
    judged = block["judged_selection"]
    # The nominee itself beat its panel, and so does the mean over the three
    # seeds; the mean's lower bound does not clear zero.
    assert block["raw_readings"]["plain_selection"] > 0
    assert block["seed_mean"]["raw_readings"]["plain_selection"] > 0
    assert judged["members"] == 3
    assert judged["mean"] == pytest.approx(block["seed_mean"]["plain_excess"])
    assert judged["lower_bound"] < 0 < judged["mean"]
    assert forward["verdict"]["reasons"] == ["forward_plain_selection_lower_bound_not_positive"]
    assert forward["verdict"]["thresholds"]["require_forward_plain_selection"] is True
    assert experiment_verdict(pipeline.ledger.read())["status"] == "discarded"


def test_seeds_that_agree_forward_graduate_together(tmp_path: Path):
    pipeline, _evaluator = _arm(
        tmp_path, [(_source(1000), "full"), (_source(2000), "full")], replicates=(1,)
    )
    pipeline.run_research_session()
    forward = pipeline.run_forward()
    judged = forward["slices"]["forward"]["judged_selection"]
    assert judged["members"] == 2 and judged["lower_bound"] > 0
    assert forward["verdict"]["status"] == "graduated"


def _refused(tmp_path: Path, candidates, **arm) -> dict[str, object]:
    pipeline, _evaluator = _arm(tmp_path, candidates, **arm)
    record = pipeline.run_research_session()
    assert record["frozen"] is None
    assert record["arm_end"]["status"] == "no_deliverable"
    return record["freeze_gate"]


def test_a_model_nominee_without_a_replicate_is_refused(tmp_path: Path):
    gate = _refused(tmp_path, [(_source(1000), "full"), (_source(1000, swaps=3), "full")])
    assert gate["reasons"] == ["freeze_too_few_seed_replicates"]
    assert gate["seed_replicates"]["replicates"] == []


@pytest.mark.parametrize(
    ("replicate", "problem"),
    [
        ((_source(1000), "full"), "holds the same bytes as the nominee"),
        ((_source(2000, swaps=3), "full"), "more than one line"),
        ((_source(1000, swaps=3), "full"), "does not give a seed name"),
        ((_source(1000, heads=4), "full"), "does not give a seed name"),
        ((_source(2000), "Y2"), "replayed span Y2, not full"),
    ],
    ids=["same_bytes", "seed_and_a_knob", "a_knob_not_the_seed", "a_seed_count", "another_span"],
)
def test_a_replicate_that_is_not_the_nominee_on_another_seed_is_refused(
    tmp_path: Path, replicate, problem
):
    gate = _refused(
        tmp_path,
        [(_source(1000), "full"), (_source(3000, swaps=2), "full"), replicate],
        replicates=(2,),
    )
    assert gate["reasons"] == ["freeze_seed_replicate_invalid"]
    [entry] = gate["seed_replicates"]["replicates"]
    assert problem in entry["problem"]


def test_a_nominee_above_its_bar_is_refused_when_its_seed_mean_is_below(tmp_path: Path):
    gate = _refused(
        tmp_path,
        [(_source(1000), "full"), (_source(4000), "full")],
        replicates=(1,),
    )
    seeds = gate["seed_replicates"]
    assert gate["information_ratio"] >= seeds["information_ratio_bar"]
    assert seeds["mean_information_ratio"] < seeds["information_ratio_bar"]
    assert gate["reasons"] == ["freeze_seed_mean_information_ratio_below_threshold"]


def test_a_nominee_that_trains_no_model_is_judged_on_its_own_series(tmp_path: Path):
    """A nominee that trains no model needs no replicate: it freezes alone,
    its gate names the rule the arm holds with nothing to replicate, and
    forward its own series is judged, with its bound."""

    pipeline, evaluator = _arm(
        tmp_path, [(_source(1000, fit=False), "full"), (_source(2000, fit=False), "full")]
    )
    record = pipeline.run_research_session()
    gate = record["freeze_gate"]
    assert gate["passed"] is True
    assert gate["seed_replicates"]["trains_a_model"] is False
    assert gate["seed_replicates"]["replicates"] == []
    assert gate["thresholds"]["require_seed_replicates"] is True
    assert "seed_replicates" not in record["frozen"]
    research = len(evaluator.requests)
    forward = pipeline.run_forward()
    assert len(evaluator.requests) == research + 1
    block = forward["slices"]["forward"]
    assert "seed_mean" not in block
    assert block["judged_selection"]["members"] == 1
    assert block["judged_selection"]["mean"] == pytest.approx(block["plain_excess"])
    assert 0 < block["judged_selection"]["lower_bound"] < block["judged_selection"]["mean"]
    assert forward["verdict"]["status"] == "graduated"


def test_an_arm_without_the_condition_freezes_and_judges_as_before(tmp_path: Path):
    """No key, no condition: a model nominee freezes alone, its gate and
    frozen block carry no seed block, one replay judges it, and naming a
    replicate under these rules is refused rather than ignored."""

    rules = acceptance_for({"require_forward_plain_selection": True})
    pipeline, evaluator = _arm(
        tmp_path, [(_source(1000), "full"), (_source(2000), "full")], rules=rules
    )
    record = pipeline.run_research_session()
    assert record["freeze_gate"]["passed"] is True
    assert "seed_replicates" not in record["freeze_gate"]
    assert "require_seed_replicates" not in record["freeze_gate"]["thresholds"]
    assert "seed_replicates" not in record["frozen"]
    research = len(evaluator.requests)
    forward = pipeline.run_forward()
    assert len(evaluator.requests) == research + 1
    assert "seed_replicates" not in forward
    assert "seed_mean" not in forward["slices"]["forward"]
    assert forward["slices"]["forward"]["judged_selection"]["members"] == 1
    assert forward["verdict"]["status"] == "graduated"

    rows = record["steps"]
    with pytest.raises(ValueError, match="no seed-replicate condition"):
        freeze_gate_for(
            [],
            rows,
            rows[0],
            experiment_dir=pipeline.config.experiment_dir,
            acceptance=rules,
            seed_replicates=[rows[1]],
        )


@pytest.mark.parametrize(
    ("replays", "error", "message"),
    [
        (
            {"fail_seed": 2000},
            RuntimeError,
            r"seed replicate research_step_1 \(.*\) did not complete",
        ),
        ({"bare_seed": 2000}, ValueError, "no plain selection to judge"),
    ],
    ids=["its_replay_fails", "its_replay_has_no_panel"],
)
def test_a_replicate_missing_from_the_mean_fails_the_stage_and_leaves_no_verdict(
    tmp_path: Path, replays, error, message
):
    pipeline, _evaluator = _arm(
        tmp_path,
        [(_source(1000), "full"), (_source(2000), "full")],
        replicates=(1,),
        **replays,
    )
    pipeline.run_research_session()
    with pytest.raises(error, match=message):
        pipeline.run_forward()
    rows = pipeline.ledger.read()
    assert rows[-1]["record_type"] == "attempt_failed"
    assert forward_record(rows) is None
    assert experiment_verdict(rows) is None
