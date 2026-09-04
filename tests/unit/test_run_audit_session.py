from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_audit_session
from autotrade.environment.sandbox import SandboxSpec
from autotrade.pipelines import worker as worker_module
from autotrade.pipelines.config import ModificationConstraints
from autotrade.pipelines.pit_views_seed import DEFAULT_PIT_VIEWS_SEED

# Constructor arguments the console's session loop supplies and a single audited
# session has no place for: the smoke-test command runner, and the sink that
# hands a Meta session's rebuilt image to the Folds that would come after it.
# Everything else must reach the audited session, or the audit reports on a
# session the console never runs.
WORKER_LOOP_ONLY = {"command_runner_factory", "use_docker", "sandbox_spec_sink"}


class _ProviderConstructed(RuntimeError):
    pass


def test_audit_pipeline_uses_the_default_pit_view_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def build_provider(**kwargs: object) -> object:
        captured.update(kwargs)
        raise _ProviderConstructed

    monkeypatch.setattr(run_audit_session, "ExperimentLedger", lambda path: object())
    monkeypatch.setattr(
        run_audit_session, "FilesystemArtifactStore", lambda path: object()
    )
    monkeypatch.setattr(
        run_audit_session, "ResearchPITSnapshotProvider", build_provider
    )

    llm = SimpleNamespace(
        compact_enabled=False,
        build_gateway=lambda role: object(),
    )
    options = SimpleNamespace(
        rolling=SimpleNamespace(
            ledger_path=tmp_path / "ledger.jsonl",
            strategy_fit_timeout_seconds=3600,
            nl_failure_policy="fail",
        ),
        experiment_dir=tmp_path / "experiments" / "audit",
        llm=llm,
        agent_sandbox=SandboxSpec(),
        data_backend="pit",
        raw_dir=tmp_path / "raw",
        fundamental_events_root=tmp_path / "fundamentals",
        fundamental_events_status=tmp_path / "fundamentals-status.json",
        snapshot_config=object(),
        pit_cache_root=tmp_path / "pit-cache",
        repo_root=tmp_path,
    )

    with pytest.raises(_ProviderConstructed):
        run_audit_session._build_pipeline(options)

    assert captured["pit_views_seed"] == tmp_path / DEFAULT_PIT_VIEWS_SEED


def _constructor_arguments(path: Path, class_name: str) -> set[str]:
    """Keyword names one module passes to ``class_name(...)``."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == class_name
        ):
            names |= {kw.arg for kw in node.keywords if kw.arg}
    return names


@pytest.mark.parametrize("class_name", ("LLMFoldDeveloper", "LLMMetaLearner"))
def test_the_audit_session_is_built_like_the_console_session(class_name: str) -> None:
    """The module promises a session configured identically to the console's.
    An argument the worker passes and this script drops is a silently different
    session: a different image, no refs pack, no operating memory."""

    worker_arguments = _constructor_arguments(
        Path(worker_module.__file__), class_name
    ) - WORKER_LOOP_ONLY
    audit_arguments = _constructor_arguments(
        Path(run_audit_session.__file__), class_name
    )
    assert worker_arguments
    assert worker_arguments <= audit_arguments, sorted(
        worker_arguments - audit_arguments
    )


def test_the_audited_session_mounts_the_image_refs_and_memory_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--sandbox-image has to reach the Meta learner as well as the Fold
    developer, and the strategy wall clocks the Agent is promised have to be the
    ones the options carry rather than library defaults."""

    captured: dict[str, dict[str, object]] = {}

    def capture(name: str):
        def build(**kwargs: object) -> object:
            captured[name] = kwargs
            return object()

        return build

    monkeypatch.setattr(run_audit_session, "ExperimentLedger", lambda path: object())
    monkeypatch.setattr(
        run_audit_session, "FilesystemArtifactStore", lambda path: object()
    )
    monkeypatch.setattr(
        run_audit_session, "LocalDailySnapshotProvider", lambda path: object()
    )
    monkeypatch.setattr(
        run_audit_session,
        "LocalDailyEvaluationBackend",
        lambda *args, **kwargs: SimpleNamespace(
            trading_days=["20240102"], sandbox=kwargs.get("sandbox")
        ),
    )
    monkeypatch.setattr(run_audit_session, "LLMFoldDeveloper", capture("developer"))
    monkeypatch.setattr(run_audit_session, "LLMMetaLearner", capture("meta"))
    monkeypatch.setattr(
        run_audit_session,
        "RollingExperimentPipeline",
        lambda *args, **kwargs: SimpleNamespace(**kwargs),
    )

    spec = SandboxSpec(image="audit-image:test")
    options = SimpleNamespace(
        rolling=SimpleNamespace(
            ledger_path=tmp_path / "ledger.jsonl",
            schedule=object(),
            broker_profile=object(),
            step_tree_enabled=True,
            fold_exploration_directive="directive",
            meta_learning_directive="meta directive",
            max_llm_calls=800,
            max_fold_minutes=20,
            strategy_fit_timeout_seconds=1800,
            nl_failure_policy="fail",
            workspace_reference="configs/workspace_refs/pack",
            operating_memory="curated+graduated",
            regularization_constraints=ModificationConstraints(),
            meta_sandbox_rebuild_enabled=False,
            meta_sandbox_rebuild_timeout_seconds=900,
            meta_sandbox_image_keep=2,
        ),
        experiment_dir=tmp_path / "experiments" / "audit",
        experiment_id="audit",
        work_root=tmp_path / "work",
        repo_root=tmp_path,
        baseline_strategy=tmp_path / "strategy",
        llm=SimpleNamespace(
            compact_enabled=False,
            compaction=object(),
            compaction_for=lambda role: object(),
            build_gateway=lambda role: object(),
            max_tokens_for=lambda role: 4096,
        ),
        agent_sandbox=spec,
        data_backend="daily",
        daily_path=tmp_path / "daily.parquet",
        execution_mode="sandbox",
    )

    pipeline, trading_days = run_audit_session._build_pipeline(options)

    assert trading_days == ["20240102"]
    assert pipeline.evaluator.sandbox.image == "audit-image:test"
    assert pipeline.evaluator.sandbox.limits.fit_timeout_seconds == 1800
    developer, meta = captured["developer"], captured["meta"]
    assert developer["sandbox_spec"] is spec
    # The regression the audit memo reported: the Meta session fell back to the
    # default image while the Fold developer used the requested one.
    assert meta["sandbox_spec"] is spec
    assert meta["fit_timeout_seconds"] == 1800
    for session in (developer, meta):
        assert session["workspace_reference"] == "configs/workspace_refs/pack"
        assert session["operating_memory"] == "curated+graduated"
        assert session["repo_root"] == tmp_path
    assert meta["regularization_constraints"] is options.rolling.regularization_constraints
    assert meta["rebuild_enabled"] is False


def test_the_default_cadence_fills_the_console_period_labels() -> None:
    """The audit CLI and the console must launch the same research calendar."""
    from argparse import ArgumentParser

    from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
    from scripts.experiments._cli import DEFAULT_FOLD_PERIOD, resolve_period_args

    args = SimpleNamespace(
        fold_period=DEFAULT_FOLD_PERIOD,
        development_first_period=None,
        development_last_period=None,
        heldout_first_period=None,
        heldout_last_period=None,
    )
    resolve_period_args(ArgumentParser(), args)
    assert (
        args.development_first_period,
        args.development_last_period,
        args.heldout_first_period,
        args.heldout_last_period,
    ) == (
        WEB_CREATE_DEFAULTS["development_first_period"],
        WEB_CREATE_DEFAULTS["development_last_period"],
        WEB_CREATE_DEFAULTS["heldout_first_period"],
        WEB_CREATE_DEFAULTS["heldout_last_period"],
    )


def test_another_cadence_demands_every_period_label(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A label written for one cadence mis-parses under another rather than
    failing, so the CLI refuses to guess instead of defaulting."""
    from argparse import ArgumentParser

    from scripts.experiments._cli import resolve_period_args

    args = SimpleNamespace(
        fold_period="month",
        development_first_period="202401",
        development_last_period=None,
        heldout_first_period=None,
        heldout_last_period="202407",
    )
    with pytest.raises(SystemExit):
        resolve_period_args(ArgumentParser(), args)
    message = capsys.readouterr().err
    assert "--development-last-period" in message
    assert "--heldout-first-period" in message
    assert "--development-first-period" not in message
