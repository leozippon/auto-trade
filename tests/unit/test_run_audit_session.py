from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_audit_session
from autotrade.environment.sandbox import SandboxSpec
from autotrade.pipelines import worker as worker_module
from autotrade.pipelines.config import ModificationConstraints


class _ProviderConstructed(RuntimeError):
    pass


def _options(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    """One options set, shaped like the worker's validated options."""

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
        developer_mode="llm",
        llm=SimpleNamespace(
            compact_enabled=True,
            compaction=object(),
            compaction_for=lambda role: object(),
            build_gateway=lambda role, **kwargs: object(),
            max_tokens_for=lambda role: 4096,
        ),
        agent_sandbox=SandboxSpec(image="audit-image:test"),
        data_backend="daily",
        daily_path=tmp_path / "daily.parquet",
        execution_mode="sandbox",
        max_intraday_row_group_rows=2_000_000,
        nl_config=object(),
    )
    for key, value in overrides.items():
        setattr(options, key, value)
    return options


def _build(options: SimpleNamespace):
    return worker_module.build_experiment_pipeline(
        options,
        ledger=object(),
        store=object(),
        ref_store=object(),
    )


def test_the_audit_entrypoint_assembles_through_the_shared_builder() -> None:
    """The audit script must not grow a second assembly.

    A hand-written copy is what drifted last time: the console built the
    compaction gateway without retries and the audit path did not, so the
    audited session was not the session the console runs. The script may
    therefore name the Agent adapters and the backends nowhere.
    """

    source = Path(run_audit_session.__file__).read_text(encoding="utf-8")
    assert "build_experiment_pipeline(" in source
    for name in (
        "LLMFoldDeveloper",
        "LLMMetaLearner",
        "RollingExperimentPipeline",
        "ResearchPITSnapshotProvider",
        "PITDailyEvaluationBackend",
        "LocalDailySnapshotProvider",
        "LocalDailyEvaluationBackend",
    ):
        assert name not in source, name


def test_the_compaction_gateway_is_built_without_provider_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed compaction falls through to the emergency fit path by design,
    so provider retries would only add their full latency to that failure.
    Every surface that assembles an experiment gets that decision, because
    there is exactly one assembly."""

    roles: list[tuple[str, dict[str, object]]] = []

    def build_gateway(role: str, **kwargs: object) -> object:
        roles.append((role, kwargs))
        return object()

    monkeypatch.setattr(
        worker_module, "LocalDailySnapshotProvider", lambda path: object()
    )
    monkeypatch.setattr(
        worker_module,
        "LocalDailyEvaluationBackend",
        lambda *args, **kwargs: SimpleNamespace(
            trading_days=["20240102"], sandbox=kwargs.get("sandbox")
        ),
    )
    monkeypatch.setattr(
        worker_module, "LLMFoldDeveloper", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        worker_module, "LLMMetaLearner", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        worker_module,
        "RollingExperimentPipeline",
        lambda *args, **kwargs: SimpleNamespace(**kwargs),
    )
    options = _options(tmp_path)
    options.llm.build_gateway = build_gateway

    build = _build(options)

    assert build.meta_enabled is True
    assert build.developer_label == "llm_fold_meta_agent"
    assert dict(roles)["compact"] == {"max_retries": 0}
    assert [role for role, _kwargs in roles] == [
        "main",
        "meta",
        "subagent",
        "nl",
        "compact",
    ]


def test_the_assembled_session_mounts_the_image_refs_and_memory_it_was_given(
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

    monkeypatch.setattr(
        worker_module, "LocalDailySnapshotProvider", lambda path: object()
    )
    monkeypatch.setattr(
        worker_module,
        "LocalDailyEvaluationBackend",
        lambda *args, **kwargs: SimpleNamespace(
            trading_days=["20240102"], sandbox=kwargs.get("sandbox")
        ),
    )
    monkeypatch.setattr(worker_module, "LLMFoldDeveloper", capture("developer"))
    monkeypatch.setattr(worker_module, "LLMMetaLearner", capture("meta"))
    monkeypatch.setattr(
        worker_module,
        "RollingExperimentPipeline",
        lambda *args, **kwargs: SimpleNamespace(**kwargs),
    )

    options = _options(tmp_path)
    build = _build(options)

    assert build.trading_days == ["20240102"]
    assert build.pipeline.evaluator.sandbox.image == "audit-image:test"
    assert build.pipeline.evaluator.sandbox.limits.fit_timeout_seconds == 1800
    developer, meta = captured["developer"], captured["meta"]
    assert developer["sandbox_spec"] is options.agent_sandbox
    # The regression the audit memo reported: the Meta session fell back to the
    # default image while the Fold developer used the requested one.
    assert meta["sandbox_spec"] is options.agent_sandbox
    assert meta["fit_timeout_seconds"] == 1800
    for session in (developer, meta):
        assert session["workspace_reference"] == "configs/workspace_refs/pack"
        assert session["operating_memory"] == "curated+graduated"
        assert session["repo_root"] == tmp_path
    assert meta["regularization_constraints"] is options.rolling.regularization_constraints
    assert meta["rebuild_enabled"] is False


def test_the_pipeline_uses_the_experiments_own_pit_view_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seed is the experiment's resolved one, not a hardcoded default.

    An arm whose dataset selection needs its own prebuilt tree must read that
    tree here too, and on the same terms: an explicitly chosen seed has to
    apply rather than fall back to a cold build.
    """

    captured: dict[str, object] = {}

    def build_provider(**kwargs: object) -> object:
        captured.update(kwargs)
        raise _ProviderConstructed

    monkeypatch.setattr(
        worker_module, "ResearchPITSnapshotProvider", build_provider
    )
    options = _options(
        tmp_path,
        data_backend="pit",
        raw_dir=tmp_path / "raw",
        fundamental_events_root=tmp_path / "fundamentals",
        fundamental_events_status=tmp_path / "fundamentals-status.json",
        snapshot_config=object(),
        pit_cache_root=tmp_path / "pit-cache",
        pit_views_seed=tmp_path / "data/pit_views_seed_ext",
        pit_views_seed_required=True,
    )

    with pytest.raises(_ProviderConstructed):
        _build(options)

    assert captured["pit_views_seed"] == tmp_path / "data/pit_views_seed_ext"
    assert captured["pit_views_seed_required"] is True


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
