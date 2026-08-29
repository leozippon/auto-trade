from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_audit_session
from autotrade.pipelines.pit_views_seed import DEFAULT_PIT_VIEWS_SEED


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
        rolling=SimpleNamespace(ledger_path=tmp_path / "ledger.jsonl"),
        experiment_dir=tmp_path / "experiments" / "audit",
        llm=llm,
        agent_sandbox=object(),
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
