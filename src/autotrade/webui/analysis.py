"""Background fold-analysis regeneration for the HITL console.

Task control (pending-set + worker threads) for the LLM strategy analysis —
process/state management like ``ExperimentManager``, kept out of the HTTP
route module. Results land in the analysis sidecar files. ``analyze_fold``
records failures of its own provider call there; every failure before that
point (proxy construction, strategy-file reads) is recorded by the thread
wrapper here, so no failure is ever silent.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from autotrade.environment.llm import LOCAL_QWEN_MODEL, model_profile
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import utc_now_iso
from autotrade.environment.step_tree import StepTree
from autotrade.pipelines.fold_analysis import (
    ANALYSIS_SCHEMA_VERSION,
    analysis_paths,
    analyze_fold,
    analyze_step,
)
from autotrade.pipelines.hitl_state import (
    ANALYSIS_DIR_NAME,
    HITL_DIR_NAME,
    PARAMS_NAME,
    read_json,
)

from . import registry
from .manager import ManagerError


class AnalysisService:
    """Background (re)generation of fold analyses, one at a time per fold."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = Path(repo_root)
        self._pending: set[tuple[str, str, str]] = set()
        self._lock = threading.Lock()

    def pending(self, experiment_id: str, epoch_id: str, fold_id: str) -> bool:
        with self._lock:
            return (experiment_id, epoch_id, fold_id) in self._pending

    def pending_for_experiment(self, experiment_id: str) -> bool:
        """Whether ANY analysis for this experiment is still in flight; its
        worker thread writes under experiments/<id>/hitl/analysis/, so the
        manager refuses to delete the experiment until this drains."""
        with self._lock:
            return any(key[0] == experiment_id for key in self._pending)

    def _run_recorded(
        self,
        key: tuple[str, str, str],
        out_dir: Path,
        analysis_kind: str,
        provider: str,
        model: str,
        call,
    ) -> None:
        """Thread body: run ``call`` and guarantee every failure is recorded.

        ``analyze_fold`` writes its own error sidecar for failures inside its
        provider try, but everything before that — gateway construction (for
        example a missing .env key) and its own strategy-file reads — raises
        without recording, which used to end the thread silently. When the
        failing call did not rewrite the sidecar itself, persist the same
        error shape here so the UI's meta view surfaces the failure. The
        console process itself never crashes: the exception stops here.

        ``key[1:]`` is the sidecar identity for both kinds — (epoch, fold)
        for folds and ("step", node_id) for steps.
        """
        _experiment_id, epoch_id, fold_id = key
        _md_path, meta_path = analysis_paths(Path(out_dir), epoch_id, fold_id)
        try:
            before = meta_path.stat().st_mtime_ns if meta_path.exists() else None
            try:
                call()
            except Exception as exc:  # noqa: BLE001 - analysis failure must not kill the console
                after = meta_path.stat().st_mtime_ns if meta_path.exists() else None
                if after == before:
                    meta_path.parent.mkdir(parents=True, exist_ok=True)
                    meta_path.write_text(
                        json.dumps(
                            {
                                "schema_version": ANALYSIS_SCHEMA_VERSION,
                                "epoch_id": epoch_id,
                                "fold_id": fold_id,
                                "provider": provider,
                                "model": model,
                                "created_at": utc_now_iso(),
                                "guarded_view": "validation_only",
                                "analysis_kind": analysis_kind,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            ensure_ascii=False,
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
        finally:
            with self._lock:
                self._pending.discard(key)

    def regenerate(
        self, experiments_root: Path, experiment_id: str, epoch_id: str, fold_id: str
    ) -> None:
        key = (experiment_id, epoch_id, fold_id)
        with self._lock:
            if key in self._pending:
                raise ManagerError("analysis for this fold is already being generated")
            self._pending.add(key)
        try:
            experiment_dir = registry.resolve_experiment_dir(
                experiments_root, experiment_id
            )
            ref_store = AgentRefStore(experiment_dir)
            detail = registry.fold_detail(
                experiments_root, experiment_id, epoch_id, fold_id
            )
            strategy_dir = detail.get("strategy_dir")
            if not strategy_dir or not Path(str(strategy_dir)).is_dir():
                raise ManagerError("fold has no frozen strategy artifact on disk")
            params = read_json(experiment_dir / HITL_DIR_NAME / PARAMS_NAME)
            model = str(params.get("analysis_model") or LOCAL_QWEN_MODEL)
            provider = model_profile(model).provider
            max_tokens = int(params.get("analysis_max_tokens") or 6000)
            record = dict(detail["record"])
            model_dir = record.get("frozen_model_artifact_path")
            out_dir = experiment_dir / HITL_DIR_NAME / ANALYSIS_DIR_NAME
        except Exception:
            with self._lock:
                self._pending.discard(key)
            raise

        def _call() -> None:
            from autotrade.environment.llm import build_model_gateway

            proxy = build_model_gateway(
                model,
                env_file=str(self.repo_root / ".env"),
                max_tokens=max_tokens,
                thinking_enabled=True,
                reasoning_effort="xhigh",
            )
            analyze_fold(
                proxy,
                ledger_record=record,
                ref_store=ref_store,
                strategy_dir=Path(str(strategy_dir)),
                model_dir=Path(str(model_dir)) if model_dir else None,
                out_dir=out_dir,
                max_tokens=max_tokens,
            )

        threading.Thread(
            target=self._run_recorded,
            args=(key, out_dir, "fold", provider, model, _call),
            name=f"analysis-{experiment_id}-{fold_id}",
            daemon=True,
        ).start()

    def regenerate_step(
        self,
        *,
        experiment_dir: Path,
        experiment_id: str,
        node_id: str,
        node_dir: Path,
        status: dict[str, object],
    ) -> None:
        """Generate an optional researcher-only review of the current Step snapshot."""
        key = (experiment_id, "step", node_id)
        with self._lock:
            if key in self._pending:
                raise ManagerError("analysis for this Step is already being generated")
            self._pending.add(key)
        try:
            ref_store = AgentRefStore(experiment_dir)
            strategy_dir = Path(node_dir) / "output"
            if not strategy_dir.is_dir():
                raise ManagerError("current Step has no strategy snapshot on disk")
            model_dir = Path(node_dir) / "models"
            params = read_json(Path(experiment_dir) / HITL_DIR_NAME / PARAMS_NAME)
            model = str(params.get("analysis_model") or LOCAL_QWEN_MODEL)
            provider = model_profile(model).provider
            max_tokens = int(params.get("analysis_max_tokens") or 6000)
            node = StepTree(Path(node_dir).parent).get_node(node_id)
            step_index = status.get("awaiting_step")
            step_record: dict[str, object] = {
                "epoch_id": status.get("epoch_id"),
                "fold_id": status.get("fold_id"),
                "step_id": f"step_{int(step_index):03d}"
                if step_index is not None
                else None,
                "validation_result": dict(
                    node.get("metrics") or status.get("step_summary") or {}
                ),
                "selected_step_id": node.get("result_name"),
            }
            out_dir = Path(experiment_dir) / HITL_DIR_NAME / ANALYSIS_DIR_NAME
        except Exception:
            with self._lock:
                self._pending.discard(key)
            raise

        def _call() -> None:
            from autotrade.environment.llm import build_model_gateway

            proxy = build_model_gateway(
                model,
                env_file=str(self.repo_root / ".env"),
                max_tokens=max_tokens,
                thinking_enabled=True,
                reasoning_effort="xhigh",
            )
            analyze_step(
                proxy,
                step_record=step_record,
                ref_store=ref_store,
                strategy_dir=strategy_dir,
                model_dir=model_dir if model_dir.is_dir() else None,
                out_dir=out_dir,
                node_id=node_id,
                max_tokens=max_tokens,
            )

        threading.Thread(
            target=self._run_recorded,
            args=(key, out_dir, "step", provider, model, _call),
            name=f"analysis-{experiment_id}-{node_id}",
            daemon=True,
        ).start()
