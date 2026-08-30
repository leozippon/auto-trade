"""FastAPI application for the ADM-Cube research, HITL, and Paper console.

JSON API + static SPA. The server is a thin control plane: pipeline execution
happens in detached worker processes; state flows through the hitl/ files and
the append-only ledger. There is no auth layer, so the console only accepts a
loopback host or a Unix socket.
"""

from __future__ import annotations

import ipaddress
import json
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from autotrade.environment.data.contracts import RAW_GENERATION_FILENAME
from autotrade.environment.llm.model_profiles import model_profile
from autotrade.environment.step_tree import StepTree
from autotrade.pipelines.fold_analysis import analysis_paths
from autotrade.pipelines.hitl_state import (
    ANALYSIS_DIR_NAME,
    HITL_DIR_NAME,
    PARAMS_NAME,
    read_json,
    read_status,
)
from autotrade.pipelines.ledger import latest_fold_records

from . import equity, memory, registry, steps, traces, trading
from .analysis import AnalysisService
from .manager import (
    MAX_RUNNING_EXPERIMENTS,
    ExperimentManager,
    ManagerDeleteError,
    ManagerError,
)
from .params_schema import parameter_schema
from .prompt_preview import build_prompt_preview
from .public_identity import PublicIdentity, redact_host_paths

STATIC_DIR = Path(__file__).resolve().parent / "static"


def is_loopback_host(host: str) -> bool:
    """Return whether a TCP bind target is unambiguously local-only."""

    value = str(host).strip()
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _raw_generation_status(repo_root: Path) -> dict[str, object]:
    """Observability view of the raw-lake generation stamp for /api/health.

    Lenient by design: health must report a broken or non-committed stamp,
    never 500 on it. The strict consumer contract stays in
    environment/data/contracts.py and is deliberately not reused here."""
    path = Path(repo_root) / "data" / "raw" / RAW_GENERATION_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"state": "absent"}  # dev/test roots without a stamped lake
    except (OSError, ValueError) as exc:
        return {"state": "unreadable", "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"state": "unreadable", "error": "stamp is not a JSON object"}
    info: dict[str, object] = {"state": str(payload.get("state") or "unreadable")}
    for key in ("generation_id", "updated_at", "completed_at"):
        if payload.get(key):
            info[key] = payload[key]
    return info


_UNREADABLE_HITL_HEALTH_ERROR = "HITL control plane is unreadable"


def _health_unreadable_experiments(
    items: list[dict[str, object]],
) -> list[dict[str, object]]:
    """HTTP-boundary view of broken experiment control planes.

    Health is a pathless contract. ``read_status`` ValueError text embeds the
    absolute HITL path; keep experiment id and exception category, replace the
    body with a stable phrase.
    """
    public: list[dict[str, object]] = []
    for item in items:
        raw = str(item.get("error") or "").strip()
        category, sep, _detail = raw.partition(": ")
        error = (
            f"{category}: {_UNREADABLE_HITL_HEALTH_ERROR}"
            if sep and category
            else _UNREADABLE_HITL_HEALTH_ERROR
        )
        public.append(
            {
                "experiment_id": str(item.get("experiment_id") or ""),
                "error": error,
            }
        )
    return public


def create_app(repo_root: Path, experiments_root: Path | None = None) -> FastAPI:
    root = Path(repo_root).resolve()
    experiment_root = Path(experiments_root or root / "experiments").resolve()
    analysis_service = AnalysisService(root)
    # The manager must see the analysis service's pending work: its background
    # threads write into experiments/<id>/hitl/analysis/, so deletion is
    # refused (409) while an analysis for that experiment is still running.
    manager = ExperimentManager(
        root, experiment_root, analysis_pending=analysis_service.pending_for_experiment
    )
    app = FastAPI(
        title="ADM-Cube Console", docs_url=None, redoc_url=None, openapi_url=None
    )
    trading_days_cache: dict[str, object] = {}

    @app.middleware("http")
    async def revalidate_frontend_assets(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            # Keep clean, unversioned asset URLs and never retain stale UI
            # code in a browser cache.
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    def _experiment_dir(experiment_id: str) -> Path:
        try:
            return registry.resolve_experiment_dir(experiment_root, experiment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _public_identity(experiment_id: str) -> tuple[Path, PublicIdentity]:
        directory = _experiment_dir(experiment_id)
        try:
            return directory, PublicIdentity(directory)
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=409, detail="experiment identity state is unreadable"
            ) from exc

    def _trace_target(
        experiment_id: str, public_ref: str | None
    ) -> tuple[Path, str, str, PublicIdentity]:
        directory, identity = _public_identity(experiment_id)
        if public_ref is None:
            status = read_status(directory / "hitl/status.json")
            raw_run_id = str(status.get("run_id") or "")
            if not raw_run_id:
                raise HTTPException(status_code=404, detail="no trace available for this run")
        else:
            try:
                raw_run_id = identity.raw_run_id(public_ref)
            except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=404, detail="unknown run or trace reference") from exc
        path = traces.resolve_trace_path(directory, raw_run_id)
        if path is None:
            raise HTTPException(status_code=404, detail="no trace available for this run")
        return path, raw_run_id, identity.trace_ref(raw_run_id), identity

    def _public_trace_blocks(
        payload: dict[str, object], identity: PublicIdentity
    ) -> dict[str, object]:
        public = dict(payload)
        blocks = public.get("blocks")
        if isinstance(blocks, list):
            public["blocks"] = [
                identity.public_record(block, heldout_revealed=False)
                for block in blocks
                if isinstance(block, dict)
            ]
        header = public.get("header")
        if isinstance(header, dict):
            public["header"] = identity.public_record(header, heldout_revealed=False)
        return public

    def _public_trace_download(path: Path, identity: PublicIdentity) -> Path:
        """Materialize a public JSONL copy without exposing the host trace file."""

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".jsonl", delete=False
        ) as handle:
            output = Path(handle.name)
            try:
                with path.open("r", encoding="utf-8", errors="replace") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except ValueError:
                            event = None
                        if not isinstance(event, dict):
                            # The line itself never leaves the host: it could not
                            # be parsed, so it could not be redacted either.
                            handle.write(
                                json.dumps(
                                    {
                                        "event_type": "unreadable_line",
                                        "bytes": len(line.encode("utf-8")),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            continue
                        public = identity.public_record(
                            event, heldout_revealed=False
                        )
                        handle.write(
                            json.dumps(public, ensure_ascii=False, default=str) + "\n"
                        )
            except Exception:
                output.unlink(missing_ok=True)
                raise
        return output

    @app.get("/api/health")
    def health() -> dict[str, object]:
        unreadable = _health_unreadable_experiments(manager.unreadable_experiments())
        raw_generation = _raw_generation_status(root)
        # Honest status: degraded when the raw lake's last mutation did not
        # commit (absent = dev/test roots without a lake), or when an
        # experiment's control plane is unreadable.
        healthy = raw_generation["state"] in ("committed", "absent") and not unreadable
        return {
            "status": "ok" if healthy else "degraded",
            "max_running_experiments": MAX_RUNNING_EXPERIMENTS,
            "running": manager.running_experiments(),
            "unreadable_experiments": unreadable,
            "raw_generation": raw_generation,
        }

    def _trading_days() -> list[str]:
        # Loaded once per process (registry.clamped_trading_days does the
        # coverage clamping; None = no calendar, pickers degrade to text).
        if "days" not in trading_days_cache:
            trading_days_cache["days"] = registry.clamped_trading_days(root)
        return trading_days_cache["days"] or []

    def _inherit_sources() -> list[str]:
        """Experiments with at least one recorded fold (inherit_from choices)."""
        if not experiment_root.is_dir():
            return []
        sources = []
        for entry in sorted(experiment_root.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            try:
                if latest_fold_records(registry.read_ledger_records(entry)):
                    sources.append(entry.name)
            except Exception:  # noqa: BLE001 - a broken experiment cannot seed a new one;
                continue  # it stays visible (state=unreadable) in the list instead
        return sources

    @app.get("/api/parameter-schema")
    def get_parameter_schema() -> dict[str, object]:
        return parameter_schema(trading_days=_trading_days(), inherit_sources=_inherit_sources())

    @app.get("/api/gpus")
    def get_gpus() -> dict[str, object]:
        try:
            from autotrade.environment.gpu import list_gpus

            return {"gpus": list_gpus()}
        except Exception as exc:  # noqa: BLE001 - CPU-only workstations retain the console
            return {"gpus": [], "error": f"{type(exc).__name__}: {exc}"}

    @app.get("/api/experiments")
    def get_experiments() -> dict[str, object]:
        return {
            "experiments": registry.list_experiments(experiment_root),
            "running": manager.running_experiments(),
            "max_running_experiments": MAX_RUNNING_EXPERIMENTS,
        }

    @app.post("/api/experiments")
    def post_experiment(payload: dict = Body(...)) -> dict[str, object]:
        params = payload.get("params") if isinstance(payload.get("params"), dict) else payload
        try:
            return manager.create_experiment(dict(params))
        except (ManagerError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, object]:
        _experiment_dir(experiment_id)
        try:
            return registry.experiment_detail(experiment_root, experiment_id)
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=409, detail="experiment identity state is unreadable"
            ) from exc

    @app.get("/api/experiments/{experiment_id}/status")
    def get_status(experiment_id: str) -> dict[str, object]:
        directory, identity = _public_identity(experiment_id)
        state = registry.experiment_state(directory)
        raw_status = state.get("status")
        public = identity.public_record(
            {key: value for key, value in state.items() if key != "status"},
            heldout_revealed=False,
        )
        if isinstance(raw_status, dict):
            public["status"] = identity.public_status(raw_status)
        return public

    @app.get("/api/experiments/{experiment_id}/trace/stats")
    def get_trace_stats(
        experiment_id: str,
        run_id: str | None = Query(None),
    ) -> dict[str, object]:
        path, _raw_run_id, trace_ref, _identity = _trace_target(experiment_id, run_id)
        payload: dict[str, object] = {**traces.trace_stats(path), "trace_ref": trace_ref}
        params = read_json(_experiment_dir(experiment_id) / HITL_DIR_NAME / PARAMS_NAME)
        model = str(params.get("model") or "")
        if model:
            try:
                window = model_profile(model).context_window_tokens
            except ValueError:
                window = None
            if isinstance(window, int) and window > 0:
                payload["context_window_tokens"] = window
        return payload

    @app.get("/api/experiments/{experiment_id}/trace/blocks")
    def get_trace_blocks(
        experiment_id: str,
        run_id: str | None = Query(None),
        offset: int = Query(0, ge=0),
        max_bytes: int | None = Query(None, ge=1, le=traces.MAX_BLOCK_READ_BYTES),
        tail_events: int | None = Query(None, ge=1, le=500),
    ) -> dict[str, object]:
        path, _raw_run_id, trace_ref, identity = _trace_target(experiment_id, run_id)
        blocks = traces.read_trace_blocks(
            path,
            offset=offset,
            max_bytes=max_bytes,
            tail_events=tail_events,
        )
        return {**_public_trace_blocks(blocks, identity), "trace_ref": trace_ref}

    @app.get("/api/experiments/{experiment_id}/trace/subagents/{task_id}")
    def get_subagent_trace(
        experiment_id: str,
        task_id: str,
        run_id: str | None = Query(None),
    ) -> dict[str, object]:
        """One sub-agent's own rounds, redacted like the parent projection."""

        if not traces.SUBAGENT_TASK_ID.fullmatch(task_id):
            raise HTTPException(status_code=400, detail="invalid sub-agent task id")
        path, _raw_run_id, trace_ref, identity = _trace_target(experiment_id, run_id)
        projected = traces.read_subagent_trace(path, task_id)
        if not projected.get("found"):
            raise HTTPException(status_code=404, detail="unknown sub-agent task")
        return {**_public_trace_blocks(projected, identity), "trace_ref": trace_ref}

    @app.get("/api/experiments/{experiment_id}/trace/stream")
    def get_trace_stream(
        request: Request,
        experiment_id: str,
        run_id: str | None = Query(None),
        offset: int = Query(0, ge=0),
    ) -> StreamingResponse:
        directory, _unused_identity = _public_identity(experiment_id)
        _path, raw_run_id, _trace_ref, _identity = _trace_target(experiment_id, run_id)
        resume = request.headers.get("last-event-id")
        if resume is not None and resume.isdigit():
            offset = max(offset, int(resume))
        return StreamingResponse(
            traces.stream_trace(directory, raw_run_id, offset=offset),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/experiments/{experiment_id}/trace/download")
    def download_trace(
        experiment_id: str,
        run_id: str | None = Query(None),
    ) -> FileResponse:
        path, _raw_run_id, trace_ref, identity = _trace_target(experiment_id, run_id)
        try:
            public_path = _public_trace_download(path, identity)
        except (OSError, UnicodeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail="trace is unreadable") from exc
        return FileResponse(
            public_path,
            media_type="application/x-ndjson",
            filename=f"{experiment_id}__{trace_ref}__agent-trace.jsonl",
            background=BackgroundTask(public_path.unlink, missing_ok=True),
        )

    @app.delete("/api/experiments/{experiment_id}")
    def delete_experiment(experiment_id: str, confirm: str = Query("")) -> dict[str, object]:
        _experiment_dir(experiment_id)
        if confirm != experiment_id:
            raise HTTPException(status_code=400, detail="confirm query param must equal the experiment id")
        try:
            return manager.delete_experiment(experiment_id)
        except ManagerDeleteError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ManagerError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/experiments/{experiment_id}/control")
    def post_control(experiment_id: str, payload: dict = Body(...)) -> dict[str, object]:
        _experiment_dir(experiment_id)
        try:
            return manager.control(
                experiment_id,
                str(payload.get("action") or ""),
                session_key=payload.get("session_key"),
                step_index=payload.get("step_index"),
                directive=payload.get("directive"),
                mode=payload.get("mode"),
                text=payload.get("text"),
                interrupt=payload.get("interrupt", False),
                at=payload.get("at"),
            )
        except ManagerError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _zip_response(members: Iterable[tuple[Path, Path]], filename: str) -> FileResponse:
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as handle:
            archive_path = Path(handle.name)
        try:
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for archive_name, file_path in members:
                    archive.write(file_path, archive_name)
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
        return FileResponse(
            archive_path,
            media_type="application/zip",
            filename=filename,
            background=BackgroundTask(archive_path.unlink, missing_ok=True),
        )

    def _strategy_zip_response(strategy_dir: Path, filename: str) -> FileResponse:
        strategy_dir = Path(strategy_dir)
        model_dir = strategy_dir.parent / "models"

        def members() -> Iterator[tuple[Path, Path]]:
            for file_path in sorted(strategy_dir.rglob("*")):
                if file_path.is_file():
                    yield Path("output") / file_path.relative_to(strategy_dir), file_path
            if model_dir.is_dir():
                for file_path in sorted(model_dir.rglob("*")):
                    if file_path.is_file():
                        yield Path("models") / file_path.relative_to(model_dir), file_path

        return _zip_response(members(), filename)

    @app.get("/api/experiments/{experiment_id}/folds/{epoch_id}/{fold_id}")
    def get_fold(experiment_id: str, epoch_id: str, fold_id: str) -> dict[str, object]:
        _experiment_dir(experiment_id)
        try:
            detail = registry.fold_detail(experiment_root, experiment_id, epoch_id, fold_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="unknown fold reference") from exc
        analysis = detail.get("analysis")
        if isinstance(analysis, dict):
            analysis["pending"] = analysis_service.pending(
                experiment_id, epoch_id, fold_id
            )
        return detail

    @app.get("/api/experiments/{experiment_id}/folds/{epoch_id}/{fold_id}/initial-prompt")
    def get_fold_initial_prompt(
        experiment_id: str,
        epoch_id: str,
        fold_id: str,
    ) -> dict[str, object]:
        directory, identity = _public_identity(experiment_id)
        try:
            raw_run_id = registry.fold_run_id(
                experiment_root,
                experiment_id,
                epoch_id,
                fold_id,
            )
            path = traces.resolve_trace_path(directory, raw_run_id)
            if path is None:
                raise KeyError("no agent trace recorded for this fold")
            prompt = traces.read_initial_prompt(path)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="unknown fold reference") from exc
        public_prompt = identity.public_record(
            prompt, heldout_revealed=False
        )
        public_prompt.pop("run_ref", None)
        return {
            "experiment_id": experiment_id,
            "epoch_id": epoch_id,
            "fold_ref": fold_id,
            "run_ref": identity.run_ref(raw_run_id),
            "trace_ref": identity.trace_ref(raw_run_id),
            **public_prompt,
        }

    @app.get("/api/experiments/{experiment_id}/folds/{epoch_id}/{fold_id}/strategy.zip")
    def get_fold_strategy(experiment_id: str, epoch_id: str, fold_id: str) -> FileResponse:
        try:
            strategy_dir = registry.fold_strategy_dir(
                experiment_root, experiment_id, epoch_id, fold_id
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="unknown fold reference") from exc
        return _strategy_zip_response(
            strategy_dir, f"{experiment_id}__{epoch_id}__{fold_id}.zip"
        )

    # ---- step tree ---------------------------------------------------------------
    @app.get("/api/experiments/{experiment_id}/steps")
    def get_step_tree(experiment_id: str) -> dict[str, object]:
        try:
            return steps.step_tree_view(_experiment_dir(experiment_id))
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=409, detail="experiment identity state is unreadable"
            ) from exc

    @app.get("/api/experiments/{experiment_id}/steps/{node_id}/source.zip")
    def get_step_node_zip(experiment_id: str, node_id: str) -> FileResponse:
        directory, _identity = _public_identity(experiment_id)
        try:
            node_dir = steps.node_export_dir(directory, node_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _zip_response(
            (
                (file_path.relative_to(node_dir), file_path)
                for file_path in sorted(node_dir.rglob("*"))
                if file_path.is_file()
            ),
            f"{experiment_id}__{node_id}.zip",
        )

    def current_step(experiment_id: str) -> tuple[Path, dict[str, object], dict[str, object], Path]:
        directory = _experiment_dir(experiment_id)
        state = registry.experiment_state(directory)
        if not state.get("worker_alive"):
            raise ValueError("current-step requires a live worker")
        status = state.get("status")
        if not isinstance(status, dict) or status.get("state") not in {
            "waiting_step_user",
            "waiting_user_reply",
        }:
            raise ValueError("experiment is not waiting for researcher input")
        run_id = str(status.get("run_id") or "")
        if not run_id or Path(run_id).name != run_id or run_id.startswith("."):
            raise ValueError("live Step run is unavailable")
        tree_root = root / ".runtime/sandboxes" / experiment_id / run_id / "artifacts/steps"
        node_id, node_dir = steps.current_node_export_dir(tree_root)
        return directory, status, StepTree(tree_root).get_node(node_id), node_dir

    @app.get("/api/experiments/{experiment_id}/current-step")
    def get_current_step(experiment_id: str) -> dict[str, object]:
        _directory, identity = _public_identity(experiment_id)
        try:
            _directory, _status, node, _node_dir = current_step(experiment_id)
        except (OSError, ValueError):
            return {"available": False, "reason": "current Step is unavailable"}
        return {
            "available": True,
            "node": steps.public_step_node(node, identity=identity),
        }

    @app.get("/api/experiments/{experiment_id}/current-step/source.zip")
    def get_current_step_source(experiment_id: str) -> FileResponse:
        _public_identity(experiment_id)
        try:
            _directory, _status, node, node_dir = current_step(experiment_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _zip_response(
            ((file_path.relative_to(node_dir), file_path) for file_path in sorted(node_dir.rglob("*")) if file_path.is_file()),
            f"{experiment_id}__{node['node_id']}.zip",
        )

    @app.get("/api/experiments/{experiment_id}/current-step/analysis")
    def get_current_step_analysis(experiment_id: str) -> dict[str, object]:
        try:
            directory, status, node, _node_dir = current_step(experiment_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        node_id = str(node["node_id"])
        md_path, meta_path = analysis_paths(directory / HITL_DIR_NAME / ANALYSIS_DIR_NAME, "step", node_id)
        meta = read_json(meta_path) if meta_path.exists() else None
        _unused_directory, identity = _public_identity(experiment_id)
        identity.public_status(status)
        return {
            "available": md_path.exists(),
            "pending": analysis_service.pending(experiment_id, "step", node_id),
            "content": identity.public_text(md_path.read_text(encoding="utf-8"))
            if md_path.exists()
            else None,
            "meta": identity.public_analysis_meta(meta) if isinstance(meta, dict) else None,
        }

    @app.post("/api/experiments/{experiment_id}/current-step/analysis")
    def post_current_step_analysis(experiment_id: str) -> dict[str, object]:
        _public_identity(experiment_id)
        try:
            directory, status, node, node_dir = current_step(experiment_id)
            analysis_service.regenerate_step(
                experiment_dir=directory,
                experiment_id=experiment_id,
                node_id=str(node["node_id"]),
                node_dir=node_dir,
                status=status,
            )
        except (ManagerError, OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "started"}

    @app.get("/api/experiments/{experiment_id}/equity")
    def get_equity(experiment_id: str, epoch_id: str | None = Query(None)) -> dict[str, object]:
        _experiment_dir(experiment_id)
        try:
            return equity.experiment_equity_payload(experiment_root, experiment_id, epoch_id=epoch_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/experiments/{experiment_id}/folds/{epoch_id}/{fold_id}/equity")
    def get_fold_equity(experiment_id: str, epoch_id: str, fold_id: str) -> dict[str, object]:
        _experiment_dir(experiment_id)
        try:
            return equity.fold_equity_payload(experiment_root, experiment_id, epoch_id, fold_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="unknown fold reference") from exc

    @app.get("/api/experiments/{experiment_id}/style")
    def get_style(experiment_id: str, run_id: str = Query(...), prefix: str = Query(...)) -> dict[str, object]:
        _experiment_dir(experiment_id)
        try:
            return registry.style_payload(
                experiment_root,
                experiment_id,
                run_ref=run_id,
                prefix=prefix,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/experiments/{experiment_id}/folds/{epoch_id}/{fold_id}/orders")
    def get_fold_orders(
        experiment_id: str,
        epoch_id: str,
        fold_id: str,
        result: str | None = Query(None),
    ) -> dict[str, object]:
        _experiment_dir(experiment_id)
        try:
            return registry.fold_orders(experiment_root, experiment_id, epoch_id, fold_id, result=result)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="unknown fold reference") from exc

    @app.get("/api/experiments/{experiment_id}/folds/{epoch_id}/{fold_id}/orders.csv")
    def get_fold_orders_csv(
        experiment_id: str,
        epoch_id: str,
        fold_id: str,
        result: str = Query(...),
    ) -> PlainTextResponse:
        _experiment_dir(experiment_id)
        try:
            filename, content = registry.fold_orders_csv(
                experiment_root,
                experiment_id,
                epoch_id,
                fold_id,
                result=result,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="unknown fold reference") from exc
        return PlainTextResponse(
            content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ---- analysis -----------------------------------------------------------------
    @app.get("/api/experiments/{experiment_id}/analysis/{epoch_id}/{fold_id}")
    def get_analysis(experiment_id: str, epoch_id: str, fold_id: str) -> dict[str, object]:
        # Resolve first so an arbitrary UUID-shaped token cannot probe sidecar names.
        directory, identity = _public_identity(experiment_id)
        try:
            raw_fold_id = identity.raw_fold_id(epoch_id, fold_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="unknown fold reference") from exc
        record = latest_fold_records(registry.read_ledger_records(directory)).get(
            (epoch_id, raw_fold_id)
        )
        if record is not None:
            identity.public_record(record, heldout_revealed=False)
        md_path, meta_path = analysis_paths(
            directory / HITL_DIR_NAME / ANALYSIS_DIR_NAME, epoch_id, fold_id
        )
        meta = read_json(meta_path) if meta_path.exists() else None
        return {
            "available": md_path.exists(),
            "pending": analysis_service.pending(experiment_id, epoch_id, fold_id),
            "content": identity.public_text(md_path.read_text(encoding="utf-8"))
            if md_path.exists()
            else None,
            "meta": identity.public_analysis_meta(meta) if isinstance(meta, dict) else None,
        }

    @app.post("/api/experiments/{experiment_id}/analysis/{epoch_id}/{fold_id}")
    def post_analysis(experiment_id: str, epoch_id: str, fold_id: str) -> dict[str, object]:
        _experiment_dir(experiment_id)
        try:
            analysis_service.regenerate(manager.experiments_root, experiment_id, epoch_id, fold_id)
        except (ManagerError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=409, detail="analysis request was rejected") from exc
        return {"status": "started"}

    # ---- operating memory ---------------------------------------------------------
    # The curated library and the graduated exclusion list are tracked
    # repository files, so editing either is a repository write. Who may issue
    # one is decided where every other mutating console route decides it — the
    # loopback/Unix-socket bind and, for the proxied surface, the edge's login
    # gate — so these routes carry no gate of their own beyond the validation
    # each write performs.
    def _memory_write(action) -> dict[str, object]:
        """One HTTP mapping for every memory write; messages carry no host path."""

        try:
            return action()
        except FileExistsError as exc:
            raise HTTPException(
                status_code=409, detail=redact_host_paths(str(exc))
            ) from exc
        except KeyError as exc:
            detail = str(exc.args[0]) if exc.args else "unknown curated memory entry"
            raise HTTPException(
                status_code=404, detail=redact_host_paths(detail)
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=redact_host_paths(str(exc))
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=409, detail="curated memory library is not writable"
            ) from exc

    @app.get("/api/memory")
    def get_memory() -> dict[str, object]:
        return memory.memory_overview(root, experiment_root)

    @app.get("/api/memory/curated/{name}")
    def get_curated_memory(name: str) -> dict[str, object]:
        try:
            return memory.curated_entry(root, name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid memory entry name") from exc
        except (KeyError, OSError) as exc:
            raise HTTPException(
                status_code=404, detail="unknown curated memory entry"
            ) from exc

    @app.get("/api/memory/graduated/{experiment_id}/{skill}")
    def get_graduated_memory(experiment_id: str, skill: str) -> dict[str, object]:
        """One graduated candidate's body, behind the same gate as its promotion."""

        try:
            return memory.graduated_entry(root, experiment_root, experiment_id, skill)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid memory entry name") from exc
        except (KeyError, OSError) as exc:
            raise HTTPException(
                status_code=404, detail="unknown graduated memory entry"
            ) from exc

    @app.post("/api/memory/graduated/{experiment_id}/{skill}/exclude")
    def post_graduated_exclusion(
        experiment_id: str, skill: str, payload: dict = Body(default={})
    ) -> dict[str, object]:
        """Withdraw one graduated skill from every future mount.

        A graduated skill is another experiment's immutable artifact, so the
        console never edits it; this records that sessions must stop mounting it.
        """

        return _memory_write(
            lambda: memory.exclude_graduated_skill(
                root,
                experiment_root,
                experiment_id=experiment_id,
                skill=skill,
                reason=str(payload.get("reason") or ""),
            )
        )

    @app.delete("/api/memory/graduated/{experiment_id}/{skill}/exclude")
    def delete_graduated_exclusion(
        experiment_id: str, skill: str
    ) -> dict[str, object]:
        return _memory_write(
            lambda: memory.restore_graduated_skill(
                root, experiment_root, experiment_id=experiment_id, skill=skill
            )
        )

    @app.post("/api/memory/curated")
    def post_curated_memory(payload: dict = Body(...)) -> dict[str, object]:
        return _memory_write(
            lambda: memory.create_curated_entry(
                root,
                str(payload.get("name") or ""),
                str(payload.get("content") or ""),
                experiments_root=experiment_root,
                live_experiments=manager.running_experiments(),
            )
        )

    @app.put("/api/memory/curated/{name}")
    def put_curated_memory(name: str, payload: dict = Body(...)) -> dict[str, object]:
        return _memory_write(
            lambda: memory.update_curated_entry(
                root, name, str(payload.get("content") or "")
            )
        )

    @app.delete("/api/memory/curated/{name}")
    def delete_curated_memory(name: str) -> dict[str, object]:
        return _memory_write(lambda: memory.delete_curated_entry(root, name))

    @app.post("/api/memory/curated/{name}/promote")
    def post_curated_promotion(
        name: str, payload: dict = Body(...)
    ) -> dict[str, object]:
        """Copy one admitted graduated skill in under the name in the path."""

        return _memory_write(
            lambda: memory.promote_curated_entry(
                root,
                experiment_root,
                name=name,
                experiment_id=str(payload.get("experiment_id") or ""),
                skill=str(payload.get("skill") or ""),
                live_experiments=manager.running_experiments(),
            )
        )

    @app.get("/api/experiments/{experiment_id}/memory")
    def get_experiment_memory(experiment_id: str) -> dict[str, object]:
        _experiment_dir(experiment_id)
        try:
            return memory.experiment_memory(experiment_root, experiment_id)
        except (OSError, TypeError, ValueError, KeyError) as exc:
            raise HTTPException(
                status_code=409, detail="experiment identity state is unreadable"
            ) from exc

    @app.post("/api/experiments/{experiment_id}/prompt-preview")
    def post_prompt_preview(experiment_id: str, payload: dict = Body(...)) -> dict[str, object]:
        directory, identity = _public_identity(experiment_id)
        try:
            raw_session_key = identity.raw_session_key(
                str(payload.get("session_key") or "")
            )
            preview = build_prompt_preview(
                directory,
                raw_session_key,
                str(payload.get("directive") or ""),
            )
            return {
                **preview,
                "prompt": identity.public_text(str(preview.get("prompt") or "")),
                "session_key": identity.public_session_key(raw_session_key),
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0] if exc.args else exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _trading_env(env: str) -> str:
        if env not in trading.TRADING_ENVS:
            raise HTTPException(status_code=404, detail=f"unknown trading environment: {env}")
        return env

    def _trading_date(value: str | None) -> str | None:
        if value is not None and not (len(value) == 8 and value.isdigit()):
            raise HTTPException(status_code=400, detail="date must be YYYYMMDD")
        return value

    @app.get("/api/trading/environments")
    def trading_environments():
        return trading.environments_payload(root)

    @app.get("/api/trading/{env}/snapshot")
    def trading_snapshot(env: str):
        return trading.snapshot_payload(root, _trading_env(env))

    @app.get("/api/trading/{env}/orders")
    def trading_orders(env: str, date: str | None = Query(None)):
        return trading.orders_payload(root, _trading_env(env), _trading_date(date))

    @app.get("/api/trading/{env}/deals")
    def trading_deals(env: str, date: str | None = Query(None)):
        return trading.deals_payload(root, _trading_env(env), _trading_date(date))

    @app.get("/api/trading/{env}/series")
    def trading_series(env: str):
        return trading.series_payload(root, _trading_env(env))

    @app.get("/api/trading/{env}/health")
    def trading_health(env: str):
        return trading.health_payload(root, _trading_env(env))

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index() -> HTMLResponse:
            return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

        @app.get("/favicon.ico", include_in_schema=False)
        def favicon() -> FileResponse:
            logo = STATIC_DIR / "logo.png"
            if not logo.is_file():
                raise HTTPException(status_code=404, detail="favicon missing")
            return FileResponse(logo, media_type="image/png")
    return app


def run(
    repo_root: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 38888,
    uds: Path | None = None,
    experiments_root: Path | None = None,
) -> None:
    import signal

    import uvicorn

    # Auto-reap detached workers so exited experiments never linger as zombies
    # (their liveness is judged via status.json pid checks).
    # CAUTION: with SIGCHLD ignored, every subprocess.run() in THIS process
    # returns returncode 0 regardless of the child's real exit status. Current
    # callers (docker cleanup, nvidia-smi) parse stdout or check filesystem
    # state only; new code here must not rely on returncode/check=True.
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    options: dict[str, object] = {
        "app": create_app(repo_root, experiments_root),
        "access_log": False,
        "log_level": "warning",
    }
    if uds is not None:
        # Unix-socket bind: local access control is the parent directory's
        # filesystem permissions (loopback TCP is reachable by every local
        # user on a shared host). uvicorn chmods the socket itself to 666,
        # so the directory is created 0700 here.
        uds.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        options["uds"] = str(uds)
    else:
        if not is_loopback_host(host):
            raise ValueError("ADM-Cube console only accepts a loopback host or a Unix socket")
        options.update({"host": host, "port": port})
    uvicorn.run(**options)


__all__ = ["create_app", "is_loopback_host", "run"]
