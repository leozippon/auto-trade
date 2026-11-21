"""FastAPI application for the ADM-Cube research and Paper console.

JSON API + static SPA. The server is a thin control plane: pipeline execution
happens in detached worker processes; state flows through the hitl/ files and
the append-only ledger. There is no auth layer, so the console only accepts a
loopback host or a Unix socket.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import logging
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask
from starlette.middleware.gzip import GZipMiddleware

from autotrade.common.records import read_json_object
from autotrade.common.time import cn_today
from autotrade.environment.data.contracts import RAW_GENERATION_FILENAME
from autotrade.environment.gpu import list_gpus
from autotrade.environment.llm.model_profiles import model_profile
from autotrade.paper import fills
from autotrade.paper.books import delete_book, open_registered_book
from autotrade.paper.engine import PAPER_STATE_NAME, PaperWriterBusy
from autotrade.pipelines.hitl_state import (
    HITL_DIR_NAME,
    PARAMS_NAME,
    STATUS_NAME,
    read_status,
)

from . import equity, feedback, memory, registry, steps, traces, trading
from .manager import (
    ExperimentManager,
    ManagerDeleteError,
    ManagerError,
)
from .params_schema import parameter_schema
from .prompt_preview import build_prompt_preview
from .public_identity import PublicIdentity, omit_private_paths

STATIC_DIR = Path(__file__).resolve().parent / "static"
# The console's own log file under the repository root, size-rotated, beside
# its stderr: the supervisor that keeps the console up discards stdout and
# stderr. Not logs/webui/console.log, which a ZGCA instance appends the
# console's stdout and stderr to (ops/zgca/remote.sh).
CONSOLE_LOG = Path("logs/webui/webui.log")
CONSOLE_LOG_MAX_BYTES = 10 * 1024 * 1024
CONSOLE_LOG_BACKUPS = 4


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
        payload = read_json_object(path)
    except FileNotFoundError:
        return {"state": "absent", "issues": []}  # dev/test roots without a stamped lake
    except (OSError, ValueError) as exc:
        # The reader names the file by its host path, which the console omits.
        return {"state": "unreadable", "error": omit_private_paths(str(exc)), "issues": []}
    info: dict[str, object] = {"state": str(payload.get("state") or "unreadable")}
    for key in ("generation_id", "updated_at", "completed_at"):
        if payload.get(key):
            info[key] = payload[key]
    issues = payload.get("issues") or []
    info["issues"] = [
        {key: str(issue.get(key) or "") for key in ("kind", "dataset", "partition", "since", "detail", "version")}
        for issue in issues
        if isinstance(issue, dict)
    ] if isinstance(issues, list) else [{"kind": "unreadable", "dataset": "", "partition": "", "since": "", "detail": "issues is not a list", "version": ""}]
    return info


# Raw-lake states that leave health ok. ``updating`` is a write in progress,
# not a fault: no console path reads the live lake -- an experiment pins an
# already-published release (its seed's, or the last complete one while the
# updater holds the lake) and Paper reads the newest committed one -- so it is
# reported in ``raw_generation`` without degrading. ``unreadable`` and any
# state this code does not know degrade, and so does every open lake issue (a
# defective vendor answer, a staged version awaiting an operator, a pending
# whole-history re-pull), which health names.
_HEALTHY_RAW_STATES = frozenset({"committed", "absent", "updating"})

_UNREADABLE_HITL_HEALTH_ERROR = "HITL control plane is unreadable"
_UNREADABLE_IDENTITY = "experiment identity state is unreadable"


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


class _Body(BaseModel):
    """A request body read as written: a field of another JSON type, or one
    the route does not take, is refused with 422 naming it, never coerced."""

    model_config = ConfigDict(strict=True, extra="forbid")


class ControlRequest(_Body):
    """``ExperimentManager.control``'s arguments, which it checks further."""

    action: str
    session_key: str | None = None
    directive: str | None = None
    text: str | None = None
    interrupt: bool | None = False
    at: str | None = None


class CuratedMemoryCreate(_Body):
    name: str
    content: str


class CuratedMemoryUpdate(_Body):
    content: str


class PromptPreviewRequest(_Body):
    session_key: str
    directive: str = ""


class OpenBookRequest(_Body):
    experiment_id: str


class FillsRequest(_Body):
    """``paper.fills.record``'s arguments; it checks each fill's fields."""

    trade_date: str
    outcome: str | None = None
    fills: list[dict[str, object]] = []


_Kinds = type[Exception] | tuple[type[Exception], ...]


@contextmanager
def _http_errors(*rules: tuple[_Kinds, int] | tuple[_Kinds, int, str]) -> Iterator[None]:
    """Answer an exception raised inside with the status of the first rule
    ``(exception types, status[, detail])`` it matches: the rule's fixed
    detail, or else the exception's own message without host paths (a
    KeyError's message is its argument; ``str(KeyError)`` is its repr).
    An exception no rule names propagates."""

    try:
        yield
    except Exception as exc:
        for kinds, status, *fixed in rules:
            if isinstance(exc, kinds):
                message = str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)
                detail = fixed[0] if fixed else omit_private_paths(message)
                raise HTTPException(status_code=status, detail=detail) from exc
        raise


def create_app(repo_root: Path, experiments_root: Path | None = None) -> FastAPI:
    root = Path(repo_root).resolve()
    experiment_root = Path(experiments_root or root / "experiments").resolve()
    manager = ExperimentManager(root, experiment_root)
    app = FastAPI(
        title="ADM-Cube Console", docs_url=None, redoc_url=None, openapi_url=None
    )

    @app.middleware("http")
    async def revalidate_frontend_assets(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            # Clean, unversioned asset URLs that are never used stale: the
            # browser keeps a copy but asks again on every load, and an
            # unchanged file answers 304 on its ETag instead of resending
            # a few hundred kilobytes of UI code each time.
            response.headers["Cache-Control"] = "no-cache"
        return response

    # The console keeps no access log, so each 5xx it answers is logged here:
    # when, the method and the path, never the query or a body.
    server_log = logging.getLogger("uvicorn.error")

    def log_server_error(request: Request, status: int) -> None:
        stamp = datetime.now(UTC).isoformat(timespec="seconds")
        server_log.warning("%s %s %s -> %d", stamp, request.method, request.url.path, status)

    @app.middleware("http")
    async def log_server_errors(request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception:
            log_server_error(request, 500)  # what an escaped exception is answered with
            raise
        if response.status_code >= 500:
            log_server_error(request, response.status_code)
        return response

    # The listing, a replay's curve and the UI code are JSON and text that
    # shrink four- to five-fold, and the public console is read over a thin
    # link. Level 6: level 9 saves 1% more of the 1 MB listing for a third
    # more time on the event loop. The live trace stream (text/event-stream)
    # is never compressed.
    app.add_middleware(GZipMiddleware, compresslevel=6)

    @app.exception_handler(RequestValidationError)
    async def refuse_malformed_request(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """A malformed request answers 422 with one line naming each bad field
        (``body.at: Input should be a valid string``): ``detail`` is text in
        every refusal, which the page shows as written."""
        problems = [f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "; ".join(problems)})

    def _experiment_dir(experiment_id: str) -> Path:
        with _http_errors((KeyError, 404), (ValueError, 400)):
            return registry.resolve_experiment_dir(experiment_root, experiment_id)

    def _public_identity(experiment_id: str) -> tuple[Path, PublicIdentity]:
        directory = _experiment_dir(experiment_id)
        with _http_errors(((OSError, ValueError), 409, _UNREADABLE_IDENTITY)):
            return directory, PublicIdentity(directory)

    def _trace_target(
        experiment_id: str, public_ref: str | None
    ) -> tuple[Path, str, str, PublicIdentity]:
        directory, identity = _public_identity(experiment_id)
        if public_ref is None:
            status = read_status(directory / HITL_DIR_NAME / STATUS_NAME)
            raw_run_id = str(status.get("run_id") or "")
            if not raw_run_id:
                raise HTTPException(status_code=404, detail="no trace available for this run")
        else:
            with _http_errors(((KeyError, ValueError), 404, "unknown run or trace reference")):
                raw_run_id = identity.raw_run_id(public_ref)
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
                identity.public_record(block) for block in blocks if isinstance(block, dict)
            ]
        header = public.get("header")
        if isinstance(header, dict):
            public["header"] = identity.public_record(header)
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
                        public = identity.public_record(event)
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
        # Honest status: degraded while the raw lake has an open issue or its
        # stamp cannot be read (absent = dev/test roots without a lake), or
        # when an experiment's control plane is unreadable.
        healthy = (
            raw_generation["state"] in _HEALTHY_RAW_STATES
            and not raw_generation["issues"]
            and not unreadable
        )
        return {
            "status": "ok" if healthy else "degraded",
            **manager.running_slots(),
            # The GPU side of a start, which the research fill reads beside
            # the two running-experiment limits.
            **manager.gpu_slots(),
            "unreadable_experiments": unreadable,
            "raw_generation": raw_generation,
        }

    @app.get("/api/parameter-schema")
    def get_parameter_schema() -> dict[str, object]:
        return parameter_schema()

    @app.get("/api/gpus")
    def get_gpus() -> dict[str, object]:
        try:
            return {"gpus": list_gpus()}
        except Exception as exc:  # noqa: BLE001 - CPU-only workstations retain the console
            return {"gpus": [], "error": f"{type(exc).__name__}: {exc}"}

    @app.get("/api/experiments")
    def get_experiments(
        kept: str | None = Query(None), archived: bool = Query(False)
    ) -> dict[str, object]:
        return {
            **registry.experiment_listing(experiment_root, kept, archived=archived),
            **manager.running_slots(),
        }

    @app.post("/api/experiments")
    def post_experiment(params: dict = Body(..., embed=True)) -> dict[str, object]:
        with _http_errors(((ManagerError, ValueError), 400)):
            return manager.create_experiment(dict(params))

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict[str, object]:
        _experiment_dir(experiment_id)
        with _http_errors(((OSError, ValueError, KeyError), 409, _UNREADABLE_IDENTITY)):
            return registry.experiment_detail(experiment_root, experiment_id)

    @app.get("/api/experiments/{experiment_id}/status")
    def get_status(experiment_id: str) -> dict[str, object]:
        directory, identity = _public_identity(experiment_id)
        state = registry.experiment_state(directory)
        raw_status = state.get("status")
        public = identity.public_record(
            {key: value for key, value in state.items() if key != "status"}
        )
        if isinstance(raw_status, dict):
            public["status"] = identity.public_status(raw_status)
        # The experiment page polls this route, so the budget bars follow the
        # live session without a rebuild, and a new Validation or test step
        # (a moved revision) makes the page re-read the experiment.
        public["budget_used"] = registry.research_budget_used(directory)
        public["revision"] = registry.research_revision(directory)
        return public

    @app.get("/api/experiments/{experiment_id}/research-depth")
    def get_research_depth(experiment_id: str) -> dict[str, object]:
        _experiment_dir(experiment_id)
        with _http_errors(((OSError, ValueError), 409, "experiment research records are unreadable")):
            return registry.research_depth(experiment_root, experiment_id)

    @app.get("/api/experiments/{experiment_id}/trace/stats")
    def get_trace_stats(
        experiment_id: str,
        run_id: str | None = Query(None),
    ) -> dict[str, object]:
        path, _raw_run_id, _trace_ref, _identity = _trace_target(experiment_id, run_id)
        payload: dict[str, object] = traces.trace_stats(path)
        params = read_json_object(_experiment_dir(experiment_id) / HITL_DIR_NAME / PARAMS_NAME, missing_ok=True)
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
        path, raw_run_id, _trace_ref, identity = _trace_target(experiment_id, run_id)
        blocks = traces.read_trace_blocks(
            path,
            offset=offset,
            max_bytes=max_bytes,
            tail_events=tail_events,
            writer_alive=traces.trace_writer_alive(identity.experiment_dir, raw_run_id),
        )
        return _public_trace_blocks(blocks, identity)

    @app.get("/api/experiments/{experiment_id}/trace/subagents/{task_id}")
    def get_subagent_trace(
        experiment_id: str,
        task_id: str,
        run_id: str | None = Query(None),
    ) -> dict[str, object]:
        """One sub-agent's own turns, redacted like the parent projection."""

        if not traces.TRACE_ID.fullmatch(task_id):
            raise HTTPException(status_code=400, detail="invalid sub-agent task id")
        path, _raw_run_id, _trace_ref, identity = _trace_target(experiment_id, run_id)
        projected = traces.read_subagent_trace(path, task_id)
        if not projected.get("found"):
            raise HTTPException(status_code=404, detail="unknown sub-agent task")
        return _public_trace_blocks(projected, identity)

    @app.get("/api/experiments/{experiment_id}/trace/initial-prompt")
    def get_initial_prompt(
        experiment_id: str,
        run_id: str = Query(...),
    ) -> dict[str, object]:
        """The system prompt and opening message a session actually started with."""

        path, _raw_run_id, _trace_ref, identity = _trace_target(experiment_id, run_id)
        with _http_errors((KeyError, 404)):
            prompt = traces.read_initial_prompt(path)
        public = identity.public_record(prompt)
        public.pop("run_ref", None)
        public.pop("trace_ref", None)
        return public

    @app.get("/api/experiments/{experiment_id}/trace/stream")
    def get_trace_stream(
        request: Request,
        experiment_id: str,
        run_id: str | None = Query(None),
        offset: int = Query(0, ge=0),
    ) -> StreamingResponse:
        _path, raw_run_id, _trace_ref, identity = _trace_target(experiment_id, run_id)
        resume = request.headers.get("last-event-id")
        if resume is not None and resume.isdigit():
            offset = max(offset, int(resume))
        return StreamingResponse(
            traces.stream_trace(identity.experiment_dir, raw_run_id, offset=offset),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/experiments/{experiment_id}/trace/download")
    def download_trace(
        experiment_id: str,
        run_id: str | None = Query(None),
    ) -> FileResponse:
        path, _raw_run_id, trace_ref, identity = _trace_target(experiment_id, run_id)
        with _http_errors(((OSError, UnicodeError, ValueError), 409, "trace is unreadable")):
            public_path = _public_trace_download(path, identity)
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
            # As written: it names what is left on disk for the operator to remove.
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ManagerError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/experiments/{experiment_id}/control")
    def post_control(experiment_id: str, body: ControlRequest) -> dict[str, object]:
        _experiment_dir(experiment_id)
        with _http_errors((ManagerError, 400)):
            return manager.control(experiment_id, **body.model_dump())

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

    @app.get("/api/experiments/{experiment_id}/registered/strategy.zip")
    def get_registered_strategy(experiment_id: str) -> FileResponse:
        _experiment_dir(experiment_id)
        with _http_errors((FileNotFoundError, 404), ((KeyError, ValueError), 404, "the experiment has no registered artifact")):
            strategy_dir = registry.registered_strategy_dir(experiment_root, experiment_id)
        return _strategy_zip_response(strategy_dir, f"{experiment_id}__registered.zip")

    # ---- step tree ---------------------------------------------------------------
    @app.get("/api/experiments/{experiment_id}/steps")
    def get_step_tree(experiment_id: str) -> dict[str, object]:
        directory = _experiment_dir(experiment_id)
        with _http_errors(((OSError, ValueError), 409, _UNREADABLE_IDENTITY)):
            return steps.step_tree_view(directory)

    @app.get("/api/experiments/{experiment_id}/steps/{node_id}/source.zip")
    def get_step_node_zip(experiment_id: str, node_id: str) -> FileResponse:
        directory, _identity = _public_identity(experiment_id)
        with _http_errors((ValueError, 404)):
            node_dir = steps.node_export_dir(directory, node_id)
        return _zip_response(
            (
                (file_path.relative_to(node_dir), file_path)
                for file_path in sorted(node_dir.rglob("*"))
                if file_path.is_file()
            ),
            f"{experiment_id}__{node_id}.zip",
        )

    # ---- replay results named by the ledger -----------------------------------------
    # A result no ledger record names answers 404 exactly like a missing one, so
    # the forward replay stays unreadable until its verdict is recorded. A
    # named result whose file is not in the experiment directory says so.
    def _ledger_result(read, experiment_id: str, name: str):
        _experiment_dir(experiment_id)
        with _http_errors((FileNotFoundError, 404), ((KeyError, ValueError), 404, f"unknown result: {name}")):
            return read()

    @app.get("/api/experiments/{experiment_id}/results/{name}/equity")
    def get_result_equity(
        experiment_id: str,
        name: str,
        points: int | None = Query(None, ge=2, le=10_000),
    ) -> dict[str, object]:
        return _ledger_result(
            lambda: equity.result_equity_payload(
                experiment_root, experiment_id, name, points=points
            ),
            experiment_id,
            name,
        )

    @app.get("/api/experiments/{experiment_id}/results/{name}/style")
    def get_result_style(experiment_id: str, name: str) -> dict[str, object]:
        return _ledger_result(
            lambda: registry.result_style(experiment_root, experiment_id, name),
            experiment_id,
            name,
        )

    @app.get("/api/experiments/{experiment_id}/results/{name}/orders")
    def get_result_orders(experiment_id: str, name: str) -> dict[str, object]:
        return _ledger_result(
            lambda: registry.result_orders(experiment_root, experiment_id, name),
            experiment_id,
            name,
        )

    @app.get("/api/experiments/{experiment_id}/results/{name}/orders.csv")
    def get_result_orders_csv(experiment_id: str, name: str) -> PlainTextResponse:
        filename, content = _ledger_result(
            lambda: registry.result_orders_csv(experiment_root, experiment_id, name),
            experiment_id,
            name,
        )
        return PlainTextResponse(
            content,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ---- skill library ---------------------------------------------------------
    # The curated library is a tracked repository directory, so editing it is
    # a repository write. Who may issue one is decided where every other
    # mutating console route decides it — the loopback/Unix-socket bind and,
    # for the proxied surface, the edge's login gate — so these routes carry no
    # gate of their own beyond the validation each write performs.
    def _memory_write(action) -> dict[str, object]:
        """One HTTP mapping for every memory write."""

        with _http_errors(
            (FileExistsError, 409), (KeyError, 404), (ValueError, 400),
            (OSError, 409, "curated memory library is not writable"),
        ):
            return action()

    @app.get("/api/memory")
    def get_memory() -> dict[str, object]:
        return memory.memory_overview(root, experiment_root)

    @app.get("/api/memory/curated/{name}")
    def get_curated_memory(name: str) -> dict[str, object]:
        with _http_errors(
            (ValueError, 400, "invalid memory entry name"), ((KeyError, OSError), 404, "unknown curated memory entry")
        ):
            return memory.curated_entry(root, name)

    @app.post("/api/memory/curated")
    def post_curated_memory(body: CuratedMemoryCreate) -> dict[str, object]:
        return _memory_write(lambda: memory.create_curated_entry(root, body.name, body.content))

    @app.put("/api/memory/curated/{name}")
    def put_curated_memory(name: str, body: CuratedMemoryUpdate) -> dict[str, object]:
        return _memory_write(lambda: memory.update_curated_entry(root, name, body.content))

    @app.delete("/api/memory/curated/{name}")
    def delete_curated_memory(name: str) -> dict[str, object]:
        return _memory_write(lambda: memory.delete_curated_entry(root, name))

    @app.get("/api/experiments/{experiment_id}/memory")
    def get_experiment_memory(experiment_id: str) -> dict[str, object]:
        _experiment_dir(experiment_id)
        with _http_errors(((OSError, TypeError, ValueError, KeyError), 409, _UNREADABLE_IDENTITY)):
            return memory.experiment_memory(experiment_root, experiment_id)

    @app.get("/api/experiments/{experiment_id}/memory/{source}/{name}")
    def get_experiment_memory_entry(
        experiment_id: str, source: str, name: str
    ) -> dict[str, object]:
        """One entry as this experiment holds it, not as the library reads now."""

        _experiment_dir(experiment_id)
        with _http_errors(
            (ValueError, 400, "invalid memory entry name"), ((KeyError, OSError), 404, "unknown mounted memory entry")
        ):
            return memory.experiment_memory_entry(experiment_root, experiment_id, source, name)

    # ---- Agent feedback channels --------------------------------------------------
    @app.get("/api/issue-reports")
    def get_issue_reports(
        experiment_id: str | None = Query(None),
        include_resolved: bool = Query(False),
    ) -> dict[str, object]:
        """Agent-filed issue reports across experiments, newest first, resolved
        ones only when asked for. Read-only: a report and its resolution are
        append-only operator telemetry, and the resolution is recorded from the
        shell (``scripts/experiments/resolve_feedback.py issue``), never from a
        browser."""

        if experiment_id is not None:
            _experiment_dir(experiment_id)
        return feedback.issue_reports(experiment_root, experiment_id=experiment_id, include_resolved=include_resolved)

    @app.get("/api/skill-feedback")
    def get_skill_feedback(
        experiment_id: str | None = Query(None),
        include_resolved: bool = Query(False),
    ) -> dict[str, object]:
        """Agent-filed skill feedback across experiments, newest first, resolved
        ones only when asked for. Read-only on the same terms as the issue
        reports: the report and its resolution are append-only operator
        telemetry, and the resolution is recorded from the shell
        (``scripts/experiments/resolve_feedback.py skill``), never from a
        browser."""

        if experiment_id is not None:
            _experiment_dir(experiment_id)
        return feedback.skill_feedback(experiment_root, experiment_id=experiment_id, include_resolved=include_resolved)

    @app.post("/api/experiments/{experiment_id}/prompt-preview")
    def post_prompt_preview(experiment_id: str, body: PromptPreviewRequest) -> dict[str, object]:
        directory, identity = _public_identity(experiment_id)
        with _http_errors((KeyError, 404), (ValueError, 400)):
            raw_session_key = identity.raw_session_key(body.session_key)
            preview = build_prompt_preview(directory, raw_session_key, body.directive, repo_root=root)
            return {**preview, "prompt": identity.public_text(str(preview.get("prompt") or ""))}

    def _trading_env(env: str) -> str:
        if env not in trading.TRADING_ENVS:
            raise HTTPException(status_code=404, detail=f"unknown trading environment: {env}")
        return env

    def _trading_book(env: str, book: str, panel):
        with _http_errors((KeyError, 404)):
            return panel(root, book, _trading_env(env))

    @app.get("/api/trading/{env}/books")
    def trading_books(env: str):
        return trading.books_payload(root, _trading_env(env))

    @app.post("/api/trading/{env}/books")
    def trading_open_book(env: str, body: OpenBookRequest) -> dict[str, object]:
        _trading_env(env)
        directory = _experiment_dir(body.experiment_id)
        with _http_errors((FileNotFoundError, 404), ((ValueError, OSError, RuntimeError), 400)):
            book_id = open_registered_book(root, directory)
        return {"env": env, "book_id": book_id}

    @app.delete("/api/trading/{env}/books/{book}")
    def trading_delete_book(env: str, book: str) -> dict[str, object]:
        state_root = trading.env_dir(root, _trading_env(env))
        with _http_errors(
            (KeyError, 404), (ValueError, 400),
            (PaperWriterBusy, 409, f"Paper account {book} is running; delete it after the run"), (OSError, 500),
        ):
            delete_book(state_root, book)
        return {"env": env, "deleted": book}

    @app.get("/api/trading/{env}/books/{book}/status")
    def trading_book_status(env: str, book: str):
        return _trading_book(env, book, trading.book_status)

    @app.get("/api/trading/{env}/books/{book}/book")
    def trading_book_identity(env: str, book: str):
        return _trading_book(env, book, trading.book_payload)

    @app.get("/api/trading/{env}/books/{book}/signal")
    def trading_signal(env: str, book: str):
        return _trading_book(env, book, trading.signal_payload)

    @app.get("/api/trading/{env}/books/{book}/history")
    def trading_history(env: str, book: str):
        return _trading_book(env, book, trading.history_payload)

    @app.get("/api/trading/{env}/books/{book}/performance")
    def trading_performance(env: str, book: str):
        return _trading_book(env, book, trading.performance_payload)

    @app.get("/api/trading/{env}/books/{book}/snapshot")
    def trading_snapshot(env: str, book: str):
        return _trading_book(env, book, trading.snapshot_payload)

    @app.get("/api/trading/{env}/books/{book}/pnl")
    def trading_pnl(env: str, book: str):
        return _trading_book(env, book, trading.pnl_payload)

    @app.get("/api/trading/{env}/books/{book}/fills")
    def trading_fills(env: str, book: str):
        return _trading_book(env, book, trading.fills_payload)

    # The owner's record of what the book's orders actually did: two appends
    # to the book's own fills.jsonl, gated like every other console write
    # (the loopback bind and the edge's login). The run reads the file; the
    # console never writes the book's state.
    def _book_root(env: str, book: str) -> Path:
        with _http_errors((KeyError, 404)):
            return trading.book_dir(root, book, _trading_env(env))

    @app.post("/api/trading/{env}/books/{book}/fills")
    def trading_record_fills(env: str, book: str, body: FillsRequest) -> dict[str, object]:
        directory = _book_root(env, book)
        with _http_errors(((TypeError, ValueError), 400)):
            lines = fills.record(
                directory,
                read_json_object(directory / PAPER_STATE_NAME, missing_ok=True),
                body.trade_date,
                outcome=body.outcome,
                fills=body.fills,
                today=cn_today(),
            )
        return {"env": env, "book_id": book, "recorded": len(lines)}

    @app.post("/api/trading/{env}/books/{book}/fills/enable")
    def trading_enable_fills(env: str, book: str) -> dict[str, object]:
        directory = _book_root(env, book)
        with _http_errors((ValueError, 400)):
            fills.enable(directory)
        return {"env": env, "book_id": book, "real_fills": True}

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
    from uvicorn.config import LOGGING_CONFIG

    if uds is None and not is_loopback_host(host):
        raise ValueError("ADM-Cube console only accepts a loopback host or a Unix socket")
    # Auto-reap detached workers so exited experiments never linger as zombies
    # (their liveness is judged via status.json pid checks).
    # CAUTION: with SIGCHLD ignored, every subprocess.run() in THIS process
    # returns returncode 0 regardless of the child's real exit status. Current
    # callers (docker cleanup, nvidia-smi) parse stdout or check filesystem
    # state only; new code here must not rely on returncode/check=True.
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    log_file = Path(repo_root) / CONSOLE_LOG
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_config = copy.deepcopy(LOGGING_CONFIG)
    log_config["formatters"]["file"] = {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}
    log_config["handlers"]["file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "formatter": "file",
        "filename": str(log_file),
        "maxBytes": CONSOLE_LOG_MAX_BYTES,
        "backupCount": CONSOLE_LOG_BACKUPS,
        "encoding": "utf-8",
    }
    # uvicorn.error, which the 5xx lines and escaped exceptions are logged
    # on, propagates to uvicorn.
    log_config["loggers"]["uvicorn"]["handlers"].append("file")
    options: dict[str, object] = {
        "app": create_app(repo_root, experiments_root),
        "access_log": False,
        "log_level": "warning",
        "log_config": log_config,
    }
    if uds is not None:
        # Unix-socket bind: local access control is the parent directory's
        # filesystem permissions (loopback TCP is reachable by every local
        # user on a shared host). uvicorn chmods the socket itself to 666,
        # so the directory is created 0700 here.
        uds.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        options["uds"] = str(uds)
    else:
        options.update({"host": host, "port": port})
    uvicorn.run(**options)


__all__ = ["create_app", "is_loopback_host", "run"]
