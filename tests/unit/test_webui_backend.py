"""Console backend tests: lifecycle guards, API routes and shared read models.

No worker subprocesses, Docker, or LLM calls: worker spawn is patched out and
experiment state is synthesized on disk exactly as the orchestrator writes it
(``tests/unit/webui_research_arm.py``). The research-arm projections, the
forward seal and the best-experiment ranking are in
``test_webui_research_console.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from autotrade.environment.identity import AgentRefStore
from autotrade.environment.llm import LOCAL_QWEN_MODEL, MODEL_CHOICES
from autotrade.environment.runtime import (
    TRACE_PAYLOAD_HEAD_CHARS,
    AgentTraceWriter,
    write_json_atomic,
)
from autotrade.pipelines.config import DEFAULT_RESEARCH_GEOMETRY
from autotrade.pipelines.hitl_state import (
    WEB_CREATE_DEFAULTS,
    StatusReporter,
    proc_start_ticks,
    read_control,
    read_status,
    status_pid_alive,
)
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.webui.manager import (
    MAX_RUNNING_EXPERIMENTS,
    ExperimentManager,
    ManagerError,
)
from autotrade.webui.public_identity import PublicIdentity
from autotrade.webui.server import create_app, is_loopback_host
from tests.unit.gpu_probe import stubbed_gpu_probe
from tests.unit.webui_research_arm import build_arm


def test_local_webui_health_schema_and_brand(tmp_path: Path):
    daily = tmp_path / "data/raw/daily"
    daily.mkdir(parents=True)
    for trade_date in (
        "20240102",
        "20240103",
        "20240201",
        "20240202",
        "20240301",
        "20240304",
        "20240429",
        "20240430",
    ):
        (daily / f"trade_date={trade_date}.parquet").touch()
    client = TestClient(create_app(tmp_path))
    # The loopback console reports operational health only: the closed
    # code-version/keepalive fields exist to serve a remote deployment stack
    # this version does not have.
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert "experiments_root" not in health
    assert str(tmp_path) not in json.dumps(health)
    assert health["max_running_experiments"] == MAX_RUNNING_EXPERIMENTS
    assert health["running"] == []
    assert health["unreadable_experiments"] == []
    assert health["raw_generation"] == {"state": "absent"}
    schema = client.get("/api/parameter-schema").json()
    fields = {
        field["key"]: field for group in schema["groups"] for field in group["fields"]
    }
    assert "fields" not in schema
    assert schema["schema_version"] == 3
    assert [group["name"] for group in schema["groups"]] == [
        "基本与排程",
        "数据窗口",
        "数据域",
        "股票筛选",
        "预算与验收",
        "Broker 账户",
        "运行控制",
        "模型与上下文",
    ]
    assert (
        not {
            "strategy_path",
            "data_backend",
            "daily_path",
            "raw_dir",
            "fundamental_events_root",
            "fundamental_events_status",
            "execution_mode",
            "developer_mode",
        }
        & fields.keys()
    )
    for key, value in DEFAULT_RESEARCH_GEOMETRY.to_record().items():
        assert fields[key]["default"] == value
    assert fields["strategy_period"]["choices"] == ["day", "month", "quarter", "year"]
    assert fields["inference_time"]["default"] == "08:30"
    assert fields["daily_window_months"]["optional"] is True
    assert fields["include_intraday"]["default"] is WEB_CREATE_DEFAULTS["include_intraday"]
    assert fields["fundamental_datasets"]["type"] == "multi"
    assert fields["macro_datasets"]["choices"]
    assert fields["events_datasets"]["choices"]
    assert fields["text_datasets"]["choices"]
    assert fields["screen_boards"]["choices"] == ["main", "gem", "star", "bj"]
    model_fields = [
        field["key"]
        for group in schema["groups"]
        if group["name"] == "模型与上下文"
        for field in group["fields"]
    ]
    assert model_fields == [
        "model",
        "subagent_model",
        "nl_model",
        "compact_model",
        "reasoning_effort",
        "no_thinking",
        "disable_context_compact",
        "compact_token_threshold",
        "compact_keep_recent_messages",
        "compact_max_tokens",
        "compact_max_calls",
    ]
    assert fields["model"]["choices"] == list(MODEL_CHOICES)
    assert fields["model"]["default"] == LOCAL_QWEN_MODEL
    assert fields["subagent_model"]["choices"] == fields["model"]["choices"]
    assert fields["subagent_model"]["default"] == LOCAL_QWEN_MODEL
    assert fields["nl_model"]["default"] == LOCAL_QWEN_MODEL
    assert fields["compact_model"]["default"] == LOCAL_QWEN_MODEL
    assert fields["reasoning_effort"]["default"] == "xhigh"
    # Empty = derived from the model window by the worker.
    assert fields["compact_token_threshold"]["default"] is None
    assert fields["compact_token_threshold"]["optional"] is True
    assert fields["compact_keep_recent_messages"]["default"] == 10
    assert fields["compact_max_tokens"]["default"] == 10_000
    assert fields["compact_max_calls"]["default"] == 10
    assert (
        not {
            "llm_model",
            "llm_max_response_tokens",
            "llm_temperature",
            "nl_max_results",
            "nl_max_calls_per_decision",
            "nl_max_total_calls",
            "nl_deadline_seconds",
        }
        & fields.keys()
    )
    for key in (
        "max_research_minutes",
        "window_months",
        "max_replay_years",
        "max_null_controls",
        "max_llm_calls",
    ):
        assert not fields[key].get("advanced", False)
    page = client.get("/")
    assert page.status_code == 200
    assert "ADM-Cube" in page.text
    assert "no-store" in page.headers["cache-control"]
    # Brand mark is a CSS background, so a failed fetch paints nothing rather
    # than the browser placeholder glyph in the top-left corner.
    assert '<span class="logo" aria-hidden="true"></span>' in page.text
    assert "<img" not in page.text.split("</header>")[0]
    assert 'url("/static/logo.png")' in client.get("/static/style.css").text
    assert 'rel="icon" href="/static/logo.png"' in page.text
    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 200
    assert favicon.headers["content-type"] == "image/png"
    # The nav strip carries a frontend-only 实盘交易 entry on its own #/qmt
    # route. What must stay absent is the live-trading console itself: no
    # backend, no communication, no execution path behind it.
    assert '<a href="#/qmt" data-nav="qmt">实盘交易</a>' in page.text
    assert "#/trading/live" not in page.text and 'data-nav="live"' not in page.text
    logo = client.get("/static/logo.png")
    assert logo.status_code == 200
    assert logo.headers["content-type"] == "image/png"


def test_health_incompatible_hitl_state_does_not_echo_paths(tmp_path: Path):
    hitl = tmp_path / "experiments" / "exp_incompat" / "hitl"
    hitl.mkdir(parents=True)
    (hitl / "status.json").write_text(
        json.dumps({"schema_version": 99, "state": "created"}),
        encoding="utf-8",
    )
    response = TestClient(create_app(tmp_path)).get("/api/health")
    assert response.status_code == 200
    health = response.json()
    assert health["status"] == "degraded"
    assert health["unreadable_experiments"] == [
        {
            "experiment_id": "exp_incompat",
            "error": "ValueError: HITL control plane is unreadable",
        }
    ]
    dumped = json.dumps(health)
    assert str(tmp_path) not in dumped
    assert str(hitl) not in dumped


def test_local_webui_disables_openapi_docs(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    for path in ("/openapi.json", "/docs", "/redoc"):
        assert client.get(path).status_code == 404


def test_site_footer_shows_icp_and_public_security_filings(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    page = client.get("/").text
    assert "津ICP备2024017854号-2" in page
    assert 'href="https://beian.miit.gov.cn/"' in page
    assert "津公网安备12010402002613号" in page
    assert (
        'href="https://beian.mps.gov.cn/#/query/webSearch?code=12010402002613"'
        in page
    )
    assert 'src="/static/gongan.png"' in page
    assert "hugo-next" not in page
    gongan = client.get("/static/gongan.png")
    assert gongan.status_code == 200
    assert gongan.headers["content-type"] == "image/png"
    assert gongan.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_exposes_distinct_session_and_subagent_model_choices() -> None:
    from scripts.experiments._cli import add_model_arguments

    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    defaults = parser.parse_args([])
    assert defaults.model == defaults.subagent_model == LOCAL_QWEN_MODEL
    assert not hasattr(defaults, "meta_model")
    mixed = parser.parse_args(
        ["--model", "deepseek-v4-flash", "--subagent-model", LOCAL_QWEN_MODEL]
    )
    assert mixed.model == "deepseek-v4-flash"
    assert mixed.subagent_model == LOCAL_QWEN_MODEL


def test_agent_trace_is_redacted_bounded_and_private(tmp_path: Path):
    path = tmp_path / "artifacts/traces/run_001.jsonl"
    writer = AgentTraceWriter(
        path,
        ids={"experiment_id": "demo", "run_id": "run_001"},
        max_bytes=900,
        max_event_bytes=450,
    )
    writer.emit(
        "tool_call",
        {
            "authorization": "Bearer should-not-appear",
            "arguments": {"api_key": "sk-abcdefghijk"},
            "content": "x" * 2_000,
        },
    )
    writer.emit("tool_call", {"content": "y" * 2_000})
    text = path.read_text(encoding="utf-8")
    assert "should-not-appear" not in text and "abcdefghijk" not in text
    assert path.stat().st_size <= 900
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700


def test_oversized_tool_call_event_keeps_a_readable_bounded_stub(tmp_path: Path):
    """A tool_call carries no ``content``: without a payload head the stub kept
    only identifiers, and the largest events — exactly the ones worth reading
    back — said nothing about what the tool was asked to do."""
    path = tmp_path / "artifacts/traces/run_002.jsonl"
    writer = AgentTraceWriter(
        path,
        ids={"experiment_id": "demo", "run_id": "run_002"},
        max_bytes=1_000_000,
        max_event_bytes=64 * 1024,
    )
    writer.emit(
        "tool_call",
        {
            "call_index": 7,
            "tool": "daily_backtest",
            "arguments": {"note": "rebalance", "api_key": "sk-abcdefghijk"},
            "result": {"ok": True, "per_stock": ["row" * 40 for _ in range(4_000)]},
        },
    )
    record = json.loads(path.read_text(encoding="utf-8").strip())
    assert record["truncated"] is True
    assert record["tool"] == "daily_backtest" and record["call_index"] == 7
    assert record["original_bytes"] > 64 * 1024
    head = record["payload_head"]
    assert len(head) <= TRACE_PAYLOAD_HEAD_CHARS
    assert '"note": "rebalance"' in head and '"per_stock"' in head
    assert "abcdefghijk" not in head
    # The reply preview stays empty: this event never had a ``content`` field.
    assert record["content_preview"] == ""


def test_experiment_endpoint_rejects_console_managed_unknown_and_missing_parameters(
    tmp_path: Path,
):
    client = TestClient(create_app(tmp_path))
    closed = client.post("/api/experiments", json={"data_backend": "daily"})
    assert closed.status_code == 400
    assert (
        closed.json()["detail"]
        == "console-managed parameters are not accepted: data_backend"
    )
    old_model = client.post("/api/experiments", json={"llm_model": "deepseek-chat"})
    assert old_model.status_code == 400
    assert (
        old_model.json()["detail"]
        == "console-managed parameters are not accepted: llm_model"
    )
    endpoint = client.post(
        "/api/experiments",
        json={"llm_base_url": "https://untrusted.example.test/v1"},
    )
    assert endpoint.status_code == 400
    assert endpoint.json()["detail"] == (
        "console-managed parameters are not accepted: llm_base_url"
    )
    unknown = client.post("/api/experiments", json={"period": "month"})
    assert unknown.status_code == 400
    assert unknown.json()["detail"] == "unknown experiment parameters: period"
    missing = client.post("/api/experiments", json={})
    assert missing.status_code == 400
    assert "experiment_id" in missing.json()["detail"]


def test_experiment_endpoint_creates_only_persistent_sandbox_research(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    with stubbed_gpu_probe():
        response = client.post(
            "/api/experiments",
            json={
                "params": {
                    "experiment_id": "persistent_demo",
                    **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                    "strategy_period": "quarter",
                    "inference_time": "23:59",
                    "daily_window_months": 18,
                    "include_macro": False,
                    "events_datasets": ["margin", "moneyflow"],
                    "screen_boards": ["main", "gem"],
                    "model": "deepseek-v4-flash",
                    "subagent_model": LOCAL_QWEN_MODEL,
                    "nl_model": "deepseek-v4-pro",
                    "compact_model": "deepseek-v4-pro",
                    "reasoning_effort": "high",
                    "no_thinking": True,
                    "disable_context_compact": False,
                    "compact_token_threshold": 90_000,
                    "compact_keep_recent_messages": 10,
                    "compact_max_tokens": 1_200,
                    "compact_max_calls": 4,
                }
            },
        )
    assert response.status_code == 200
    assert response.json()["experiment_id"] == "persistent_demo"
    assert response.json()["spawned"] is False
    params = json.loads(
        (tmp_path / "experiments/persistent_demo/hitl/params.json").read_text(
            encoding="utf-8"
        )
    )
    assert params["strategy_path"] == "configs/agent_output_template/main.py"
    assert params["data_backend"] == "pit"
    assert params["execution_mode"] == "sandbox"
    assert params["developer_mode"] == "llm"
    assert params["strategy_period"] == "quarter"
    assert params["inference_time"] == "23:59"
    assert params["daily_window_months"] == 18
    assert params["include_macro"] is False
    assert params["events_datasets"] == ["margin", "moneyflow"]
    assert params["screen_boards"] == ["main", "gem"]
    assert params["model"] == "deepseek-v4-flash"
    assert params["subagent_model"] == LOCAL_QWEN_MODEL
    assert params["nl_model"] == "deepseek-v4-pro"
    assert params["compact_model"] == "deepseek-v4-pro"
    assert params["reasoning_effort"] == "high"
    assert params["no_thinking"] is True
    assert params["compact_token_threshold"] == 90_000


def test_static_console_keeps_macro_style_surfaces_without_closed_capabilities(
    tmp_path: Path,
):
    client = TestClient(create_app(tmp_path))
    page = client.get("/").text
    script = client.get("/static/app.js").text
    assert "ADM-Cube" in page and "/static/logo.png" in page
    for label in ("系统提示词预览", "模拟交易"):
        assert label in script
    for label in (
        "实验流程",
        "Step 产物树",
        "冻结产物",
        "Paper 建账户",
    ):
        assert label in script
    # The Fold-era console is gone.
    for retired in ("Fold 策略分析", "样本外过渡", "揭示测试结果", "提前收官", "回滚到此 Fold"):
        assert retired not in script
    # The console has no research-report surface.
    assert "研究报告" not in script
    assert "实时 Agent Trace" in script and "/trace/download" in script
    assert "实盘交易" in page and "实盘交易" in script
    assert "后端未连接" in script
    assert "仅前端" not in script
    qmt_source = script.split("function renderQmtPage()", 1)[1]
    assert not any(
        text in qmt_source
        for text in (
            "连接不可用",
            "交易不可用",
            "暂无可显示",
            "未连接 QMT 后端",
            "委托（不可用）",
            "查询与下单",
        )
    )
    assert 'el("div", { class: "empty" }, "后端未连接")' in qmt_source
    assert "status.awaiting_question" not in script
    assert "status.awaiting_step" not in script
    assert "gpus" in client.get("/api/gpus").json()


def test_gpu_allocation_bar_uses_csp_compatible_native_control(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    script = client.get("/static/app.js").text
    stylesheet = client.get("/static/style.css").text
    source = script.split("function gpuAllocationRow(", 1)[1].split(
        "async function sendControlAction(", 1
    )[0]
    assert re.search(r'el\(\s*"progress",', source)
    assert "value: gpu.memory_free_mib" in source
    assert "max: gpu.memory_total_mib" in source
    assert "style:" not in source.split("const renderGpus", 1)[1]
    assert ".gpu-bar > span" not in stylesheet
    assert 'select.value === "" ? experimentDefault' in source


def test_qmt_backend_is_absent_and_loopback_validation_is_single_source(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/trading/live/health").status_code == 404
    assert client.get("/api/trading/sim/health").status_code == 404
    assert is_loopback_host("127.0.0.2")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("example.test")


def test_worker_liveness_rejects_unstamped_or_recycled_process_ids():
    pid = os.getpid()
    ticks = proc_start_ticks(pid)
    assert ticks is not None
    assert status_pid_alive({"pid": pid, "pid_start_ticks": ticks})
    assert not status_pid_alive({"pid": pid})
    assert not status_pid_alive({"pid": pid, "pid_start_ticks": ticks + 1})


def test_public_identity_keeps_cjk_slash_lists_and_redacts_host_paths(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "experiments/demo"
    AgentRefStore(directory)
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    (hitl / "schedule.json").write_text(
        json.dumps({"schema_version": 1, "sessions": []}), encoding="utf-8"
    )
    identity = PublicIdentity(directory)
    omitted = "[host path omitted]"
    chinese_list = (
        "原型 + 机械测试通过（30买/无churn/卖出路径/冷热同单/严格JSON）。"
    )
    buy_sell = "买入路径与卖出路径均保留，冷热同单。"
    assert identity.public_text(chinese_list) == chinese_list
    assert identity.public_text(buy_sell) == buy_sell
    assert identity.public_status({"error": chinese_list})["error"] == chinese_list

    assert identity.public_text("打开/Data2/lzp/secret") == f"打开{omitted}"
    assert identity.public_text("中文/mnt/private/secret") == f"中文{omitted}"
    assert identity.public_text("/var/lib/private/result.json") == omitted
    assert identity.public_text("file:///tmp/private/result.json") == omitted
    assert identity.public_text("file://server/share/private.txt") == omitted
    assert identity.public_text(r"C:\Users\private\result.json") == omitted
    assert identity.public_text("/mnt/private/secret") == omitted
    assert identity.public_text("/无churn/secret") == omitted

    kept = identity.public_text(
        "see https://example.test/docs/path and/or 3/4 relative/path; "
        "GET /api/experiments/x; /mnt/agent/workspace/main.py; "
        "/mnt/artifacts/a; /mnt/snapshot/s; /mnt/snapshots/t; "
        "python /mnt/tools/screen.py --signal notes/idea.py"
    )
    for token in (
        "https://example.test/docs/path",
        "and/or",
        "3/4",
        "relative/path",
        "/api/experiments/x",
        "/mnt/agent/workspace/main.py",
        "/mnt/artifacts/a",
        "/mnt/snapshot/s",
        "/mnt/snapshots/t",
        "python /mnt/tools/screen.py --signal notes/idea.py",
    ):
        assert token in kept
    assert omitted not in kept

    # Two segments is the whole threshold: the shortest host path still goes.
    assert identity.public_text("cat /srv/dump.log 后再看") == f"cat {omitted} 后再看"
    # A single-segment token is prose or arithmetic, never a host path.
    formulas = identity.public_text(
        'k_mid=(C-O)/C；z=(x-mean)/std；pd.read_parquet(asof_dir + "/daily")'
    )
    assert omitted not in formulas
    # An allow-listed sandbox root followed by CJK must not be absorbed.
    assert identity.public_text("读 /mnt/snapshot。pandas") == "读 /mnt/snapshot。pandas"
    # Container FHS frames stay readable: they are image-fixed, not host identity.
    traceback_frame = "/usr/local/lib/python3.11/x.py:12: RuntimeWarning"
    assert identity.public_text(traceback_frame) == traceback_frame
    assert identity.public_text("2>/dev/null 会丢证据") == "2>/dev/null 会丢证据"


class WebuiBackendTest(unittest.TestCase):
    """Console backend: lifecycle guards, result routes and traces.

    No worker subprocesses, Docker, or LLM calls: worker spawn is patched out
    and experiment state is synthesized on disk exactly as the orchestrator
    writes it.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo_root = Path(self._tmp.name)
        self.experiments_root = self.repo_root / "experiments"
        self.experiments_root.mkdir(parents=True)
        self._build_hitl_experiment("exp_hitl")
        self.app = create_app(self.repo_root, self.experiments_root)
        self.client = TestClient(self.app)

    def _identity(self, experiment_id: str = "exp_hitl") -> PublicIdentity:
        return PublicIdentity(self.experiments_root / experiment_id)

    def _run_ref(self, raw_run_id: str, experiment_id: str = "exp_hitl") -> str:
        return self._identity(experiment_id).run_ref(raw_run_id)

    # ---- fixtures ------------------------------------------------------------
    def _build_hitl_experiment(self, experiment_id: str, stage: str = "research") -> Path:
        experiment_dir = build_arm(self.experiments_root, experiment_id, stage)
        write_json_atomic(
            experiment_dir / "hitl" / "status.json",
            {
                "schema_version": 1,
                "pid": 999_999_999,
                "state": "running_session",
                "session_key": "research",
            },
        )
        trace_dir = experiment_dir / "artifacts" / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        events = [
            {
                "event_type": "llm_call",
                "seq": 0,
                "run_id": "run_s1",
                "content": f"run_s1 inspected s1 under {experiment_dir}",
                "usage": {
                    "total_tokens": 1000,
                    "prompt_tokens": 800,
                    "completion_tokens": 200,
                },
            },
            {
                "event_type": "llm_call",
                "seq": 1,
                "content": "/mnt/agent/output uses run_s1",
                "usage": {
                    "total_tokens": 2000,
                    "prompt_tokens": 1500,
                    "completion_tokens": 500,
                },
            },
            {"event_type": "tool_call", "seq": 2, "tool": "shell"},
        ]
        (trace_dir / "run_s1.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
        return experiment_dir

    def _validation_result(self, experiment_id: str) -> Path:
        """The first research Validation the ledger names, which a recorded
        session (a sealed or judged arm) has and a running one does not."""
        records = ExperimentLedger(
            self.experiments_root / experiment_id / "ledgers/experiment_ledger.jsonl"
        ).read()
        return Path(str(records[0]["steps"][0]["validation_result_ref"]))

    # ---- schema & listing ----------------------------------------------------
    def test_frontend_assets_use_clean_urls_and_revalidate(self) -> None:
        index = self.client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn('src="/static/app.js"', index.text)
        self.assertNotIn("app.js?v=", index.text)

    def test_parameter_schema_defaults_track_worker_defaults(self) -> None:
        schema = self.client.get("/api/parameter-schema").json()
        fields = {
            field["key"]: field
            for group in schema["groups"]
            for field in group["fields"]
        }
        self.assertEqual(
            fields["max_research_minutes"]["default"],
            WEB_CREATE_DEFAULTS["max_research_minutes"],
        )
        self.assertEqual(fields["model"]["default"], WEB_CREATE_DEFAULTS["model"])
        for hidden in (
            "experiments_root",
            "work_root",
            "raw_dir",
            "fundamental_events_root",
            "fundamental_events_status",
            "template_dir",
            "local_dev",
        ):
            self.assertNotIn(hidden, fields, hidden)
        for model_field in ("model", "subagent_model", "nl_model", "compact_model"):
            self.assertNotIn("deepseek-chat", fields[model_field]["choices"])
            self.assertNotIn("deepseek-reasoner", fields[model_field]["choices"])
        visible_copy = "\n".join(
            str(field.get(key, ""))
            for field in fields.values()
            for key in ("label", "help")
        )
        self.assertNotIn("DeepSeek", visible_copy)
        self.assertNotIn("provider", visible_copy)
        self.assertEqual(fields["no_thinking"]["label"], "禁用推理模式")
        # The geometry dates are plain YYYYMMDD strings the worker validates.
        self.assertEqual(fields["research_start"]["type"], "string")
        for retired in (
            "meta_learning_directive",
            "meta_learning_fold_interval",
            "meta_model",
            "analysis_model",
        ):
            self.assertNotIn(retired, fields)
        # Every field is a parameter the worker accepts.
        from autotrade.pipelines.worker import _ALLOWED_PARAMS

        self.assertLessEqual(set(fields), _ALLOWED_PARAMS)
        self.assertEqual(fields["research_directive"]["type"], "text")
        self.assertEqual(fields["research_directive"]["default"], "")
        self.assertTrue(fields["research_directive"]["wide"])

    def test_public_params_never_echo_hidden_keys(self) -> None:
        # params.json is also a worker-side ops channel where operator-only keys
        # legitimately exist; the read model must not echo them. The geometry is
        # configuration, not a result, and stays public.
        from autotrade.webui.registry import _public_params

        operator_only = (
            "raw_dir",
            "fundamental_events_root",
            "fundamental_events_status",
            "experiments_root",
            "work_root",
            "llm_api_key_env",
            "llm_env_file",
            "llm_base_url",
        )
        params = {
            "model": "deepseek-v4-pro",
            **DEFAULT_RESEARCH_GEOMETRY.to_record(),
            "_created_at": "2026-09-13T00:00:00+00:00",
            **{key: f"secret-{key}" for key in operator_only},
        }
        self.assertEqual(
            _public_params(params),
            {"model": "deepseek-v4-pro", **DEFAULT_RESEARCH_GEOMETRY.to_record()},
        )

    def test_historical_endpoint_is_absent_from_list_and_detail_api(self) -> None:
        params_path = self.experiments_root / "exp_hitl/hitl/params.json"
        params = json.loads(params_path.read_text(encoding="utf-8"))
        private_endpoint = "https://historical-private.example.test/v1"
        params["llm_base_url"] = private_endpoint
        write_json_atomic(params_path, params)

        listed = self.client.get("/api/experiments")
        detail = self.client.get("/api/experiments/exp_hitl")
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(detail.status_code, 200)
        self.assertNotIn(private_endpoint, listed.text)
        self.assertNotIn(private_endpoint, detail.text)
        self.assertNotIn("llm_base_url", detail.json()["params"])

    # ---- result routes ---------------------------------------------------------
    def test_result_orders_rows_and_csv_export(self) -> None:
        self._build_hitl_experiment("exp_sealed", "sealed")
        result = self._validation_result("exp_sealed")
        name = result.parent.name
        base = f"/api/experiments/exp_sealed/results/{name}"
        data = self.client.get(f"{base}/orders").json()
        self.assertEqual(sorted(data), ["result", "row_count", "rows", "stats"])
        self.assertEqual(data["result"], name)
        self.assertEqual(data["row_count"], 2)
        self.assertEqual([row["status"] for row in data["rows"]], ["filled", "rejected"])
        self.assertEqual(data["stats"]["reject_reasons"], {"limit_up_blocked_buy": 1})
        csv_response = self.client.get(f"{base}/orders.csv")
        self.assertEqual(csv_response.status_code, 200)
        self.assertIn("attachment", csv_response.headers.get("content-disposition", ""))
        self.assertEqual(len(csv_response.text.strip().splitlines()), 3)  # header + 2
        missing = self.client.get("/api/experiments/exp_sealed/results/valid_nope/orders.csv")
        self.assertEqual(missing.status_code, 404)

    def test_result_orders_cap_the_table_at_five_hundred_rows_but_not_the_export(
        self,
    ) -> None:
        """`row_count` and the stats describe the whole stream, `rows` is capped.

        The pane renders 「共 M 条」 from `row_count`, so a cap that also shrank
        the count would report a lie; and the CSV is the escape hatch from the
        cap, so it must stay uncapped.
        """
        self._build_hitl_experiment("exp_sealed", "sealed")
        result = self._validation_result("exp_sealed")
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["executions"] = [
            {
                "symbol": "000001.SZ",
                "action": "buy",
                "quantity": 100,
                "execute_at": "2024-07-01T09:32:00+08:00",
                "status": "filled",
                "price": 10.0,
            }
            for _ in range(501)
        ]
        result.write_text(json.dumps(payload), encoding="utf-8")
        base = f"/api/experiments/exp_sealed/results/{result.parent.name}"
        data = self.client.get(f"{base}/orders").json()
        self.assertEqual(len(data["rows"]), 500)
        self.assertEqual(data["row_count"], 501)
        self.assertEqual(data["stats"]["orders"], 501)
        self.assertEqual(data["stats"]["filled"], 501)
        csv_response = self.client.get(f"{base}/orders.csv")
        self.assertEqual(csv_response.status_code, 200)
        self.assertEqual(len(csv_response.text.strip().splitlines()), 502)

    def test_a_result_the_ledger_names_outside_the_experiment_is_not_served(self) -> None:
        self._build_hitl_experiment("exp_sealed", "sealed")
        ledger = ExperimentLedger(
            self.experiments_root / "exp_sealed/ledgers/experiment_ledger.jsonl"
        )
        records = ledger.read()
        outside = self.repo_root / "outside/valid_outside/result.json"
        outside.parent.mkdir(parents=True)
        outside.write_text("{}", encoding="utf-8")
        (outside.parent / "style_analysis.json").write_text(
            json.dumps({"schema_version": 1, "mode": "valid"}), encoding="utf-8"
        )
        records[0]["steps"][0]["validation_result_ref"] = str(outside)
        ledger.rewrite(records)
        for route in ("equity", "style", "orders", "orders.csv"):
            response = self.client.get(
                f"/api/experiments/exp_sealed/results/valid_outside/{route}"
            )
            self.assertEqual(response.status_code, 404, route)

    # ---- lifecycle guards ------------------------------------------------------
    def test_corrupt_identity_store_fails_closed_without_host_paths(self) -> None:
        directory = self._build_hitl_experiment("exp_bad_refs")
        store_path = directory / ".host/agent-refs.json"
        store_path.write_text("{broken", encoding="utf-8")

        responses = [
            self.client.get("/api/experiments"),
            self.client.get("/api/experiments/exp_bad_refs"),
            self.client.get("/api/experiments/exp_bad_refs/status"),
            self.client.get("/api/experiments/exp_bad_refs/steps"),
            self.client.post(
                "/api/experiments/exp_bad_refs/control", json={"action": "pause"}
            ),
        ]
        self.assertEqual(responses[0].status_code, 200)
        bad_summary = next(
            row
            for row in responses[0].json()["experiments"]
            if row["experiment_id"] == "exp_bad_refs"
        )
        self.assertEqual(bad_summary["state"], "unreadable")
        self.assertEqual(responses[1].json()["state"], "unreadable")
        self.assertEqual(responses[2].status_code, 409)
        self.assertEqual(responses[3].status_code, 409)
        self.assertEqual(responses[4].status_code, 400)
        for response in responses:
            public_text = response.text
            self.assertNotIn(str(self.repo_root), public_text)
            self.assertNotIn(str(directory), public_text)
        self.assertEqual(store_path.read_text(encoding="utf-8"), "{broken")

    def test_dead_live_session_degrades_to_interrupted(self) -> None:
        status = self.client.get("/api/experiments/exp_hitl/status").json()
        self.assertEqual(status["state"], "interrupted")
        self.assertFalse(status["worker_alive"])
        self.assertEqual(status["status"]["state"], "running_session")
        self.assertEqual(status["status"]["session_key"], "research")
        # The status poll also carries the research budget block (none yet).
        self.assertIn("budget_used", status)

    def test_frozen_strategy_zip_contains_output_tree(self) -> None:
        self.assertEqual(
            self.client.get("/api/experiments/exp_hitl/frozen/strategy.zip").status_code,
            404,
        )
        self._build_hitl_experiment("exp_frozen", "sealed")
        response = self.client.get("/api/experiments/exp_frozen/frozen/strategy.zip")
        self.assertEqual(response.status_code, 200)
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        self.assertEqual(archive.namelist(), ["output/main.py"])

    def test_delete_requires_confirm_and_no_live_worker(self) -> None:
        missing_confirm = self.client.delete("/api/experiments/exp_hitl")
        self.assertEqual(missing_confirm.status_code, 400)
        # Simulate a live worker on the HITL experiment (our own pid is alive;
        # liveness requires the recorded kernel start ticks to match).
        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "pid_start_ticks": proc_start_ticks(os.getpid()),
                "state": "running_session",
            },
        )
        alive = self.client.delete(
            "/api/experiments/exp_hitl", params={"confirm": "exp_hitl"}
        )
        self.assertEqual(alive.status_code, 409)
        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "stopped"},
        )
        gone = self.client.delete(
            "/api/experiments/exp_hitl", params={"confirm": "exp_hitl"}
        )
        self.assertEqual(gone.status_code, 200)
        self.assertFalse((self.experiments_root / "exp_hitl").exists())

    def test_delete_reclaims_the_experiments_own_image_tags(self) -> None:
        """The tags the experiment's image state lists as owned go with it;
        one already removed by hand is not an error."""

        import subprocess

        owned = [
            "autotrade-sandbox:exp_hitl-base-11111111-1111-4111-8111-111111111111",
            "autotrade-sandbox:exp_hitl-base-22222222-2222-4222-8222-222222222222",
        ]
        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "sandbox_image.json",
            {
                "experiment_id": "exp_hitl",
                "image_ref": owned[1],
                "owned_image_refs": owned,
                "kind": "base_clone",
            },
        )
        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "stopped"},
        )
        removals: list[list[str]] = []

        def fake_run(argv, **kwargs):
            argv = list(argv)
            if argv[1:3] == ["image", "rm"]:
                removals.append(argv)
                # The first tag is gone already: docker reports it, delete goes on.
                return subprocess.CompletedProcess(argv, 1 if argv[-1] == owned[0] else 0, "", "")
            # Every other docker call here (the container listing) finds nothing.
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch("subprocess.run", side_effect=fake_run):
            response = self.client.delete(
                "/api/experiments/exp_hitl", params={"confirm": "exp_hitl"}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["removed_image_refs"], owned)
        self.assertEqual(removals, [["docker", "image", "rm", ref] for ref in owned])
        self.assertFalse((self.experiments_root / "exp_hitl").exists())

    def test_failed_experiment_with_readonly_hardlinked_artifacts_can_be_deleted(
        self,
    ) -> None:
        directory = self.experiments_root / "exp_readonly"
        hitl = directory / "hitl"
        readonly = directory / "artifacts/results/valid_repro/asof"
        hitl.mkdir(parents=True)
        readonly.mkdir(parents=True)
        write_json_atomic(
            hitl / "status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        shared = self.repo_root / "shared-cache.parquet"
        shared.write_bytes(b"shared")
        shared.chmod(0o444)
        os.link(shared, readonly / "part.parquet")
        readonly.chmod(0o555)

        response = self.client.delete(
            "/api/experiments/exp_readonly", params={"confirm": "exp_readonly"}
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(directory.exists())
        self.assertEqual(shared.read_bytes(), b"shared")
        self.assertEqual(shared.stat().st_mode & 0o777, 0o444)

    def test_delete_filesystem_failure_returns_detail_without_false_success(
        self,
    ) -> None:
        directory = self.experiments_root / "exp_delete_error"
        hitl = directory / "hitl"
        hitl.mkdir(parents=True)
        write_json_atomic(
            hitl / "status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        with patch(
            "autotrade.webui.manager._remove_readonly_tree",
            side_effect=PermissionError("readonly artifact"),
        ):
            response = self.client.delete(
                "/api/experiments/exp_delete_error",
                params={"confirm": "exp_delete_error"},
            )

        self.assertEqual(response.status_code, 500)
        self.assertIn("was not fully deleted", response.json()["detail"])
        self.assertIn("PermissionError", response.json()["detail"])
        self.assertTrue(directory.exists())

    def test_delete_rejects_derived_sandbox_symlink_without_touching_target(
        self,
    ) -> None:
        directory = self._build_hitl_experiment("exp_sandbox_link")
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        external = self.repo_root / "external-sandbox"
        external.mkdir()
        marker = external / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        expected = self.repo_root / ".runtime/sandboxes/exp_sandbox_link"
        expected.parent.mkdir(parents=True)
        expected.symlink_to(external, target_is_directory=True)

        with patch("autotrade.webui.manager._remove_sandbox_tree") as remove:
            response = self.client.delete(
                "/api/experiments/exp_sandbox_link",
                params={"confirm": "exp_sandbox_link"},
            )

        self.assertEqual(response.status_code, 500, response.text)
        self.assertIn("symbolic link", response.json()["detail"])
        remove.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertTrue(directory.exists())

    def test_delete_rejects_symlinked_sandbox_root_without_touching_external_tree(
        self,
    ) -> None:
        experiment_id = "exp_sandbox_root_link"
        directory = self._build_hitl_experiment(experiment_id)
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        runtime_root = self.repo_root / ".runtime"
        runtime_root.mkdir()
        with tempfile.TemporaryDirectory() as external_tmp:
            external = Path(external_tmp)
            outside = external / experiment_id
            outside.mkdir()
            marker = outside / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            (runtime_root / "sandboxes").symlink_to(external, target_is_directory=True)

            with patch("autotrade.webui.manager._remove_sandbox_tree") as remove:
                response = self.client.delete(
                    f"/api/experiments/{experiment_id}",
                    params={"confirm": experiment_id},
                )

            self.assertEqual(response.status_code, 500, response.text)
            self.assertIn("symbolic-link sandbox root", response.json()["detail"])
            remove.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertTrue(directory.exists())

    def test_delete_rejects_symlinked_runtime_root_without_touching_external_tree(
        self,
    ) -> None:
        experiment_id = "exp_runtime_root_link"
        directory = self._build_hitl_experiment(experiment_id)
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        with tempfile.TemporaryDirectory() as external_tmp:
            external = Path(external_tmp)
            outside = external / "sandboxes" / experiment_id
            outside.mkdir(parents=True)
            marker = outside / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            (self.repo_root / ".runtime").symlink_to(external, target_is_directory=True)

            with patch("autotrade.webui.manager._remove_sandbox_tree") as remove:
                response = self.client.delete(
                    f"/api/experiments/{experiment_id}",
                    params={"confirm": experiment_id},
                )

            self.assertEqual(response.status_code, 500, response.text)
            self.assertIn("symbolic-link runtime root", response.json()["detail"])
            remove.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertTrue(directory.exists())

    def test_delete_with_corrupt_params_uses_only_derived_sandbox_path(self) -> None:
        directory = self._build_hitl_experiment("exp_corrupt_params")
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        explicit = self.repo_root / "explicit-work-root"
        explicit.mkdir()
        marker = explicit / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        (directory / "hitl/params.json").write_text(
            f'{{"work_root": {json.dumps(str(explicit))}, broken',
            encoding="utf-8",
        )
        sandbox = self.repo_root / ".runtime/sandboxes/exp_corrupt_params"
        sandbox.mkdir(parents=True)
        (sandbox / "runtime.txt").write_text("remove", encoding="utf-8")

        response = self.client.delete(
            "/api/experiments/exp_corrupt_params",
            params={"confirm": "exp_corrupt_params"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(directory.exists())
        self.assertFalse(sandbox.exists())
        self.assertEqual(response.json()["removed_work_root"], str(sandbox))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_delete_with_unreadable_params_uses_only_derived_sandbox_path(self) -> None:
        directory = self._build_hitl_experiment("exp_unreadable_params")
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        params_path = directory / "hitl/params.json"
        explicit = self.repo_root / "unreadable-explicit-work-root"
        explicit.mkdir()
        marker = explicit / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        write_json_atomic(params_path, {"work_root": str(explicit)})
        sandbox = self.repo_root / ".runtime/sandboxes/exp_unreadable_params"
        sandbox.mkdir(parents=True)
        (sandbox / "runtime.txt").write_text("remove", encoding="utf-8")
        original_read_text = Path.read_text

        def deny_params(path: Path, *args, **kwargs):
            if path == params_path:
                raise PermissionError("params unreadable")
            return original_read_text(path, *args, **kwargs)

        with patch.object(Path, "read_text", new=deny_params):
            response = self.client.delete(
                "/api/experiments/exp_unreadable_params",
                params={"confirm": "exp_unreadable_params"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(directory.exists())
        self.assertFalse(sandbox.exists())
        self.assertEqual(response.json()["removed_work_root"], str(sandbox))
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_delete_removes_console_sandbox_tree_under_shared_work_root(self) -> None:
        # Console-created experiments record the shared sandbox root in
        # params.json and the worker runs under <work_root>/<experiment_id>.
        # Deletion must reclaim exactly that subtree and nothing around it.
        experiment_id = "exp_console_layout"
        directory = self._build_hitl_experiment(experiment_id)
        write_json_atomic(
            directory / "hitl/status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "failed"},
        )
        params_path = directory / "hitl/params.json"
        shared_root = self.repo_root / ".runtime/sandboxes"
        params = json.loads(params_path.read_text(encoding="utf-8"))
        params["work_root"] = str(shared_root)
        write_json_atomic(params_path, params)
        sandbox = shared_root / experiment_id
        (sandbox / "run_0001").mkdir(parents=True)
        (sandbox / "run_0001/runtime.txt").write_text("remove", encoding="utf-8")
        sibling = shared_root / "exp_other_layout"
        sibling.mkdir()
        keep = sibling / "keep.txt"
        keep.write_text("keep", encoding="utf-8")

        response = self.client.delete(
            f"/api/experiments/{experiment_id}",
            params={"confirm": experiment_id},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["removed_work_root"], str(sandbox))
        self.assertFalse(sandbox.exists())
        self.assertFalse(directory.exists())
        self.assertTrue(shared_root.is_dir())
        self.assertEqual(keep.read_text(encoding="utf-8"), "keep")

    def test_delete_rejects_symlink_alias_to_another_experiment(self) -> None:
        target = self.experiments_root / "exp_target"
        target.mkdir()
        (target / "marker").write_text("keep", encoding="utf-8")
        (self.experiments_root / "exp_alias").symlink_to(
            target, target_is_directory=True
        )
        manager = ExperimentManager(self.repo_root, self.experiments_root)

        with self.assertRaisesRegex(ManagerError, "invalid experiment ID"):
            manager.delete_experiment("exp_alias")
        self.assertEqual((target / "marker").read_text(encoding="utf-8"), "keep")

    def test_health_raw_generation_states_drive_status(self) -> None:
        from autotrade.environment.data.contracts import RAW_GENERATION_FILENAME

        # No stamp at all (dev/test roots without a lake): still healthy.
        payload = self.client.get("/api/health").json()
        self.assertEqual(payload["raw_generation"], {"state": "absent"})
        self.assertEqual(payload["status"], "ok")
        stamp = self.repo_root / "data" / "raw" / RAW_GENERATION_FILENAME
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "state": "committed",
                    "generation_id": "gen42",
                    "completed_at": "2026-07-28T04:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        payload = self.client.get("/api/health").json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["raw_generation"]["state"], "committed")
        self.assertEqual(payload["raw_generation"]["generation_id"], "gen42")
        self.assertEqual(
            payload["raw_generation"]["completed_at"], "2026-07-28T04:00:00+00:00"
        )
        # A dirty lake (aborted mutating cron) must degrade health — the
        # 6-day production outage stayed green behind the hardcoded literal.
        stamp.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "state": "dirty",
                    "generation_id": "gen43",
                    "updated_at": "2026-07-23T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        payload = self.client.get("/api/health").json()
        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["raw_generation"]["state"], "dirty")
        self.assertEqual(
            payload["raw_generation"]["updated_at"], "2026-07-23T00:00:00+00:00"
        )
        # A malformed stamp is reported, never a 500.
        stamp.write_text("{oops", encoding="utf-8")
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["raw_generation"]["state"], "unreadable")
        self.assertEqual(response.json()["status"], "degraded")

    def test_assert_no_live_writer_guards_migrations(self) -> None:
        from autotrade.pipelines.hitl_state import assert_no_live_writer

        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {
                "schema_version": 1,
                "pid": os.getpid(),
                "pid_start_ticks": proc_start_ticks(os.getpid()),
                "state": "running_session",
            },
        )
        with self.assertRaises(RuntimeError):
            assert_no_live_writer(self.experiments_root / "exp_hitl")
        write_json_atomic(
            self.experiments_root / "exp_hitl" / "hitl" / "status.json",
            {"schema_version": 1, "pid": 999_999_999, "state": "stopped"},
        )
        assert_no_live_writer(self.experiments_root / "exp_hitl")

    def test_broken_experiment_is_isolated_from_creation_and_detail(self) -> None:
        broken = self.experiments_root / "exp_broken" / "hitl"
        broken.mkdir(parents=True)
        (broken / "status.json").write_text("{not json", encoding="utf-8")
        listing = {
            entry["experiment_id"]: entry
            for entry in self.client.get("/api/experiments").json()["experiments"]
        }
        # A broken experiment renders as a structured error, never a 500, and
        # stays inspectable and deletable from the console.
        self.assertEqual(listing["exp_broken"]["state"], "unreadable")
        self.assertIn("exp_hitl", listing)
        detail = self.client.get("/api/experiments/exp_broken")
        self.assertEqual(detail.status_code, 200, detail.text)
        self.assertEqual(detail.json()["state"], "unreadable")
        self.assertEqual(detail.json()["sessions"], [])
        health = self.client.get("/api/health").json()
        self.assertIn(
            "exp_broken",
            [item["experiment_id"] for item in health["unreadable_experiments"]],
        )
        self.assertEqual(health["status"], "degraded")
        # A broken neighbour must never block creating a new experiment.
        with stubbed_gpu_probe():
            created = self.client.post(
                "/api/experiments",
                json={
                    "params": {
                        "experiment_id": "exp_new",
                        **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                    }
                },
            )
        self.assertEqual(created.status_code, 200, created.text)

    def test_running_cap_allows_last_slot_and_blocks_overflow(self) -> None:
        manager = ExperimentManager(self.repo_root, self.experiments_root)
        running = [f"running_{index}" for index in range(MAX_RUNNING_EXPERIMENTS - 1)]
        with (
            patch.object(manager, "running_experiments", return_value=running),
            patch.object(manager, "_preflight"),
            patch.object(manager, "start_worker", return_value={"spawned": False}),
        ):
            created = manager.create_experiment(
                {
                    "experiment_id": "exp_last_slot",
                    **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                }
            )
        self.assertEqual(created["experiment_id"], "exp_last_slot")

        running.append("exp_last_slot")
        with (
            patch.object(manager, "running_experiments", return_value=running),
            self.assertRaisesRegex(
                ManagerError,
                rf"parallel experiment cap reached \({MAX_RUNNING_EXPERIMENTS}\)",
            ),
        ):
            manager.create_experiment(
                {
                    "experiment_id": "exp_overflow",
                    **DEFAULT_RESEARCH_GEOMETRY.to_record(),
                }
            )
        self.assertFalse((self.experiments_root / "exp_overflow").exists())

    def test_running_cap_also_guards_worker_restart(self) -> None:
        manager = ExperimentManager(self.repo_root, self.experiments_root)
        running = [f"running_{index}" for index in range(MAX_RUNNING_EXPERIMENTS)]
        with (
            patch.object(manager, "running_experiments", return_value=running),
            self.assertRaisesRegex(
                ManagerError,
                rf"parallel experiment cap reached \({MAX_RUNNING_EXPERIMENTS}\)",
            ),
        ):
            manager.start_worker("exp_hitl")

    # ---- traces ----------------------------------------------------------------
    def test_trace_pagination_and_partial_tail(self) -> None:
        run_ref = self._run_ref("run_s1")
        first = self.client.get(
            "/api/experiments/exp_hitl/trace/blocks", params={"run_id": run_ref}
        ).json()
        self.assertTrue(first["blocks"])
        self.assertTrue(first["eof"])
        again = self.client.get(
            "/api/experiments/exp_hitl/trace/blocks",
            params={"run_id": run_ref, "offset": first["next_offset"]},
        ).json()
        self.assertEqual(again["blocks"], [])
        self.assertEqual(again["next_offset"], first["next_offset"])

    def test_trace_run_id_traversal_is_rejected(self) -> None:
        for run_id in ("../run_s1", "run_s1/../..", "/etc/passwd"):
            response = self.client.get(
                "/api/experiments/exp_hitl/trace/blocks", params={"run_id": run_id}
            )
            # Never served, and never distinguishable from an absent trace.
            self.assertIn(response.status_code, (400, 404), run_id)
            self.assertNotIn("etc", response.text)
        # A traversing result name is never joined onto a path.
        self.assertEqual(
            self.client.get(
                "/api/experiments/exp_hitl/results/..%2F..%2Fhitl/style"
            ).status_code,
            404,
        )

    def test_trace_tail_returns_recent_events_and_stream_offset(self) -> None:
        tail = self.client.get(
            "/api/experiments/exp_hitl/trace/blocks",
            params={"run_id": self._run_ref("run_s1"), "tail_events": 2},
        ).json()
        self.assertEqual(
            [block["kind"] for block in tail["blocks"]],
            ["agent_output", "tool_group"],
        )
        self.assertTrue(tail["next_offset"] > 0)

    def test_trace_stats_counts_tokens_and_tool_calls(self) -> None:
        stats = self.client.get(
            "/api/experiments/exp_hitl/trace/stats",
            params={"run_id": self._run_ref("run_s1")},
        ).json()
        self.assertEqual(stats["counts"]["llm_call"], 2)
        self.assertEqual(stats["tool_counts"], {"shell": 1})
        self.assertEqual(stats["llm_total_tokens"], 3000)
        self.assertEqual(stats["llm_prompt_tokens"], 2300)
        self.assertEqual(stats["llm_completion_tokens"], 700)
        self.assertEqual(stats["subagent_tasks"], 0)

    def test_trace_download_serves_public_jsonl(self) -> None:
        trace_ref = self._identity().trace_ref("run_s1")
        response = self.client.get(
            "/api/experiments/exp_hitl/trace/download",
            params={"run_id": trace_ref},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.text.strip().splitlines()), 3)
        self.assertIn(trace_ref, response.headers["content-disposition"])
        self.assertNotIn("run_s1", response.headers["content-disposition"])

    def test_initial_prompt_is_read_off_the_session_start_event(self) -> None:
        trace = self.experiments_root / "exp_hitl/artifacts/traces/run_s2.jsonl"
        AgentTraceWriter(trace, ids={"experiment_id": "exp_hitl", "run_id": "run_s2"}).emit(
            "session_start",
            {
                "system_prompt": f"遵守 PIT 合同；工作区 {self.experiments_root}/exp_hitl",
                "instruction": "开始研究会话 s2",
            },
        )
        url = "/api/experiments/exp_hitl/trace/initial-prompt"
        prompt = self.client.get(url, params={"run_id": self._run_ref("run_s2")})
        self.assertEqual(prompt.status_code, 200, prompt.text)
        messages = prompt.json()["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "开始研究会话 s2")
        self.assertNotIn(str(self.experiments_root), prompt.text)
        self.assertNotIn("run_s2", prompt.text)
        # A trace without a session_start event has no initial prompt.
        missing = self.client.get(url, params={"run_id": self._run_ref("run_s1")})
        self.assertEqual(missing.status_code, 404)


class HitlControlActionTest(unittest.TestCase):
    """The control actions a research arm still has, and their refusals.

    Every test asserts the control-state change the worker actually reads
    back, not only the HTTP status.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo_root = Path(self._tmp.name)
        self.experiments_root = self.repo_root / "experiments"
        self.experiments_root.mkdir(parents=True)
        self.directory = build_arm(self.experiments_root, "exp_ctl", "research")
        self.client = TestClient(create_app(self.repo_root, self.experiments_root))

    def _control(self, directory: Path | None = None):
        return read_control((directory or self.directory) / "hitl/control.json")

    def _post(self, experiment_id: str = "exp_ctl", **payload):
        return self.client.post(f"/api/experiments/{experiment_id}/control", json=payload)

    def test_set_directive_stores_and_clears_a_per_session_directive(self) -> None:
        response = self._post(action="set_directive", session_key="research", directive="控制回撤")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self._control().directives["research"], "控制回撤")
        # An empty directive clears it rather than storing a blank.
        self._post(action="set_directive", session_key="research", directive="")
        self.assertNotIn("research", self._control().directives)
        # The operator defines the PIT policy, so a directive that names a date
        # is stored as written: the rule against it is written guidance (see
        # docs/agent-design.md), not a server-side refusal.
        dated = self._post(action="set_directive", session_key="research", directive="在 2022Q1 减仓")
        self.assertEqual(dated.status_code, 200, dated.text)
        self.assertEqual(self._control().directives["research"], "在 2022Q1 减仓")

    def test_per_session_settings_target_research_sessions_while_research_lasts(
        self,
    ) -> None:
        for action, directive in (("set_directive", "x"), ("set_gpu_count", "1")):
            with self.subTest(action=action):
                forward = self._post(action=action, session_key="forward", directive=directive)
                self.assertEqual(forward.status_code, 400)
                self.assertIn("research sessions only", forward.json()["detail"])
        # Once an artifact froze, no Agent session remains to read a setting.
        frozen = build_arm(self.experiments_root, "exp_frozen", "sealed")
        refused = self._post(
            "exp_frozen", action="set_directive", session_key="research", directive="x"
        )
        self.assertEqual(refused.status_code, 400)
        self.assertIn("research is over", refused.json()["detail"])
        self.assertEqual(self._control(frozen).directives, {})

    def test_inject_message_reaches_only_a_live_research_session(self) -> None:
        live = {
            "schema_version": 1,
            "pid": os.getpid(),
            "pid_start_ticks": proc_start_ticks(os.getpid()),
            "state": "running_session",
        }
        write_json_atomic(
            self.directory / "hitl/status.json", {**live, "session_key": "forward"}
        )
        refused = self._post(action="inject_message", session_key="forward", text="停")
        self.assertEqual(refused.status_code, 400)
        self.assertFalse((self.directory / "hitl/agent_inbox.jsonl").exists())
        write_json_atomic(self.directory / "hitl/status.json", {**live, "session_key": "research"})
        queued = self._post(action="inject_message", session_key="research", text="先看回撤")
        self.assertEqual(queued.status_code, 200, queued.text)
        self.assertEqual(queued.json()["session_key"], "research")

    def test_the_control_projection_is_dated_by_the_file_not_by_the_read(self) -> None:
        """`updated_at` says when the control was last written.

        The projection serialises a state read off disk, and the serialiser
        stamped the current time, so every read dated the control to the
        moment it was fetched while the file it claimed to project said
        something else — and nothing rewrote the file to agree.
        """

        path = self.directory / "hitl/control.json"
        before = json.loads(path.read_text(encoding="utf-8"))
        detail = self.client.get("/api/experiments/exp_ctl").json()
        self.assertEqual(detail["control"]["updated_at"], before["updated_at"])
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), before)
        # A control action is a write: it re-dates the file and returns that
        # same stamp.
        changed = self._post(
            action="set_directive", session_key="research", directive="控制回撤"
        )
        written = json.loads(path.read_text(encoding="utf-8"))["updated_at"]
        self.assertGreater(
            datetime.fromisoformat(written), datetime.fromisoformat(before["updated_at"])
        )
        self.assertEqual(changed.json()["control"]["updated_at"], written)
        # A file that carries no stamp is dated by nothing at all.
        write_json_atomic(
            path,
            {
                key: value
                for key, value in json.loads(path.read_text(encoding="utf-8")).items()
                if key != "updated_at"
            },
        )
        reread = self.client.get("/api/experiments/exp_ctl").json()
        self.assertIsNone(reread["control"]["updated_at"])

    def test_set_gpu_count_round_trips_and_refuses_everything_else(self) -> None:
        allocated = self._post(action="set_gpu_count", session_key="research", directive="2")
        self.assertEqual(allocated.status_code, 200, allocated.text)
        self.assertEqual(allocated.json()["control"]["gpu_counts"], {"research": 2})
        self.assertEqual(self._control().gpu_counts, {"research": 2})
        cleared = self._post(action="set_gpu_count", session_key="research", directive="")
        self.assertEqual(cleared.json()["control"]["gpu_counts"], {})
        zero = self._post(action="set_gpu_count", session_key="research", directive="0")
        self.assertEqual(zero.status_code, 200, zero.text)
        # 0 is a CPU-only allocation, not an absent one: it must survive the
        # write/read round trip through control.json, or the session silently
        # falls back to the experiment default the researcher just overrode.
        self.assertEqual(self._control().gpu_counts, {"research": 0})
        self._post(action="set_gpu_count", session_key="research", directive="")
        for directive, fragment in (("5", "0..4"), ("abc", "整数")):
            with self.subTest(directive=directive):
                refused = self._post(
                    action="set_gpu_count", session_key="research", directive=directive
                )
                self.assertEqual(refused.status_code, 400, refused.text)
                self.assertIn(fragment, refused.json()["detail"])
        missing = self._post(action="set_gpu_count", directive="2")
        self.assertEqual(missing.status_code, 400)
        self.assertIn("set_gpu_count requires session_key", missing.json()["detail"])
        unplanned = self._post(action="set_gpu_count", session_key="s9", directive="2")
        self.assertEqual(unplanned.status_code, 400)
        self.assertIn("unknown session key", unplanned.json()["detail"])
        self.assertEqual(
            self._control().gpu_counts, {}, "a refused request must persist nothing"
        )

    def test_the_action_set_is_pinned_and_retired_actions_are_unknown(self) -> None:
        from autotrade.webui.manager import _ACTIONS

        self.assertEqual(
            sorted(_ACTIONS),
            [
                "inject_message",
                "pause",
                "restart",
                "resume",
                "set_directive",
                "set_gpu_count",
                "stop",
                "terminate",
            ],
        )
        for retired in (
            "skip_to_heldout",
            "cancel_skip_to_heldout",
            "rerun_fold",
            "rollback_fold",
            "reveal_test_results",
            "not_a_real_action",
        ):
            with self.subTest(action=retired):
                refused = self._post(action=retired, session_key="research")
                self.assertEqual(refused.status_code, 400)
                self.assertIn("unknown control action", refused.json()["detail"])

    def test_lifecycle_actions_work_on_a_finished_arm(self) -> None:
        build_arm(self.experiments_root, "exp_done", "graduated")
        done = self.experiments_root / "exp_done"
        self.assertEqual(self._post("exp_done", action="pause").status_code, 200)
        self.assertEqual(self._control(done).request, "pause")
        self.assertEqual(self._post("exp_done", action="stop").status_code, 200)
        self.assertEqual(self._control(done).request, "stop")
        # `terminate` reaches its handler and refuses on its own terms.
        dead = self._post("exp_done", action="terminate")
        self.assertEqual(dead.status_code, 400)
        self.assertEqual(dead.json()["detail"], "no live worker to terminate")


def test_public_worker_log_is_repo_relative_and_never_a_host_path(tmp_path: Path):
    """The worker log location crosses the public boundary; a host path must not.

    ``worker_log`` is written into hitl/status.json and returned by the create
    and restart endpoints, both of which the console projects.
    """
    from autotrade.webui.registry import WORKER_LOG_DIR, worker_log_ref

    directory = tmp_path / "experiments" / "demo"
    AgentRefStore(directory)
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    (hitl / "schedule.json").write_text(
        json.dumps({"schema_version": 1, "sessions": []}), encoding="utf-8"
    )

    relative = worker_log_ref("demo")
    assert relative == f"{WORKER_LOG_DIR}/demo.log"
    assert not Path(relative).is_absolute()

    identity = PublicIdentity(directory)
    public = identity.public_status(
        {"schema_version": 1, "state": "launching", "worker_log": relative}
    )
    # A relative location survives the projection unchanged...
    assert public["worker_log"] == relative
    # ...while the absolute form it replaced would have been redacted, which is
    # exactly why it must never be stored.
    assert identity.public_status(
        {"worker_log": f"/Data2/lzp/ADMCubeQuant/{relative}"}
    )["worker_log"] == "[host path omitted]"
    assert "/" != json.dumps(public)[0]
    assert not any(
        isinstance(value, str) and Path(value).is_absolute()
        for value in public.values()
    )

    # The launch write is transient: the worker replaces status.json with its
    # own record, so a running experiment must still carry the reference.
    from autotrade.webui.registry import experiment_state, summarize_experiment

    (hitl / "status.json").write_text(
        json.dumps(
            {"schema_version": 1, "state": "running_session", "run_id": "run_001"}
        ),
        encoding="utf-8",
    )
    assert "worker_log" not in experiment_state(directory)
    log_path = tmp_path / relative
    log_path.parent.mkdir(parents=True)
    log_path.write_text("boom\n", encoding="utf-8")
    running = experiment_state(directory)
    # A recorded running state whose pid is gone reads as interrupted — the
    # case where the log is the only explanation left.
    assert running["state"] == "interrupted"
    assert running["worker_log"] == relative
    assert "worker_log" not in running["status"]
    listed = summarize_experiment(directory)
    assert listed["worker_log"] == relative


def test_a_worker_that_dies_during_boot_reads_as_interrupted(tmp_path: Path) -> None:
    """The console's active vocabulary must contain the worker's boot state.

    ``StatusReporter`` writes that state before the first session opens. A
    boot state the console does not count as active never degrades when the
    pid dies, so the experiment keeps its boot badge forever: no 已中断, and
    no resume button on a worker that is gone. The state is read back from the
    writer rather than spelled a second time here, so the two cannot drift.
    """
    from autotrade.webui.manager import _TERMINAL_RESUMABLE_STATES
    from autotrade.webui.registry import ACTIVE_STATES, experiment_state

    directory = tmp_path / "experiments/demo"
    hitl = directory / "hitl"
    hitl.mkdir(parents=True)
    reporter = StatusReporter(hitl / "status.json")
    reporter.start()
    reporter.stop()
    boot = read_status(hitl / "status.json")
    assert boot["state"] in ACTIVE_STATES

    write_json_atomic(
        hitl / "status.json", {**boot, "pid": 999_999_999, "pid_start_ticks": 1}
    )
    state = experiment_state(directory)
    assert state["state"] == "interrupted"
    # …and the console offers the researcher a way out of it.
    assert state["state"] in _TERMINAL_RESUMABLE_STATES

