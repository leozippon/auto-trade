"""Phase 2C experiment-page inject-message UI contracts.

Backend enqueue/reject behaviour lives in test_agent_inbox.py; this file
covers the console payload, enablement, queued copy, and the rule that a
queued send must not be painted as a consumed Trace user bubble.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
STYLE_CSS = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/style.css"


def _fn(script: str, name: str) -> str:
    start = script.index(f"function {name}(")
    rest = script[start + 1 :]
    nxt = len(rest)
    for marker in ("\nfunction ", "\nasync function "):
        index = rest.find(marker)
        if index != -1:
            nxt = min(nxt, index)
    return script[start : start + 1 + nxt]


def _const_block(script: str, name: str) -> str:
    start = script.index(f"const {name} =")
    end = script.index(";", start)
    return script[start : end + 1]


def _panel_source(script: str) -> str:
    return _fn(script, "injectMessagePanel")


def _live_source(script: str) -> str:
    return _fn(script, "liveTracePanel")


def _session_source(script: str) -> str:
    return _fn(script, "sessionDetailPanel")


def test_inject_message_constants_and_csp() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    style = STYLE_CSS.read_text(encoding="utf-8")
    panel = _panel_source(script)
    assert "const INJECT_MESSAGE_MAX_CHARS = 8192;" in script
    assert (
        'const INJECT_MESSAGE_QUEUED_NOTE = "已排队，将在 Agent 下一安全点生效";'
        in script
    )
    assert "LIVE_RUN_STATES" in script
    assert "style:" not in panel
    assert "onclick:" not in panel
    assert "addEventListener" in panel
    assert "textarea.directive.inject-input" in style
    assert ".inject-count" in style
    assert ".inject-queue" in style


def test_inject_panel_is_not_inside_live_trace_and_does_not_fake_user_trace() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    panel = _panel_source(script)
    live = _live_source(script)
    session = _session_source(script)
    assert "injectMessagePanel(detail, session)" in session
    assert "liveTracePanel(detail, session)" in session
    assert "inject_message" not in live
    assert "injectMessagePanel" not in live
    assert "injectDrafts" not in live
    assert "renderTraceBlocks" not in panel
    assert "renderUserBlock" not in panel
    assert "user_message" not in panel
    assert 'kind: "user"' not in panel
    assert "trace-block" not in panel
    assert "inbox.text" not in panel
    assert "inbox.messages" not in panel
    assert "pending_count" in panel
    assert "queued_ids" in panel
    assert "INJECT_MESSAGE_QUEUED_NOTE" in panel
    assert "sendControlAction(" in panel
    assert "buildInjectMessagePayload(session.key, checked.text, interrupt)" in panel
    assert "submit(false)" in panel
    assert "submit(true)" in panel
    assert "injectDrafts" in panel
    assert "不会取消已在途的模型调用" in panel
    assert "已取消" not in panel
    assert "已打断" not in panel
    assert "已中断" not in panel


def test_send_control_action_still_toasts_api_errors_and_reports_success() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    source = script.split("async function sendControlAction(", 1)[1].split(
        "\nasync function ", 1
    )[0]
    assert "toast(error.message, true)" in source
    assert "return true;" in source
    assert "return false;" in source
    assert script.count("async function sendControlAction(") == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node required for JS contract")
def test_inject_message_js_contract_via_node() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    harness = "\n".join(
        [
            _const_block(script, "LIVE_RUN_STATES"),
            _const_block(script, "INJECT_MESSAGE_MAX_CHARS"),
            _const_block(script, "TERMINAL_INJECT_STATES"),
            _fn(script, "injectMessageEnabled"),
            _fn(script, "injectMessageDisableReason"),
            _fn(script, "buildInjectMessagePayload"),
            _fn(script, "validateInjectMessageText"),
            _fn(script, "inboxQueueSummary"),
            """
const live = {
  state: "running_session",
  worker_alive: true,
  status: { session_key: "epoch_001/fold_2022Q1" },
};
const fold = { key: "epoch_001/fold_2022Q1", kind: "fold" };
const meta = { key: "epoch_001/meta_learning", kind: "meta_learning" };
const cases = [];
function check(name, got, want) {
  const a = JSON.stringify(got);
  const b = JSON.stringify(want);
  if (a !== b) cases.push({ name, got, want });
}
check("live fold", injectMessageEnabled(live, fold), true);
check("live meta", injectMessageEnabled({
  ...live,
  status: { session_key: "epoch_001/meta_learning" },
}, meta), true);
check("paused", injectMessageEnabled({ ...live, state: "paused" }, fold), false);
check("paused reason", injectMessageDisableReason({ ...live, state: "paused" }, fold),
  "实验已暂停。请先恢复运行后再发送。");
for (const state of ["completed", "stopped", "failed", "interrupted", "terminated"]) {
  check("terminal " + state, injectMessageEnabled({ ...live, state }, fold), false);
  check("terminal reason " + state,
    injectMessageDisableReason({ ...live, state }, fold),
    "会话已结束，无法发送。");
}
check("heldout", injectMessageEnabled(live, { key: fold.key, kind: "heldout" }), false);
check("other session", injectMessageEnabled(live, {
  key: "epoch_001/fold_2022Q2", kind: "fold",
}), false);
check("dead worker", injectMessageEnabled({ ...live, worker_alive: false }, fold), false);
check("no session_key", injectMessageEnabled({
  ...live, status: {},
}, fold), false);
check("no-agent reason", injectMessageDisableReason(live, {
  key: fold.key, kind: "heldout",
}), "当前没有可接收消息的 Agent 会话。");
check("payload false", buildInjectMessagePayload(fold.key, "继续验证", false), {
  action: "inject_message",
  session_key: fold.key,
  text: "继续验证",
  interrupt: false,
});
check("payload true", buildInjectMessagePayload(fold.key, "停一下", true), {
  action: "inject_message",
  session_key: fold.key,
  text: "停一下",
  interrupt: true,
});
check("empty", validateInjectMessageText(""), { ok: false, error: "消息不能为空" });
check("blank", validateInjectMessageText("  \\n\\t"), { ok: false, error: "消息不能为空" });
check("max", validateInjectMessageText("x".repeat(8192)).ok, true);
check("over", validateInjectMessageText("x".repeat(8193)), {
  ok: false,
  error: "消息不能超过 8192 个字符",
});
check("inbox hides body", inboxQueueSummary({
  pending_count: 1,
  queued_ids: ["abc"],
  text: "secret-body",
  messages: [{ text: "secret-body" }],
}), { pending_count: 1, queued_ids: ["abc"] });
if (INJECT_MESSAGE_MAX_CHARS !== 8192) {
  cases.push({ name: "max const", got: INJECT_MESSAGE_MAX_CHARS, want: 8192 });
}
if (cases.length) {
  console.error(JSON.stringify(cases, null, 2));
  process.exit(1);
}
""",
        ]
    )
    result = subprocess.run(
        ["node", "-e", harness],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(result.stderr or result.stdout or "node contract failed")
