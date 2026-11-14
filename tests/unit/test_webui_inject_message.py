"""The experiment page's inject-message client logic, run under node.

Backend enqueue/reject behaviour lives in test_agent_inbox.py; this file
runs the console's enablement, payload, length check and inbox summary (which
never carries a queued message's body).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"


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
  status: { session_key: "s1" },
};
const session = { key: "s1", kind: "research" };
const cases = [];
function check(name, got, want) {
  const a = JSON.stringify(got);
  const b = JSON.stringify(want);
  if (a !== b) cases.push({ name, got, want });
}
check("live research session", injectMessageEnabled(live, session), true);
check("paused", injectMessageEnabled({ ...live, state: "paused" }, session), false);
check("paused reason", injectMessageDisableReason({ ...live, state: "paused" }, session),
  "实验已暂停。请先恢复运行后再发送。");
for (const state of ["completed", "stopped", "failed", "interrupted", "terminated"]) {
  check("terminal " + state, injectMessageEnabled({ ...live, state }, session), false);
  check("terminal reason " + state,
    injectMessageDisableReason({ ...live, state }, session),
    "会话已结束，无法发送。");
}
check("forward replay", injectMessageEnabled({
  ...live, status: { session_key: "forward" },
}, { key: "forward", kind: "forward" }), false);
check("other session", injectMessageEnabled(live, {
  key: "s2", kind: "research",
}), false);
check("dead worker", injectMessageEnabled({ ...live, worker_alive: false }, session), false);
check("no session_key", injectMessageEnabled({
  ...live, status: {},
}, session), false);
check("no-agent reason", injectMessageDisableReason(live, {
  key: session.key, kind: "forward",
}), "当前没有可接收消息的 Agent 会话。");
check("payload false", buildInjectMessagePayload(session.key, "继续验证", false), {
  action: "inject_message",
  session_key: session.key,
  text: "继续验证",
  interrupt: false,
});
check("payload true", buildInjectMessagePayload(session.key, "停一下", true), {
  action: "inject_message",
  session_key: session.key,
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
