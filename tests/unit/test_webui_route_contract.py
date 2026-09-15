"""The client↔server route contract.

A live 404 once reached production behind a wholly green suite: `app.js` called
`/api/trading/{env}/executions` while the server registered `/deals`, and
because `api()` throws inside `Promise.all`, the whole 模拟 route died. Nothing
compared the two sides.

This module generalises that check rather than pinning the one instance: every
API path the SPA can request is parsed out of `app.js` and matched against the
routes `create_app()` actually registers, and the payload keys the SPA reads
are asserted against a real response.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrade.webui.server import create_app

APP_JS = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
# `api(`…`)`, `new EventSource(`…`)`, and bare "/api/…" string literals.
_TEMPLATE_CALL = re.compile(r"(?:api|EventSource)\(\s*`")
_PLAIN_LITERAL = re.compile(r'"(/api/[^"]*)"')
# `const base = `/api/…`;` then `api(base)` / `api(`${base}/orders`)`.
_CONST_TEMPLATE = re.compile(r"const\s+(\w+)\s*=\s*`(/api/[^`]*)`")
_BARE_CALL = re.compile(r"(?:api|EventSource)\(\s*(\w+)\s*[,)]")
_INTERPOLATION = re.compile(r"\$\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}")


def _template_literals(source: str) -> list[tuple[int, str]]:
    """Every template literal passed to api()/EventSource, backtick-balanced.

    A literal may nest another template inside `${ … }` (the orders route
    does), so a naive regex to the next backtick truncates it."""
    literals: list[tuple[int, str]] = []
    for match in _TEMPLATE_CALL.finditer(source):
        start = match.start()
        index = match.end()
        depth = 0
        buffer: list[str] = []
        while index < len(source):
            char = source[index]
            if char == "\\":
                buffer.append(source[index : index + 2])
                index += 2
                continue
            if char == "`" and depth == 0:
                break
            if source.startswith("${", index):
                depth += 1
                buffer.append("${")
                index += 2
                continue
            if char == "}" and depth:
                depth -= 1
                buffer.append("}")
                index += 1
                continue
            buffer.append(char)
            index += 1
        literals.append((start, "".join(buffer)))
    return literals


def _normalize(path: str) -> str:
    """A comparable path shape: parameters collapsed, query string removed.

    Interpolations collapse FIRST: a `${query ? "&" : "?"}` separator contains
    a literal `?`, so splitting on `?` before collapsing truncates the path."""
    previous = None
    while previous != path:  # nested ${ … ${ … } … }
        previous = path
        path = _INTERPOLATION.sub("{}", path)
    path = re.sub(r"\{[^{}]*\}", "{}", path)
    # A placeholder glued to a path segment (`…/equity{}`, `…/deals{}`,
    # `…/stream{}{}offset={}`) is an interpolated query string, not a path
    # parameter: the path ends where it starts.
    path = re.sub(r"(?<=[^/]){\}.*$", "", path)
    path = path.split("?", 1)[0].split("#", 1)[0]
    return path.rstrip("/") or "/"


def client_api_paths() -> set[str]:
    source = APP_JS.read_text(encoding="utf-8")
    # A route may be assembled in a local `const base = `/api/…`` and then
    # requested bare or extended. The same local name is reused in different
    # functions, so a binding is resolved by lexical scope — the nearest one
    # ABOVE the call site — not by cross-product, which would invent routes.
    bindings: list[tuple[int, str, str]] = [
        (match.start(), match.group(1), match.group(2))
        for match in _CONST_TEMPLATE.finditer(source)
    ]

    def resolve(position: int, name: str) -> str | None:
        nearest = [value for start, bound, value in bindings if bound == name and start < position]
        return nearest[-1] if nearest else None

    paths: set[str] = set()
    for position, literal in _template_literals(source):
        resolved = literal
        for _start, name, _value in bindings:
            token = "${" + name + "}"
            if token in resolved:
                value = resolve(position, name)
                if value is not None:
                    resolved = resolved.replace(token, value)
        normalized = _normalize(resolved)
        if normalized.startswith("/api/"):
            paths.add(normalized)
    for match in _BARE_CALL.finditer(source):
        value = resolve(match.start(), match.group(1))
        if value is not None:
            paths.add(_normalize(value))
    for literal in _PLAIN_LITERAL.findall(source):
        paths.add(_normalize(literal))
    return paths


def server_api_paths() -> set[str]:
    app = create_app(Path("."))
    return {
        _normalize(route.path)
        for route in app.routes
        if getattr(route, "path", "").startswith("/api/")
    }


def test_the_extractor_finds_the_routes_the_console_really_calls():
    """A parser that silently found nothing would make the check vacuous."""
    paths = client_api_paths()
    assert len(paths) >= 20, sorted(paths)
    for expected in (
        "/api/experiments",
        "/api/experiments/{}",
        "/api/experiments/{}/control",
        "/api/experiments/{}/trace/stream",
        "/api/trading/{}/books",
        "/api/trading/{}/books/{}/signal",
        "/api/parameter-schema",
        # Assembled from a local `const base`, not written inline.
        "/api/experiments/{}/results/{}/orders",
        "/api/experiments/{}/results/{}/equity",
        "/api/experiments/{}/results/{}/style",
        "/api/experiments/{}/trace/initial-prompt",
        # 运行记忆: the page bundle, one entry's body, one experiment's mounts,
        # and the curated writes the page issues (create, edit, delete, promote).
        "/api/memory",
        "/api/memory/curated",
        "/api/memory/curated/{}",
        "/api/memory/curated/{}/promote",
        "/api/memory/graduated/{}/{}",
        "/api/experiments/{}/memory",
    ):
        assert expected in paths, sorted(paths)


def test_every_api_path_the_console_calls_is_a_registered_route():
    missing = sorted(client_api_paths() - server_api_paths())
    assert missing == [], (
        "app.js calls API paths the server does not register (a live 404): "
        f"{missing}"
    )


def test_the_route_check_fails_on_a_renamed_route():
    """The mutation: the exact C1 defect must be detectable."""
    server = server_api_paths()
    # Fills were once requested as `/executions` on the client only.
    assert "/api/trading/{}/books/{}/history" in server
    assert "/api/trading/{}/executions" not in server
    assert sorted({"/api/trading/{}/executions"} - server) == ["/api/trading/{}/executions"]


PAPER_PANEL_ROUTES = ("status", "book", "signal", "history", "performance", "snapshot")


def test_paper_bundle_serves_the_key_names_the_console_reads(tmp_path: Path):
    """A status-only smoke test would have missed `payload.executions`: the
    SPA reads named keys, so the contract is the key names."""
    root = tmp_path / "data/trading/paper/exp"
    root.mkdir(parents=True)
    (root / "book.json").write_text(json.dumps({"experiment_id": "exp", "artifact_id": "art"}), encoding="utf-8")
    (root / "executions_20260102.jsonl").write_text(
        json.dumps(
            {
                "event_id": "e1",
                "symbol": "000001.SZ",
                "action": "buy",
                "quantity": 100,
                "execute_at": "2026-01-02T09:30:00+08:00",
                "matched_at": "2026-01-02T09:30:00+08:00",
                "status": "filled",
                "price": 10.25,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))

    # The requests a book's page issues together; api() throws inside
    # Promise.all, so ONE 404 blanks the whole route.
    for route in PAPER_PANEL_ROUTES:
        response = client.get(f"/api/trading/paper/books/exp/{route}")
        assert response.status_code == 200, route
    assert client.get("/api/trading/paper/health").status_code == 200

    history = client.get("/api/trading/paper/books/exp/history").json()
    assert history["days"][0]["fills"][0]["status"] == "filled"
    assert {"state", "error"} <= history.keys()
    day = history["days"][0]
    assert {
        "trade_date", "orders", "target", "cash_after", "cash_weight", "fills", "skipped_lines",
    } <= day.keys(), "a past day renders through the same order-sheet renderer as today"
    assert "executions" not in day, "the SPA reads day.fills"
    assert {"state", "error", "signal"} <= client.get("/api/trading/paper/books/exp/signal").json().keys()
    assert {"state", "error", "book", "start_date", "settled_through", "last_fit_date"} <= (
        client.get("/api/trading/paper/books/exp/book").json().keys()
    )
    performance = client.get("/api/trading/paper/books/exp/performance").json()
    assert {"state", "error", "chart", "statistics", "min_days", "benchmark_error"} <= performance.keys()
    assert "snapshot" in client.get("/api/trading/paper/books/exp/snapshot").json()
    status = client.get("/api/trading/paper/books/exp/status").json()
    for key in ("book_id", "state", "error", "age_seconds", "stale_threshold_seconds"):
        assert key in status, key

    [row] = client.get("/api/trading/paper/books").json()["books"]
    for key in (
        "book_id", "experiment_id", "artifact_id", "candidate_source", "start_date",
        "initial_cash", "equity", "position_count", "total_return", "excess_return",
        "max_drawdown", "curve", "signal_date", "order_count", "state", "error",
    ):
        assert key in row, key


@pytest.mark.parametrize("route", PAPER_PANEL_ROUTES)
def test_the_paper_routes_the_console_reads_are_named_on_both_sides(route: str):
    assert f"/api/trading/{{}}/books/{{}}/{route}" in client_api_paths()
    assert f"/api/trading/{{}}/books/{{}}/{route}" in server_api_paths()


def test_the_paper_health_route_is_an_external_probe_only() -> None:
    """``/health`` returns the roster entry the page already reads plus an
    ``ok`` flag. Polling it from the console would spend a request per refresh
    on data the page never renders, so it stays server-side only."""

    assert "/api/trading/{}/health" in server_api_paths()
    assert "/api/trading/{}/health" not in client_api_paths()


def _js_function_body(name: str) -> str:
    """Source of one top-level ``function name(...) { … }`` in app.js."""
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index(f"function {name}(")
    index = source.index("{", start)
    depth = 0
    for end in range(index, len(source)):
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
            if not depth:
                return source[index : end + 1]
    raise AssertionError(f"unbalanced braces in {name}")


def _dom_append_arguments(source: str) -> list[str]:
    """Every top-level argument of every DOM append/prepend/replaceChildren call.

    Scanned with balanced brackets and quotes, because an argument routinely
    holds nested calls, template literals and commas of its own.
    """

    arguments: list[str] = []
    for match in re.finditer(r"\.(?:append|prepend|replaceChildren)\(", source):
        index = match.end()
        depth = 0
        quote = ""
        current: list[str] = []
        while index < len(source):
            char = source[index]
            if quote:
                if char == "\\":
                    current.append(source[index : index + 2])
                    index += 2
                    continue
                if char == quote:
                    quote = ""
            elif char in "\"'`":
                quote = char
            elif char in "([{":
                depth += 1
            elif char in ")]}":
                if depth == 0:
                    arguments.append("".join(current))
                    break
                depth -= 1
            elif char == "," and depth == 0:
                arguments.append("".join(current))
                current = []
                index += 1
                continue
            current.append(char)
            index += 1
    return [argument.strip() for argument in arguments if argument.strip()]


def test_no_page_appends_a_renderer_that_can_return_nothing() -> None:
    """DOM append() stringifies null, el() drops it.

    The homepage hero once printed the literal "null" under its title because
    the tiles renderer — which answers "no out-of-sample figures" with null —
    was passed straight to panel.append(). A renderer that can answer nothing
    reaches the page through el(), a `||` fallback or an explicit guard.
    """

    source = APP_JS.read_text(encoding="utf-8")
    nullable = (
        "forwardTiles",
        "evidenceTiles",
        "cardEquityNode",
        "verdictBadge",
        "verdictPanel",
        "frozenPanel",
        "sliceTable",
        "subWindowSection",
        "elapsedClockNode",
        "subagentClockNode",
        "skippedChip",
    )
    # The list cannot quietly become a no-op: each name must still answer
    # nothing itself, or hand the answer on to another renderer that does.
    for name in nullable:
        body = _js_function_body(name)
        assert re.search(r"\bnull\b", body) or any(
            other != name and f"{other}(" in body for other in nullable
        ), name
    offenders = [
        argument
        for argument in _dom_append_arguments(source)
        if (match := re.match(r"(\w+)\(", argument))
        and match.group(1) in nullable
        and "||" not in argument
        and "??" not in argument
    ]
    assert offenders == [], offenders


def test_the_sub_window_columns_the_console_reads_are_the_ones_produced() -> None:
    """Same class of defect as the renamed route, one level down: the fold
    panel renders per-quarter rows the replay reducer writes, and a renamed or
    dropped column would silently render an empty column instead of failing."""

    from autotrade.environment.replay.stats import sub_window_stats

    body = _js_function_body("subWindowSection")
    read = set(re.findall(r"\brow\.([a-z_]+)", body))
    produced = set(
        sub_window_stats(
            (
                {"trade_date": "20220104", "initial_equity": 100.0, "equity": 110.0},
                {"trade_date": "20220331", "initial_equity": 100.0, "equity": 105.0},
            ),
            (),
            initial=100.0,
        )[0]
    )
    assert read, "the sub-window table reads no row field"
    assert read <= produced, sorted(read - produced)


def _js_literal(opening: str, closing: str) -> str:
    """Source of one top-level literal, from its opening line to its close."""
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index(opening)
    return source[start : source.index(closing, start) + len(closing)]


def test_every_progress_stage_the_pipeline_publishes_has_a_console_label() -> None:
    """An unlabelled stage renders its raw token in the live status line, and
    a preparation stage missing from the prep set makes the session panel offer
    Agent controls for a session that has not started yet."""

    pipeline = (
        Path(__file__).resolve().parents[2] / "src/autotrade/pipelines/experiment.py"
    ).read_text(encoding="utf-8")
    stages = set(re.findall(r'_publish_progress\(\s*progress,\s*"([a-z_]+)"', pipeline))
    labels = set(
        re.findall(
            r'^  ([a-z_]+): "',
            _js_literal("const ENVIRONMENT_STAGE_LABELS = {", "\n};"),
            re.MULTILINE,
        )
    )
    assert labels and "forward_replay" in stages
    assert stages <= labels, sorted(stages - labels)
    # The forward replay runs with no Agent session at all.
    prep = _js_literal("const PREP_ENVIRONMENT_STAGES = new Set([", "\n]);")
    assert '"forward_replay"' in prep and '"verdict"' in prep


def test_every_reason_the_pipeline_records_has_a_console_label() -> None:
    """A failed freeze-gate or verdict condition renders as its raw token when
    the console has no label for it, so the label map is checked against the
    tokens pipelines/verdict.py can emit, and every session outcome too."""

    from autotrade.pipelines.config import SESSION_OUTCOMES

    verdict = (
        Path(__file__).resolve().parents[2] / "src/autotrade/pipelines/verdict.py"
    ).read_text(encoding="utf-8")
    emitted = set(re.findall(r'reasons\.append\("([a-z_]+)"\)', verdict))
    emitted |= {f"{where}_strategy_error" for where in ("forward", "heldout")}
    assert "forward_lower_bound_not_positive" in emitted
    labels = set(
        re.findall(
            r"^  ([a-z_]+): \"",
            _js_literal("const REASON_LABELS = {", "\n};"),
            re.MULTILINE,
        )
    )
    assert emitted <= labels, sorted(emitted - labels)
    outcomes = set(
        re.findall(
            r"^  ([a-z_]+): \"",
            _js_literal("const OUTCOME_LABELS = {", "\n};"),
            re.MULTILINE,
        )
    )
    assert set(SESSION_OUTCOMES) <= outcomes, sorted(set(SESSION_OUTCOMES) - outcomes)


def test_the_research_arm_fields_the_console_reads_are_served(tmp_path: Path) -> None:
    """The panels read the registry's arm projections by field name; a renamed
    field would render an empty cell instead of failing. Checked against a
    projection of a synthetic arm that carries its verdict."""

    from autotrade.webui.registry import experiment_detail
    from tests.unit.webui_research_arm import build_arm

    build_arm(tmp_path, "arm", "graduated")
    detail = experiment_detail(tmp_path, "arm")
    frozen = set(detail["frozen"])
    record = set(detail["sessions"][1]["record"])
    best = set(detail["sessions"][1]["record"]["best"])
    validation = set(detail["sessions"][1]["record"]["validations"][0])
    forward = set(detail["forward"])
    for name, read, served in (
        ("frozenPanel", set(re.findall(r"\bfrozen\.([a-z_]+)", _js_function_body("frozenPanel"))), frozen),
        ("researchSessionPanel", set(re.findall(r"\brecord\.([a-z_]+)", _js_function_body("researchSessionPanel"))), record),
        ("researchSessionPanel", set(re.findall(r"\bbest\.([a-z_]+)", _js_function_body("researchSessionPanel"))), best),
        ("researchSessionPanel", set(re.findall(r"\brow\.([a-z_]+)", _js_function_body("researchSessionPanel"))), validation),
        ("verdictPanel", set(re.findall(r"\bforward\.([a-z_]+)", _js_function_body("verdictPanel"))), forward),
    ):
        assert read, name
        assert read <= served, (name, sorted(read - served))
    # The slice table's rows are statistics the verdict slices carry.
    slice_fields = set(re.findall(r'^  \["([a-z_]+)",', _js_literal("const SLICE_ROWS = [", "\n];"), re.MULTILINE))
    slices = detail["forward"]["slices"]
    assert slice_fields <= set(slices["forward"]) | set(slices["heldout"])
