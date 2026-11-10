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

import ast
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
        # 演化: the page bundle, one entry's body, one experiment's mounts,
        # and the curated writes the page issues (create, edit, delete).
        "/api/memory",
        "/api/memory/curated",
        "/api/memory/curated/{}",
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


PAPER_PANEL_ROUTES = ("status", "book", "signal", "history", "performance", "snapshot", "pnl")


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

    history = client.get("/api/trading/paper/books/exp/history").json()
    assert history["days"][0]["fills"][0]["status"] == "filled"
    assert {"state", "error"} <= history.keys()
    day = history["days"][0]
    assert {
        "trade_date", "orders", "target", "cash_after", "cash_weight", "fills", "skipped_lines",
    } <= day.keys(), "a past day renders through the same order-sheet renderer as today"
    assert "executions" not in day, "the SPA reads day.fills"
    assert {"state", "error", "signal"} <= client.get("/api/trading/paper/books/exp/signal").json().keys()
    assert {"state", "error", "book", "verdict", "start_date", "settled_through", "last_fit_date"} <= (
        client.get("/api/trading/paper/books/exp/book").json().keys()
    )
    performance = client.get("/api/trading/paper/books/exp/performance").json()
    assert {
        "state", "error", "chart", "statistics", "min_days", "benchmark_label", "benchmark_error",
        "source", "source_error",
    } <= performance.keys()
    assert "snapshot" in client.get("/api/trading/paper/books/exp/snapshot").json()
    assert {
        "state", "error", "mark_date", "total_pnl", "total_return", "instruments", "instruments_error",
        "instruments_pnl", "residual",
    } <= client.get("/api/trading/paper/books/exp/pnl").json().keys()
    status = client.get("/api/trading/paper/books/exp/status").json()
    for key in ("book_id", "state", "error", "age_seconds", "stale_threshold_seconds"):
        assert key in status, key

    [row] = client.get("/api/trading/paper/books").json()["books"]
    for key in (
        "book_id", "experiment_id", "artifact_id", "verdict", "start_date",
        "initial_cash", "equity", "cash", "position_count", "total_return", "excess_return",
        "benchmark_label", "curve", "source", "signal_date", "order_count", "state", "error",
    ):
        assert key in row, key


def test_the_paper_page_names_each_books_own_benchmark() -> None:
    """A book is measured against the index it registered, which the payloads name;
    no Paper tile or note may name a fixed index in its place."""
    for name, payload in (("bookCard", "row"), ("paperEquityPanel", "payload")):
        body = _js_function_body(name)
        assert "沪深300" not in body, name
        assert f"{payload}.benchmark_label" in body, name


@pytest.mark.parametrize("route", PAPER_PANEL_ROUTES)
def test_the_paper_routes_the_console_reads_are_named_on_both_sides(route: str):
    assert f"/api/trading/{{}}/books/{{}}/{route}" in client_api_paths()
    assert f"/api/trading/{{}}/books/{{}}/{route}" in server_api_paths()


def _js_function_body(name: str) -> str:
    """Source of one top-level ``function name(...) { … }`` in app.js. The
    parameter list is skipped whole, so a destructured option object in it is
    not mistaken for the body."""
    source = APP_JS.read_text(encoding="utf-8")
    start = source.index(f"function {name}(")
    parens = 0
    for close in range(start, len(source)):
        if source[close] == "(":
            parens += 1
        elif source[close] == ")":
            parens -= 1
            if not parens:
                break
    index = source.index("{", close)
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
        "recordPanel",
        "validationWindowBlock",
        "forwardPoolBlock",
        "cardEquityNode",
        "endingBadge",
        "endingReason",
        "experimentEquityHost",
        "runningSubagentDock",
        "registeredPanel",
        "sliceStats",
        "subWindowSection",
        "elapsedClockNode",
        "subagentClockNode",
        "skippedChip",
        "activityNode",
        "budgetBars",
        "spanBar",
        "replaySpanBar",
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
            year_windows=(("20210801", "20220731"),),
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
    # The session's own stages: its preparation phases, and the one its event
    # sink publishes for each model call, tool (``test`` has its own) and wait.
    session = (
        Path(__file__).resolve().parents[2] / "src/autotrade/pipelines/research_session.py"
    ).read_text(encoding="utf-8")
    stages |= set(re.findall(r'_environment_phase\(\s*request\.progress_hook,\s*"([a-z_]+)"', session))

    def literals(node: ast.expr) -> set[str]:
        if isinstance(node, ast.IfExp):
            return literals(node.body) | literals(node.orelse)
        return {node.value} if isinstance(node, ast.Constant) and isinstance(node.value, str) else set()

    sink = next(
        node
        for node in ast.walk(ast.parse(session))
        if isinstance(node, ast.FunctionDef) and node.name == "_agent_event_sink"
    )
    published = set().union(
        *(
            literals(node.value)
            for node in ast.walk(sink)
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "stage"
        )
    )
    assert {"llm_call", "backtest", "test"} <= published
    stages |= published
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
    assert '"forward_replay"' in prep and '"readout"' in prep


def test_every_reason_the_pipeline_records_has_a_console_label() -> None:
    """A failed register-gate or verdict condition renders as its raw token when
    the console has no label for it, so the label map is checked against every
    reason a record can carry -- the verdict's condition table and the gate
    that could not be measured -- and every session outcome too."""

    from autotrade.pipelines.config import SESSION_OUTCOMES
    from autotrade.pipelines.verdict import CONDITIONS, ConditionReason
    recorded = {condition.reason for condition in CONDITIONS} | {ConditionReason.UNMEASURABLE}
    assert {"registration_nominee_is_control", "max_drawdown_above_limit"} <= recorded
    labels = _reason_tokens()
    assert recorded <= labels, sorted(recorded - labels)
    outcomes = set(
        re.findall(
            r"^  ([a-z_]+): \"",
            _js_literal("const OUTCOME_LABELS = {", "\n};"),
            re.MULTILINE,
        )
    )
    assert set(SESSION_OUTCOMES) <= outcomes, sorted(set(SESSION_OUTCOMES) - outcomes)


def test_every_track_status_and_verdict_reason_has_a_console_label() -> None:
    """The Paper page words the book's track, the verdict's status and the
    reason of its terminal row; a code added to paper/verdict.py without a
    label would render raw."""

    from autotrade.paper import verdict

    def keys(literal: str) -> set[str]:
        return set(re.findall(r"^  ([a-z_]+): ", _js_literal(literal, "\n};"), re.MULTILINE))

    # Every reason string the checkpoint logic can write.
    reasons: set[str] = set()
    source = ast.parse(Path(verdict.__file__).read_text(encoding="utf-8"))
    transitions = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "_first_transition")
    for node in ast.walk(transitions):
        is_pair = isinstance(node, ast.Tuple) and len(node.elts) == 2 and all(isinstance(e, ast.Constant) for e in node.elts)
        if is_pair and node.elts[0].value in ("killed", "confirmed"):
            reasons.add(node.elts[1].value)
    assert len(reasons) == 4, reasons
    assert reasons <= keys("const VERDICT_REASON_LABELS = {"), sorted(reasons)
    assert keys("const VERDICT_STATUS = {") == {"observing", "confirmed", "killed"}


def test_a_criterion_that_names_a_threshold_words_it_from_the_record() -> None:
    """A creation parameter must not be registered into a Chinese label. 毕业成本压力倍数
    is per-experiment, so an experiment run at ×3 used to read "前推加倍滑点后超额"
    beside a threshold column that correctly said ×3 — a reproducible mislabel.
    A label that names a threshold is a function of the `thresholds` block, and
    the multiplier is stated in the name alone, not twice."""

    labels = _js_literal("const REASON_LABELS = {", "\n};")
    rows = _js_literal("const SLICE_ROWS = [", "\n];")
    [line] = [row for row in rows.splitlines() if row.strip().startswith('["excess_at_cost_stress",')]
    assert "(t) =>" in line and "t.cost_stress_multiplier" in line, line
    # A missing threshold reads as the console's own dash, not as a default.
    assert '?? "—"' in line, line
    assert "加倍" not in labels + _js_literal("const SLICE_ROWS = [", "\n];")
    # No surface names the confidence in plain text: the lower bound's label is
    # worded from the slice's own parameters (``forward_confidence``), so a
    # change to the pipeline constant cannot leave a page quoting the old one.
    written = set(re.findall(r"(\d+)% 下界", APP_JS.read_text(encoding="utf-8")))
    assert written == set(), sorted(written)
    # The registration check is worded the same way: its record carries its limits, so
    # raising one cannot leave the checklist quoting the old number.
    gate = _js_function_body("registrationChecklist")
    assert not re.search(r"≥ [0-9]", gate), gate


def test_the_research_experiment_fields_the_console_reads_are_served(tmp_path: Path) -> None:
    """The panels read the registry's experiment projections by field name; a renamed
    field would render an empty cell instead of failing. Checked against a
    projection of a synthetic experiment that carries its readout, frozen with seed
    replicates so the fields only such an experiment has are served too."""

    from autotrade.webui.registry import experiment_detail, summarize_experiment
    from tests.unit.webui_research_experiment import build_experiment

    build_experiment(tmp_path, "experiment", "registered_with_record", seeded=True)
    detail = experiment_detail(tmp_path, "experiment")
    # The home page's row of the same experiment (the cards and the hero).
    row = summarize_experiment(tmp_path / "experiment")
    [research, replay] = detail["sessions"]
    assert (research["kind"], replay["kind"]) == ("research", "forward")
    registered = set(detail["registered"])
    record = set(research["record"])
    validation = set(research["record"]["validations"][0])
    rolling = set(detail["record"])
    rolling_row = set(detail["record"]["rows"][0])
    readout = set(detail["forward"]["readout"])
    gate = set(research["record"]["registration"])
    forward = set(detail["forward"])
    slices = detail["forward"]["slices"]
    # A slice's statistics table reads its holder's readings, and its residual
    # reading's, as rows of its own.
    slice_served = {
        name: set(block) | set(block["raw_readings"]) | set(block["residual_reading"])
        for name, block in slices.items()
    }
    slice_fields = slice_served["forward"] | slice_served["heldout"]
    budget = set(detail["budget"]) & set(detail["budget_used"])
    def reads(name: str, var: str) -> set[str]:
        return set(re.findall(rf"\b{var}\.([a-z_]+)", _js_function_body(name)))

    # A criterion or a statistics row whose own name states a threshold reads it
    # from the same block, so its wording cannot register at today's defaults.
    labelled = set(
        re.findall(
            r"\bt\.([a-z_]+)",
            _js_literal("const REASON_LABELS = {", "\n};")
            + _js_literal("const SLICE_ROWS = [", "\n];"),
        )
    )
    assert "cost_stress_multiplier" in labelled
    # A statistics row reads the slice's own parameters.
    slice_labelled = set(re.findall(r"\bt\.([a-z_]+)", _js_literal("const SLICE_ROWS = [", "\n];")))

    for name, read, served in (
        ("registeredPanel", reads("registeredPanel", "registered"), registered),
        ("researchSessionPanel", reads("researchSessionPanel", "record"), record),
        ("evidenceTiles", reads("evidenceTiles", "record"), rolling),
        ("evidenceTiles", reads("evidenceTiles", "item"), set(row)),
        ("evidenceTiles", reads("evidenceTiles", "forward"), set(row["forward"])),
        ("evidenceTiles", reads("evidenceTiles", "heldout"), set(row["forward"]["slices"]["heldout"]["residual_reading"])),
        ("forwardTiles", reads("forwardTiles", "f"), set(row["forward"]["slices"]["forward"])),
        ("heroSignature", reads("heroSignature", "item"), set(row)),
        ("paperHandoff", reads("paperHandoff", "detail"), set(row)),
        ("recordPanel", reads("recordPanel", "record"), rolling),
        ("recordPanel", reads("recordPanel", "detail"), set(detail)),
        ("processRows", reads("processRows", "detail"), set(detail)),
        ("recordPanel", reads("recordPanel", "row"), rolling_row),
        ("readoutBlock", reads("readoutBlock", "readout"), readout),
        ("readoutBlock", reads("readoutBlock", "record"), set(detail["forward"]["readout"]["record"])),
        ("readoutBlock", reads("readoutBlock", "forward"), set(detail["forward"]["readout"]["forward"])),
        ("accountTiles", reads("accountTiles", "raw"), set(slices["forward"]["raw_readings"])),
        ("researchSessionPanel", reads("researchSessionPanel", "row"), validation),
        ("registrationChecklist", reads("registrationChecklist", "gate"), gate),
        ("forwardStagePanel", reads("forwardStagePanel", "forward"), forward),
        ("replayContext", reads("replayContext", "forward"), forward),
        ("readoutStagePanel", reads("readoutStagePanel", "forward"), forward),
        ("readoutStagePanel", reads("readoutStagePanel", "detail"), set(detail)),
        ("sliceStats", slice_labelled, set(slices["forward"]["parameters"])),
        ("validationWindowBlock", reads("validationWindowBlock", "w"), set(detail["readouts"]["validation_window"])),
        ("validationWindowBlock", reads("validationWindowBlock", "row"), set(detail["readouts"]["validation_window"]["readings"][0])),
        ("forwardPoolBlock", reads("forwardPoolBlock", "p"), set(detail["readouts"]["forward_pool"])),
        ("readoutStagePanel", reads("readoutStagePanel", "attempts"), set(detail["replay_attempts"])),
        ("readoutStagePanel", reads("readoutStagePanel", "validations"), set(detail["research_validation_failures"])),
        ("readoutStagePanel", reads("readoutStagePanel", "research"), record),
        ("replaySpanBar", reads("replaySpanBar", "replay"), set(replay["replay"])),
        ("experimentEquityHost", reads("experimentEquityHost", "item"), set(detail)),
        ("stepper", set(re.findall(r'^  (\w+): "', _js_literal("const STEP_LABELS = {", "\n};"), re.MULTILINE)), {"research", "registered", "forward", "heldout", "readout"}),
    ):
        assert read, name
        assert read <= served, (name, sorted(read - served))
    # The same threshold keys are served before the replay, from the plan.
    assert {"alpha", "threshold"} <= set(replay["thresholds"])
    # The stage views' statistics rows are statistics their own slice carries,
    # and every row defined is one a view selects: two rows (残差跟踪误差, IR)
    # sat in the table definition that neither stage ever drew.
    slice_rows = set(re.findall(r'^  \["([a-z_]+)",', _js_literal("const SLICE_ROWS = [", "\n];"), re.MULTILINE))
    holder_rows = set(re.findall(r'"([a-z_]+)"', _js_literal("const HOLDER_STAT_FIELDS = [", "\n];")))

    def stat_rows(literal: str) -> set[str]:
        """A view's rows: its own names and the holder's readings it spreads."""
        rows = set(re.findall(r'"([a-z_]+)"', literal))
        return rows | holder_rows if "...HOLDER_STAT_FIELDS" in literal else rows

    forward_rows = stat_rows(_js_literal("const FORWARD_STAT_FIELDS = [", "\n];"))
    heldout_rows = stat_rows(_js_literal("const HELDOUT_STAT_FIELDS = [", "];"))
    assert slice_rows <= slice_fields
    assert forward_rows <= slice_served["forward"]
    assert heldout_rows <= slice_served["heldout"]
    assert slice_rows == forward_rows | heldout_rows, sorted(slice_rows ^ (forward_rows | heldout_rows))
    # The budget bars read the keys both the totals and the usage block carry.
    bars = set(re.findall(r'^  \["([a-z_]+)",', _js_literal("const BUDGET_ROWS = [", "\n];"), re.MULTILINE))
    assert bars and bars <= budget, sorted(bars - budget)


def _reason_tokens() -> set[str]:
    """Every labelled criterion token. A label is a string, or a function of the
    thresholds when its own wording states one."""

    return set(
        re.findall(r"^  ([a-z_]+): ", _js_literal("const REASON_LABELS = {", "\n};"), re.MULTILINE)
    )
