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
        "/api/trading/{}/deals",
        "/api/parameter-schema",
        # Assembled from a local `const base`, not written inline.
        "/api/experiments/{}/analysis/{}/{}",
        "/api/experiments/{}/folds/{}/{}/orders",
        # 运行记忆: the page bundle, one entry's body, one experiment's mounts.
        "/api/memory",
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
    # `/deals` was once called `/executions` on the client only.
    assert "/api/trading/{}/deals" in server
    assert "/api/trading/{}/executions" not in server
    assert sorted({"/api/trading/{}/executions"} - server) == ["/api/trading/{}/executions"]


def test_paper_bundle_serves_the_key_names_the_console_reads(tmp_path: Path):
    """A status-only smoke test would have missed `payload.executions`: the
    SPA reads named keys, so the contract is the key names."""
    root = tmp_path / "data/trading/paper"
    root.mkdir(parents=True)
    (root / "orders_20260102.jsonl").write_text(
        json.dumps(
            {
                "event_id": "o1",
                "symbol": "000001.SZ",
                "action": "buy",
                "quantity": 100,
                "execute_at": "2026-01-02T09:30:00+08:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
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

    # The four requests the Paper page issues together; api() throws inside
    # Promise.all, so ONE 404 blanks the whole route.
    for route in ("snapshot", "orders", "deals", "series"):
        response = client.get(f"/api/trading/paper/{route}")
        assert response.status_code == 200, route
    assert client.get("/api/trading/paper/health").status_code == 200

    orders = client.get("/api/trading/paper/orders").json()
    assert "orders" in orders and orders["count"] == 1
    assert {"env", "trade_date", "available_dates", "state", "skipped_lines"} <= orders.keys()

    deals = client.get("/api/trading/paper/deals").json()
    assert "deals" in deals, "the SPA reads payload.deals, not payload.executions"
    assert "executions" not in deals
    assert deals["deals"][0]["status"] == "filled"

    roster = client.get("/api/trading/environments").json()["environments"][0]
    for key in ("env", "label", "state", "trade_date", "order_count", "deal_count",
                "skipped_lines", "stale_threshold_seconds", "snapshot"):
        assert key in roster, key
    assert "execution_count" not in roster, "the SPA reads summary.deal_count"

    series = client.get("/api/trading/paper/series").json()
    assert "series" in series and "state" in series


@pytest.mark.parametrize("route", ["orders", "deals", "series", "snapshot"])
def test_the_paper_routes_the_console_reads_are_named_on_both_sides(route: str):
    assert f"/api/trading/{{}}/{route}" in client_api_paths()
    assert f"/api/trading/{{}}/{route}" in server_api_paths()


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
    assert labels and "parent_control" in stages
    assert stages <= labels, sorted(stages - labels)
    # The control runs before the Agent session opens.
    prep = _js_literal("const PREP_ENVIRONMENT_STAGES = new Set([", "\n]);")
    assert '"parent_control"' in prep


def test_the_parent_control_and_walk_forward_surfaces_are_mounted() -> None:
    """A rendered-but-never-called panel is invisible and wholly green."""

    source = APP_JS.read_text(encoding="utf-8")
    assert "parentControlSection(" in _js_function_body("foldResultPanel")
    for name in ("walkForwardPanel", "walkForwardTerm"):
        assert source.count(f"{name}(") > 1, f"{name} is never called"


def test_the_parent_control_fields_the_console_reads_are_served() -> None:
    """The fold panel's baseline row and the walk-forward table read the
    registry's parent-control view by name; a renamed field would render an
    empty column instead of failing."""

    from autotrade.webui.registry import _parent_control_view

    served = set(
        _parent_control_view(
            {
                "parent_control": {
                    "status": "ok",
                    "validation_result": {
                        "total_return": 0.08,
                        "sharpe": 0.6,
                        "max_drawdown": 0.07,
                        "benchmark": {"benchmark_return": 0.03},
                    },
                }
            }
        )
    )
    read = set(
        re.findall(r"\bmetrics\.([a-z_]+)", _js_function_body("parentControlSection"))
    )
    read |= set(
        re.findall(r"\bcontrol\.([a-z_]+)", _js_function_body("walkForwardPanel"))
    )
    assert read, "the console reads no parent-control field"
    assert read <= served, sorted(read - served)


def test_the_walk_forward_counts_the_console_reads_are_served() -> None:
    """Two producers, two shapes: the per-Epoch table reads the registry's
    transition counts, the term beside the graduation badge reads the block
    the acceptance rules stamp into every Held-out verdict."""

    from autotrade.pipelines.config import AcceptanceRules
    from autotrade.webui.registry import _walk_forward_view

    counts = {"source": "parent_control", "transitions": 3, "positive_excess": 1}
    epoch_served = set(
        _walk_forward_view([], "epoch_001", test_stage=False, revealed=False)
    )
    verdict_served = set(AcceptanceRules.walk_forward_consistency(counts))
    epoch_read = set(
        re.findall(r"\bterm\.([a-z_]+)", _js_function_body("walkForwardPanel"))
    )
    verdict_read = set(
        re.findall(r"\bterm\.([a-z_]+)", _js_function_body("walkForwardTerm"))
    )
    assert epoch_read and verdict_read
    assert epoch_read <= epoch_served, sorted(epoch_read - epoch_served)
    assert verdict_read <= verdict_served, sorted(verdict_read - verdict_served)


def test_the_benchmark_fields_the_fold_panel_reads_are_served() -> None:
    """The raw and the size/beta-neutralized excess are read side by side, so
    both must exist in the block the evaluation summary actually carries."""

    from autotrade.environment.replay.style import benchmark_summary_block

    body = _js_function_body("foldResultPanel")
    read = set(re.findall(r"\bbenchmark\.([a-z_]+)", body))
    served = set(
        benchmark_summary_block(
            {
                "compact": {
                    "benchmark_return": 0.01,
                    "excess_return": 0.02,
                    "neutralized_excess_return": 0.03,
                    "neutralized_excess_method": "…",
                    "beta": 0.9,
                    "n_days": 60,
                    "size_tilt": -0.2,
                }
            }
        )
    )
    assert read, "the fold panel reads no benchmark field"
    assert read <= served, sorted(read - served)
