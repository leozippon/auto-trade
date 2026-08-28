"""The console's shipped assets must survive the public site's CSP.

Both public vhosts send `style-src 'self' data:` with no `'unsafe-inline'`,
which makes the browser refuse every inline `style` attribute -- including one
written by `setAttribute("style", ...)`. That silently dropped each
`el(..., { style })` declaration on the deployed site while leaving it working
on the CSP-free local console; the loudest symptom was the chart tooltip
painting at viewport (0,0) over the header on phones, where no `mousemove`
ever arrives to hide it.

Text checks on purpose: the invariant belongs to the shipped files, and the
two halves (a strict CSP, and assets that never need an inline style) only
mean anything together, so they are asserted in one place.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP_JS = REPO / "src/autotrade/webui/static/app.js"
STYLE_CSS = REPO / "src/autotrade/webui/static/style.css"
INDEX_HTML = REPO / "src/autotrade/webui/static/index.html"
VHOSTS = (
    REPO / "ops/nginx/aliyun/admcubequant-https.conf",
    REPO / "ops/nginx/aliyun/admcube-https.conf",
)
CSSOM_BRANCH = 'else if (key === "style") node.style.cssText = value;'


def _el_body() -> str:
    text = APP_JS.read_text(encoding="utf-8")
    start = text.index("function el(tag")
    return text[start : text.index("\n}\n", start)]


def test_el_applies_style_through_the_cssom() -> None:
    body = _el_body()
    assert CSSOM_BRANCH in body
    # Ordering matters: the generic branch below it would otherwise win.
    assert body.index(CSSOM_BRANCH) < body.index("node.setAttribute(key, value);")


def test_no_asset_writes_an_inline_style_attribute() -> None:
    for path in (APP_JS, STYLE_CSS, INDEX_HTML):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"""setAttribute\(\s*['"]style['"]""", text), path
        assert "style=" not in text, path


def test_chart_tip_is_hidden_by_the_stylesheet_not_by_a_runtime_style() -> None:
    rule = re.search(
        r"^\.chart-tip \{(.*?)^\}",
        STYLE_CSS.read_text(encoding="utf-8"),
        re.DOTALL | re.MULTILINE,
    )
    assert rule is not None
    assert re.search(r"^\s*display: none;$", rule.group(1), re.MULTILINE)
    # The tooltip node itself must carry no hidden state of its own.
    assert 'el("div", { class: "chart-tip" })' in APP_JS.read_text(encoding="utf-8")


def test_public_vhosts_keep_style_src_strict() -> None:
    for path in VHOSTS:
        policy = re.search(
            r'add_header Content-Security-Policy "([^"]+)"',
            path.read_text(encoding="utf-8"),
        )
        assert policy is not None, path
        directive = next(
            part.strip()
            for part in policy.group(1).split(";")
            if part.strip().startswith("style-src")
        )
        assert "'unsafe-inline'" not in directive, path
