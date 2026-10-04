"""What the first load of the console costs and how it looks.

A dark-mode reader used to see a light frame on every load: the stylesheet's
defaults were the light palette and the module script that switched to dark
ran only after the document had painted. The guarantees here are the order of
the document's first bytes, the behaviour of the one script that runs before
anything is drawn, a palette that defines every colour once per theme, and UI
code that is revalidated and compressed rather than resent whole each time.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrade.webui.server import create_app

STATIC = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static"
INDEX_HTML = STATIC / "index.html"
PREFERENCES_JS = STATIC / "preferences.js"
APP_JS = STATIC / "app.js"
STYLE_CSS = STATIC / "style.css"
HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")


def test_the_theme_script_runs_before_anything_can_paint() -> None:
    head = INDEX_HTML.read_text(encoding="utf-8").split("</head>")[0]
    tags = re.findall(r"<(script|link)\b([^>]*)>", head)
    scripts = [attrs for tag, attrs in tags if tag == "script"]
    # One classic, parser-blocking script in the head (no module, defer or
    # async), ahead of the stylesheet so it never waits for it either.
    assert scripts == [' src="/static/preferences.js"'], scripts
    order = [attrs for _tag, attrs in tags]
    stylesheet = next(i for i, attrs in enumerate(order) if 'rel="stylesheet"' in attrs)
    assert order.index(scripts[0]) < stylesheet


def _resolve(stored: str | None, system_dark: bool, *, storage_fails: bool = False) -> dict:
    """Run preferences.js against a stub document and report what it set."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    harness = f"""
const set = {{}};
globalThis.document = {{ documentElement: {{ dataset: set, style: {{
  setProperty: (name, value) => {{ set[name] = value; }} }} }} }};
globalThis.localStorage = {{ getItem: (key) => {{
  if ({json.dumps(storage_fails)}) throw new Error("blocked");
  return {{ ch_theme: {json.dumps(stored)}, ch_zoom: "1.15" }}[key] ?? null; }} }};
globalThis.window = {{ matchMedia: () => ({{ matches: {json.dumps(system_dark)} }}) }};
{PREFERENCES_JS.read_text(encoding="utf-8")}
console.log(JSON.stringify(set));
"""
    result = subprocess.run(
        [node, "-e", harness], capture_output=True, text=True, check=True, timeout=30
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    ("stored", "system_dark", "theme"),
    [
        ("dark", False, "dark"),
        ("light", True, "light"),
        (None, True, "dark"),
        (None, False, "light"),
        ("purple", True, "dark"),
    ],
)
def test_the_stored_choice_wins_and_the_system_decides_without_one(
    stored: str | None, system_dark: bool, theme: str
) -> None:
    applied = _resolve(stored, system_dark)
    assert applied["theme"] == theme
    assert applied["--ui-zoom"] == "1.15"


def test_blocked_storage_still_follows_the_system() -> None:
    assert _resolve("light", True, storage_fails=True) == {"theme": "dark"}


def _blocks(css: str) -> dict[str, str]:
    """Top-level rule bodies of the stylesheet, keyed by selector."""

    return {
        match.group(1).strip(): match.group(2)
        for match in re.finditer(r"(?m)^([^\s@}][^{]*)\{\n(.*?)^\}", css, re.DOTALL)
    }


def test_both_themes_define_every_colour_and_paint_the_canvas() -> None:
    css = re.sub(r"/\*.*?\*/", "", STYLE_CSS.read_text(encoding="utf-8"), flags=re.DOTALL)
    blocks = _blocks(css)
    light, dark = blocks[":root"], blocks[':root[data-theme="dark"]']
    assert "color-scheme: light;" in light and "color-scheme: dark;" in dark
    def colours(body: str) -> set[str]:
        return {
            name
            for name, value in re.findall(r"(--[\w-]+):\s*([^;]+);", body)
            if HEX.search(value) or "rgba(" in value
        }

    # A colour token the dark block leaves out would paint its light value.
    assert colours(light) <= colours(dark), sorted(colours(light) - colours(dark))
    assert "background: var(--bg);" in blocks["html"]
    # Colours live in those two blocks only, so no rule can carry a light
    # constant into the dark theme.
    elsewhere = HEX.findall(css.replace(light, "").replace(dark, ""))
    assert elsewhere == [], elsewhere
    # A theme is never animated: an eased switch is a flash of its own.
    for selector in ("html", "body"):
        assert "transition" not in blocks[selector], selector


def test_charts_read_their_inks_from_the_stylesheet() -> None:
    script = APP_JS.read_text(encoding="utf-8")
    assert HEX.findall(script) == []
    for token in re.findall(r'token\("(--[\w-]+)"\)', script):
        assert f"{token}:" in STYLE_CSS.read_text(encoding="utf-8"), token


def test_ui_code_is_revalidated_and_compressed(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    for path in ("/static/app.js", "/static/style.css", "/static/preferences.js"):
        first = client.get(path)
        assert first.headers["cache-control"] == "no-cache", path
        assert first.headers["content-encoding"] == "gzip", path
        again = client.get(path, headers={"If-None-Match": first.headers["etag"]})
        assert again.status_code == 304, path
    # A client that does not ask for gzip (the round script's urllib) reads
    # the same bytes uncompressed.
    plain = client.get("/static/app.js", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
    assert plain.text == APP_JS.read_text(encoding="utf-8")
