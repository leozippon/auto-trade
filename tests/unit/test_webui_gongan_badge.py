"""Console static route for the public-security filing badge."""

from pathlib import Path

from fastapi.testclient import TestClient

from autotrade.webui.server import create_app

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
CANONICAL = "/static/gongan.png"
MPS_HREF = "https://beian.mps.gov.cn/#/query/webSearch?code=12010402002613"


def test_unauthenticated_console_badge_is_png_at_stable_url(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    page = client.get("/")
    assert page.status_code == 200
    assert f'src="{CANONICAL}"' in page.text
    assert 'src="/gongan.png"' not in page.text
    assert f'href="{MPS_HREF}"' in page.text
    assert "津公网安备12010402002613号" in page.text

    badge = client.get(CANONICAL)
    assert badge.status_code == 200
    assert badge.headers["content-type"] == "image/png"
    assert badge.content[:8] == PNG_MAGIC
    assert "no-store" in badge.headers["cache-control"]

    head = client.head(CANONICAL)
    assert head.status_code == 200
    assert head.headers["content-type"] == "image/png"

    # Local console does not grow a second unauthenticated badge path.
    missing = client.get("/gongan.png")
    assert missing.status_code == 404
