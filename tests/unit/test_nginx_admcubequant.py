"""Nginx text contract for the public login gate, filing badge, and edge install order."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOMAIN_CONF = REPO / "ops/nginx/aliyun/admcubequant-https.conf"
IP_CONF = REPO / "ops/nginx/aliyun/admcube-https.conf"
INSTALL_SCRIPT = REPO / "ops/nginx/install-admcubequant.sh"
INDEX_HTML = REPO / "src/autotrade/webui/static/index.html"
IP_FULLCHAIN = "/etc/letsencrypt/live/admcube-ip/fullchain.pem"
IP_PRIVKEY = "/etc/letsencrypt/live/admcube-ip/privkey.pem"
IP_AVAILABLE_CP = "sudo cp ~/admcube-nginx/admcube-https.conf /etc/nginx/sites-available/admcube-https"
IP_ENABLED_LN = "sudo ln -sfn /etc/nginx/sites-available/admcube-https /etc/nginx/sites-enabled/admcube-https"

GONGAN_BACKEND = "http://127.0.0.1:19080/static/gongan.png"
CANONICAL_SRC = 'src="/static/gongan.png"'
LOGIN_SRC_REWRITE = "sub_filter 'src=\"/gongan.png\"' 'src=\"/static/gongan.png\"';"
MPS_HREF = "https://beian.mps.gov.cn/#/query/webSearch?code=12010402002613"

# Exact public locations that must not run auth_request. Adding a path here
# is an explicit allowlist change, not a silent expansion.
DOMAIN_UNAUTHENTICATED = (
    "= /_admcube_auth",
    "= /login",
    "= /logout",
    "= /login.css",
    "= /login-logo.png",
    "= /gongan.png",
    "= /favicon.ico",
    "= /static/logo.png",
    "= /static/gongan.png",
    "@login_required",
    "@service_unavailable",
)
IP_UNAUTHENTICATED = (
    "= /_admcube_auth",
    "= /login",
    "= /logout",
    "= /login.css",
    "= /login-logo.png",
    "= /gongan.png",
    "= /static/gongan.png",
    "@login_required",
    "@service_unavailable",
)


def _location_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, str] = {}
    for match in re.finditer(r"(?m)^\s*location\s+([^{]+)\s*\{", text):
        name = match.group(1).strip()
        depth = 1
        index = match.end()
        while index < len(text) and depth:
            char = text[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            index += 1
        blocks[name] = text[match.end() : index - 1]
    return blocks


def _assert_gongan_locations(blocks: dict[str, str]) -> None:
    for name in ("= /gongan.png", "= /static/gongan.png"):
        body = blocks[name]
        assert "auth_request" not in body
        assert f"proxy_pass {GONGAN_BACKEND};" in body
        assert "proxy_set_header Host 127.0.0.1:38888;" in body
        assert 'proxy_set_header Cookie "";' in body


def _assert_login_rewrites_badge_src(blocks: dict[str, str]) -> None:
    login = blocks["= /login"]
    assert "auth_request" not in login
    assert LOGIN_SRC_REWRITE in login
    assert 'proxy_set_header Accept-Encoding "";' in login


def _assert_catch_all_is_authenticated(blocks: dict[str, str]) -> None:
    catch_all = blocks["/"]
    assert "auth_request /_admcube_auth;" in catch_all
    assert "location /static/" not in catch_all


def test_domain_vhost_serves_filing_badge_without_auth() -> None:
    text = DOMAIN_CONF.read_text(encoding="utf-8")
    blocks = _location_blocks(text)
    assert tuple(name for name, body in blocks.items() if "auth_request" not in body) == DOMAIN_UNAUTHENTICATED
    _assert_gongan_locations(blocks)
    _assert_login_rewrites_badge_src(blocks)
    _assert_catch_all_is_authenticated(blocks)
    assert "location /static/" not in text


def test_ip_vhost_serves_filing_badge_without_auth() -> None:
    text = IP_CONF.read_text(encoding="utf-8")
    blocks = _location_blocks(text)
    assert tuple(name for name, body in blocks.items() if "auth_request" not in body) == IP_UNAUTHENTICATED
    _assert_gongan_locations(blocks)
    _assert_login_rewrites_badge_src(blocks)
    _assert_catch_all_is_authenticated(blocks)
    assert "location /static/" not in text


def test_console_footer_uses_the_same_absolute_badge_url() -> None:
    page = INDEX_HTML.read_text(encoding="utf-8")
    assert page.count(CANONICAL_SRC) == 1
    assert 'src="/gongan.png"' not in page
    assert f'href="{MPS_HREF}"' in page
    assert "津公网安备12010402002613号" in page


def test_install_script_checks_ip_certs_before_enabling_ip_vhost() -> None:
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    fullchain_check = text.index(f"[ ! -f {IP_FULLCHAIN} ]")
    privkey_check = text.index(f"[ ! -f {IP_PRIVKEY} ]")
    cp_at = text.index(IP_AVAILABLE_CP)
    ln_at = text.index(IP_ENABLED_LN)
    assert fullchain_check < cp_at
    assert privkey_check < cp_at
    assert fullchain_check < ln_at
    assert privkey_check < ln_at
    gate = text[min(fullchain_check, privkey_check) : cp_at]
    assert "exit 1" in gate
    assert IP_AVAILABLE_CP not in text[: min(fullchain_check, privkey_check)]
    assert IP_ENABLED_LN not in text[: min(fullchain_check, privkey_check)]
    assert "-d admcube-ip" not in text
    for line in text.splitlines():
        if "certbot" in line:
            assert "admcube-ip" not in line
