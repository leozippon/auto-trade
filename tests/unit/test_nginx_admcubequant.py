"""Nginx text contract for the public login gate, filing badge, and edge install order."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOMAIN_CONF = REPO / "ops/nginx/aliyun/admcubequant-https.conf"
IP_CONF = REPO / "ops/nginx/aliyun/admcube-https.conf"
INSTALL_SCRIPT = REPO / "ops/nginx/install-admcubequant.sh"
INDEX_HTML = REPO / "src/autotrade/webui/static/index.html"
DOMAIN_FULLCHAIN = "/etc/letsencrypt/live/admcubequant.tj.cn/fullchain.pem"
DOMAIN_AVAILABLE_CP = "sudo cp ~/admcube-nginx/admcubequant-https.conf /etc/nginx/sites-available/admcubequant-https"
DOMAIN_ENABLED_LN = "sudo ln -sfn /etc/nginx/sites-available/admcubequant-https /etc/nginx/sites-enabled/admcubequant-https"
IP_FULLCHAIN = "/etc/letsencrypt/live/admcube-ip/fullchain.pem"
IP_PRIVKEY = "/etc/letsencrypt/live/admcube-ip/privkey.pem"
IP_CERT_CHECK = (
    f"if ! sudo -n test -f {IP_FULLCHAIN} || ! sudo -n test -f {IP_PRIVKEY}; then"
)
IP_CERT_REFUSAL = (
    f'echo "missing {IP_FULLCHAIN} or {IP_PRIVKEY}; refusing to enable IP vhost" >&2'
)
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
    assert "sub_filter_once on;" in login
    assert "sub_filter_types" not in login
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


def test_install_script_uses_sudo_to_detect_domain_cert_before_certbot() -> None:
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    domain_check = f"sudo -n test -f {DOMAIN_FULLCHAIN}"
    guarded_check = f"if ! {domain_check}; then"
    check_at = text.index(guarded_check)
    certbot_at = text.index("sudo certbot certonly")
    fi_at = text.index("\nfi\n", check_at)
    cp_at = text.index(DOMAIN_AVAILABLE_CP)
    ln_at = text.index(DOMAIN_ENABLED_LN)
    first_reload = text.index("sudo systemctl reload nginx")
    assert first_reload < check_at < certbot_at < fi_at < cp_at
    assert fi_at < ln_at
    assert f"[ ! -f {DOMAIN_FULLCHAIN} ]" not in text
    assert "sudo certbot" not in text[:check_at]
    assert "sudo certbot" not in text[fi_at:]
    assert DOMAIN_AVAILABLE_CP not in text[:check_at]
    assert DOMAIN_ENABLED_LN not in text[:check_at]
    assert "-d admcubequant.tj.cn" in text[check_at:fi_at]


def test_install_script_checks_ip_certs_before_enabling_ip_vhost() -> None:
    text = INSTALL_SCRIPT.read_text(encoding="utf-8")
    gate_at = text.index(IP_CERT_CHECK)
    echo_at = text.index(IP_CERT_REFUSAL, gate_at)
    exit_at = text.index("exit 1", echo_at)
    fi_at = text.index("\nfi\n", exit_at)
    cp_at = text.index(IP_AVAILABLE_CP)
    ln_at = text.index(IP_ENABLED_LN)
    fullchain_check = text.index(f"sudo -n test -f {IP_FULLCHAIN}", gate_at)
    privkey_check = text.index(f"sudo -n test -f {IP_PRIVKEY}", fullchain_check)
    assert gate_at <= fullchain_check < privkey_check < echo_at < exit_at < fi_at < cp_at
    assert fi_at < ln_at
    assert IP_AVAILABLE_CP not in text[:gate_at]
    assert IP_ENABLED_LN not in text[:gate_at]
    assert f"[ ! -f {IP_FULLCHAIN} ]" not in text
    assert f"[ ! -f {IP_PRIVKEY} ]" not in text
    assert "-d admcube-ip" not in text
    for line in text.splitlines():
        if "certbot" in line:
            assert "admcube-ip" not in line
