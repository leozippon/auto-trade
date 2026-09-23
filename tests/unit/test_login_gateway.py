"""Login gate on the edge host (ops/login-gateway/gateway.py): sessions, cookies, CSRF and source checks.

Each test serves the gateway on an ephemeral loopback port with a fake
password verifier, so neither the edge host, the network nor a bcrypt
library is needed.
"""

from __future__ import annotations

import contextlib
import http.client
import importlib.util
import io
import json
import os
import re
import stat
import sys
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode

MODULE_PATH = Path(__file__).resolve().parents[2] / "ops" / "login-gateway" / "gateway.py"
SPEC = importlib.util.spec_from_file_location("admcube_login_gateway", MODULE_PATH)
gateway = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = gateway
SPEC.loader.exec_module(gateway)


class Clock:
    def __init__(self) -> None:
        self.value = 1_800_000_000.0

    def __call__(self) -> float:
        return self.value


class Verifier:
    def verify(self, username: str, password: str) -> bool:
        return username == "operator" and password == "correct-password"


class Audit:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, message: str) -> None:
        self.lines.append(message)


class GatewayIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "login.css").write_text("body{}", encoding="utf-8")
        (root / "logo.png").write_bytes(b"png")
        (root / "gongan.png").write_bytes(b"png")
        self.store = root / "sessions.json"
        self.clock = Clock()
        self.audit = Audit()
        self.config = gateway.Config(
            public_origin="https://example.test",
            public_host="example.test",
            css_path=root / "login.css",
            logo_path=root / "logo.png",
            gongan_path=root / "gongan.png",
            session_store=self.store,
            session_seconds=30,
        )
        self.start_gateway()

    def start_gateway(self) -> None:
        self.state = gateway.GatewayState(self.config, Verifier(), self.audit, self.clock)
        self.server = gateway.LoginServer(("127.0.0.1", 0), self.state)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        self.thread.start()

    def stop_gateway(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def restart_gateway(self) -> None:
        self.stop_gateway()
        self.start_gateway()

    def tearDown(self) -> None:
        self.stop_gateway()
        self.temp.cleanup()

    def request(
        self,
        method: str,
        path: str,
        body: str = "",
        cookie: str = "",
        origin: str | None = None,
        referer: str | None = None,
    ):
        headers = {"Host": "example.test", "X-ADM-Cube-Client-IP": "203.0.113.8"}
        if cookie:
            headers["Cookie"] = cookie
        if origin is not None:
            headers["Origin"] = origin
        if referer is not None:
            headers["Referer"] = referer
        if body:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        result = response.status, response.getheaders(), payload
        connection.close()
        return result

    @staticmethod
    def cookie(headers: list[tuple[str, str]], name: str) -> str:
        for key, value in headers:
            if key.lower() == "set-cookie" and value.startswith(name + "="):
                return value.split(";", 1)[0]
        raise AssertionError(f"cookie {name} absent")

    def login_form(self):
        status, headers, body = self.request("GET", "/login?next=%2F%3Fview%3Dpaper")
        self.assertEqual(status, 200)
        csrf = re.search(rb'name="csrf" value="([A-Za-z0-9_-]+)"', body).group(1).decode()
        cookie = self.cookie(headers, gateway.CSRF_COOKIE)
        return csrf, cookie, body

    def post_login(
        self,
        username: str,
        password: str,
        csrf: str,
        cookie: str,
        next_path: str = "/",
        origin: str | None = "https://example.test",
        referer: str | None = None,
    ):
        body = urlencode({"username": username, "password": password, "csrf": csrf, "next": next_path})
        return self.request("POST", "/login", body=body, cookie=cookie, origin=origin, referer=referer)

    def test_csrf_rejected_and_username_not_reflected(self) -> None:
        status, headers, body = self.post_login("visible-user", "secret-value", "bad-token", "")
        self.assertEqual(status, 400)
        self.assertNotIn(b"visible-user", body)
        self.assertNotIn(b"secret-value", body)
        self.assertIn("页面已过期".encode(), body)
        self.assertTrue(self.cookie(headers, gateway.CSRF_COOKIE))
        self.assertEqual(json.loads(self.audit.lines[-1])["result"], "failure")

    def test_hidden_favicon_login_get_keeps_the_visible_form_valid(self) -> None:
        csrf, cookie, visible_body = self.login_form()
        self.assertIn(b'<link rel="icon" href="/login-logo.png">', visible_body)

        status, headers, hidden_body = self.request(
            "GET", "/login?next=%2Ffavicon.ico", cookie=cookie
        )
        self.assertEqual(status, 200)
        hidden_csrf = re.search(
            rb'name="csrf" value="([A-Za-z0-9_-]+)"', hidden_body
        ).group(1).decode()
        self.assertEqual(hidden_csrf, csrf)
        self.assertFalse(
            any(
                key.lower() == "set-cookie" and value.startswith(gateway.CSRF_COOKIE + "=")
                for key, value in headers
            )
        )

        status, _, _ = self.post_login("operator", "correct-password", csrf, cookie)
        self.assertEqual(status, 303)

    def test_invalid_csrf_cookie_is_rotated_with_the_full_security_attributes(self) -> None:
        for invalid in ("too-short", "x" * 40 + "."):
            with self.subTest(invalid=invalid):
                status, headers, body = self.request(
                    "GET", "/login", cookie=f"{gateway.CSRF_COOKIE}={invalid}"
                )
                self.assertEqual(status, 200)
                csrf = re.search(
                    rb'name="csrf" value="([A-Za-z0-9_-]+)"', body
                ).group(1).decode()
                raw_cookie = next(
                    value
                    for key, value in headers
                    if key.lower() == "set-cookie"
                    and value.startswith(gateway.CSRF_COOKIE + "=")
                )
                self.assertEqual(
                    self.cookie(headers, gateway.CSRF_COOKIE),
                    f"{gateway.CSRF_COOKIE}={csrf}",
                )
                self.assertNotEqual(csrf, invalid)
                self.assertIn(
                    "Path=/; Max-Age=600; Secure; HttpOnly; SameSite=Strict",
                    raw_cookie,
                )
                self.assertNotIn("Domain=", raw_cookie)

    def test_wrong_password_is_audited_without_secret_or_cookie(self) -> None:
        csrf, cookie, _ = self.login_form()
        status, _, body = self.post_login("operator", "wrong-secret", csrf, cookie)
        self.assertEqual(status, 401)
        self.assertNotIn(b"operator", body)
        joined = "\n".join(self.audit.lines)
        self.assertNotIn("wrong-secret", joined)
        self.assertNotIn(csrf, joined)
        self.assertNotIn(cookie, joined)
        self.assertEqual(json.loads(self.audit.lines[-1]), {"time": "2027-01-15T08:00:00Z", "ip": "203.0.113.8", "user": "operator", "result": "failure"})

    def test_session_check_logout_and_open_redirect_defense(self) -> None:
        csrf, cookie, _ = self.login_form()
        status, headers, _ = self.post_login("operator", "correct-password", csrf, cookie, "https://attacker.test/")
        self.assertEqual(status, 303)
        self.assertEqual(dict(headers)["Location"], "/")
        session = self.cookie(headers, gateway.SESSION_COOKIE)
        session_value = session.split("=", 1)[1]
        self.assertNotEqual(session_value, csrf)
        status, _, _ = self.request("GET", "/check", cookie=session)
        self.assertEqual(status, 204)
        status, headers, _ = self.request("POST", "/logout", cookie=session, origin="https://example.test")
        self.assertEqual(status, 303)
        self.assertIn("Max-Age=0", next(value for key, value in headers if key.lower() == "set-cookie" and value.startswith(gateway.SESSION_COOKIE)))
        status, headers, _ = self.request("GET", "/check", cookie=session)
        self.assertEqual(status, 401)
        self.assertEqual(dict(headers)["X-Login-Location"], "/login?next=%2F")

    def login(self) -> str:
        csrf, cookie, _ = self.login_form()
        status, headers, _ = self.post_login("operator", "correct-password", csrf, cookie)
        self.assertEqual(status, 303)
        return self.cookie(headers, gateway.SESSION_COOKIE)

    def test_session_cookie_is_persistent_lax_and_lasts_its_lifetime_without_activity(self) -> None:
        csrf, cookie, _ = self.login_form()
        _, headers, _ = self.post_login("operator", "correct-password", csrf, cookie)
        raw_cookie = next(
            value
            for key, value in headers
            if key.lower() == "set-cookie" and value.startswith(gateway.SESSION_COOKIE + "=")
        )
        self.assertIn("Path=/; Max-Age=30; Secure; HttpOnly; SameSite=Lax", raw_cookie)
        self.assertNotIn("Domain=", raw_cookie)
        session = raw_cookie.split(";", 1)[0]
        # No idle expiry: a session untouched for almost its whole lifetime
        # still opens the console, and it ends exactly at the lifetime.
        self.clock.value += 29
        self.assertEqual(self.request("GET", "/check", cookie=session)[0], 204)
        self.clock.value += 1
        self.assertEqual(self.request("GET", "/check", cookie=session)[0], 401)
        status, headers, _ = self.request("GET", "/login", cookie=session)
        self.assertEqual(status, 200)

    def test_sessions_and_logout_survive_a_gateway_restart(self) -> None:
        kept = self.login()
        ended = self.login()
        self.restart_gateway()
        self.assertEqual(self.request("GET", "/check", cookie=kept)[0], 204)
        self.assertEqual(self.request("GET", "/check", cookie=ended)[0], 204)
        status, _, _ = self.request("POST", "/logout", cookie=ended, origin="https://example.test")
        self.assertEqual(status, 303)
        self.restart_gateway()
        self.assertEqual(self.request("GET", "/check", cookie=kept)[0], 204)
        self.assertEqual(self.request("GET", "/check", cookie=ended)[0], 401)
        # The store is private and holds token digests, never a usable cookie.
        self.assertEqual(stat.S_IMODE(os.stat(self.store).st_mode), 0o600)
        stored = self.store.read_text(encoding="utf-8")
        self.assertNotIn(kept.split("=", 1)[1], stored)
        self.assertEqual(len(json.loads(stored)["sessions"]), 1)
        # Past its lifetime a restart drops the session from the store too.
        self.clock.value += 30
        self.restart_gateway()
        self.assertEqual(self.request("GET", "/check", cookie=kept)[0], 401)
        self.assertEqual(json.loads(self.store.read_text(encoding="utf-8")), {"sessions": {}})

    def test_forged_or_altered_session_token_is_rejected(self) -> None:
        name, token = self.login().split("=", 1)
        altered = token[:-1] + ("A" if token[-1] != "A" else "B")
        for value in (altered, gateway.secrets.token_urlsafe(32), gateway._token_digest(token)):
            with self.subTest(value=value):
                self.assertEqual(self.request("GET", "/check", cookie=f"{name}={value}")[0], 401)
        # A quoted octal escape makes the previous cookie non-ASCII; it cannot
        # break the next login.
        csrf, cookie, _ = self.login_form()
        status, _, _ = self.post_login("operator", "correct-password", csrf, f'{cookie}; {name}="caf\\351"')
        self.assertEqual(status, 303)

    def test_deleting_the_store_invalidates_every_session(self) -> None:
        session = self.login()
        self.stop_gateway()
        self.store.unlink()
        self.start_gateway()
        self.assertEqual(self.request("GET", "/check", cookie=session)[0], 401)

    def test_unreadable_store_fails_startup(self) -> None:
        for content in ("not json", '{"sessions": []}', '{"sessions": {"abc": {"username": "operator", "created_at": 1}}}'):
            with self.subTest(content=content):
                self.store.write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "session store"):
                    gateway.GatewayState(self.config, Verifier(), self.audit, self.clock)

    def test_input_boundaries_and_origin_host_checks(self) -> None:
        csrf, cookie, _ = self.login_form()
        status, _, _ = self.post_login("x" * (gateway.MAX_USERNAME_CHARS + 1), "password", csrf, cookie)
        self.assertEqual(status, 401)
        status, _, _ = self.request("POST", "/login", body="x=1", cookie=cookie)
        self.assertEqual(status, 403)
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        connection.request("GET", "/login", headers={"Host": "attacker.test", "X-ADM-Cube-Client-IP": "203.0.113.8"})
        self.assertEqual(connection.getresponse().status, 403)
        connection.close()

    def test_unknown_route_and_assets_are_no_store(self) -> None:
        status, headers, _ = self.request("GET", "/login.css")
        self.assertEqual(status, 200)
        self.assertEqual(dict(headers)["Cache-Control"], "no-store, max-age=0")
        self.assertEqual(self.request("GET", "/missing")[0], 404)

    def test_exact_origin_is_allowed(self) -> None:
        csrf, cookie, _ = self.login_form()
        status, _, _ = self.post_login("operator", "correct-password", csrf, cookie)
        self.assertEqual(status, 303)

    def test_missing_origin_with_same_origin_referer_is_allowed(self) -> None:
        csrf, cookie, _ = self.login_form()
        status, _, _ = self.post_login(
            "operator",
            "correct-password",
            csrf,
            cookie,
            origin=None,
            referer="https://example.test/login?next=%2Fpaper",
        )
        self.assertEqual(status, 303)

    def test_login_page_permits_same_origin_referer_fallback(self) -> None:
        status, headers, _ = self.request("GET", "/login")
        self.assertEqual(status, 200)
        self.assertEqual(dict(headers)["Referrer-Policy"], "same-origin")

    def test_explicit_evil_origin_is_rejected_even_with_same_origin_referer(self) -> None:
        csrf, cookie, _ = self.login_form()
        status, _, _ = self.post_login(
            "operator",
            "correct-password",
            csrf,
            cookie,
            origin="https://evil.test",
            referer="https://example.test/login",
        )
        self.assertEqual(status, 403)

    def test_missing_origin_and_referer_is_rejected(self) -> None:
        csrf, cookie, _ = self.login_form()
        status, _, _ = self.post_login(
            "operator", "correct-password", csrf, cookie, origin=None, referer=None
        )
        self.assertEqual(status, 403)

    def test_referer_prefix_userinfo_and_port_variants_are_rejected(self) -> None:
        for referer in (
            "https://example.test.evil.test/login",
            "https://example.test@evil.test/login",
            "https://evil.test@example.test/login",
            "https://example.test:443/login",
        ):
            with self.subTest(referer=referer):
                csrf, cookie, _ = self.login_form()
                status, _, _ = self.post_login(
                    "operator",
                    "correct-password",
                    csrf,
                    cookie,
                    origin=None,
                    referer=referer,
                )
                self.assertEqual(status, 403)

    def test_delete_uses_the_same_source_policy(self) -> None:
        self.assertEqual(
            self.request("DELETE", "/missing", origin="https://example.test")[0], 404
        )
        self.assertEqual(
            self.request(
                "DELETE", "/missing", referer="https://example.test/arbitrary/path?value=1"
            )[0],
            404,
        )
        self.assertEqual(
            self.request(
                "DELETE",
                "/missing",
                origin="https://evil.test",
                referer="https://example.test/arbitrary/path",
            )[0],
            403,
        )


class ParseConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        root = Path(self.temp.name)
        for name in ("login.css", "logo.png", "gongan.png"):
            (root / name).write_bytes(b"x")
        self.base = [
            "--public-origin", "https://example.test",
            "--htpasswd", str(root / "users"),
            "--audit-log", str(root / "audit.jsonl"),
            "--css", str(root / "login.css"),
            "--logo", str(root / "logo.png"),
            "--gongan", str(root / "gongan.png"),
        ]
        self.store = root / "sessions.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_session_lifetime_defaults_to_thirty_days_and_is_bounded(self) -> None:
        _, config = gateway.parse_config([*self.base, "--session-store", str(self.store)])
        self.assertEqual(config.session_seconds, 30 * 86400)
        self.assertEqual(config.session_store, self.store.resolve())
        _, config = gateway.parse_config([*self.base, "--session-store", str(self.store), "--session-days", "7"])
        self.assertEqual(config.session_seconds, 7 * 86400)
        for arguments in (
            self.base,
            [*self.base, "--session-store", str(self.store), "--session-days", "0"],
            [*self.base, "--session-store", str(self.store), "--session-days", "401"],
        ):
            with (
                self.subTest(arguments=arguments[-2:]),
                self.assertRaises(SystemExit),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                gateway.parse_config(arguments)


class HtpasswdVerifierTest(unittest.TestCase):
    def test_unknown_user_still_runs_bcrypt_backend(self) -> None:
        class CryptBackend:
            def __init__(self) -> None:
                self.calls = 0

            def verify(self, password: str, encoded: str) -> bool:
                self.calls += 1
                return True

        backend = CryptBackend()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "users"
            path.write_text("operator:$2y$05$" + "." * 53 + "\n", encoding="utf-8")
            verifier = gateway.HtpasswdVerifier(path, backend)
            self.assertFalse(verifier.verify("unknown", "anything"))
            self.assertEqual(backend.calls, 1)


if __name__ == "__main__":
    unittest.main()
