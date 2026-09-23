#!/usr/bin/env python3
"""Loopback-only authentication gateway for the ADM-Cube WebUI.

A login issues an opaque random session token that stays valid for
``--session-days`` (default 30) from login, or until logout. Sessions live in
``--session-store`` as SHA-256 digests of their tokens, so they survive a
gateway restart and the file holds no usable cookie; deleting it while the
service is stopped invalidates every session.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import hmac
import html
import ipaddress
import json
import logging
import logging.handlers
import os
import re
import secrets
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, quote, urlsplit

SESSION_COOKIE = "__Host-admcube_session"
CSRF_COOKIE = "__Host-admcube_csrf"
# The session cookie is Lax so a link from another site still opens the
# console; cross-site writes stay refused by the edge Origin/Referer map and by
# this gateway's own source check. The login-form CSRF cookie stays Strict.
SAME_SITE = {SESSION_COOKIE: "Lax", CSRF_COOKIE: "Strict"}
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{40,128}$")
TOKEN_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_SESSION_DAYS = 30
# Browsers cap a cookie's lifetime at 400 days; a longer setting would not hold.
MAX_SESSION_DAYS = 400
BCRYPT_RE = re.compile(r"^\$2[aby]\$\d\d\$[./A-Za-z0-9]{53}$")
MAX_BODY_BYTES = 4096
MAX_COOKIE_BYTES = 8192
MAX_NEXT_BYTES = 2048
MAX_USERNAME_CHARS = 64
MAX_USERNAME_BYTES = 128
MAX_PASSWORD_CHARS = 256
MAX_PASSWORD_BYTES = 1024


class PasswordVerifier(Protocol):
    def verify(self, username: str, password: str) -> bool: ...


class AuditSink(Protocol):
    def info(self, message: str) -> None: ...


class SystemCrypt:
    """Thread-safe wrapper around the host libcrypt implementation."""

    def __init__(self) -> None:
        library = ctypes.util.find_library("crypt")
        if not library:
            raise RuntimeError("system libcrypt is unavailable")
        self._library = ctypes.CDLL(library, use_errno=True)
        self._crypt = self._library.crypt
        self._crypt.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
        self._crypt.restype = ctypes.c_void_p
        self._lock = threading.Lock()

    def verify(self, password: str, encoded: str) -> bool:
        password_bytes = password.encode("utf-8")
        encoded_bytes = encoded.encode("ascii")
        with self._lock:
            pointer = self._crypt(password_bytes, encoded_bytes)
            candidate = ctypes.string_at(pointer) if pointer else b""
        return hmac.compare_digest(candidate, encoded_bytes)


class HtpasswdVerifier:
    """Immutable bcrypt credential set loaded once at process startup."""

    def __init__(self, path: Path, crypt_backend: SystemCrypt | None = None) -> None:
        users: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                if ":" not in line:
                    raise RuntimeError(f"invalid htpasswd entry at line {line_number}")
                username, encoded = line.split(":", 1)
                if not _valid_username(username) or not BCRYPT_RE.fullmatch(encoded):
                    raise RuntimeError(f"invalid bcrypt htpasswd entry at line {line_number}")
                if username in users:
                    raise RuntimeError(f"duplicate htpasswd user at line {line_number}")
                users[username] = encoded
        if not users:
            raise RuntimeError("htpasswd contains no bcrypt users")
        self._users = users
        self._dummy = next(iter(users.values()))
        self._crypt = crypt_backend or SystemCrypt()

    def verify(self, username: str, password: str) -> bool:
        encoded = self._users.get(username, self._dummy)
        matches = self._crypt.verify(password, encoded)
        return bool(username in self._users) & matches


@dataclass(frozen=True)
class Config:
    public_origin: str
    public_host: str
    css_path: Path
    logo_path: Path
    gongan_path: Path
    session_store: Path
    extra_origins: tuple[str, ...] = ()
    session_seconds: int = DEFAULT_SESSION_DAYS * 86400
    csrf_seconds: int = 600
    max_sessions: int = 1024

    @property
    def allowed_origins(self) -> frozenset[str]:
        return frozenset((self.public_origin, *self.extra_origins))

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return frozenset(
            (self.public_host, *(urlsplit(origin).netloc for origin in self.extra_origins))
        )


@dataclass(frozen=True)
class Session:
    username: str
    created_at: float


class GatewayState:
    def __init__(
        self,
        config: Config,
        verifier: PasswordVerifier,
        audit: AuditSink,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.verifier = verifier
        self.audit_sink = audit
        self.clock = clock
        self.lock = threading.Lock()
        # Rewriting the store at startup purges expired sessions and proves the
        # store is writable before the first login depends on it.
        self.sessions: dict[str, Session] = {}
        with self.lock:
            self._commit_locked(
                _load_sessions(config.session_store, config.session_seconds, clock())
            )

    def audit_login(self, client_ip: str, username: str, result: str) -> None:
        timestamp = datetime.fromtimestamp(self.clock(), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        record = {
            "time": timestamp,
            "ip": client_ip[:64],
            "user": _audit_username(username),
            "result": result,
        }
        self.audit_sink.info(json.dumps(record, ensure_ascii=True, separators=(",", ":")))

    def create_session(self, username: str, previous_token: str | None) -> str:
        now = self.clock()
        token = secrets.token_urlsafe(32)
        with self.lock:
            sessions = self._live_locked(now)
            if previous_token and TOKEN_RE.fullmatch(previous_token):
                sessions.pop(_token_digest(previous_token), None)
            if len(sessions) >= self.config.max_sessions:
                oldest = min(sessions, key=lambda item: sessions[item].created_at)
                sessions.pop(oldest)
            sessions[_token_digest(token)] = Session(username=username, created_at=now)
            self._commit_locked(sessions)
        return token

    def check_session(self, token: str | None) -> bool:
        if not token or not TOKEN_RE.fullmatch(token):
            return False
        with self.lock:
            session = self.sessions.get(_token_digest(token))
        return session is not None and self.clock() - session.created_at < self.config.session_seconds

    def remove_session(self, token: str | None) -> None:
        if not token or not TOKEN_RE.fullmatch(token):
            return
        digest = _token_digest(token)
        with self.lock:
            if digest not in self.sessions:
                return
            sessions = self._live_locked(self.clock())
            sessions.pop(digest, None)
            self._commit_locked(sessions)

    def _live_locked(self, now: float) -> dict[str, Session]:
        return {
            digest: session
            for digest, session in self.sessions.items()
            if now - session.created_at < self.config.session_seconds
        }

    def _commit_locked(self, sessions: dict[str, Session]) -> None:
        # Memory changes only after the store holds the change, so a failed
        # write fails the request instead of leaving a session that a restart
        # would silently drop or a logout that a restart would undo.
        _save_sessions(self.config.session_store, sessions)
        self.sessions = sessions


class LoginServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: GatewayState) -> None:
        self.state = state
        super().__init__(address, LoginHandler)


class LoginHandler(BaseHTTPRequestHandler):
    server: LoginServer
    server_version = "ADM-Cube Login"
    sys_version = ""

    def log_message(self, _format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        if not self._trusted_request():
            return
        split = urlsplit(self.path)
        if split.path == "/login":
            self._get_login(split.query)
        elif split.path == "/check" and not split.query:
            self._get_check()
        elif split.path == "/login.css" and not split.query:
            self._serve_file(self.server.state.config.css_path, "text/css; charset=utf-8")
        elif split.path == "/login-logo.png" and not split.query:
            self._serve_file(self.server.state.config.logo_path, "image/png")
        elif split.path == "/gongan.png" and not split.query:
            self._serve_file(self.server.state.config.gongan_path, "image/png")
        else:
            self._send_text(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if not self._trusted_request(require_origin=True):
            return
        split = urlsplit(self.path)
        if split.query:
            self._send_text(HTTPStatus.NOT_FOUND, "Not found")
        elif split.path == "/login":
            self._post_login()
        elif split.path == "/logout":
            self._post_logout()
        else:
            self._send_text(HTTPStatus.NOT_FOUND, "Not found")

    def do_HEAD(self) -> None:
        self._send_text(HTTPStatus.NOT_FOUND, "Not found", head_only=True)

    def do_DELETE(self) -> None:
        if not self._trusted_request(require_origin=True):
            return
        self._send_text(HTTPStatus.NOT_FOUND, "Not found", head_only=True)

    do_OPTIONS = do_HEAD
    do_PUT = do_HEAD
    do_PATCH = do_HEAD
    do_TRACE = do_HEAD
    do_CONNECT = do_HEAD

    def _trusted_request(self, require_origin: bool = False) -> bool:
        try:
            peer = ipaddress.ip_address(self.client_address[0])
        except ValueError:
            peer = None
        if peer is None or not peer.is_loopback:
            self._send_text(HTTPStatus.FORBIDDEN, "Forbidden")
            return False
        if self.headers.get("Host", "") not in self.server.state.config.allowed_hosts:
            self._send_text(HTTPStatus.FORBIDDEN, "Forbidden")
            return False
        client_ip = self.headers.get("X-ADM-Cube-Client-IP", "")
        try:
            ipaddress.ip_address(client_ip)
        except ValueError:
            self._send_text(HTTPStatus.FORBIDDEN, "Forbidden")
            return False
        if require_origin:
            origins = self.headers.get_all("Origin", [])
            allowed = self.server.state.config.allowed_origins
            if origins:
                source_allowed = len(origins) == 1 and origins[0] in allowed
            else:
                referers = self.headers.get_all("Referer", [])
                source_allowed = len(referers) == 1 and any(
                    _same_origin_referer(referers[0], origin) for origin in allowed
                )
            if not source_allowed:
                self._send_text(HTTPStatus.FORBIDDEN, "Forbidden")
                return False
        return True

    def _get_login(self, query: str) -> None:
        next_path = "/"
        try:
            values = parse_qs(query, keep_blank_values=True, strict_parsing=True, max_num_fields=2)
            if set(values) - {"next"} or any(len(items) != 1 for items in values.values()):
                raise ValueError("invalid query")
            next_path = _safe_next(values.get("next", ["/"])[0])
        except (UnicodeError, ValueError):
            next_path = "/"
        if self.server.state.check_session(self._cookie(SESSION_COOKIE)):
            self._redirect(next_path)
            return
        csrf = self._cookie(CSRF_COOKIE)
        cookies: list[str] = []
        if not csrf or not TOKEN_RE.fullmatch(csrf):
            csrf = secrets.token_urlsafe(32)
            cookies.append(_set_cookie(CSRF_COOKIE, csrf, self.server.state.config.csrf_seconds))
        # Reusing a valid cookie without reissuing it preserves its original
        # expiry while keeping duplicate login GETs from invalidating a form.
        self._send_login_page(
            HTTPStatus.OK,
            csrf=csrf,
            next_path=next_path,
            error="",
            cookies=cookies,
        )

    def _post_login(self) -> None:
        client_ip = self.headers["X-ADM-Cube-Client-IP"]
        username = ""
        try:
            form = self._read_login_form()
            username = form["username"]
            password = form["password"]
            csrf = form["csrf"]
            next_path = _safe_next(form["next"])
            cookie_csrf = self._cookie(CSRF_COOKIE) or ""
            if not TOKEN_RE.fullmatch(csrf) or not hmac.compare_digest(csrf, cookie_csrf):
                self._login_failure(client_ip, username, next_path, "页面已过期，请重新登录", HTTPStatus.BAD_REQUEST)
                return
            if not _valid_username(username) or not _valid_password(password):
                self._login_failure(client_ip, username, next_path, "用户名或密码不正确", HTTPStatus.UNAUTHORIZED)
                return
            if not self.server.state.verifier.verify(username, password):
                self._login_failure(client_ip, username, next_path, "用户名或密码不正确", HTTPStatus.UNAUTHORIZED)
                return
        except (KeyError, UnicodeError, ValueError):
            self._login_failure(client_ip, username, "/", "请求无效，请重试", HTTPStatus.BAD_REQUEST)
            return

        previous = self._cookie(SESSION_COOKIE)
        session = self.server.state.create_session(username, previous)
        self.server.state.audit_login(client_ip, username, "success")
        self._redirect(
            next_path,
            cookies=[
                _set_cookie(SESSION_COOKIE, session, self.server.state.config.session_seconds),
                _clear_cookie(CSRF_COOKIE),
            ],
            status=HTTPStatus.SEE_OTHER,
        )

    def _login_failure(
        self,
        client_ip: str,
        username: str,
        next_path: str,
        error: str,
        status: HTTPStatus,
    ) -> None:
        self.server.state.audit_login(client_ip, username, "failure")
        csrf = secrets.token_urlsafe(32)
        self._send_login_page(
            status,
            csrf=csrf,
            next_path=_safe_next(next_path),
            error=error,
            cookies=[_set_cookie(CSRF_COOKIE, csrf, self.server.state.config.csrf_seconds)],
        )

    def _post_logout(self) -> None:
        try:
            content_length = self._content_length()
        except ValueError:
            self._send_text(HTTPStatus.BAD_REQUEST, "Bad request", cookies=[_clear_cookie(SESSION_COOKIE)])
            return
        if content_length > MAX_BODY_BYTES:
            self._send_text(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request too large")
            return
        if content_length:
            self.rfile.read(content_length)
        self.server.state.remove_session(self._cookie(SESSION_COOKIE))
        self._redirect("/login", cookies=[_clear_cookie(SESSION_COOKIE), _clear_cookie(CSRF_COOKIE)], status=HTTPStatus.SEE_OTHER)

    def _get_check(self) -> None:
        if self.server.state.check_session(self._cookie(SESSION_COOKIE)):
            self._send_empty(HTTPStatus.NO_CONTENT)
            return
        original = _safe_next(self.headers.get("X-Original-URI", "/"))
        login_location = "/login?next=" + quote(original, safe="")
        self._send_empty(HTTPStatus.UNAUTHORIZED, headers=[("X-Login-Location", login_location)])

    def _read_login_form(self) -> dict[str, str]:
        if self.headers.get_content_type() != "application/x-www-form-urlencoded":
            raise ValueError("unsupported content type")
        content_length = self._content_length()
        if content_length <= 0 or content_length > MAX_BODY_BYTES:
            raise ValueError("invalid body size")
        raw = self.rfile.read(content_length).decode("ascii")
        values = parse_qs(
            raw,
            keep_blank_values=True,
            strict_parsing=True,
            encoding="utf-8",
            errors="strict",
            max_num_fields=4,
        )
        if set(values) != {"username", "password", "csrf", "next"}:
            raise ValueError("unexpected form fields")
        if any(len(items) != 1 for items in values.values()):
            raise ValueError("duplicate form field")
        return {key: items[0] for key, items in values.items()}

    def _content_length(self) -> int:
        value = self.headers.get("Content-Length")
        if value is None or not value.isascii() or not value.isdecimal():
            raise ValueError("invalid content length")
        return int(value)

    def _cookie(self, name: str) -> str | None:
        header = self.headers.get("Cookie", "")
        if not header or len(header.encode("latin-1", "ignore")) > MAX_COOKIE_BYTES:
            return None
        cookie = SimpleCookie()
        try:
            cookie.load(header)
        except CookieError:
            return None
        morsel = cookie.get(name)
        return morsel.value if morsel else None

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._send_text(HTTPStatus.SERVICE_UNAVAILABLE, "Unavailable")
            return
        self._send(HTTPStatus.OK, body, content_type)

    def _send_login_page(
        self,
        status: HTTPStatus,
        csrf: str,
        next_path: str,
        error: str,
        cookies: list[str],
    ) -> None:
        body = _login_html(csrf, next_path, error).encode("utf-8")
        self._send(status, body, "text/html; charset=utf-8", cookies=cookies)

    def _redirect(
        self,
        location: str,
        cookies: list[str] | None = None,
        status: HTTPStatus = HTTPStatus.FOUND,
    ) -> None:
        self._send_empty(status, headers=[("Location", _safe_next(location))], cookies=cookies)

    def _send_text(
        self,
        status: HTTPStatus,
        message: str,
        cookies: list[str] | None = None,
        head_only: bool = False,
    ) -> None:
        body = message.encode("utf-8")
        self._send(status, b"" if head_only else body, "text/plain; charset=utf-8", cookies=cookies)

    def _send_empty(
        self,
        status: HTTPStatus,
        headers: list[tuple[str, str]] | None = None,
        cookies: list[str] | None = None,
    ) -> None:
        self._send(status, b"", "text/plain; charset=utf-8", headers=headers, cookies=cookies)

    def _send(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        headers: list[tuple[str, str]] | None = None,
        cookies: list[str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Security-Policy", "default-src 'none'; img-src 'self'; style-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        for name, value in headers or []:
            self.send_header(name, value)
        for cookie in cookies or []:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        if body:
            self.wfile.write(body)


def _valid_username(username: str) -> bool:
    if not username or len(username) > MAX_USERNAME_CHARS or len(username.encode("utf-8")) > MAX_USERNAME_BYTES:
        return False
    return ":" not in username and "\x00" not in username and not any(ord(char) < 32 or ord(char) == 127 for char in username)


def _valid_password(password: str) -> bool:
    return (
        bool(password)
        and len(password) <= MAX_PASSWORD_CHARS
        and len(password.encode("utf-8")) <= MAX_PASSWORD_BYTES
        and "\x00" not in password
    )


def _audit_username(username: str) -> str:
    return "".join(char if ord(char) >= 32 and ord(char) != 127 else "?" for char in username[:MAX_USERNAME_CHARS])


def _safe_next(value: str) -> str:
    if not value or len(value.encode("utf-8")) > MAX_NEXT_BYTES or any(ord(char) < 32 or ord(char) == 127 for char in value):
        return "/"
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.fragment or not parsed.path.startswith("/") or parsed.path.startswith("//"):
        return "/"
    if parsed.path in {"/login", "/logout", "/check", "/login.css", "/login-logo.png", "/gongan.png"}:
        return "/"
    return value


def _same_origin_referer(referer: str, public_origin: str) -> bool:
    if not referer or any(ord(char) < 32 or ord(char) == 127 for char in referer):
        return False
    try:
        parsed = urlsplit(referer)
    except ValueError:
        return False
    if parsed.username is not None or parsed.password is not None:
        return False
    return bool(parsed.scheme and parsed.netloc) and f"{parsed.scheme}://{parsed.netloc}" == public_origin


def _set_cookie(name: str, value: str, max_age: int) -> str:
    return f"{name}={value}; Path=/; Max-Age={max_age}; Secure; HttpOnly; SameSite={SAME_SITE[name]}"


def _clear_cookie(name: str) -> str:
    return f"{name}=; Path=/; Max-Age=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Secure; HttpOnly; SameSite={SAME_SITE[name]}"


def _token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def _load_sessions(path: Path, lifetime: int, now: float) -> dict[str, Session]:
    """Unexpired sessions from the store; a missing store is an empty one."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except ValueError as exc:
        raise RuntimeError(f"unreadable session store {path}; delete it to invalidate all sessions") from exc
    entries = raw.get("sessions") if isinstance(raw, dict) and set(raw) == {"sessions"} else None
    if not isinstance(entries, dict) or not all(
        _valid_session_entry(digest, entry) for digest, entry in entries.items()
    ):
        raise RuntimeError(f"invalid session store {path}; delete it to invalidate all sessions")
    return {
        digest: Session(username=entry["username"], created_at=float(entry["created_at"]))
        for digest, entry in entries.items()
        if now - entry["created_at"] < lifetime
    }


def _valid_session_entry(digest: str, entry: object) -> bool:
    return bool(
        TOKEN_DIGEST_RE.fullmatch(digest)
        and isinstance(entry, dict)
        and set(entry) == {"username", "created_at"}
        and isinstance(entry["username"], str)
        and _valid_username(entry["username"])
        and type(entry["created_at"]) in (int, float)
    )


def _save_sessions(path: Path, sessions: dict[str, Session]) -> None:
    """Atomically replace the store with a 0600 file (mkstemp's mode)."""

    payload = {
        "sessions": {
            digest: {"username": session.username, "created_at": session.created_at}
            for digest, session in sessions.items()
        }
    }
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _login_html(csrf: str, next_path: str, error: str) -> str:
    error_html = f'<p class="error" role="alert">{html.escape(error)}</p>' if error else ""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>登录 ADM-Cube</title>
  <link rel="icon" href="/login-logo.png">
  <link rel="stylesheet" href="/login.css">
</head>
<body>
  <main class="login-shell">
    <section class="login-card" aria-labelledby="login-title">
      <div class="brand"><img src="/login-logo.png" alt="ADM-Cube Logo"><span>ADM-Cube</span></div>
      <h1 id="login-title">登录</h1>
      {error_html}
      <form method="post" action="/login">
        <input type="hidden" name="csrf" value="{html.escape(csrf, quote=True)}">
        <input type="hidden" name="next" value="{html.escape(next_path, quote=True)}">
        <label for="username">用户名</label>
        <input id="username" name="username" type="text" autocomplete="username" maxlength="{MAX_USERNAME_CHARS}" required autofocus>
        <label for="password">密码</label>
        <input id="password" name="password" type="password" autocomplete="current-password" maxlength="{MAX_PASSWORD_CHARS}" required>
        <button type="submit">登录</button>
      </form>
    </section>
    <footer class="site-footer">
      <a href="https://beian.miit.gov.cn/" target="_blank" rel="noopener noreferrer">津ICP备2024017854号-2</a>
      <a href="https://beian.mps.gov.cn/#/query/webSearch?code=12010402002613" target="_blank" rel="noopener noreferrer"><img src="/gongan.png" alt="公安备案" width="18" height="18">津公网安备12010402002613号</a>
    </footer>
  </main>
</body>
</html>
"""


def make_audit_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("admcube-login-audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    handler = logging.handlers.WatchedFileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def parse_config(arguments: list[str] | None = None) -> tuple[argparse.Namespace, Config]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19081)
    parser.add_argument("--public-origin", required=True)
    parser.add_argument("--extra-public-origin", action="append", default=[])
    parser.add_argument("--htpasswd", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--session-store", type=Path, required=True)
    parser.add_argument("--session-days", type=int, default=DEFAULT_SESSION_DAYS)
    parser.add_argument("--css", type=Path, default=Path(__file__).with_name("login.css"))
    parser.add_argument("--logo", type=Path, default=Path(__file__).with_name("logo.png"))
    parser.add_argument("--gongan", type=Path, default=Path(__file__).with_name("gongan.png"))
    args = parser.parse_args(arguments)
    if args.listen != "127.0.0.1" or not 1 <= args.port <= 65535:
        parser.error("the service must listen on 127.0.0.1 and a valid TCP port")
    if not 1 <= args.session_days <= MAX_SESSION_DAYS:
        parser.error(f"--session-days must be between 1 and {MAX_SESSION_DAYS}")
    extra_origins: list[str] = []
    for raw in [args.public_origin, *args.extra_public_origin]:
        origin = urlsplit(raw)
        if origin.scheme != "https" or not origin.netloc or origin.path not in {"", "/"} or origin.query or origin.fragment:
            parser.error("public origins must be HTTPS without a path")
        extra_origins.append(raw.rstrip("/"))
    primary = urlsplit(extra_origins[0])
    config = Config(
        public_origin=extra_origins[0],
        public_host=primary.netloc,
        extra_origins=tuple(extra_origins[1:]),
        css_path=args.css.resolve(strict=True),
        logo_path=args.logo.resolve(strict=True),
        gongan_path=args.gongan.resolve(strict=True),
        session_store=args.session_store.resolve(),
        session_seconds=args.session_days * 86400,
    )
    return args, config


def main(arguments: list[str] | None = None) -> None:
    args, config = parse_config(arguments)
    verifier = HtpasswdVerifier(args.htpasswd)
    audit = make_audit_logger(args.audit_log)
    server = LoginServer((args.listen, args.port), GatewayState(config, verifier, audit))
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
