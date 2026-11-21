"""The TuShare data API: the HTTP client pinned to one endpoint, the token
readers and offset paging.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .versions import DefectiveAnswer

DEFAULT_TUSHARE_RELAY_URL = "https://fast.xiaodefa.cn"

# The official TuShare service, reached over TLS (the SDK's own default is the
# same host over plain HTTP, which would put the token on the wire in clear).
# Only datasets whose spec says ``official=True`` go here, with their own
# token; everything else stays on the relay with TUSHARE_TOKEN. The variable's
# name says the access may be temporary: without it those datasets alone stop.
OFFICIAL_TUSHARE_URL = "https://api.waditu.com/dataapi"
OFFICIAL_TOKEN_ENV = "TUSHARE_TOKEN_TMP"


# Minimum seconds between vendor requests, for every entry point that talks to
# TuShare. 0.80s (75 req/min) is the rate the 2026-09-10 catch-up sustained for
# ~2.5 hours without incident. The former 0.22s (272 req/min) was inside the
# stated 400/min ceiling but now trips a SIX-MINUTE vendor cooldown, and
# ``TuShareClient.query`` retries five times over ~20 seconds — a budget no
# cooldown can fit inside, so every request in the window fails and the step
# dies. Anything faster than this must be a deliberate, supervised override.
MIN_REQUEST_INTERVAL_SECONDS = 0.80

# The largest offset the vendor serves: a request past it is refused (code
# 50101, "offset不能大于100000"), so offset paging reaches at most this offset
# plus one page. A per-day answer can grow past it unnoticed (a since-retired
# board-membership table did on 20261103: 105,452 rows at 8,000 a page);
# query_paged never requests past it.
VENDOR_MAX_OFFSET = 100_000


@dataclass
class ApiResult:
    fields: list[str]
    items: list[list[Any]]


class TuShareClient:
    """The TuShare data API over HTTP, pinned to one endpoint.

    The endpoint is the relay unless a caller names another (the official
    service, for the datasets that use it). Requests follow the vendor SDK's
    wire protocol (a JSON POST to ``<endpoint>/<api_name>``), sent here rather
    than through the SDK because the SDK answers an HTTP error status with an
    empty frame, indistinguishable from an empty result. The token lives only
    in this object and is scrubbed from any error this adapter raises.

    An answer that lacks a requested field (compared case-insensitively, as
    the macro tier normalises vendor casing) is a :class:`DefectiveAnswer`.
    """

    def __init__(
        self,
        token: str,
        min_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        timeout: int = 60,
        endpoint: str | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("TuShare timeout must be a positive integer number of seconds")
        self._token = token
        self.endpoint = (endpoint or load_relay_url()).rstrip("/")
        self.min_interval = max(0.0, float(min_interval))
        self.timeout = timeout
        self.last_call = 0.0

    def _post(self, api_name: str, params: dict[str, Any], fields: str) -> ApiResult:
        payload = {
            "api_name": api_name,
            "token": self._token,
            # The SDK sends its endpoint as ts_type_name; the services expect it.
            "params": {**params, "ts_type_name": self.endpoint},
            "fields": fields,
        }
        response = requests.post(f"{self.endpoint}/{api_name}", json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"{api_name} answered HTTP {response.status_code}: {response.text[:200]}")
        body = response.json()
        if body.get("code") != 0:
            raise RuntimeError(f"{api_name} answered code {body.get('code')}: {body.get('msg')}")
        # An empty answer still carries its fields and an empty item list; a
        # missing or null block is a malformed answer (the SDK raised on it),
        # never zero rows.
        data = body.get("data")
        if not (isinstance(data, dict) and isinstance(data.get("fields"), list) and isinstance(data.get("items"), list)):
            raise RuntimeError(f"{api_name} answered code 0 without a data block of fields and items")
        return ApiResult(list(data["fields"]), list(data["items"]))

    def query(self, api_name: str, params: dict[str, Any] | None = None, fields: str = "", retries: int = 5) -> ApiResult:
        for attempt in range(1, retries + 1):
            self._throttle()
            try:
                result = self._post(api_name, dict(params or {}), fields)
            except Exception as exc:  # noqa: BLE001 - every vendor error is retried, then re-raised scrubbed
                if attempt == retries:
                    # Raised without the chained original, whose text the
                    # traceback would print unscrubbed.
                    raise RuntimeError(
                        f"{api_name} failed after {retries} attempts: {self._scrub(str(exc))}"
                    ) from None
                time.sleep(2 * attempt)
                continue
            requested = {name.strip().lower() for name in fields.split(",") if name.strip()}
            missing = requested - {str(name).lower() for name in result.fields}
            if result.items and missing:
                raise DefectiveAnswer(
                    f"{api_name} answered without requested fields {sorted(missing)} (got {result.fields[:20]})"
                )
            return result
        raise RuntimeError(f"{api_name} failed unexpectedly")

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.monotonic()

    def _scrub(self, text: str) -> str:
        return text.replace(self._token, "<token>") if self._token else text


def load_token(repo_root: Path, name: str = "TUSHARE_TOKEN") -> str:
    """The token in environment variable ``name``, else its line in the ignored .env."""
    token = os.environ.get(name, "").strip()
    if token:
        return token
    env_file = repo_root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(f"{name}="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                if token:
                    return token
    raise RuntimeError(f"{name} is not set in the environment or ignored .env")


def official_client(repo_root: Path, min_interval: float, timeout: int) -> TuShareClient:
    """A client on the official service with its own token (``OFFICIAL_TOKEN_ENV``)."""
    return TuShareClient(
        load_token(repo_root, OFFICIAL_TOKEN_ENV), min_interval, timeout, endpoint=OFFICIAL_TUSHARE_URL
    )


def load_relay_url() -> str:
    return os.environ.get("TUSHARE_RELAY_URL", DEFAULT_TUSHARE_RELAY_URL).strip() or DEFAULT_TUSHARE_RELAY_URL

def frame(result: ApiResult) -> pd.DataFrame:
    return pd.DataFrame(result.items, columns=result.fields)

def query_paged(client: TuShareClient, api_name: str, params: dict[str, Any], fields: str = "", page_limit: int | None = 10000, allow_repeated_pages: bool = False) -> tuple[ApiResult, int]:
    """Every row the vendor serves for ``params``, by offset paging.

    An answer that runs past VENDOR_MAX_OFFSET raises."""
    page_limit = page_limit or 10000
    all_items: list[list[Any]] = []
    result_fields: list[str] = []
    page_payloads: set[str] = set()
    pages = 0
    offset = 0
    while True:
        page_params = dict(params)
        page_params.update({"limit": page_limit, "offset": offset})
        result = client.query(api_name, page_params, fields)
        if not result_fields:
            result_fields = result.fields
        elif result.items and result.fields != result_fields:
            raise DefectiveAnswer(f"{api_name} returned inconsistent fields while paging")
        page_payload = json.dumps(
            {"fields": result.fields, "items": result.items},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if result.items and page_payload in page_payloads and not allow_repeated_pages:
            # A repeated full page normally means broken pagination. Sources
            # that serve heavily duplicated rows (eco_cal repeats each event
            # many times) legitimately produce identical pages; callers that
            # dedup by key opt out and rely on the offset cap below.
            raise DefectiveAnswer(
                f"{api_name} returned a repeated page while paging params={params}; "
                f"offset={offset} page_limit={page_limit}"
            )
        page_payloads.add(page_payload)
        all_items.extend(result.items)
        pages += 1
        if len(result.items) < page_limit:
            if not result.items and all_items:
                # The answer ended on a page boundary: an empty page is the
                # only sign it is complete, and an error answered as an empty
                # page reads the same. Read the boundary once more.
                again = client.query(api_name, page_params, fields)
                if again.items:
                    raise DefectiveAnswer(
                        f"{api_name} served rows at offset {offset} on a second read after an empty page; "
                        f"params={params} page_limit={page_limit}"
                    )
            break
        offset += page_limit
        if offset > VENDOR_MAX_OFFSET:
            # The vendor would refuse the next page and the rows past it are
            # unreachable with these params: fail instead of retrying a refusal.
            raise RuntimeError(
                f"{api_name} answer for params={params} runs past the vendor's offset cap "
                f"{VENDOR_MAX_OFFSET} (page_limit={page_limit}); narrow the request"
            )
    return ApiResult(result_fields, all_items), pages
