"""A transport double for the model-gateway tests."""

from __future__ import annotations

import json


class FakeTransport:
    """Answers each ``post`` with the next outcome -- raised when it is an
    exception, returned when it is bytes, encoded as JSON otherwise -- and
    records every request as ``(url, headers, body, timeout)``."""

    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[tuple[str, dict[str, str], dict[str, object], float]] = []

    @property
    def urls(self) -> list[str]:
        return [request[0] for request in self.requests]

    @property
    def headers(self) -> list[dict[str, str]]:
        return [request[1] for request in self.requests]

    @property
    def bodies(self) -> list[dict[str, object]]:
        return [request[2] for request in self.requests]

    def post(self, url, headers, body, timeout):
        self.requests.append((url, dict(headers), json.loads(body), timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome if isinstance(outcome, bytes) else json.dumps(outcome).encode()
