"""Shared effective-time budget for one Agent session.

The budget counts active inference wall time.  Explicitly exempt operations
may pause it without changing the amount of active time left; nested pauses
extend the deadline only once for their union.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass


class InferenceTimeBudget:
    """A monotonic deadline that can be paused by named blocking operations."""

    def __init__(
        self,
        *,
        duration_seconds: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if (
            isinstance(duration_seconds, bool)
            or not math.isfinite(float(duration_seconds))
            or duration_seconds <= 0
        ):
            raise ValueError("duration_seconds must be a positive finite number")
        self._clock = clock or time.monotonic
        self._deadline = self._clock() + float(duration_seconds)
        self._pause_depth = 0
        self._pause_started: float | None = None
        self._lock = threading.RLock()

    def remaining(self) -> float:
        with self._lock:
            now = self._clock()
            active_now = self._pause_started if self._pause_started is not None else now
            return self._deadline - active_now

    def check(self) -> None:
        if self.remaining() <= 0:
            raise TimeoutError("Agent session deadline exceeded")

    @contextmanager
    def pause(self) -> Iterator[None]:
        """Exclude the context's wall time, including nested/exceptional exits."""

        with self._lock:
            self.check()
            if self._pause_depth == 0:
                self._pause_started = self._clock()
            self._pause_depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._pause_depth -= 1
                if self._pause_depth == 0:
                    assert self._pause_started is not None
                    self._deadline += max(0.0, self._clock() - self._pause_started)
                    self._pause_started = None


class SessionTimeBudgetAware:
    """Opt-in interface for components governed by a session time budget."""

    @property
    def session_time_budget(self) -> InferenceTimeBudget | None:
        raise NotImplementedError


@dataclass(frozen=True)
class TimeBudgetBinding:
    component: str
    budget: InferenceTimeBudget | None


def validate_time_budget_bindings(
    explicit: InferenceTimeBudget | None,
    bindings: tuple[TimeBudgetBinding, ...],
    *,
    owner: str,
) -> InferenceTimeBudget | None:
    """Resolve one budget and reject every opted-in component that differs."""

    expected = explicit or next(
        (binding.budget for binding in bindings if binding.budget is not None),
        None,
    )
    if expected is None:
        if bindings:
            names = ", ".join(binding.component for binding in bindings)
            raise ValueError(f"{owner} budget-aware component is unbound: {names}")
        return None
    for binding in bindings:
        if binding.budget is not expected:
            state = "unbound" if binding.budget is None else "bound to another budget"
            raise ValueError(
                f"{owner} must share one inference time budget; "
                f"{binding.component} is {state}"
            )
    return expected


__all__ = [
    "InferenceTimeBudget",
    "SessionTimeBudgetAware",
    "TimeBudgetBinding",
    "validate_time_budget_bindings",
]
