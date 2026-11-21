"""Slices of the console's app.js source that the JS contract tests run or read."""

from __future__ import annotations


def js_function(script: str, name: str) -> str:
    """``function name(...)`` up to the next top-level function."""
    start = script.index(f"function {name}(")
    rest = script[start + 1 :]
    nxt = len(rest)
    for marker in ("\nfunction ", "\nasync function "):
        index = rest.find(marker)
        if index != -1:
            nxt = min(nxt, index)
    return script[start : start + 1 + nxt]


def js_const(script: str, name: str) -> str:
    """The ``const name = ...;`` statement."""
    start = script.index(f"const {name} =")
    end = script.index(";", start)
    return script[start : end + 1]
