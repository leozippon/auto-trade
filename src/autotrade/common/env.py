"""The one reader of a value from the process environment or a ``.env`` file.

The file is parsed, never evaluated: a line is ``NAME=value`` or
``export NAME=value``, with blanks around the name and the value ignored
and quote characters at either end of the value dropped; blank lines,
``#`` comments and lines without ``=`` are skipped, and nothing in a value
is expanded or executed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def env_value(name: str, env_file: str | Path = ".env") -> str:
    """``name`` from the process environment, else its first assignment in
    ``env_file`` (which may be missing); "" when neither has it."""

    if not _NAME.fullmatch(name):
        raise ValueError("env_var must be an environment variable name")
    direct = os.environ.get(name, "").strip()
    if direct:
        return direct
    path = Path(env_file)
    if not path.is_file():
        return ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if key == name:
            return value.strip().strip("'\"")
    return ""


__all__ = ["env_value"]
