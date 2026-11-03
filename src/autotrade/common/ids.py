"""The experiment-id rule.

An experiment id is also its directory name under ``experiments/``, so the
rule is a safe path component: a letter or digit, then up to 99 letters,
digits, underscores or dashes.
"""

from __future__ import annotations

import re

EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}")


def is_experiment_id(value: object) -> bool:
    return isinstance(value, str) and EXPERIMENT_ID.fullmatch(value) is not None


__all__ = ["EXPERIMENT_ID", "is_experiment_id"]
