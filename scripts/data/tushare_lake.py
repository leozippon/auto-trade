#!/usr/bin/env python3
"""Operator commands over the versioned TuShare raw lake (revert, promote, repull, migrate)."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.data_sources.tushare.lake import main

if __name__ == "__main__":
    raise SystemExit(main())
