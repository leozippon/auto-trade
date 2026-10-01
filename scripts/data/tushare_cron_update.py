#!/usr/bin/env python3
"""Thin CLI wrapper for scheduled AutoTrade TuShare updates."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.data_sources.tushare.cron_update import main

if __name__ == "__main__":
    raise SystemExit(main())
