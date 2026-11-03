"""The market's timezone, for host code.

The sandbox-baked strategy contract (``environment/strategy.py``) keeps its
own copy, because the image carries no other ``autotrade`` module; the two are
the same zone (``tests/unit/test_common.py``).
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

CN_TZ = ZoneInfo("Asia/Shanghai")

__all__ = ["CN_TZ"]
