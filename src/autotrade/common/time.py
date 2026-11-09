"""The market's timezone and the shared clock helpers, for host code.

The sandbox-baked strategy contract (``environment/strategy.py``) keeps its
own copy of the zone, because the image carries no other ``autotrade``
module; the two are the same zone (``tests/unit/test_common.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

CN_TZ = ZoneInfo("Asia/Shanghai")


def cn_today() -> str:
    """Today's market date as YYYYMMDD, whatever the host's own timezone."""
    return datetime.now(CN_TZ).strftime("%Y%m%d")


def utc_now(timespec: str = "seconds") -> str:
    """The current UTC instant in ISO form, at ``timespec`` precision."""
    return datetime.now(UTC).isoformat(timespec=timespec)


__all__ = ["CN_TZ", "cn_today", "utc_now"]
