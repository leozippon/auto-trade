"""Transferable-content policy for PRIOR.md and shared skills.

Pure text checks shared by the ``finish_meta`` gate, the skills write tools,
the PRIOR store, and the console: calendar dates, Held-out mentions, per-Fold
Test figures, and Test-based selection must not leak into cross-Fold memory.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

# A bare 4-digit number followed by one of these is a count/threshold, not a
# date (e.g. "2000 只股票"), so it must not trip the visible-window year check.
_COUNT_UNITS = "只家个支亿万元点名股条款行列页倍"

# A year welded to date syntax (年 / -MM / .MM / Qn / 季度), an 8-digit YYYYMMDD,
# or QnYYYY — a calendar date regardless of which year, so it stays correct when
# the visible fold moves to another year. Bare 4-digit numbers are NOT matched,
# and an 8-digit run inside an ASCII identifier (``factor_cs_20260901``, the
# experiment id pattern) is a name, not a date; ``截至20210930`` still is one.
_DATE_EXPR = re.compile(
    r"(?<![A-Za-z0-9_])(?:19|20)\d{2}\s*(?:年|[/.\-]\s*\d{1,2}|[Qq][1-4](?![A-Za-z0-9_])|\s*[一二三四]\s*季度)"
    r"|(?<![A-Za-z0-9_])(?:19|20)\d{6}(?![A-Za-z0-9_])"
    r"|(?<![A-Za-z0-9_])[Qq][1-4]\s*(?:19|20)\d{2}(?![A-Za-z0-9_])"
)

# PRIOR is free-format strategy direction and process memory published by Meta.
# This is a resource bound, not a schema.
PRIOR_MAX_CHARS = 16_000


def visible_window_dates(manifest: Mapping[str, object]) -> set[str]:
    """Years and YYYYMMDD period bounds of the meta-learning visible fold, read
    from the manifest so the leak check targets the real window whatever year it
    is (e.g. ``{"2020", "2021", "20200101", "20210930", ...}``)."""
    fold = manifest.get("meta_learning_visible_fold") or {}
    if not isinstance(fold, Mapping):
        fold = {}
    blob = " ".join(
        str(value)
        for value in (
            fold.get("input_window"),
            fold.get("validation_period"),
            fold.get("valid_decision_time"),
            manifest.get("valid_decision_time"),
        )
        if value
    )
    return set(re.findall(r"(?:19|20)\d{6}", blob)) | set(
        re.findall(r"(?:19|20)\d{2}", blob)
    )


def calendar_policy_violation(
    text: str, *, window_dates: set[str] | None = None
) -> str:
    """Why this text contains a forbidden calendar date, or "" when it does not.

    Welded date expressions (_DATE_EXPR) are always rejected. When window_dates
    is given, the visible-window years/bounds are also rejected even if written
    bare. Cadence words (季度/月/周) and plain counts/percentages are unaffected.
    """
    dates = set(window_dates or ())
    bare_window = (
        re.compile(
            r"\b(?:"
            + "|".join(re.escape(token) for token in sorted(dates, key=len, reverse=True))
            + r")\b"
            rf"(?!\s*[{_COUNT_UNITS}])"
        )
        if dates
        else None
    )
    for lineno, line in enumerate(text.splitlines(), start=1):
        if _DATE_EXPR.search(line) or (bare_window and bare_window.search(line)):
            return (
                f"line {lineno} contains a calendar date (non-transferable): "
                f"{line.strip()[:80]!r}"
            )
    return ""


_PRIOR_BOUNDARY_RE = re.compile(
    r"不得|不要|禁止|不可见|不能用于|不得按|不得用|不得读取|不得使用|不得写入|"
    r"永远不可见|排除|不进入|不读取|不挂载"
)
_HELDOUT_MENTION_RE = re.compile(r"held-?out|holdout|持有期外|隐藏区间", re.I)
_TEST_NUMBER_RE = re.compile(
    r"(?:逐\s*fold|每个\s*fold|fold[_\s-]?(?:ref)?\s*\d*).{0,48}(?:test|测试).{0,40}\d|"
    r"(?:test|测试).{0,24}(?:sharpe|收益|回撤|夏普|total_return|超额).{0,16}\d",
    re.I,
)
_TEST_SELECTION_RE = re.compile(
    r"(根据|按照|基于|凭).{0,20}(?:test|测试).{0,20}(选|选择|保留|淘汰|采用)|"
    r"(?:test|测试).{0,20}(更好|更差|更优|更稳).{0,16}(所以|因此|于是|选择|保留)|"
    r"(?:based on|according to).{0,20}test.{0,20}(?:select|choose|retain|reject|adopt)|"
    r"test.{0,20}(?:better|worse|superior|stable).{0,16}(?:so|therefore|select|retain)",
    re.I,
)
_STRICT_TEST_NUMBER_RE = re.compile(
    r"(?:test|测试).{0,48}\d|\d.{0,48}(?:test|测试)", re.I
)
_STRICT_BOUNDARY_LINE_RE = re.compile(
    r"^(?:[-*]\s*)?(?:"
    r"(?:不得|严禁|禁止)(?:读取|使用|依赖|写入|泄露)?\s*"
    r"(?:test\s*/\s*held-?out|held-?out\s*/\s*test)"
    r"(?:\s*(?:数据|结果|指标|原始记录))?|"
    r"(?:do not|must not|never)\s+(?:read|use|rely on|write|leak)\s+"
    r"(?:test\s*/\s*held-?out|held-?out\s*/\s*test)"
    r"(?:\s*(?:data|results?|metrics?))?"
    r")[。.!！]?$",
    re.I,
)


def strict_transferable_content_violation(text: str) -> str:
    """Fail closed on hidden-stage content while allowing a pure boundary rule."""

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or _STRICT_BOUNDARY_LINE_RE.fullmatch(stripped):
            continue
        if _HELDOUT_MENTION_RE.search(stripped):
            return f"line {lineno} leaks Held-out into shared skills"
        if _STRICT_TEST_NUMBER_RE.search(stripped):
            return f"line {lineno} contains a Test figure in shared skills"
        if _TEST_SELECTION_RE.search(stripped):
            return f"line {lineno} uses Test to choose shared skill content"
    return ""


def prior_content_violation(text: str) -> str:
    """Held-out leaks, per-Fold Test figures, or Test-based selection."""
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or _PRIOR_BOUNDARY_RE.search(stripped):
            continue
        if _HELDOUT_MENTION_RE.search(stripped):
            return (
                f"line {lineno} leaks Held-out into PRIOR; "
                "use a boundary sentence such as 不得使用 Test/Held-out"
            )
        if _TEST_NUMBER_RE.search(stripped):
            return (
                f"line {lineno} contains a per-Fold Test figure; "
                "remove Test numbers then call finish_meta again"
            )
        if _TEST_SELECTION_RE.search(stripped):
            return (
                f"line {lineno} uses Test to choose a strategy; "
                "state a transferable process rule instead"
            )
    return ""


def prior_policy_violation(
    prior_path: Path, *, window_dates: set[str] | None = None
) -> str:
    """Why the fixed PRIOR.md cannot be accepted, or "" when it is acceptable."""
    if not prior_path.exists():
        return "write PRIOR.md before finishing"
    text = prior_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return "PRIOR.md must be non-empty before finishing"
    nchars = len(text.strip())
    if nchars > PRIOR_MAX_CHARS:
        return (
            f"PRIOR.md is {nchars} characters; keep it to {PRIOR_MAX_CHARS} "
            "as transferable direction and memory, then call finish_meta again"
        )
    calendar_leak = calendar_policy_violation(
        text, window_dates=set(window_dates or ())
    )
    if calendar_leak:
        return (
            f"PRIOR.md {calendar_leak}; state it qualitatively "
            "with no year or visible-window date, then call finish_meta again"
        )
    content_leak = prior_content_violation(text)
    if content_leak:
        return f"PRIOR.md {content_leak}"
    return ""


__all__ = [
    "PRIOR_MAX_CHARS",
    "calendar_policy_violation",
    "prior_content_violation",
    "prior_policy_violation",
    "strict_transferable_content_violation",
    "visible_window_dates",
]
