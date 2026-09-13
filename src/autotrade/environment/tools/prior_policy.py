"""The PRIOR length bound and the hidden-stage content check for shared skills.

PRIOR.md is a research session's handoff to the next session and gets only a
length bound. Shared skills can reach other experiments through the graduated
memory layer, so every skill write refuses Held-out mentions, Test figures and
Test-based selection.
"""

from __future__ import annotations

import re

# PRIOR is free-format handoff text. This is a resource bound, not a schema.
PRIOR_MAX_CHARS = 16_000

_HELDOUT_MENTION_RE = re.compile(r"held-?out|holdout|持有期外|隐藏区间", re.I)
# What a leaked Test result actually looks like: a performance word next to a
# number. The Test-figure check below requires one, because "test"/"测试" near a
# bare digit is ordinary prose ("边界单测覆盖 3 个用例", "H+1", "09:30") far more
# often than it is a hidden-stage figure, and rejecting that prose taught
# sessions to avoid the word rather than the leak.
_PERFORMANCE_WORD = (
    r"sharpe|夏普|calmar|sortino|索提诺|total_return|收益|回报|回撤|drawdown|超额|"
    r"alpha|年化|胜率|净值|盈亏|波动|换手|turnover|信息比|表现|"
    r"excess|annualized|volatility|profit|performance|"
    r"(?<![A-Za-z_])returns?(?![A-Za-z_])|(?<![A-Za-z])(?:ic|ir|pnl|cagr)(?![A-Za-z])"
)
# A signed percentage is a reported figure on its own ("Test 段 +8%"); a bare
# number still needs a performance word beside it.
_SIGNED_PERCENT = r"[-+±]\s*\d+(?:\.\d+)?\s*%"
_PERFORMANCE_FIGURE = (
    rf"(?:(?:{_PERFORMANCE_WORD}).{{0,12}}[-+±]?\d"
    rf"|\d.{{0,12}}(?:{_PERFORMANCE_WORD})"
    rf"|{_SIGNED_PERCENT})"
)
# The stage reference itself must be a whole word: `test` is a substring of the
# domain's most common loanwords (`backtest`, `latest`, `contest`), and matching
# inside them rejected ordinary prose that never named the hidden stage.
_TEST_WORD = r"(?:(?<![A-Za-z_])tests?(?![A-Za-z_])|测试)"
_TEST_SELECTION_RE = re.compile(
    rf"(根据|按照|基于|凭).{{0,20}}{_TEST_WORD}.{{0,20}}(选|选择|保留|淘汰|采用)|"
    rf"{_TEST_WORD}.{{0,20}}(更好|更差|更优|更稳).{{0,16}}(所以|因此|于是|选择|保留)|"
    rf"(?:based on|according to).{{0,20}}{_TEST_WORD}.{{0,20}}(?:select|choose|retain|reject|adopt)|"
    rf"{_TEST_WORD}.{{0,20}}(?:better|worse|superior|stable).{{0,16}}(?:so|therefore|select|retain)",
    re.I,
)
# A shared skill leaks whether the figure precedes or follows the Test
# reference, but it still has to be a figure. Held-out mentions are rejected
# outright by _HELDOUT_MENTION_RE above.
_STRICT_TEST_NUMBER_RE = re.compile(
    rf"{_TEST_WORD}.{{0,24}}{_PERFORMANCE_FIGURE}|"
    rf"{_PERFORMANCE_FIGURE}.{{0,24}}{_TEST_WORD}",
    re.I,
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
        figure = _STRICT_TEST_NUMBER_RE.search(stripped)
        if figure:
            # Name the matched span: the check is a same-line pattern, not a
            # reading of the sentence, so the fix is only obvious once the
            # Agent can see which words tripped it.
            return (
                f"line {lineno} contains a Test figure in shared skills: "
                f"{figure.group(0)[:60]!r}"
            )
        if _TEST_SELECTION_RE.search(stripped):
            return f"line {lineno} uses Test to choose shared skill content"
    return ""


__all__ = [
    "PRIOR_MAX_CHARS",
    "strict_transferable_content_violation",
]
