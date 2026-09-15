"""The sealed-stage content check for shared skills.

Shared skills can reach other experiments through the graduated memory layer,
so every skill write refuses Held-out mentions, forward-period figures and
forward-period-based selection.
"""

from __future__ import annotations

import re

_HELDOUT_MENTION_RE = re.compile(r"held-?out|holdout|持有期外|隐藏区间", re.I)
# What a leaked forward result actually looks like: a performance word next to
# a number. Naming the forward period is ordinary knowledge (every session is
# told it exists and is sealed), so only a figure or a selection built on it is
# refused.
_PERFORMANCE_WORD = (
    r"sharpe|夏普|calmar|sortino|索提诺|total_return|收益|回报|回撤|drawdown|超额|"
    r"alpha|年化|胜率|净值|盈亏|波动|换手|turnover|信息比|表现|"
    r"excess|annualized|volatility|profit|performance|"
    r"(?<![A-Za-z_])returns?(?![A-Za-z_])|(?<![A-Za-z])(?:ic|ir|pnl|cagr)(?![A-Za-z])"
)
# A signed percentage is a reported figure on its own ("前推期 +8%"); a bare
# number still needs a performance word beside it.
_SIGNED_PERCENT = r"[-+±]\s*\d+(?:\.\d+)?\s*%"
_PERFORMANCE_FIGURE = (
    rf"(?:(?:{_PERFORMANCE_WORD}).{{0,12}}[-+±]?\d"
    rf"|\d.{{0,12}}(?:{_PERFORMANCE_WORD})"
    rf"|{_SIGNED_PERCENT})"
)
# The stage reference must name the stage, not the bare word: "forward return"
# is the research label for next-open returns, "walk-forward" is an in-sample
# method, and "前推 20 日" counts days backwards. Matching those rejected
# ordinary research prose that never named the sealed period.
_FORWARD_STAGE = (
    r"(?:前推(?:期|段|检验|回放|阶段)"
    r"|(?<![A-Za-z_-])forward[\s_-]*(?:period|test|testing|replay|stage)s?(?![A-Za-z_]))"
)
_FORWARD_SELECTION_RE = re.compile(
    rf"(根据|按照|基于|凭).{{0,20}}{_FORWARD_STAGE}.{{0,20}}(选|选择|保留|淘汰|采用)|"
    rf"{_FORWARD_STAGE}.{{0,20}}(更好|更差|更优|更稳).{{0,16}}(所以|因此|于是|选择|保留)|"
    rf"(?:based on|according to).{{0,20}}{_FORWARD_STAGE}.{{0,20}}(?:select|choose|retain|reject|adopt)|"
    rf"{_FORWARD_STAGE}.{{0,20}}(?:better|worse|superior|stable).{{0,16}}(?:so|therefore|select|retain)",
    re.I,
)
# A shared skill leaks whether the figure precedes or follows the forward
# reference, but it still has to be a figure. Held-out mentions are rejected
# outright by _HELDOUT_MENTION_RE above.
_FORWARD_FIGURE_RE = re.compile(
    rf"{_FORWARD_STAGE}.{{0,24}}{_PERFORMANCE_FIGURE}|"
    rf"{_PERFORMANCE_FIGURE}.{{0,24}}{_FORWARD_STAGE}",
    re.I,
)
_SEALED_PAIR = (
    rf"(?:{_FORWARD_STAGE}\s*(?:/|与|和|and)\s*held-?out"
    rf"|held-?out\s*(?:/|与|和|and)\s*{_FORWARD_STAGE})"
)
_STRICT_BOUNDARY_LINE_RE = re.compile(
    rf"^(?:[-*]\s*)?(?:"
    rf"(?:不得|严禁|禁止)(?:读取|使用|依赖|写入|泄露)?\s*{_SEALED_PAIR}"
    rf"(?:\s*(?:数据|结果|指标|原始记录))?|"
    rf"(?:do not|must not|never)\s+(?:read|use|rely on|write|leak)\s+{_SEALED_PAIR}"
    rf"(?:\s*(?:data|results?|metrics?))?"
    rf")[。.!！]?$",
    re.I,
)


def strict_transferable_content_violation(text: str) -> str:
    """Fail closed on sealed-stage content while allowing a pure boundary rule."""

    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or _STRICT_BOUNDARY_LINE_RE.fullmatch(stripped):
            continue
        if _HELDOUT_MENTION_RE.search(stripped):
            return f"line {lineno} leaks Held-out into shared skills"
        figure = _FORWARD_FIGURE_RE.search(stripped)
        if figure:
            # Name the matched span: the check is a same-line pattern, not a
            # reading of the sentence, so the fix is only obvious once the
            # Agent can see which words tripped it.
            return (
                f"line {lineno} contains a forward-period figure in shared skills: "
                f"{figure.group(0)[:60]!r}"
            )
        if _FORWARD_SELECTION_RE.search(stripped):
            return f"line {lineno} uses the forward period to choose shared skill content"
    return ""


__all__ = ["strict_transferable_content_violation"]
