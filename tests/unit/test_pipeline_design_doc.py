"""The condition table of docs/pipeline-design.md is ``verdict.CONDITIONS``.

The design document states every condition the freeze gate and the
graduation verdict judge in one table. This keeps that table the code's: the
same rows in the same order, each with its stage, its graduation code, the
rule that turns it on and the reason a failure records. ``docs/`` is local and
not tracked, so the test skips where it is absent.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from autotrade.pipelines.verdict import CONDITIONS

DOC = Path(__file__).resolve().parents[2] / "docs" / "pipeline-design.md"
HEADER = "| 阶段 | 代码 | 条件 | 阈值参数 | 开关 | 原因 |"
NONE = "—"


def _cell(text: str) -> str:
    """A cell's text, without the backticks around a single code span."""

    text = text.strip()
    match = re.fullmatch(r"`([^`]+)`", text)
    return match.group(1) if match else text


def documented_conditions(text: str) -> list[tuple[str, str, str, str]]:
    """``(stage, code, rule, reason)`` of every row of the condition table."""

    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == HEADER]
    assert len(starts) == 1, f"expected one condition table headed {HEADER!r}"
    rows = []
    for line in lines[starts[0] + 2 :]:
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert len(cells) == 6, f"a condition row has six cells: {line!r}"
        rows.append((_cell(cells[0]), cells[1], _cell(cells[4]), _cell(cells[5])))
    return rows


@pytest.mark.skipif(not DOC.is_file(), reason="docs/ is local and not tracked")
def test_the_documented_condition_table_is_the_code_table():
    expected = [
        (
            condition.stage,
            condition.code or NONE,
            condition.requires or NONE,
            condition.reason,
        )
        for condition in CONDITIONS
    ]
    assert documented_conditions(DOC.read_text(encoding="utf-8")) == expected


def test_a_reordered_or_missing_row_is_caught():
    rows = [
        f"| `{stage}` | {code} | x | — | {f'`{rule}`' if rule != NONE else NONE} | `{reason}` |"
        for stage, code, rule, reason in (
            (c.stage, c.code or NONE, c.requires or NONE, c.reason) for c in CONDITIONS
        )
    ]
    table = "\n".join([HEADER, "| --- | --- | --- | --- | --- | --- |", *rows, ""])
    expected = [
        (c.stage, c.code or NONE, c.requires or NONE, c.reason) for c in CONDITIONS
    ]
    assert documented_conditions(table) == expected
    swapped = "\n".join([HEADER, "| --- |", rows[1], rows[0], *rows[2:]])
    assert documented_conditions(swapped) != expected
    assert documented_conditions("\n".join([HEADER, "| --- |", *rows[1:]])) != expected
