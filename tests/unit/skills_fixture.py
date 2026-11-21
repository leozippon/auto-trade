"""A skill-feedback record filed the way the Agent's skill_feedback tool files one."""

from __future__ import annotations

from pathlib import Path

from autotrade.environment.runtime import append_versioned_jsonl
from autotrade.environment.tools.skill_feedback import (
    SKILL_FEEDBACK_SCHEMA_VERSION,
    skill_feedback_path,
)

SESSION_KEY = "research"
CURATED = "curated/offline-readings"
# A real report: what was tried, what the reading was, what it contradicts.
EVIDENCE = (
    "条目说离线引擎在已验证节点上逐笔对上 executions 就可以读它的收益；本会话照做"
    "（node_7f3a），名单与订单逐笔一致，年化收益却差 3 个百分点，差在离线引擎没扣最低佣金。"
)


def file_skill_feedback(directory: Path, **overrides: object) -> dict[str, object]:
    record = {
        "report_id": "skillfb_abc",
        "experiment_id": directory.name,
        "session_key": SESSION_KEY,
        "run_id": "run_raw",
        "kind": "research",
        "skill": CURATED,
        "origin": "curated",
        "claim": "outdated",
        "evidence": EVIDENCE,
        **overrides,
    }
    return append_versioned_jsonl(
        skill_feedback_path(directory),
        record,
        schema_version=SKILL_FEEDBACK_SCHEMA_VERSION,
    )
