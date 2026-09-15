"""Contradict a mounted operating-memory entry, so the researcher can fix it.

Mounted skills are read-only advice carried in from the curated library or a
graduated experiment. A session that measures something the entry denies has no
way to say so: it cannot edit the mount, and the finding dies in the trace.
``skill_feedback`` is that return path, and it is deliberately negative only —
the channel it replaces also let a session endorse an entry, and of its 651
calls 522 endorsed one, 3 said "outdated" and 0 said "wrong", which is not a
signal. Silence means the entry was used without contradiction; there is no
verdict to file for that.

So a call has to cost something: it names one mounted entry, claims it is
``outdated`` or ``wrong``, and carries evidence long enough to say what was
tried and what the data showed. One report per entry per run — a second one
about the same entry would tell the researcher nothing new.

Host-side telemetry only, exactly like ``report_issue``: the report changes no
artifact, budget, or result, and is never mounted or projected back into any
session's inputs. Nothing about the mounted entry changes because of it; the
researcher edits the curated library from the console and the change reaches
experiments created afterwards.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.runtime import (
    RunManifest,
    append_versioned_jsonl,
    new_id,
)

from .base import ToolError, ToolResult, ToolSpec
from .feedback_log import (
    append_resolution,
    read_feedback_log,
    session_link_fields,
)

# Stamped on every appended report; bump when the record shape changes.
SKILL_FEEDBACK_SCHEMA_VERSION = 1
# Negative only: an entry the session's own measurements contradict outright
# (``wrong``), or one whose advice no longer matches the current contract,
# data or tooling (``outdated``).
SKILL_CLAIMS = ("outdated", "wrong")
# The researcher's verdict. ``skill_updated`` names the library edit, the other
# two close a report that needed none; the note carries the reason.
SKILL_FEEDBACK_OUTCOMES = ("skill_updated", "not_a_defect", "accepted_limitation")
# Names this log's records in the shared reader's errors.
SKILL_FEEDBACK_LABEL = "skill feedback"
# The floor is the point of the redesign: a verdict without a reproduction is
# the noise this channel exists to keep out. The ceiling keeps one report to a
# paragraph — the reproduction belongs in the skill the session writes.
MIN_SKILL_EVIDENCE_CHARS = 80
MAX_SKILL_EVIDENCE_CHARS = 1_000
# Bounds the refusal hint on a checkout mounting many graduated experiments.
_MAX_HINT_ENTRIES = 20
SKILL_FEEDBACK_NAME = "skill_feedback.jsonl"


def skill_feedback_path(experiment_dir: str | Path) -> Path:
    """Single source for the writer and the console reader."""

    return Path(experiment_dir) / "ledgers" / SKILL_FEEDBACK_NAME


def read_skill_feedback(path: str | Path) -> list[dict[str, object]]:
    """All reports in append order, each carrying its resolution when it has one."""

    return read_feedback_log(
        path,
        schema_version=SKILL_FEEDBACK_SCHEMA_VERSION,
        label=SKILL_FEEDBACK_LABEL,
        outcomes=SKILL_FEEDBACK_OUTCOMES,
    )


def append_skill_feedback_resolution(
    path: str | Path, *, report_id: str, outcome: str, note: str
) -> dict[str, object]:
    """Record how one filed report was answered; refuse anything else."""

    return append_resolution(
        path,
        schema_version=SKILL_FEEDBACK_SCHEMA_VERSION,
        label=SKILL_FEEDBACK_LABEL,
        outcomes=SKILL_FEEDBACK_OUTCOMES,
        report_id=report_id,
        outcome=outcome,
        note=note,
    )


class SkillFeedbackTool:
    spec = ToolSpec(
        "skill_feedback",
        "Tell the human operators that one mounted operating-memory entry is "
        "contradicted by what this session measured. skill is the mounted "
        "reference '<source>/<name>' exactly as inputs/skills_index.json lists "
        "it (source 'curated' or the graduated experiment's id). claim is "
        "outdated (the advice no longer matches the current data contract, "
        "tooling or figures) or wrong (this session's own measurements "
        "contradict it). evidence says what was tried and what the data showed "
        f"in {MIN_SKILL_EVIDENCE_CHARS}-{MAX_SKILL_EVIDENCE_CHARS} characters: "
        "the reading, the node id or path, expected vs observed. Negative only: "
        "there is no positive verdict and none is wanted — an entry used "
        "without contradiction needs no call. One report per entry per "
        "session. Host-side telemetry: it changes no artifact, budget, or "
        "result, changes nothing about the mounted entry, and is never read "
        "back by any session.",
        {
            "type": "object",
            "properties": {
                "skill": {"type": "string", "minLength": 1},
                "claim": {"type": "string", "enum": list(SKILL_CLAIMS)},
                "evidence": {
                    "type": "string",
                    "minLength": MIN_SKILL_EVIDENCE_CHARS,
                    "maxLength": MAX_SKILL_EVIDENCE_CHARS,
                },
            },
            "required": ["skill", "claim", "evidence"],
            "additionalProperties": False,
        },
        mutating=False,
        example={
            "skill": "curated/grid-plateau-selection",
            "claim": "outdated",
            "evidence": (
                "条目把单参数网格限死在四点；本会话把同一机制的 5 点网格一次提交 "
                "batch_validate（node_7f3a），replay-year 与单日耗时都在限额内。"
                "四点上限来自已废弃的 daily_backtest，不是当前工具的约束。"
            ),
        },
    )

    def __init__(
        self,
        path: str | Path,
        manifest: RunManifest,
        mounted: Mapping[str, str],
    ) -> None:
        self.path = Path(path)
        self.manifest = manifest
        # ``<source>/<name>`` -> origin, built from what this run actually
        # mounted: the tool can only speak about entries the session can read.
        self.mounted = dict(mounted)
        # The spec is non-mutating, so calls in one assistant turn may dispatch
        # in parallel. The set is per tool instance, which is per run: a retried
        # session is a new run and files its own reports.
        self._lock = threading.Lock()
        self._filed: set[str] = set()

    def _unknown_skill(self, skill: str) -> ValueError:
        if not self.mounted:
            return ValueError(
                f"unknown mounted skill: {skill!r}; this session mounts no "
                "operating memory"
            )
        names = sorted(self.mounted)
        listed = ", ".join(names[:_MAX_HINT_ENTRIES])
        if len(names) > _MAX_HINT_ENTRIES:
            listed += f", … ({len(names)} mounted)"
        return ValueError(f"unknown mounted skill: {skill!r}; mounted: {listed}")

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        try:
            skill = str(arguments["skill"]).strip()
            origin = self.mounted.get(skill)
            if origin is None:
                raise self._unknown_skill(skill)
            claim = str(arguments["claim"]).strip()
            if claim not in SKILL_CLAIMS:
                raise ValueError("claim must be one of " + ", ".join(SKILL_CLAIMS))
            evidence = str(arguments["evidence"]).strip()
            if len(evidence) < MIN_SKILL_EVIDENCE_CHARS:
                raise ValueError(
                    "evidence must say what was tried and what the data showed "
                    f"in at least {MIN_SKILL_EVIDENCE_CHARS} characters"
                )
            if len(evidence) > MAX_SKILL_EVIDENCE_CHARS:
                raise ValueError(
                    f"evidence exceeds {MAX_SKILL_EVIDENCE_CHARS} characters"
                )
            with self._lock:
                if skill in self._filed:
                    raise ValueError(f"this session already filed feedback on {skill}")
                record: dict[str, object] = {
                    "report_id": new_id("skillfb"),
                    **session_link_fields(self.manifest),
                    "skill": skill,
                    "origin": origin,
                    "claim": claim,
                    "evidence": evidence,
                }
                recorded = append_versioned_jsonl(
                    self.path, record, schema_version=SKILL_FEEDBACK_SCHEMA_VERSION
                )
                self._filed.add(skill)
        except ValueError as exc:
            raise ToolError(str(exc), error_type="skill_feedback_policy") from exc
        return ToolResult(
            True,
            value={
                "report_id": record["report_id"],
                "skill": skill,
                "claim": claim,
                "recorded_at": recorded["recorded_at"],
            },
        )


__all__ = [
    "MAX_SKILL_EVIDENCE_CHARS",
    "MIN_SKILL_EVIDENCE_CHARS",
    "SKILL_CLAIMS",
    "SKILL_FEEDBACK_LABEL",
    "SKILL_FEEDBACK_NAME",
    "SKILL_FEEDBACK_OUTCOMES",
    "SKILL_FEEDBACK_SCHEMA_VERSION",
    "SkillFeedbackTool",
    "append_skill_feedback_resolution",
    "read_skill_feedback",
    "skill_feedback_path",
]
