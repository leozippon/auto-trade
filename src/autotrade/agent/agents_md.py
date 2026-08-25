"""Load the three AGENTS.md sections Fold/Meta system prompts must include.

The repository-root ``AGENTS.md`` is the only source for those section bodies.
This module extracts them at runtime; it does not copy the prose into Python.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

REQUIRED_AGENTS_MD_SECTIONS = (
    "Rules for Multi-Agent Cooperation",
    "Development Principles",
    "Operational Guardrails",
)


class AgentsMdError(ValueError):
    """The root AGENTS.md file is missing or does not contain a required section."""


@dataclass(frozen=True)
class AgentsMdSections:
    text: str
    sha256: str
    path: Path

    @property
    def version(self) -> str:
        return self.sha256[:12]


def default_agents_md_path() -> Path:
    return Path(__file__).resolve().parents[3] / "AGENTS.md"


def load_required_agents_md_sections(
    path: str | Path | None = None,
) -> AgentsMdSections:
    """Return the three required sections joined in documented order.

    Missing file or missing any named ``##`` section is an explicit failure.
    """

    source = Path(path) if path is not None else default_agents_md_path()
    if not source.is_file():
        raise AgentsMdError(f"AGENTS.md is missing: {source}")
    try:
        body = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise AgentsMdError(f"AGENTS.md cannot be read: {source}") from exc
    extracted = [_extract_section(body, title, source) for title in REQUIRED_AGENTS_MD_SECTIONS]
    text = "\n\n".join(extracted).strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return AgentsMdSections(text=text, sha256=digest, path=source.resolve())


def _extract_section(body: str, title: str, source: Path) -> str:
    heading = f"## {title}"
    lines = body.splitlines()
    start = next((index for index, line in enumerate(lines) if line.strip() == heading), -1)
    if start < 0:
        raise AgentsMdError(f"AGENTS.md is missing required section {title!r}: {source}")
    end = start + 1
    while end < len(lines):
        current = lines[end]
        if current.startswith("## ") and not current.startswith("### "):
            break
        end += 1
    section = "\n".join(lines[start:end]).strip()
    if not section[len(heading) :].strip():
        raise AgentsMdError(f"AGENTS.md section {title!r} is empty: {source}")
    return section
