"""Load the Chinese AGENTS.md sections Fold/Meta system prompts must include."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REQUIRED_AGENTS_MD_SECTIONS = ("多智能体协作", "开发原则", "操作护栏")


class AgentsMdError(ValueError):
    """The root AGENTS.md file is missing or does not contain a required section."""


@dataclass(frozen=True)
class AgentsMdSections:
    text: str


def default_agents_md_path() -> Path:
    return Path(__file__).resolve().parents[3] / "AGENTS.md"


def load_required_agents_md_sections(
    path: str | Path | None = None,
) -> AgentsMdSections:
    """Return the Fold/Meta guideline sections, headings promoted for system prompts.

    Missing file or missing any named heading is an explicit failure.
    """

    source = Path(path) if path is not None else default_agents_md_path()
    if not source.is_file():
        raise AgentsMdError(f"AGENTS.md is missing: {source}")
    try:
        body = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise AgentsMdError(f"AGENTS.md cannot be read: {source}") from exc
    extracted = [
        _promote_headings(_extract_section(body, title, source))
        for title in REQUIRED_AGENTS_MD_SECTIONS
    ]
    text = "\n\n".join(extracted).strip()
    return AgentsMdSections(text=text)


def _heading_level(line: str) -> int | None:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    hashes = len(stripped) - len(stripped.lstrip("#"))
    if hashes < 1 or hashes > 6:
        return None
    if len(stripped) <= hashes or stripped[hashes] != " ":
        return None
    return hashes


def _extract_section(body: str, title: str, source: Path) -> str:
    lines = body.splitlines()
    start = -1
    level = 0
    for index, line in enumerate(lines):
        level_here = _heading_level(line)
        if level_here is None:
            continue
        heading_title = line.strip().lstrip("#").strip()
        if heading_title == title and level_here in {2, 3}:
            start = index
            level = level_here
            break
    if start < 0:
        raise AgentsMdError(f"AGENTS.md is missing required section {title!r}: {source}")
    end = start + 1
    while end < len(lines):
        next_level = _heading_level(lines[end])
        if next_level is not None and next_level <= level:
            break
        end += 1
    heading = lines[start].strip()
    section = "\n".join(lines[start:end]).strip()
    if not section[len(heading) :].strip():
        raise AgentsMdError(f"AGENTS.md section {title!r} is empty: {source}")
    return section


def _promote_headings(section: str) -> str:
    """Turn AGENTS.md ``##``/``###`` into system-prompt ``#``/``##``."""

    lines: list[str] = []
    for line in section.splitlines():
        level = _heading_level(line)
        if level is not None and level > 1:
            title = line.strip().lstrip("#").strip()
            lines.append("#" * (level - 1) + " " + title)
        else:
            lines.append(line)
    return "\n".join(lines)
