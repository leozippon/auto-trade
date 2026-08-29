"""Experiment-level shared skills snapshots, workspace tools, and ledger reachability.

``skills/`` is writable session knowledge, not a formal strategy artifact.  The
ledger is the only current-pointer mechanism: immutable generations without a
remaining successful Fold/Meta row are deliberately unreachable orphans.

``memory/`` is the second source of session knowledge: entries the researcher
curated into ``configs/operating_memory/`` and mounted read-only for this run.
Both reach the Agent through one ``inputs/skills_index.json``, tagged by origin,
and only the writable ``skills/`` tree is ever published as a generation.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from autotrade.environment.runtime import chmod_tree, write_json_atomic
from autotrade.environment.tools.base import ToolError, ToolResult, ToolSpec
from autotrade.environment.tools.prior_policy import (
    strict_transferable_content_violation,
)
from autotrade.environment.tools.workspace import SafeWorkspace

from .ledger import ExperimentLedger, experiment_verdict

SKILLS_DIRNAME = "skills"
OPERATING_MEMORY_DIRNAME = "memory"
# Repository-relative library of human-curated cross-experiment entries, in
# the same format ``write_skill`` produces so a promoted skill copies verbatim.
OPERATING_MEMORY_LIBRARY = "configs/operating_memory"
# Reserved mount name of the curated tier; every other mounted source directory
# is named after the graduated experiment its skills came from.
CURATED_MEMORY_SOURCE = "curated"
OPERATING_MEMORY_MODES = ("none", "curated", "curated+graduated")
DEFAULT_OPERATING_MEMORY = "curated+graduated"
SKILL_FILENAME = "SKILL.md"
SKILLS_INDEX_PATH = "inputs/skills_index.json"
MAX_SKILL_NAME_CHARS = 64
MAX_SKILL_CHARS = 16_000
MAX_SKILL_FILE_BYTES = 64 * 1024
MAX_SKILLS_BYTES = 1024 * 1024
MAX_SKILLS_FILES = 128
MAX_SKILLS = 32
ALLOWED_SKILL_SUFFIXES = frozenset(
    {".md", ".py", ".sql", ".sh", ".json", ".toml", ".yaml", ".yml", ".txt"}
)
_SKILL_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class SkillsStats:
    count: int = 0
    files: int = 0
    bytes: int = 0

    def ledger_fields(self) -> dict[str, int]:
        return {
            "skills_count": self.count,
            "skills_files": self.files,
            "skills_bytes": self.bytes,
        }


@dataclass(frozen=True)
class SkillsSnapshot:
    skills_ref: str = ""
    generation_id: str = ""
    stats: SkillsStats = SkillsStats()
    root: Path | None = None


@dataclass(frozen=True)
class SkillsPublication:
    skills_ref: str
    generation_id: str
    stats: SkillsStats
    published: bool


def validate_skill_name(name: str) -> str:
    value = str(name)
    if len(value) > MAX_SKILL_NAME_CHARS or not _SKILL_NAME_RE.fullmatch(value):
        raise ValueError(
            "skill name must be lowercase kebab-case and at most "
            f"{MAX_SKILL_NAME_CHARS} characters"
        )
    violation = strict_transferable_content_violation(value)
    if violation:
        raise ValueError(f"skill name violates the transferable boundary: {violation}")
    return value


def validate_skill_path(path: str) -> PurePosixPath:
    value = str(path)
    pure = PurePosixPath(value)
    if (
        not value
        or pure.is_absolute()
        or any(part in {"", ".", ".."} or part.startswith(".") for part in pure.parts)
    ):
        raise ValueError("skill path must be a non-hidden relative path without '..'")
    if pure == PurePosixPath(SKILL_FILENAME):
        return pure
    if len(pure.parts) < 2 or pure.parts[0] not in {"scripts", "references"}:
        raise ValueError("skill path must be SKILL.md or live under scripts/ or references/")
    if pure.suffix.lower() not in ALLOWED_SKILL_SUFFIXES:
        raise ValueError(f"skill file extension is not allowed: {pure.suffix or '[none]'}")
    violation = strict_transferable_content_violation(value)
    if violation:
        raise ValueError(f"skill path violates the transferable boundary: {violation}")
    return pure


def _decode_skill_file(path: Path, payload: bytes) -> str:
    if len(payload) > MAX_SKILL_FILE_BYTES:
        raise ValueError(
            f"skill file exceeds {MAX_SKILL_FILE_BYTES} bytes: {path.as_posix()}"
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"skill file is not UTF-8: {path.as_posix()}") from exc
    violation = strict_transferable_content_violation(text)
    if violation:
        raise ValueError(f"skill content violates the transferable boundary: {violation}")
    return text


def validate_skills_tree(
    root: str | Path, *, require_writable: bool = True
) -> SkillsStats:
    """Validate one complete top-level ``skills/`` working tree."""

    tree = Path(root)
    if not tree.exists():
        raise ValueError("skills root does not exist")
    if tree.is_symlink() or not tree.is_dir():
        raise ValueError("skills root must be a regular directory, not a symlink")
    if require_writable and not stat.S_IMODE(tree.stat().st_mode) & 0o222:
        raise ValueError("skills working tree must be writable")

    item_count = 0
    file_count = 0
    total_bytes = 0
    for item in sorted(tree.iterdir(), key=lambda path: path.name):
        if item.name.startswith("."):
            raise ValueError(f"hidden skills path is forbidden: {item.name}")
        validate_skill_name(item.name)
        if item.is_symlink() or not item.is_dir():
            raise ValueError(f"skill item must be a regular directory: {item.name}")
        if require_writable and not stat.S_IMODE(item.stat().st_mode) & 0o222:
            raise ValueError(f"skill item must be writable: {item.name}")
        item_count += 1
        if item_count > MAX_SKILLS:
            raise ValueError(f"skills tree exceeds {MAX_SKILLS} items")
        skill_md = item / SKILL_FILENAME
        if skill_md.is_symlink() or not skill_md.is_file():
            raise ValueError(f"skill item is missing {SKILL_FILENAME}: {item.name}")

        for path in sorted(item.rglob("*"), key=lambda candidate: candidate.as_posix()):
            relative = path.relative_to(item)
            if any(part.startswith(".") for part in relative.parts):
                raise ValueError(f"hidden skills path is forbidden: {item.name}/{relative}")
            if path.is_symlink():
                raise ValueError(f"skill symlink is forbidden: {item.name}/{relative}")
            if path.is_dir():
                if relative.parts[0] not in {"scripts", "references"}:
                    raise ValueError(
                        f"skill directory must live under scripts/ or references/: "
                        f"{item.name}/{relative}"
                    )
                if require_writable and not stat.S_IMODE(path.stat().st_mode) & 0o222:
                    raise ValueError(f"skill directory must be writable: {item.name}/{relative}")
                continue
            if not path.is_file():
                raise ValueError(f"skill path is not a regular file: {item.name}/{relative}")
            validated_relative = validate_skill_path(relative.as_posix())
            if validated_relative.suffix.lower() not in ALLOWED_SKILL_SUFFIXES:
                raise ValueError(f"skill file extension is not allowed: {relative}")
            payload = path.read_bytes()
            text = _decode_skill_file(path, payload)
            if relative == PurePosixPath(SKILL_FILENAME) and len(text) > MAX_SKILL_CHARS:
                raise ValueError(
                    f"{item.name}/{SKILL_FILENAME} exceeds {MAX_SKILL_CHARS} characters"
                )
            file_count += 1
            total_bytes += len(payload)
            if file_count > MAX_SKILLS_FILES:
                raise ValueError(f"skills tree exceeds {MAX_SKILLS_FILES} files")
            if total_bytes > MAX_SKILLS_BYTES:
                raise ValueError(f"skills tree exceeds {MAX_SKILLS_BYTES} bytes")
    return SkillsStats(item_count, file_count, total_bytes)


def _assert_skills_tree_read_only(root: str | Path) -> None:
    tree = Path(root)
    for path in (tree.parent, tree, *tree.rglob("*")):
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            relative = "generation" if path == tree.parent else (
                "." if path == tree else path.relative_to(tree).as_posix()
            )
            raise ValueError(f"published skills path is writable: {relative}")


def _skill_title_summary(text: str) -> tuple[str, str]:
    title = ""
    lines = text.splitlines()
    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            title = match.group(1).strip()
            break
    summary_lines: list[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if started:
                break
            continue
        if _HEADING_RE.match(line):
            continue
        started = True
        summary_lines.append(stripped)
    return title, " ".join(summary_lines)[:500]


def _index_entries(tree: Path, *, directory: str, origin: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for item in sorted(tree.iterdir(), key=lambda path: path.name):
        files: list[dict[str, object]] = []
        item_bytes = 0
        for path in sorted(
            (candidate for candidate in item.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.as_posix(),
        ):
            size = path.stat().st_size
            item_bytes += size
            files.append(
                {
                    "path": f"{directory}/{item.name}/{path.relative_to(item).as_posix()}",
                    "bytes": size,
                }
            )
        body = (item / SKILL_FILENAME).read_text(encoding="utf-8")
        title, summary = _skill_title_summary(body)
        entries.append(
            {
                "name": item.name,
                "title": title,
                "summary": summary,
                "path": f"{directory}/{item.name}/{SKILL_FILENAME}",
                "files": files,
                "bytes": item_bytes,
                "origin": origin,
            }
        )
    return entries


def build_skills_index(root: str | Path) -> dict[str, object]:
    """Index the writable session skills plus the operating memory mounted beside them.

    ``count``/``files``/``bytes`` describe the writable tree alone, because they
    are the published generation's ledger fields; the read-only entries are a
    separate list, each carrying its ``origin`` (``curated`` or ``graduated``)
    and its ``source``, so a session can tell what it may rewrite from what it
    may not and can weigh a graduated experiment's knowledge accordingly.
    """

    tree = Path(root)
    stats = validate_skills_tree(tree, require_writable=False)
    memory_root = tree.parent / OPERATING_MEMORY_DIRNAME
    memory: list[dict[str, object]] = []
    if memory_root.is_dir():
        for source in sorted(memory_root.iterdir(), key=lambda path: path.name):
            validate_skills_tree(source, require_writable=False)
            origin = (
                "curated" if source.name == CURATED_MEMORY_SOURCE else "graduated"
            )
            for entry in _index_entries(
                source,
                directory=f"{OPERATING_MEMORY_DIRNAME}/{source.name}",
                origin=origin,
            ):
                entry["source"] = source.name
                memory.append(entry)
    return {
        "skills": _index_entries(tree, directory=SKILLS_DIRNAME, origin="session"),
        "operating_memory": memory,
        "count": stats.count,
        "files": stats.files,
        "bytes": stats.bytes,
    }


def write_skills_index(root: str | Path, index_path: str | Path) -> SkillsStats:
    payload = build_skills_index(root)
    write_json_atomic(Path(index_path), payload)
    return validate_skills_tree(root, require_writable=False)


def _copy_skills_tree(source: Path | None, destination: Path) -> SkillsStats:
    if destination.exists():
        raise FileExistsError(f"skills workspace already exists: {destination}")
    if source is None:
        destination.mkdir(parents=True)
    else:
        validate_skills_tree(source, require_writable=False)
        shutil.copytree(source, destination, copy_function=shutil.copyfile)
    chmod_tree(destination, file_mode=0o644, dir_mode=0o755)
    return validate_skills_tree(destination, require_writable=True)


def install_workspace_skills(
    source: str | Path | None,
    workspace: str | Path,
    *,
    index_path: str | Path | None = None,
) -> SkillsStats:
    """Copy the current immutable snapshot into a writable session workspace."""

    workspace_path = Path(workspace)
    source_path = Path(source).resolve(strict=True) if source else None
    destination = workspace_path / SKILLS_DIRNAME
    _copy_skills_tree(source_path, destination)
    return write_skills_index(
        destination,
        Path(index_path) if index_path is not None else workspace_path / SKILLS_INDEX_PATH,
    )


@dataclass(frozen=True)
class MemorySource:
    """One mounted knowledge source: where it came from and what it carries."""

    source: str
    origin: str
    root: Path
    entries: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "source": self.source,
            "origin": self.origin,
            "entries": list(self.entries),
        }


def operating_memory_entries(library: str | Path) -> tuple[str, ...]:
    """Entry names of one memory tree, validated in the shared skill format."""

    root = Path(library)
    if not root.is_dir():
        raise FileNotFoundError(f"operating memory library does not exist: {root}")
    validate_skills_tree(root, require_writable=False)
    return tuple(sorted(item.name for item in root.iterdir()))


def resolve_operating_memory(value: object) -> str:
    """Validate the create-time mode: ``none``/``curated``/``curated+graduated``."""

    text = str(value).strip() if value not in (None, "") else DEFAULT_OPERATING_MEMORY
    if text not in OPERATING_MEMORY_MODES:
        raise ValueError(
            "operating_memory must be one of " + ", ".join(OPERATING_MEMORY_MODES)
        )
    return text


def experiment_graduated(records: Sequence[Mapping[str, object]]) -> bool:
    """Whether the Pipeline's held-out verdict adopted this experiment.

    ``ledger.experiment_verdict`` is the one aggregator: every latest held-out
    period must be ``graduated``, so a single failed period keeps the whole
    experiment out. Only durable successful held-out rows reach it, so a
    running, failed or integrity-flagged experiment never qualifies. Read
    non-strict on purpose — these are other experiments' ledgers, and one that
    predates the verdict block must contribute nothing rather than break the
    session that is mounting memory.
    """

    verdict = experiment_verdict([dict(record) for record in records], strict=False)
    return verdict is not None and verdict.get("status") == "graduated"


def curated_memory_source(repo_root: str | Path | None) -> MemorySource | None:
    """The human-curated library, or ``None`` when this checkout has none."""

    if repo_root is None:
        return None
    library = Path(repo_root) / OPERATING_MEMORY_LIBRARY
    if not library.is_dir():
        return None
    entries = operating_memory_entries(library)
    if not entries:
        return None
    return MemorySource(CURATED_MEMORY_SOURCE, "curated", library, entries)


def graduated_memory_sources(
    experiments_root: str | Path, *, exclude: str = ""
) -> tuple[MemorySource, ...]:
    """Skills of every graduated experiment, read from the experiments on disk.

    The ledgers are the single source of both the verdict and the current skills
    generation, so there is no second registry to keep in sync. An experiment
    without a ledger, without a graduated held-out row, or without a published
    skills generation contributes nothing.
    """

    root = Path(experiments_root)
    if not root.is_dir():
        return ()
    sources: list[MemorySource] = []
    for directory in sorted(root.iterdir(), key=lambda path: path.name):
        if not directory.is_dir() or directory.name == exclude:
            continue
        ledger_path = directory / "ledgers" / "experiment_ledger.jsonl"
        if not ledger_path.is_file():
            continue
        records = ExperimentLedger(ledger_path).read()
        if not experiment_graduated(records):
            continue
        if directory.name == CURATED_MEMORY_SOURCE:
            raise ValueError(
                f"experiment id {CURATED_MEMORY_SOURCE!r} is reserved for the "
                "curated memory tier; rename that experiment"
            )
        snapshot = latest_skills_snapshot(records, experiment_dir=directory)
        if snapshot.root is None or not snapshot.stats.count:
            continue
        sources.append(
            MemorySource(
                directory.name,
                "graduated",
                snapshot.root,
                tuple(sorted(item.name for item in snapshot.root.iterdir())),
            )
        )
    return tuple(sources)


def install_operating_memory(
    workspace: str | Path,
    *,
    mode: str,
    repo_root: str | Path | None = None,
    experiments_root: str | Path | None = None,
    experiment_id: str = "",
) -> tuple[MemorySource, ...]:
    """Mount the selected memory tiers read-only next to the session skills.

    ``curated`` mounts the repository library; ``curated+graduated`` adds the
    skills of every experiment the held-out verdict graduated. Each source lands
    in ``workspace/memory/<source>/<name>/`` as read-only files, so a session can
    read them like its own skills but can never change or delete them: the skill
    tools refuse mounted names and the copy itself is not writable. Nothing
    mounted here is ever published as this experiment's skills generation.
    """

    if mode not in OPERATING_MEMORY_MODES:
        raise ValueError(
            "operating_memory must be one of " + ", ".join(OPERATING_MEMORY_MODES)
        )
    destination = Path(workspace) / OPERATING_MEMORY_DIRNAME
    if destination.exists():
        raise FileExistsError(f"workspace memory directory already exists: {destination}")
    if mode == "none":
        return ()
    if repo_root is None:
        raise ValueError("operating_memory needs a repository root to mount from")
    sources: list[MemorySource] = []
    curated = curated_memory_source(repo_root)
    if curated is not None:
        sources.append(curated)
    if mode == "curated+graduated" and experiments_root is not None:
        sources.extend(
            graduated_memory_sources(experiments_root, exclude=experiment_id)
        )
    if not sources:
        return ()
    destination.mkdir()
    for source in sources:
        shutil.copytree(
            source.root, destination / source.source, copy_function=shutil.copyfile
        )
    chmod_tree(destination, file_mode=0o444, dir_mode=0o555)
    for source in sources:
        validate_skills_tree(destination / source.source, require_writable=False)
    return tuple(sources)


def _reject_operating_memory_name(workspace_root: Path, name: str) -> None:
    """Mounted memory is read-only; a session may not shadow it under its own name."""

    memory = workspace_root / OPERATING_MEMORY_DIRNAME
    if not memory.is_dir():
        return
    for source in sorted(memory.iterdir(), key=lambda path: path.name):
        if (source / name).is_dir():
            raise ValueError(
                f"{name} is mounted operating memory ({source.name}) and is "
                "read-only; use a different skill name"
            )


def _file_map(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }


def skills_trees_equal(left: str | Path, right: str | Path) -> bool:
    """Compare path sets and each file's bytes directly; no digest is persisted."""

    left_path = Path(left)
    right_path = Path(right)
    validate_skills_tree(left_path, require_writable=False)
    validate_skills_tree(right_path, require_writable=False)
    left_files = _file_map(left_path)
    right_files = _file_map(right_path)
    if left_files.keys() != right_files.keys():
        return False
    return all(left_files[name].read_bytes() == right_files[name].read_bytes() for name in left_files)


class ExperimentSkillsStore:
    """Immutable generations; the experiment ledger is the only reachable head."""

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir).resolve()
        self.root = self.experiment_dir / "artifacts" / SKILLS_DIRNAME

    def publish(
        self,
        source: str | Path,
        *,
        generation_id: str,
        previous: SkillsSnapshot | None = None,
    ) -> SkillsPublication:
        source_path = Path(source).resolve(strict=True)
        stats = validate_skills_tree(source_path, require_writable=True)
        previous = previous or SkillsSnapshot()
        if previous.root is not None and skills_trees_equal(source_path, previous.root):
            return SkillsPublication(
                previous.skills_ref, previous.generation_id, previous.stats, False
            )
        if previous.root is None and stats == SkillsStats():
            return SkillsPublication("", "", stats, False)

        generation = str(generation_id).strip()
        if (
            not generation
            or Path(generation).name != generation
            or generation.startswith(".")
        ):
            raise ValueError("skills generation_id must be one non-hidden path component")
        generations = self.root / "generations"
        destination = generations / generation
        if destination.exists():
            raise FileExistsError(f"skills generation already exists: {generation}")
        generations.mkdir(parents=True, exist_ok=True)
        staging = generations / f".{generation}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            staging.mkdir()
            snapshot_root = staging / SKILLS_DIRNAME
            _copy_skills_tree(source_path, snapshot_root)
            validate_skills_tree(snapshot_root, require_writable=True)
            chmod_tree(staging, file_mode=0o444, dir_mode=0o555)
            staging.replace(destination)
            _assert_skills_tree_read_only(destination / SKILLS_DIRNAME)
        finally:
            if staging.exists():
                chmod_tree(staging, file_mode=0o644, dir_mode=0o755)
                shutil.rmtree(staging)
        snapshot_root = destination / SKILLS_DIRNAME
        ref = snapshot_root.relative_to(self.experiment_dir).as_posix()
        return SkillsPublication(ref, generation, stats, True)


def _optional_ledger_count(record: Mapping[str, object], key: str) -> int | None:
    raw = record.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(f"ledger {key} must be a non-negative integer")
    return raw


def latest_skills_snapshot(
    records: Sequence[Mapping[str, object]], *, experiment_dir: str | Path
) -> SkillsSnapshot:
    """Resolve skills from the final remaining successful Fold/Meta ledger row."""

    successful = [
        record
        for record in records
        if record.get("record_type") in {"fold", "meta_learning"}
    ]
    if not successful:
        return SkillsSnapshot()
    record = successful[-1]
    raw_ref = str(record.get("skills_ref") or "").strip()
    recorded_generation = str(record.get("skills_generation_id") or "").strip()
    recorded_counts = {
        key: _optional_ledger_count(record, key)
        for key in ("skills_count", "skills_files", "skills_bytes")
    }
    raw_published = record.get("skills_published")
    if raw_published is not None and not isinstance(raw_published, bool):
        raise ValueError("ledger skills_published must be a boolean")
    if not raw_ref:
        if (
            recorded_generation
            or any(value not in {None, 0} for value in recorded_counts.values())
            or bool(raw_published)
        ):
            raise ValueError("ledger empty skills_ref has non-empty skills metadata")
        return SkillsSnapshot()
    pure = PurePosixPath(raw_ref)
    if pure.is_absolute() or any(part in {"", ".", ".."} or part.startswith(".") for part in pure.parts):
        raise ValueError("ledger skills_ref must be a non-hidden experiment-relative path")
    experiment = Path(experiment_dir).resolve()
    root = experiment.joinpath(*pure.parts).resolve(strict=True)
    expected_parent = experiment / "artifacts" / SKILLS_DIRNAME / "generations"
    if not root.is_relative_to(expected_parent) or root.name != SKILLS_DIRNAME:
        raise ValueError("ledger skills_ref is outside the experiment skills store")
    stats = validate_skills_tree(root, require_writable=False)
    _assert_skills_tree_read_only(root)
    generation = recorded_generation or root.parent.name
    if (
        Path(generation).name != generation
        or generation.startswith(".")
        or generation != root.parent.name
    ):
        raise ValueError("ledger skills_generation_id does not match skills_ref")
    actual_counts = {
        "skills_count": stats.count,
        "skills_files": stats.files,
        "skills_bytes": stats.bytes,
    }
    for key, recorded in recorded_counts.items():
        if recorded is not None and recorded != actual_counts[key]:
            raise ValueError(f"ledger {key} does not match the published skills tree")
    return SkillsSnapshot(raw_ref, generation, stats, root)


def resolve_collected_skills_source(
    experiment_dir: str | Path, run_id: str, source_ref: str | Path
) -> Path:
    """Accept only this run's collected ``workspace/skills`` audit copy."""

    experiment = Path(experiment_dir).resolve()
    expected = (experiment / "artifacts" / run_id / "workspace" / SKILLS_DIRNAME).resolve(
        strict=True
    )
    supplied = Path(source_ref).resolve(strict=True)
    if supplied != expected:
        raise ValueError("skills source is not this run's collected workspace/skills")
    validate_skills_tree(supplied, require_writable=True)
    return supplied


class WriteSkillTool:
    spec = ToolSpec(
        "write_skill",
        "Atomically create or replace one UTF-8 file under skills/<name>/ using "
        "the bounded shared-skill contract. path is item-relative (SKILL.md, "
        "scripts/..., or references/...).",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": MAX_SKILL_NAME_CHARS},
                "path": {"type": "string", "minLength": 1, "maxLength": 500},
                "content": {"type": "string"},
            },
            "required": ["name", "path", "content"],
            "additionalProperties": False,
        },
        mutating=True,
    )

    def __init__(self, workspace: SafeWorkspace) -> None:
        self.workspace = workspace
        self.root = workspace.root / SKILLS_DIRNAME

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        try:
            name = validate_skill_name(str(arguments["name"]))
            _reject_operating_memory_name(self.workspace.root, name)
            relative = validate_skill_path(str(arguments["path"]))
            content = str(arguments["content"])
            payload = content.encode("utf-8")
            _decode_skill_file(Path(name) / Path(relative.as_posix()), payload)
            if relative == PurePosixPath(SKILL_FILENAME) and len(content) > MAX_SKILL_CHARS:
                raise ValueError(f"SKILL.md exceeds {MAX_SKILL_CHARS} characters")
            before = validate_skills_tree(self.root, require_writable=True)
            item = self.root / name
            target = item.joinpath(*relative.parts)
            if not item.exists() and relative != PurePosixPath(SKILL_FILENAME):
                raise ValueError("create SKILL.md before adding scripts or references")
            existed = target.is_file()
            previous_bytes = target.read_bytes() if existed else b""
            resulting_files = before.files + (0 if existed else 1)
            resulting_bytes = before.bytes - len(previous_bytes) + len(payload)
            resulting_count = before.count + (0 if item.exists() else 1)
            if resulting_count > MAX_SKILLS:
                raise ValueError(f"skills tree exceeds {MAX_SKILLS} items")
            if resulting_files > MAX_SKILLS_FILES:
                raise ValueError(f"skills tree exceeds {MAX_SKILLS_FILES} files")
            if resulting_bytes > MAX_SKILLS_BYTES:
                raise ValueError(f"skills tree exceeds {MAX_SKILLS_BYTES} bytes")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(
                f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
            )
            try:
                temporary.write_bytes(payload)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            after = validate_skills_tree(self.root, require_writable=True)
        except ValueError as exc:
            raise ToolError(str(exc), error_type="skill_policy") from exc
        return ToolResult(
            True,
            value={
                "name": name,
                "path": f"skills/{name}/{relative.as_posix()}",
                "created": not existed,
                "bytes_written": len(payload),
                **after.ledger_fields(),
            },
        )


class DeleteSkillTool:
    spec = ToolSpec(
        "delete_skill",
        "Atomically remove one complete skills/<name>/ item from the shared working tree.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": MAX_SKILL_NAME_CHARS}
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        mutating=True,
    )

    def __init__(self, workspace: SafeWorkspace) -> None:
        self.workspace = workspace
        self.root = workspace.root / SKILLS_DIRNAME

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        try:
            name = validate_skill_name(str(arguments["name"]))
            _reject_operating_memory_name(self.workspace.root, name)
            validate_skills_tree(self.root, require_writable=True)
            item = self.root / name
            if item.is_symlink() or not item.is_dir():
                raise ValueError(f"skill does not exist: {name}")
            staging = self.workspace.root / (
                f".deleted-skill-{name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            )
            item.replace(staging)
            try:
                after = validate_skills_tree(self.root, require_writable=True)
            except Exception:
                staging.replace(item)
                raise
            shutil.rmtree(staging)
        except ValueError as exc:
            raise ToolError(str(exc), error_type="skill_policy") from exc
        return ToolResult(
            True,
            value={"name": name, "deleted": True, **after.ledger_fields()},
        )


__all__ = [
    "ALLOWED_SKILL_SUFFIXES",
    "DeleteSkillTool",
    "ExperimentSkillsStore",
    "MAX_SKILL_CHARS",
    "MAX_SKILL_FILE_BYTES",
    "MAX_SKILL_NAME_CHARS",
    "MAX_SKILLS",
    "MAX_SKILLS_BYTES",
    "MAX_SKILLS_FILES",
    "CURATED_MEMORY_SOURCE",
    "DEFAULT_OPERATING_MEMORY",
    "OPERATING_MEMORY_DIRNAME",
    "OPERATING_MEMORY_LIBRARY",
    "OPERATING_MEMORY_MODES",
    "MemorySource",
    "SKILLS_INDEX_PATH",
    "SkillsPublication",
    "SkillsSnapshot",
    "SkillsStats",
    "WriteSkillTool",
    "build_skills_index",
    "curated_memory_source",
    "experiment_graduated",
    "graduated_memory_sources",
    "install_operating_memory",
    "install_workspace_skills",
    "latest_skills_snapshot",
    "operating_memory_entries",
    "resolve_operating_memory",
    "resolve_collected_skills_source",
    "skills_trees_equal",
    "validate_skill_name",
    "validate_skill_path",
    "validate_skills_tree",
    "write_skills_index",
]
