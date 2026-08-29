"""Strategy and model artifact contracts, revisions, and modification diffs.

The formal strategy artifact is the ``output/`` directory. ``main.py`` at the
root is the only required entrypoint; helper modules and subpackages are
ordinary Agent-editable code, not separate artifact classes. Inherited model
parameters live in the sibling ``models/`` directory so binary state is
validated and frozen separately from strategy code.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import shutil
import stat
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from .runtime import RUNTIME_CACHE_DIR_NAMES, RUNTIME_CACHE_SUFFIXES, chmod_tree

REQUIRED_FILES = ("main.py",)
ARTIFACT_METADATA_FILES = frozenset({"manifest.json"})
READONLY_FILES = frozenset({"README.md"})
ALLOWED_SUFFIXES = frozenset({".py", ".json", ".md", ".txt", ".toml", ".yaml", ".yml"})
# Deny-by-default allowlist for the frozen, inheritable ``models/`` directory.
# Covers mainstream parameter/weight/serialization formats; executables, shared
# libraries, scripts, and archives are excluded by design to keep the inherited
# artifact auditable. Anti-overfit/anti-dump is enforced by byte caps + PIT
# visibility + held-out, not by suffix.
MODEL_ARTIFACT_ALLOWED_SUFFIXES = frozenset(
    {
        ".bin",
        ".cbm",
        ".ckpt",
        ".csv",
        ".gguf",
        ".h5",
        ".hdf5",
        ".joblib",
        ".json",
        ".keras",
        ".model",
        ".msgpack",
        ".npy",
        ".npz",
        ".onnx",
        ".params",
        ".pb",
        ".pdparams",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".safetensors",
        ".tflite",
        ".toml",
        ".txt",
        ".ubj",
        ".yaml",
        ".yml",
    }
)

# Mount paths a formal strategy must never hardcode. "/mnt/snapshots/" (plural,
# the staged alias root) is not mounted into the formal run. "/mnt/runtime/"
# subpaths ARE mounted there, but they are per-run ephemeral host-managed paths
# reachable only via the context surfaces, so hardcoding them must fail fast
# just the same. The singular "/mnt/snapshot" is intentionally absent: it is the
# legitimate formal read root (see sandbox.py formal_strategy_read_roots).
FORBIDDEN_CODE_REFERENCES = (
    "/mnt/snapshots/",
    "/mnt/runtime/",
    "/mnt/artifacts",
    "/mnt/agent/workspace",
)


class ArtifactError(ValueError):
    """A strategy artifact violates the documented format contract."""


class ArtifactSnapshotUnstable(RuntimeError):
    """A working artifact kept changing while its evaluation snapshot was taken.

    An evaluation never reads the Agent's live tree, so a copy taken while a
    write is in flight could mix two versions' bytes into a strategy that never
    existed. The copy is verified against its source and retaken once; a source
    still moving after that raises this instead of evaluating the mixture.
    """


@dataclass(frozen=True)
class StrategyArtifact:
    root: Path
    files: tuple[str, ...]
    revision_id: str


@dataclass(frozen=True)
class ModelArtifacts:
    root: Path
    files: tuple[str, ...]
    revision_id: str
    total_bytes: int


def new_revision_id(prefix: str = "artifact") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class FilesystemArtifactStore:
    """Revision/Freeze backend for ``pipelines.config.ArtifactStore``.

    Revisions and frozen artifacts are immutable directories addressed by
    explicit IDs.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.revisions_root = self.root / "revisions"
        self.frozen_root = self.root / "frozen"
        self.revisions_root.mkdir(parents=True, exist_ok=True)
        self.frozen_root.mkdir(parents=True, exist_ok=True)

    def create_revision(
        self,
        output_path: str | Path,
        *,
        models_path: str | Path | None = None,
        revision_id: str | None = None,
    ):
        """Snapshot a working artifact into a new immutable revision.

        The returned record carries the snapshot's ``fingerprint``: the
        evaluated bytes are addressed by content, not by the directory they
        were copied from, so a caller can prove the revision it replays is the
        one it approved.
        """
        revision_id = revision_id or new_revision_id("revision")
        directory = self._id_path(self.revisions_root, revision_id)
        if directory.exists():
            raise FileExistsError(f"artifact revision already exists: {revision_id}")
        try:
            fingerprint = copy_artifact_snapshot(
                output_path,
                models_path,
                dest_output=directory / "output",
                dest_models=directory / "models",
            )
        except Exception:
            self.discard_revision(revision_id)
            raise
        chmod_tree(directory, file_mode=0o444, dir_mode=0o555)
        record = self.revision(revision_id)
        record.fingerprint = fingerprint
        return record

    def discard_revision(self, revision_id: str) -> None:
        """Drop a candidate revision that was never accepted."""
        directory = self._id_path(self.revisions_root, revision_id)
        if directory.is_dir():
            self._discard_directory(directory)

    def revision(self, revision_id: str):
        directory = self._id_path(self.revisions_root, revision_id)
        output = directory / "output"
        if not output.is_dir():
            raise KeyError(f"unknown artifact revision: {revision_id}")
        models = directory / "models"
        return SimpleNamespace(revision_id=revision_id, output_path=output, models_path=models if models.is_dir() else None)

    def freeze_revision(
        self,
        revision_id: str,
        *,
        artifact_id: str,
        experiment_id: str,
        epoch_id: str,
        fold_id: str,
        run_id: str,
        step_id: str,
    ):
        revision = self.revision(revision_id)
        directory = self._id_path(self.frozen_root, artifact_id)
        if directory.exists():
            raise FileExistsError(f"frozen artifact already exists: {artifact_id}")
        copy_artifact(revision.output_path, directory / "output")
        copy_model_artifacts(revision.models_path, directory / "models")
        metadata = {
            "artifact_id": artifact_id, "revision_id": revision_id, "experiment_id": experiment_id,
            "epoch_id": epoch_id, "fold_id": fold_id, "run_id": run_id,
            "source_step_id": step_id,
        }
        (directory / "revision.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        chmod_tree(directory, file_mode=0o444, dir_mode=0o555)
        return SimpleNamespace(
            artifact_id=artifact_id, path=directory / "output", model_path=directory / "models",
            source_run_id=run_id, source_fold_id=fold_id, source_step_id=step_id,
            revision_id=revision_id,
            requires_validation=False,
        )

    def prune_transient(self, *, keep_frozen_ids: tuple[str, ...] = ()) -> None:
        """Discard candidate revisions and superseded frozen artifacts."""

        keep = set(keep_frozen_ids)
        for directory in list(self.revisions_root.iterdir()):
            self._discard_directory(directory)
        for directory in list(self.frozen_root.iterdir()):
            if directory.name not in keep:
                self._discard_directory(directory)

    def frozen(
        self,
        artifact_id: str,
        *,
        expected_path: str | Path | None = None,
        experiment_id: str | None = None,
    ):
        """Load a frozen artifact after validating identity and immutability."""

        directory = self._id_path(self.frozen_root, artifact_id)
        if not directory.is_dir():
            raise KeyError(f"unknown frozen artifact: {artifact_id}")
        resolved = directory.resolve(strict=True)
        if not resolved.is_relative_to(self.frozen_root.resolve()):
            raise ArtifactError(f"frozen artifact escaped the configured store: {artifact_id}")
        output = resolved / "output"
        models = resolved / "models"
        manifest_path = resolved / "revision.json"
        if not output.is_dir() or not manifest_path.is_file():
            raise ArtifactError(f"incomplete frozen artifact: {artifact_id}")
        if expected_path is not None and Path(expected_path).resolve(strict=True) != output:
            raise ArtifactError(f"frozen artifact path mismatch: {artifact_id}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"invalid frozen artifact manifest: {artifact_id}") from exc
        if not isinstance(manifest, dict) or manifest.get("artifact_id") != artifact_id:
            raise ArtifactError(f"frozen artifact manifest identity mismatch: {artifact_id}")
        if experiment_id is not None and manifest.get("experiment_id") != experiment_id:
            raise ArtifactError(f"frozen artifact belongs to another experiment: {artifact_id}")
        for name in ("revision_id", "epoch_id", "fold_id", "run_id", "source_step_id"):
            if not isinstance(manifest.get(name), str) or not str(manifest[name]).strip():
                raise ArtifactError(f"frozen artifact manifest is missing {name}: {artifact_id}")
        load_strategy_artifact(output, revision_id=artifact_id)
        if models.exists():
            load_model_artifacts(models, revision_id=artifact_id)
        _assert_readonly_tree(resolved)
        return SimpleNamespace(
            artifact_id=artifact_id,
            path=output,
            model_path=models if models.is_dir() else None,
            source_run_id=str(manifest["run_id"]),
            source_fold_id=str(manifest["fold_id"]),
            source_step_id=str(manifest["source_step_id"]),
            revision_id=str(manifest["revision_id"]),
            requires_validation=False,
        )

    @staticmethod
    def _discard_directory(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ArtifactError(f"artifact store contains an invalid entry: {path}")
        chmod_tree(path, file_mode=0o600, dir_mode=0o700)
        shutil.rmtree(path)

    @staticmethod
    def _id_path(root: Path, identifier: str) -> Path:
        if not identifier or identifier in {".", ".."} or Path(identifier).name != identifier or "/" in identifier or "\\" in identifier:
            raise ValueError("artifact ID must be one local path component")
        return root / identifier


def load_strategy_artifact(root: str | Path, *, revision_id: str | None = None) -> StrategyArtifact:
    """Load and validate the ``output/`` strategy artifact directory."""
    root = Path(root).resolve()
    files = _artifact_files(root)
    for required in REQUIRED_FILES:
        if required not in files:
            raise ArtifactError(f"missing required artifact file: {required}")
    main = root / "main.py"
    if "generate_orders" not in defined_function_names(main):
        raise ArtifactError("main.py must define generate_orders(context)")
    for relpath in files:
        if not relpath.endswith(".py"):
            continue
        for literal in _runtime_string_constants(root / relpath):
            for forbidden in FORBIDDEN_CODE_REFERENCES:
                if forbidden in literal:
                    raise ArtifactError(f"formal strategy code must not reference stage directories: {forbidden}")
    return StrategyArtifact(root, tuple(sorted(files)), revision_id or new_revision_id())


def load_model_artifacts(root: str | Path, *, revision_id: str | None = None) -> ModelArtifacts:
    """Load and validate the optional ``models/`` artifact directory."""
    root = Path(root).resolve()
    files = _model_artifact_files(root, missing_ok=True)
    return ModelArtifacts(
        root, tuple(sorted(files)), revision_id or new_revision_id("models"),
        sum((root / item).stat().st_size for item in files),
    )


def init_from_template(template_dir: str | Path, dest_root: str | Path) -> None:
    """Initialize ``output/`` from ``configs/agent_output_template/``."""
    template_dir = Path(template_dir)
    _copy_revision(template_dir, Path(dest_root), _artifact_files(template_dir, reject_runtime_cache=False))


def copy_artifact(source_root: str | Path, dest_root: str | Path) -> None:
    """Copy one strategy artifact directory, replacing any existing copy.

    Runtime caches next to the source (including the output template) are
    skipped; official load/freeze still reject them if they remain in dest.
    """
    source_root = Path(source_root)
    _copy_revision(
        source_root,
        Path(dest_root),
        _artifact_files(source_root, reject_runtime_cache=False),
    )


def copy_model_artifacts(source_root: str | Path | None, dest_root: str | Path) -> None:
    """Copy optional model artifact directories, replacing any existing copy.

    ``source_root=None`` replaces the destination with an empty directory."""
    dest_root = Path(dest_root)
    if source_root is None:
        _replace_artifact_root(dest_root)
        return
    source_root = Path(source_root)
    _copy_revision(source_root, dest_root, _model_artifact_files(source_root, missing_ok=True))


def artifact_fingerprint(
    output_root: str | Path, models_root: str | Path | None = None
) -> str:
    """Content address of one strategy artifact: its file names and bytes.

    Covers exactly the files ``copy_artifact``/``copy_model_artifacts`` carry,
    so a copy and its source fingerprint identically and a missing ``models/``
    is the same artifact as an empty one. It identifies the evaluated bytes
    independently of the directory they were read from, which is what lets a
    formal call prove it replayed the artifact it approved.
    """
    output_root = Path(output_root)
    roots: list[tuple[str, Path, set[str]]] = [
        (
            "output",
            output_root,
            _artifact_files(output_root, reject_runtime_cache=False),
        )
    ]
    if models_root is not None:
        models = Path(models_root)
        roots.append(("models", models, _model_artifact_files(models, missing_ok=True)))
    digest = hashlib.sha256()
    for label, root, relpaths in roots:
        for relpath in sorted(relpaths):
            path = root / relpath
            digest.update(f"{label}/{relpath}\0{path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def copy_artifact_snapshot(
    output_root: str | Path,
    models_root: str | Path | None,
    *,
    dest_output: str | Path,
    dest_models: str | Path,
) -> str:
    """Copy one working artifact to an immutable destination; return its fingerprint.

    The Agent and its children keep working while a formal call replays the
    copy, so the copy has to be one point in that timeline rather than a blend
    of two. It is taken, compared with its source, and retaken once if the
    source moved meanwhile; a source still moving after that raises
    ``ArtifactSnapshotUnstable`` instead of evaluating a torn tree.
    """
    for _ in range(2):
        copy_artifact(output_root, dest_output)
        copy_model_artifacts(models_root, dest_models)
        fingerprint = artifact_fingerprint(dest_output, dest_models)
        if fingerprint == artifact_fingerprint(output_root, models_root):
            return fingerprint
    raise ArtifactSnapshotUnstable(
        "the strategy artifact kept changing while its evaluation snapshot was taken"
    )


def restore_frozen_artifact_trees(
    *,
    output_path: str | Path,
    snapshot_output: str | Path,
    models_path: str | Path | None,
    snapshot_models: str | Path,
) -> None:
    """Replace live frozen output/models with snapshot bytes and re-lock them.

    The snapshot must outlive this call. Restore is fail-closed: a copy or
    re-lock error leaves the caller to treat the trees as unrestored.
    """
    live_output = Path(output_path)
    live_models = (
        Path(models_path) if models_path is not None else live_output.parent / "models"
    )
    _atomic_replace_copied(Path(snapshot_output), live_output, copy_artifact)
    _atomic_replace_copied(Path(snapshot_models), live_models, copy_model_artifacts)
    frozen_root = live_output.parent
    if (frozen_root / "revision.json").is_file():
        chmod_tree(frozen_root, file_mode=0o444, dir_mode=0o555)
        _assert_readonly_tree(frozen_root)
        return
    chmod_tree(live_output, file_mode=0o444, dir_mode=0o555)
    _assert_readonly_tree(live_output)
    if live_models.exists():
        chmod_tree(live_models, file_mode=0o444, dir_mode=0o555)
        _assert_readonly_tree(live_models)


def restore_working_artifacts_writable(
    output_root: str | Path,
    models_root: str | Path | None = None,
) -> None:
    """Normalize copied Fold artifacts for the unprivileged Agent workspace."""

    output = Path(output_root)
    models = Path(models_root) if models_root is not None else None
    chmod_tree(output, file_mode=0o666, dir_mode=0o777)
    if models is not None:
        chmod_tree(models, file_mode=0o666, dir_mode=0o777)
    for relpath in READONLY_FILES:
        target = output / relpath
        if target.exists():
            target.chmod(0o444)
    _assert_working_tree_permissions(output, readonly_files=READONLY_FILES)
    if models is not None:
        _assert_working_tree_permissions(models)


@dataclass(frozen=True)
class ModificationDelta:
    changed_files: tuple[str, ...]
    diff_lines: int
    code_diff_lines: int
    total_files: int
    total_bytes: int
    readonly_violations: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "changed_files": list(self.changed_files),
            "changed_file_count": len(self.changed_files),
            "diff_lines": self.diff_lines,
            "code_diff_lines": self.code_diff_lines,
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "readonly_violations": list(self.readonly_violations),
        }


@dataclass(frozen=True)
class ModelArtifactDelta:
    changed_files: tuple[str, ...]
    total_files: int
    total_bytes: int

    def to_record(self) -> dict[str, object]:
        return {
            "changed_files": list(self.changed_files),
            "changed_file_count": len(self.changed_files),
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
        }


def modification_delta(parent_root: str | Path, work_root: str | Path) -> ModificationDelta:
    """Deterministic file and line counts for ``output`` changes."""
    parent_root = Path(parent_root)
    work_root = Path(work_root)
    parent_files = _artifact_files(parent_root) if parent_root.is_dir() and any(parent_root.iterdir()) else set()
    work_files = _artifact_files(work_root)
    changed: list[str] = []
    diff_lines = 0
    code_diff_lines = 0
    readonly_violations: list[str] = []
    for relpath in sorted(parent_files | work_files):
        parent_text = _read_text(parent_root / relpath) if relpath in parent_files else None
        work_text = _read_text(work_root / relpath) if relpath in work_files else None
        if parent_text == work_text:
            continue
        changed.append(relpath)
        if relpath in READONLY_FILES:
            readonly_violations.append(relpath)
        line_delta = _changed_line_count(parent_text or "", work_text or "")
        diff_lines += line_delta
        if relpath.endswith(".py"):
            code_diff_lines += line_delta
    return ModificationDelta(
        changed_files=tuple(changed),
        diff_lines=diff_lines,
        code_diff_lines=code_diff_lines,
        total_files=len(work_files),
        total_bytes=sum((work_root / relpath).stat().st_size for relpath in work_files),
        readonly_violations=tuple(readonly_violations),
    )


def model_artifact_delta(parent_root: str | Path, work_root: str | Path) -> ModelArtifactDelta:
    """Deterministic changed-file counts for optional model parameter files."""
    parent_root = Path(parent_root)
    work_root = Path(work_root)
    parent_files = _model_artifact_files(parent_root, missing_ok=True)
    work_files = _model_artifact_files(work_root, missing_ok=True)
    changed: list[str] = []
    for relpath in sorted(parent_files | work_files):
        if relpath not in parent_files or relpath not in work_files:
            changed.append(relpath)
        elif not _files_equal(parent_root / relpath, work_root / relpath):
            changed.append(relpath)
    return ModelArtifactDelta(
        changed_files=tuple(changed),
        total_files=len(work_files),
        total_bytes=sum((work_root / relpath).stat().st_size for relpath in work_files),
    )


@dataclass(frozen=True)
class ModificationConstraints:
    """Per-Step/Fold limits over ``output`` and optional model parameters."""

    max_changed_files: int = 8
    max_diff_lines: int = 600
    max_code_diff_lines: int = 500
    max_strategy_files: int = 64
    max_strategy_bytes: int = 1_000_000
    max_model_artifact_files: int = 64
    max_model_artifact_bytes: int = 1024 * 1024 * 1024
    early_epoch_count: int = 2
    early_max_changed_files: int = 12
    early_max_diff_lines: int = 1200
    early_max_code_diff_lines: int = 1000
    is_initial_artifact: bool = False

    def for_epoch(self, epoch_index: int) -> ModificationConstraints:
        if self.is_initial_artifact or epoch_index <= self.early_epoch_count:
            return replace(
                self,
                max_changed_files=self.early_max_changed_files,
                max_diff_lines=self.early_max_diff_lines,
                max_code_diff_lines=self.early_max_code_diff_lines,
            )
        return self

    def evaluate(
        self,
        delta: ModificationDelta,
        model_delta: ModelArtifactDelta | None = None,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if delta.readonly_violations:
            reasons.append(f"readonly files modified: {list(delta.readonly_violations)}")
        if not self.is_initial_artifact:
            if len(delta.changed_files) > self.max_changed_files:
                reasons.append(f"changed files {len(delta.changed_files)} > {self.max_changed_files}")
            if delta.diff_lines > self.max_diff_lines:
                reasons.append(f"diff lines {delta.diff_lines} > {self.max_diff_lines}")
            if delta.code_diff_lines > self.max_code_diff_lines:
                reasons.append(f"code diff lines {delta.code_diff_lines} > {self.max_code_diff_lines}")
        if delta.total_files > self.max_strategy_files:
            reasons.append(f"strategy files {delta.total_files} > {self.max_strategy_files}")
        if delta.total_bytes > self.max_strategy_bytes:
            reasons.append(f"strategy bytes {delta.total_bytes} > {self.max_strategy_bytes}")
        if model_delta is not None:
            if model_delta.total_files > self.max_model_artifact_files:
                reasons.append(f"model artifact files {model_delta.total_files} > {self.max_model_artifact_files}")
            if model_delta.total_bytes > self.max_model_artifact_bytes:
                reasons.append(f"model artifact bytes {model_delta.total_bytes} > {self.max_model_artifact_bytes}")
        return (not reasons, reasons)

    def to_record(self) -> dict[str, object]:
        return {
            "max_changed_files": self.max_changed_files,
            "max_diff_lines": self.max_diff_lines,
            "max_code_diff_lines": self.max_code_diff_lines,
            "max_strategy_files": self.max_strategy_files,
            "max_strategy_bytes": self.max_strategy_bytes,
            "max_model_artifact_files": self.max_model_artifact_files,
            "max_model_artifact_bytes": self.max_model_artifact_bytes,
            "early_epoch_count": self.early_epoch_count,
            "early_max_changed_files": self.early_max_changed_files,
            "early_max_diff_lines": self.early_max_diff_lines,
            "early_max_code_diff_lines": self.early_max_code_diff_lines,
            "is_initial_artifact": self.is_initial_artifact,
        }

    @classmethod
    def from_record(cls, record: dict[str, object]) -> ModificationConstraints:
        allowed = set(cls().to_record())
        return cls(**{key: record[key] for key in allowed if key in record})


def defined_function_names(main_py: Path) -> set[str]:
    try:
        tree = ast.parse(main_py.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise ArtifactError(f"{main_py.name} has a syntax error: {exc}") from exc
    return {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _artifact_files(root: Path, *, reject_runtime_cache: bool = True) -> set[str]:
    if not root.is_dir():
        raise ArtifactError(f"missing artifact directory: {root}")
    return _collect_artifact_files(
        root,
        allowed_suffixes=ALLOWED_SUFFIXES,
        metadata_files=ARTIFACT_METADATA_FILES,
        reject_runtime_cache=reject_runtime_cache,
        label="strategy artifact",
    )


def _model_artifact_files(
    root: Path,
    *,
    reject_runtime_cache: bool = True,
    missing_ok: bool = False,
) -> set[str]:
    if not root.exists():
        if missing_ok:
            return set()
        raise ArtifactError(f"missing model artifact directory: {root}")
    if not root.is_dir():
        raise ArtifactError(f"models must be a directory: {root}")
    return _collect_artifact_files(
        root,
        allowed_suffixes=MODEL_ARTIFACT_ALLOWED_SUFFIXES,
        metadata_files=frozenset(),
        reject_runtime_cache=reject_runtime_cache,
        label="models",
    )


def _collect_artifact_files(
    root: Path,
    *,
    allowed_suffixes: frozenset[str],
    metadata_files: frozenset[str],
    reject_runtime_cache: bool,
    label: str,
) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        relpath = rel.as_posix()
        if relpath in metadata_files:
            continue
        if path.is_symlink():
            raise ArtifactError(f"{label} must not contain symlinks: {relpath}")
        if _has_hidden_part(rel):
            raise ArtifactError(f"{label} must not contain hidden files or directories: {relpath}")
        if _is_runtime_cache(rel):
            if not reject_runtime_cache:
                continue
            raise ArtifactError(f"{label} must not contain runtime cache files: {relpath}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArtifactError(f"{label} must contain only regular files: {relpath}")
        if rel.suffix.lower() not in allowed_suffixes:
            raise ArtifactError(f"unsupported {label} file type: {relpath}")
        files.add(relpath)
    return files


def _replace_artifact_root(dest_root: Path) -> None:
    dest_root.mkdir(parents=True, exist_ok=True)
    for child in list(dest_root.iterdir()):
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _unlock_directory(path: Path) -> None:
    if path.exists():
        chmod_tree(path, file_mode=0o600, dir_mode=0o700)
    if path.parent.exists():
        path.parent.chmod(0o700)


def _discard_replaced_tree(path: Path) -> None:
    if not path.exists():
        return
    chmod_tree(path, file_mode=0o600, dir_mode=0o700)
    shutil.rmtree(path, ignore_errors=True)


def _atomic_replace_copied(source: Path, dest: Path, copier) -> None:
    """Copy ``source`` onto ``dest`` via a sibling staging directory."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _unlock_directory(dest)
    token = uuid.uuid4().hex[:8]
    staging = dest.with_name(f".{dest.name}.restore_{token}")
    backup = dest.with_name(f".{dest.name}.backup_{token}")
    try:
        copier(source, staging)
        if dest.exists():
            dest.rename(backup)
        staging.rename(dest)
        _discard_replaced_tree(backup)
    except Exception:
        if not dest.exists() and backup.exists():
            try:
                backup.rename(dest)
            except OSError as restore_error:
                _discard_replaced_tree(staging)
                raise OSError(
                    f"failed to restore {dest} from backup {backup} after replace failure"
                ) from restore_error
        _discard_replaced_tree(staging)
        raise


def _copy_revision(source_root: Path, dest_root: Path, relpaths: set[str]) -> None:
    _replace_artifact_root(dest_root)
    for relpath in relpaths:
        target = dest_root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relpath, target)


def _is_runtime_cache(relpath: str | Path) -> bool:
    path = Path(relpath)
    return any(name in path.parts for name in RUNTIME_CACHE_DIR_NAMES) or path.suffix in RUNTIME_CACHE_SUFFIXES


def _has_hidden_part(relpath: str | Path) -> bool:
    return any(part.startswith(".") for part in Path(relpath).parts)


def _runtime_string_constants(main_py: Path) -> list[str]:
    try:
        tree = ast.parse(main_py.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        # Surface a fixable ArtifactError (as main.py does) rather than a raw
        # SyntaxError, so modification_check reports a clear reason for any helper
        # file, not just main.py.
        raise ArtifactError(f"{main_py.name} has a syntax error: {exc}") from exc
    docstring_constants = _docstring_constant_ids(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_constants
    ]


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if not body or not isinstance(body[0], ast.Expr):
                continue
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                docstrings.add(id(value))
    return docstrings


def _changed_line_count(before: str, after: str) -> int:
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
    return sum(1 for line in diff if line[:1] in "+-" and line[:3] not in ("+++", "---"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _files_equal(left: Path, right: Path, *, chunk_size: int = 1024 * 1024) -> bool:
    """Compare regular files without materialising either file in memory."""
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_stream, right.open("rb") as right_stream:
        while True:
            left_chunk = left_stream.read(chunk_size)
            right_chunk = right_stream.read(chunk_size)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _assert_readonly_tree(root: Path) -> None:
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise ArtifactError(f"frozen artifact must not contain symlinks: {path.relative_to(root)}")
        if path.stat().st_mode & 0o222:
            raise ArtifactError(f"frozen artifact is writable: {path.relative_to(root) or '.'}")


def _assert_working_tree_permissions(
    root: Path,
    *,
    readonly_files: frozenset[str] = frozenset(),
) -> None:
    if not root.is_dir():
        raise ArtifactError(f"working artifact directory is missing: {root}")
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise ArtifactError(f"working artifact must not contain symlinks: {path.relative_to(root)}")
        relative = path.relative_to(root).as_posix() if path != root else "."
        expected = 0o777 if path.is_dir() else (0o444 if relative in readonly_files else 0o666)
        actual = stat.S_IMODE(path.stat().st_mode)
        if actual != expected:
            raise ArtifactError(
                f"working artifact has unsafe permissions: {relative} is {actual:o}, expected {expected:o}"
            )


