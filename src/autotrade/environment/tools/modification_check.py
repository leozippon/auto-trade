"""Static and bounded checks performed immediately before a formal replay."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.artifacts import (
    ArtifactError,
    ModificationConstraints,
    artifact_fingerprint,
    model_artifact_delta,
    modification_delta,
)
from autotrade.environment.replay.timeview import ASOF_DOMAIN_NAMES
from autotrade.environment.strategy_loader import (
    StrategyLoadError,
    validate_strategy_source,
)

from .base import ToolError, ToolResult, ToolSpec

# ``ctx.asof_dir/<domain>/`` is a DIRECTORY of parquet parts; only the frozen
# decision snapshot is flat ``<domain>.parquet``. Reading the rolling view with
# the snapshot's shape raises FileNotFoundError on the first decision, which is
# how official backtests kept dying on day one. Cheap static catch: a domain
# file name whose expression also mentions ``asof_dir``.
# The gap may not cross another path root, so a correct directory read followed
# by a correct flat SNAPSHOT read on the next line is not mistaken for a hit.
_ASOF_FLAT_READ = re.compile(
    r"asof_dir(?:(?!asof_dir|snapshot_dir).){0,120}?"
    r"(?P<domain>" + "|".join(ASOF_DOMAIN_NAMES) + r")\.parquet",
    re.DOTALL,
)


class ModificationCheckTool:
    spec = ToolSpec(
        "modification_check",
        "Validate the daily JSON strategy and bounded artifact changes.",
        {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    )

    def __init__(
        self,
        output_dir: str | Path,
        *,
        parent_dir: str | Path | None = None,
        models_dir: str | Path | None = None,
        parent_models_dir: str | Path | None = None,
        constraints: ModificationConstraints | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.parent_dir = Path(parent_dir) if parent_dir is not None else None
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self.parent_models_dir = (
            Path(parent_models_dir) if parent_models_dir is not None else None
        )
        # The researcher-configured limits, not literals: the same constraint
        # set the run manifest publishes is the one enforced here.
        self.constraints = constraints or ModificationConstraints()
        self.check_index = 0

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        constraints = self.constraints
        files = _formal_files(self.output_dir)
        main = self.output_dir / "main.py"
        if main not in files:
            raise ToolError("formal output must contain main.py")
        if len(files) > constraints.max_strategy_files:
            raise ToolError(f"formal output exceeds {constraints.max_strategy_files} files")
        total_bytes = sum(path.stat().st_size for path in files)
        if total_bytes > constraints.max_strategy_bytes:
            raise ToolError(f"formal output exceeds {constraints.max_strategy_bytes} bytes")
        try:
            fit_schedule = validate_strategy_source(
                main.read_text(encoding="utf-8"), filename="main.py"
            )
        except StrategyLoadError as exc:
            raise ToolError(str(exc)) from exc
        _reject_flat_asof_reads(files, self.output_dir)
        try:
            delta = modification_delta(self.parent_dir or self.output_dir, self.output_dir)
            model_delta = (
                model_artifact_delta(
                    self.parent_models_dir or self.models_dir, self.models_dir
                )
                if self.models_dir is not None
                else None
            )
        except ArtifactError as exc:
            raise ToolError(f"artifact format invalid: {exc}") from exc
        allowed, reasons = constraints.evaluate(delta, model_delta)
        if not allowed:
            raise ToolError("; ".join(reasons))
        self.check_index += 1
        return ToolResult(
            True,
            value={
                "check_index": self.check_index,
                "strategy_entry": "generate_orders",
                # The optional fit(context) entry as statically declared: None
                # when main.py has none, else its REFIT_PERIOD (None = once).
                "fit": fit_schedule.to_record() if fit_schedule is not None else None,
                # Content address of exactly what this check read. A formal
                # call snapshots the artifact and refuses to replay a snapshot
                # whose fingerprint is not this one, so an approval cannot be
                # transferred to bytes written after it.
                "fingerprint": artifact_fingerprint(self.output_dir, self.models_dir),
                "file_count": len(files),
                "total_bytes": total_bytes,
                "changed_lines": delta.diff_lines,
                "constraints": constraints.to_record(),
                "delta": delta.to_record(),
                "model_delta": model_delta.to_record() if model_delta is not None else None,
            },
        )


def _reject_flat_asof_reads(files: list[Path], root: Path) -> None:
    """Reject reading a rolling as-of domain as a flat ``<domain>.parquet``."""
    for path in files:
        if path.suffix != ".py":
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        match = _ASOF_FLAT_READ.search(source)
        if match is None:
            continue
        domain = match.group("domain")
        relative = path.relative_to(root)
        raise ToolError(
            f"{relative} reads context.asof_dir/{domain}.parquet, but every "
            f"as-of domain is a DIRECTORY of parquet parts: use "
            f'pd.read_parquet(context.asof_dir + "/{domain}"). Only '
            f"context.snapshot_dir is flat ({domain}.parquet), and falling back "
            f"to it when an as-of read fails is a point-in-time violation."
        )


def _formal_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ToolError(f"missing formal output directory: {root.name}/")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            raise ToolError(f"hidden or runtime file is forbidden: {relative}")
        if path.is_symlink():
            raise ToolError(f"symbolic links are forbidden: {relative}")
        if path.is_file():
            files.append(path)
    return files


__all__ = ["ModificationCheckTool"]
