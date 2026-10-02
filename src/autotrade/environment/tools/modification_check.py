"""Static and bounded checks performed immediately before a formal replay."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from pathlib import Path

from autotrade.environment.artifacts import (
    ArtifactError,
    ModificationConstraints,
    artifact_fingerprint,
    model_artifact_delta,
    modification_delta,
    reject_forbidden_code_references,
    restore_readonly_baseline,
)
from autotrade.environment.replay.timeview import ASOF_DOMAIN_NAMES
from autotrade.environment.strategy_loader import (
    StrategyLoadError,
    validate_strategy_package,
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
        "Validate the daily JSON strategy and bounded artifact changes. "
        "smoke_backtest and batch_validate run this check themselves before they "
        "replay anything, so call it to read the current state early, not to "
        "clear a replay.",
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
        readonly_baseline: Mapping[str, str] | None = None,
        readonly_seed: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.parent_dir = Path(parent_dir) if parent_dir is not None else None
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self.parent_models_dir = (
            Path(parent_models_dir) if parent_models_dir is not None else None
        )
        # The read-only files as this session was seeded, not as the parent
        # directory reads now: an initial artifact's parent IS the live
        # repository template, and editing it must not fail a running session.
        self.readonly_baseline = (
            dict(readonly_baseline) if readonly_baseline is not None else None
        )
        # Where the read-only files were seeded from, given only for the tree
        # the host itself seeded: the session's working copy. A lost or
        # rewritten contract file is restored from there instead of blocking
        # every replay behind bytes the Agent is not allowed to produce. A
        # candidate directory gets no seed -- it is the Agent's own layout of
        # the artifact it asks to freeze, batch_validate supplies the file
        # where it is absent, and a candidate carrying different bytes must
        # still be refused rather than corrected.
        self.readonly_seed = Path(readonly_seed) if readonly_seed is not None else None
        # The researcher-configured limits, not literals: the same constraint
        # set the run manifest publishes is the one enforced here.
        self.constraints = constraints or ModificationConstraints()

    def invoke(self, arguments: Mapping[str, object]) -> ToolResult:
        del arguments
        constraints = self.constraints
        # First, so the file counts, the fingerprint and the delta below all
        # read one tree: a host-owned contract file this session lost or
        # overwrote is rewritten from the seed it was given.
        restored = (
            restore_readonly_baseline(
                self.output_dir, self.readonly_seed, self.readonly_baseline
            )
            if self.readonly_seed is not None
            and self.readonly_baseline is not None
            and self.output_dir.is_dir()
            else ()
        )
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
            # main.py and every sibling .py: one import/I-O rule set per file.
            fit_schedule = validate_strategy_package(main)
        except StrategyLoadError as exc:
            raise ToolError(str(exc)) from exc
        _reject_flat_asof_reads(files, self.output_dir)
        _reject_unseeded_randomness(files, self.output_dir)
        # The same scan the artifact store runs when a frozen artifact is
        # reloaded as a later session's start: run it here so a hardcoded stage
        # path is reported to the Agent before any formal replay.
        try:
            reject_forbidden_code_references(files)
        except ArtifactError as exc:
            raise ToolError(str(exc)) from exc
        try:
            delta = modification_delta(
                self.parent_dir or self.output_dir,
                self.output_dir,
                readonly_baseline=self.readonly_baseline,
            )
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
            # The read-only violations travel structurally: a caller that has
            # to tell this rejection class from the others (``batch_validate``
            # bounds a repeating one) must not parse the sentence, whose text
            # carries digests and therefore changes with the file's content.
            raise ToolError(
                "; ".join(reasons),
                error_type="artifact_constraint",
                details=(
                    {"readonly_violations": list(delta.readonly_violations)}
                    if delta.readonly_violations
                    else None
                ),
            )
        value: dict[str, object] = {
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
        }
        if restored:
            # Explicit, never silent: the Agent is told its tree changed under
            # it and why, so it does not go looking for the file it lost.
            value["restored_readonly"] = (
                f"{', '.join(restored)}: absent or altered, and the host rewrote it "
                "from the bytes this session was seeded with. It is the host's "
                "read-only contract copy; nothing else in your tree was touched."
            )
        return ToolResult(True, value=value)


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


# Random draws must be a function of what the strategy is shown: two replays of
# one span that place different orders are not reproducible evidence, and a
# shuffled control or random placebo is only a control if it is the same draw
# every time. ``random``, ``time``, ``os`` and ``hashlib`` are not importable,
# so what source can prove unseeded is numpy's hidden global state, a generator
# built without a seed (seeded from OS entropy), and a seed read off the wall
# clock. Anything less certain -- a seed variable that may be None, an
# estimator left at ``random_state=None`` -- is the contract's rule, not this
# check's.
# numpy.random names that build or seed an explicit generator instead of
# drawing from the global state; the second set takes the seed first.
_NUMPY_GENERATOR_NAMES = frozenset(
    {"default_rng", "Generator", "RandomState", "SeedSequence", "BitGenerator",
     "PCG64", "PCG64DXSM", "Philox", "MT19937", "SFC64", "seed"}
)
_NUMPY_SEEDED_NAMES = _NUMPY_GENERATOR_NAMES - {"Generator", "BitGenerator"}
# Keywords that carry a seed into a constructor or a library call.
_SEED_KEYWORDS = frozenset({"seed", "random_state", "entropy"})
_WALL_CLOCK_CALLS = frozenset({"now", "today", "utcnow"})
_SEEDED_DRAW_IDIOM = (
    "Seed every draw from the decision date: day = "
    "int(context.inference_at.strftime('%Y%m%d')), then rng = "
    "numpy.random.default_rng(day), or default_rng([day, 1]) with a fixed integer "
    "for each further independent stream; draw from rng (rng.permutation, "
    "rng.choice, rng.integers) and pass rng or one of its integers as "
    "random_state= to a library."
)


def _reject_unseeded_randomness(files: list[Path], root: Path) -> None:
    """Reject a random draw whose seed the source proves is not reproducible."""

    for path in files:
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            # validate_strategy_package has already reported what it could not read.
            continue
        finding = _unseeded_draw(tree)
        if finding is None:
            continue
        line, what = finding
        raise ToolError(
            f"{path.relative_to(root)}:{line} {what}, so two replays of the same "
            "span can place different orders and the result is not reproducible "
            f"evidence. {_SEEDED_DRAW_IDIOM}"
        )


def _unseeded_draw(tree: ast.Module) -> tuple[int, str] | None:
    """The first call in ``tree`` that draws or seeds irreproducibly, if any."""

    aliases = _numpy_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted(node.func, aliases)
        parts = name.split(".") if name else []
        if len(parts) == 3 and parts[:2] == ["numpy", "random"]:
            if parts[2] not in _NUMPY_GENERATOR_NAMES:
                return node.lineno, f"draws from numpy's global random state ({name})"
            if parts[2] in _NUMPY_SEEDED_NAMES and not _has_seed(node):
                return node.lineno, f"calls {name}() without a seed, which seeds it from OS entropy"
        # The seed a call is handed: a seed keyword anywhere, and the
        # positional arguments of a numpy generator or a torch manual_seed.
        seeds = [keyword.value for keyword in node.keywords if keyword.arg in _SEED_KEYWORDS]
        if parts[:2] == ["numpy", "random"] or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "manual_seed"
        ):
            seeds.extend(node.args)
        for seed in seeds:
            for inner in ast.walk(seed):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr in _WALL_CLOCK_CALLS
                ):
                    return node.lineno, (
                        f"seeds a random generator from the wall clock ({inner.func.attr}())"
                    )
    return None


def _has_seed(call: ast.Call) -> bool:
    """Whether a numpy generator constructor is given a seed that is not ``None``."""

    values = [*call.args[:1], *(k.value for k in call.keywords if k.arg in _SEED_KEYWORDS)]
    return any(
        not (isinstance(value, ast.Constant) and value.value is None) for value in values
    )


def _numpy_aliases(tree: ast.Module) -> dict[str, str]:
    """Local names bound to numpy, ``numpy.random`` or a name imported from it."""

    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy" or alias.name.startswith("numpy."):
                    if alias.asname:
                        aliases[alias.asname] = alias.name
                    else:
                        aliases["numpy"] = "numpy"
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
            and (node.module == "numpy" or node.module.startswith("numpy."))
        ):
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _dotted(node: ast.expr, aliases: Mapping[str, str]) -> str | None:
    """The numpy dotted name an expression refers to, or None."""

    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value, aliases)
        return f"{base}.{node.attr}" if base else None
    return None


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
