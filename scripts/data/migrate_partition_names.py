#!/usr/bin/env python3
"""Merge legacy raw-lake partition directories into the canonical ones.

Until 2026-09 the partition-name builder sanitised vendor values with a regex
and fell back to a hex digest, so `热股` landed in `market=e783ade882a1` and
`A股市场` was truncated to `market=A`. The lake's own history uses the
percent-encoded names, so `common.safe_partition_value` is now percent-encoding
and this one-shot script folds the legacy trees into the canonical ones.

Dry-run by default: it prints, per affected partition, the file and row counts
on both sides (parquet footers only, no data is read) and any leaf that exists
in both trees. `--apply` moves every legacy file into the canonical tree and
removes the emptied directories; it refuses the whole migration if any leaf
collides, because merging two files that claim the same partition would need a
row-level decision this script must not make. Nothing is ever deleted or
truncated: history only changes directory.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

import pyarrow.parquet as pq

from autotrade.data_sources.tushare.common import (
    BOARD_DC_HOT_MARKETS,
    BOARD_DC_HOT_TYPES,
    BOARD_KPL_TAGS,
    BOARD_THS_HOT_MARKETS,
    BOARD_THS_LIMIT_TYPES,
    safe_partition_value,
)

# Every partition key whose values are vendor labels rather than dates or
# codes. Date/code partitions are ASCII and unchanged by the new rule.
PARTITIONED_VALUES: tuple[tuple[str, str, list[str]], ...] = (
    ("ths_hot", "market", BOARD_THS_HOT_MARKETS),
    ("dc_hot", "market", BOARD_DC_HOT_MARKETS),
    ("dc_hot", "hot_type", BOARD_DC_HOT_TYPES),
    ("kpl_list", "tag", BOARD_KPL_TAGS),
    ("limit_list_ths", "limit_type", BOARD_THS_LIMIT_TYPES),
)


def legacy_partition_value(value: str) -> str:
    """The pre-2026-09 name builder, kept only here.

    The migration has to locate directories that no production code path can
    name any more; reproducing the old rule inside the one-shot script keeps
    `common.safe_partition_value` the single authority for live writes.
    """
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value).strip())
    if cleaned.strip("_"):
        return cleaned.strip("_")
    encoded = str(value).encode("utf-8").hex()
    return encoded[:96] or "empty"


def rename_map() -> dict[str, dict[str, str]]:
    """dataset -> {legacy directory name: canonical directory name}."""
    mapping: dict[str, dict[str, str]] = {}
    for dataset, key, values in PARTITIONED_VALUES:
        for value in values:
            legacy = f"{key}={legacy_partition_value(value)}"
            canonical = f"{key}={safe_partition_value(value)}"
            if legacy != canonical:
                mapping.setdefault(dataset, {})[legacy] = canonical
    return mapping


def rows_in(path: Path) -> int:
    return int(pq.read_metadata(path).num_rows)


def canonical_path(path: Path, dataset_dir: Path, names: dict[str, str]) -> Path:
    parts = [names.get(part, part) for part in path.relative_to(dataset_dir).parts]
    return dataset_dir.joinpath(*parts)


def plan_dataset(dataset_dir: Path, names: dict[str, str]) -> list[dict[str, object]]:
    """One entry per legacy leaf directory, with its canonical counterpart."""
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(dataset_dir).parts
        if not any(part in names for part in relative):
            continue
        target = canonical_path(path, dataset_dir, names)
        key = (str(path.parent.relative_to(dataset_dir)), str(target.parent.relative_to(dataset_dir)))
        group = groups.setdefault(
            key,
            {"source": key[0], "target": key[1], "moves": [], "legacy_rows": 0, "collisions": []},
        )
        group["moves"].append((path, target))  # type: ignore[union-attr]
        if path.suffix == ".parquet":
            group["legacy_rows"] = int(group["legacy_rows"]) + rows_in(path)  # type: ignore[arg-type]
        if target.exists():
            group["collisions"].append(target)  # type: ignore[union-attr]
    for group in groups.values():
        target_dir = dataset_dir / str(group["target"])
        group["canonical_rows"] = sum(
            rows_in(path) for path in sorted(target_dir.glob("*.parquet"))
        ) if target_dir.exists() else 0
    return [groups[key] for key in sorted(groups)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--apply", action="store_true", help="move the files (default: report only)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"missing raw dir: {raw_dir}")

    plans: dict[str, list[dict[str, object]]] = {}
    for dataset, names in sorted(rename_map().items()):
        dataset_dir = raw_dir / dataset
        if not dataset_dir.is_dir():
            continue
        groups = plan_dataset(dataset_dir, names)
        if groups:
            plans[dataset] = groups

    if not plans:
        print(f"{raw_dir}: no legacy partition directories; nothing to migrate")
        return 0

    total_files = 0
    total_rows = 0
    collisions = 0
    for dataset, groups in plans.items():
        print(f"== {dataset}")
        for group in groups:
            moves = group["moves"]  # type: ignore[index]
            total_files += len(moves)  # type: ignore[arg-type]
            total_rows += int(group["legacy_rows"])
            collisions += len(group["collisions"])  # type: ignore[arg-type]
            print(
                f"   {group['source']} -> {group['target']}\n"
                f"      legacy files={len(moves)} rows={group['legacy_rows']}"  # type: ignore[arg-type]
                f" | canonical rows={group['canonical_rows']}"
                f" | colliding leaves={len(group['collisions'])}"  # type: ignore[arg-type]
            )
            for target in group["collisions"][:10]:  # type: ignore[index]
                print(f"      COLLISION {target.relative_to(raw_dir)}")
    print(f"-- total legacy files={total_files} rows={total_rows} colliding leaves={collisions}")

    if not args.apply:
        print("dry-run: re-run with --apply to move the files")
        return 1 if collisions else 0
    if collisions:
        raise SystemExit("refusing to migrate: colliding leaves need a row-level decision")

    moved = 0
    for dataset, groups in plans.items():
        for group in groups:
            for source, target in group["moves"]:  # type: ignore[index]
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise SystemExit(f"refusing to overwrite {target}")
                source.rename(target)
                moved += 1
    for dataset in plans:
        for path in sorted((raw_dir / dataset).rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()

    print(f"moved {moved} files")
    for dataset, groups in plans.items():
        after = sum(rows_in(path) for path in (raw_dir / dataset).rglob("*.parquet"))
        print(f"   {dataset}: rows after migration = {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
