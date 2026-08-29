#!/usr/bin/env python3
"""Render the unit registry to docs/units-reference.md; refresh the schema inventory.

``FIELD_RULES`` in src/autotrade/environment/data/unit_rules.py defines the unit
rules. This exporter renders the reference document; a freshness test regenerates
and compares it with the committed file.

``--refresh-inventory`` rescans the raw lake and rewrites
configs/data/snapshot_columns.json (the committed per-dataset column
inventory that tests resolve against). Run it when vendor schemas change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.environment.data.snapshot import (
    DEFAULT_DATASETS,
    SELECTABLE_DATASETS,
    SNAPSHOT_EXCLUDED_COLUMNS,
)
from autotrade.environment.data.units import (
    COMMON_FIELD_SEMANTICS,
    FIELD_RULES,
    NO_NUMERIC_DATASETS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "docs" / "units-reference.md"
INVENTORY_PATH = REPO_ROOT / "configs" / "data" / "snapshot_columns.json"

FILE_TITLES = {
    "daily.parquet": "daily.parquet（日频归一化文件）",
    "intraday_1min.parquet": "intraday_1min.parquet（历史分钟线）",
    "auction.parquet": "auction.parquet（开盘竞价）",
    "corporate_actions.parquet": "corporate_actions.parquet（回放分红送转）",
    "events.parquet": "events.parquet（事件/资金/打板多来源合并文件）",
    "macro.parquet": "macro.parquet（宏观与跨资产多来源合并文件）",
    "fundamentals.parquet": "fundamentals.parquet（财务多来源合并文件）",
    "raw_only": "仅原始湖数据（不进入快照）",
}

# Snapshot visibility is derived, never written by hand: a union file's domain
# decides whether a dataset can enter a snapshot at all and whether it is in the
# default scope, and SNAPSHOT_EXCLUDED_COLUMNS decides which of its columns do.
_FILE_DOMAINS = {
    "events.parquet": "events",
    "macro.parquet": "macro",
    "fundamentals.parquet": "fundamentals",
}
_DATASET_NOT_IN_SNAPSHOT = "该数据集不进入快照，只保留在原始湖与数据审计中"
_DATASET_NOT_DEFAULT = "该数据集默认不加载，可在创建实验时通过数据集子集显式选择"
_COLUMNS_NOT_IN_SNAPSHOT = "这些字段不进入快照，只保留在原始湖与数据审计中"

DOC_REFERENCE_LINKS = {
    "data docs §3.3": "[数据文档 §3.3](data-documentation.md#33-原始数据何时可见)",
    "data docs §4": "[数据文档 §4](data-documentation.md#4-已知数据风险与限制)",
}


def _cell(text: str) -> str:
    return text.replace("|", "\\|") if text else "—"


def _link_doc_references(text: str) -> str:
    for source, link in DOC_REFERENCE_LINKS.items():
        text = text.replace(source, link)
    return text


def _visibility_note(rule) -> str | None:
    """Whether this rule's dataset/columns reach the Agent snapshot."""
    domain = _FILE_DOMAINS.get(rule.file)
    if domain is None or rule.dataset is None:
        return None
    if rule.dataset not in SELECTABLE_DATASETS[domain]:
        return _DATASET_NOT_IN_SNAPSHOT
    if rule.dataset not in DEFAULT_DATASETS[domain]:
        return _DATASET_NOT_DEFAULT
    excluded = set(SNAPSHOT_EXCLUDED_COLUMNS.get(rule.dataset, ()))
    covered = excluded.intersection(rule.columns)
    if not covered:
        return None
    if covered != set(rule.columns):
        raise ValueError(
            f"unit rule {rule.key()} mixes snapshot-excluded and visible columns; "
            "split it so each rule is uniformly one or the other"
        )
    return _COLUMNS_NOT_IN_SNAPSHOT


def render_units_markdown() -> str:
    lines = [
        "# 单位参考表",
        "",
        (
            "本文档由 `scripts/dev/export_units.py` 从 `src/autotrade/environment/data/unit_rules.py` 的 "
            "`FIELD_RULES` 生成，禁止手工编辑。回归测试会重新生成并与本文件比较。"
        ),
        "",
        "单位怎样查找、哪些字段可以换算，见 [数据文档 §1.2](data-documentation.md#12-原始单位)。",
        "",
        "## 怎样阅读本表",
        "",
        "- 本表按列名或通配符列出注册表规则，不是某次快照的实际字段清单。",
        "- 每次快照还会生成 `/mnt/artifacts/unit_reference.json`，只列出本次实际可见的文件、数据集和字段。",
        "- 快照构建会逐列检查单位，并核对多来源合并清单与文件结构；缺少规则或字段归属时立即失败。",
        "- 回归测试会逐列检查 `configs/data/snapshot_columns.json`。该文件从供应商分区抽样汇总，不扫描全库；少数历史分区独有的字段由快照构建检查。",
        "- 注明“不进入快照”的数据集或字段只保留在原始湖与数据审计中，Agent 看不到，也不会出现在 `unit_reference.json` 里。",
        "- 注明“默认不加载”的数据集仍可被实验显式选择；选中后它的字段与规则同样生效。",
        "",
        "## 状态与换算",
        "",
        "- `verified`：已与另一数据源或已知外部事实核对，依据见“依据/说明”列。",
        "- `official`：单位来自供应商官方字段说明；“依据/说明”列可能补充本地检查结果。",
        "- `inferred`：只根据本地数值量级推断。",
        "- `unknown`：单位尚未确认。此类字段只能在所属数据集内用于排序、分位数等不依赖单位的运算；用于绝对阈值、换算或跨数据集计算前必须先核实。",
        "- `factor`：快照载入时的乘数；已经归一化的文件保存换算后的值。",
    ]
    files = list(dict.fromkeys(rule.file for rule in FIELD_RULES))
    for file in files:
        lines += ["", f"## {FILE_TITLES[file]}", ""]
        lines.append("| 数据集 (`dataset`) | 列（名/通配符） | 类型 | 源单位 | 换算 (`factor`) | 状态 | 依据/说明 |")
        lines.append("|---|---|---|---|---|---|---|")
        for rule in FIELD_RULES:
            if rule.file != file:
                continue
            factor = f"×{rule.factor:g} → {rule.normalized_unit}" if rule.factor is not None else ""
            basis = _link_doc_references(
                "；".join(
                    part
                    for part in (_visibility_note(rule), rule.evidence, rule.note)
                    if part
                )
            )
            lines.append(
                "| " + " | ".join([
                    _cell(f"`{rule.dataset}`" if rule.dataset else ""),
                    _cell("/".join(rule.columns)),
                    rule.semantic,
                    _cell(rule.source_unit or ""),
                    _cell(factor),
                    rule.status,
                    _cell(basis),
                ]) + " |"
            )
    lines += [
        "",
        "## 无数值字段的数据集",
        "",
        "以下数据集的字段都是标识、日期或文本，由通用字段分类规则解析，不需要单位规则：",
        "`" + "`、`".join(sorted(NO_NUMERIC_DATASETS)) + "`。",
        "",
        "## 通用字段分类（按顺序采用第一个匹配；数据集规则优先）",
        "",
        "| 模式 | 含义 |",
        "|---|---|",
    ]
    for pattern, semantic in COMMON_FIELD_SEMANTICS:
        lines.append(f"| `{pattern}` | {semantic} |")
    lines.append("")
    return "\n".join(lines)


def refresh_inventory() -> None:
    import pyarrow.parquet as pq

    from autotrade.environment.data.fundamental_events import (
        FUNDAMENTAL_SIDECAR_COLUMNS,
    )

    raw = REPO_ROOT / "data" / "raw"
    if not raw.exists():
        raise FileNotFoundError(f"raw lake not available at {raw}; run on the data host")
    # Every SELECTABLE dataset, not just the default scope: the unit registry is
    # fail-closed, so a dataset an experiment can opt into must stay covered.
    plans = [
        ("events.parquet", SELECTABLE_DATASETS["events"], ["dataset"]),
        ("macro.parquet", SELECTABLE_DATASETS["macro"], ["dataset"]),
        (
            "fundamentals.parquet",
            SELECTABLE_DATASETS["fundamentals"],
            list(FUNDAMENTAL_SIDECAR_COLUMNS),
        ),
    ]
    files: dict[str, dict[str, list[str]]] = {}
    for file, datasets, extra in plans:
        for dataset in datasets:
            parquets = sorted((raw / dataset).rglob("*.parquet"))
            if not parquets:
                raise FileNotFoundError(f"no raw parquet partitions for dataset {dataset}")
            indexes = sorted({0, len(parquets) // 4, len(parquets) // 2, 3 * len(parquets) // 4, len(parquets) - 1})
            columns: set[str] = set(extra)
            for index in indexes:
                columns.update(pq.read_schema(parquets[index]).names)
            files.setdefault(file, {})[dataset] = sorted(columns)
    inventory = {
        "note": (
            "SAMPLED union of raw vendor parquet schemas per snapshot dataset (up to five spread "
            "partitions each, not a full-lake scan), plus snapshot-builder provenance columns. "
            "Regenerate with scripts/dev/export_units.py --refresh-inventory. Tests resolve every "
            "listed column against the unit registry; columns appearing only in unsampled historical "
            "partitions are caught by the snapshot build's live full-column validation."
        ),
        "files": files,
    }
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_PATH.write_text(json.dumps(inventory, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {INVENTORY_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the committed document is stale")
    parser.add_argument("--refresh-inventory", action="store_true",
                        help="rescan the raw lake and rewrite configs/data/snapshot_columns.json")
    args = parser.parse_args()
    if args.refresh_inventory:
        refresh_inventory()
    rendered = render_units_markdown()
    if args.check:
        if not DOC_PATH.exists() or DOC_PATH.read_text(encoding="utf-8") != rendered:
            print("docs/units-reference.md is stale; run scripts/dev/export_units.py", file=sys.stderr)
            return 1
        return 0
    DOC_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
