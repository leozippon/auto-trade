"""Recalibrate the freeze gate's deflated Sharpe (controls, null dispersion,
effective trials, declared offline screens) on the record, replay-free.

Reads every arm's ``experiments/<arm>/ledgers/experiment_ledger.jsonl``, its
step tree's candidate names and hypotheses, and the ``style_analysis.json``
beside each recorded validation. Nothing is replayed; CPU only.

A. Every frozen or nominated candidate re-graded: the DSR it was recorded
   with (and the threshold of its era), the former rule (N = every validated
   revision, sqrt(V) = sample s.d. of the arm's full-span IRs) recomputed,
   the current rule as the code reads these ledgers (no control flagged, no
   offline screen declared), the current rule with the legs the arm's own
   hypotheses name as controls flagged, and the offline screens K a nominee
   could have declared before its DSR fell below the current threshold.
B. The IR the current threshold asks for at N_eff 1/2/3/5 over 4/6/8 years.
C. Per arm, its best non-control full-span candidate under both rules.
D. Zero skill and power through the real gate: each arm's own full-span
   graded series with every trial's neutralised intercept removed,
   block-bootstrapped jointly (20-day blocks, same dates for every trial, so
   the arm's cross-trial correlation, volatility and tails are kept); the
   best non-control trial is nominated, then again with a true IR of 1.0 in
   one non-control trial (power). The current rule is also read at stricter
   thresholds. Joint = freeze x the forward-test zero-skill pass rate
   measured in e17 (0.156), windows being disjoint.

Usage: python scripts/dev/dsr_recalibration.py [draws_per_arm] [processes] > out.md
"""

from __future__ import annotations

import glob
import json
import math
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from statistics import NormalDist

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

import numpy as np

from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR
from autotrade.pipelines import verdict
from autotrade.pipelines.calendar import FULL_SPAN
from autotrade.pipelines.config import AcceptanceRules
from autotrade.pipelines.experiment import freeze_gate_for, neutralized

# The former rule is read at the default it ran under; the current rule at
# today's default (raised to 0.975 by this recalibration) unless a column
# names another.
FORMER_THRESHOLD = 0.90
THRESHOLD = verdict.FREEZE_MIN_DSR_PROBABILITY
FORWARD_ZERO_SKILL = 0.156  # e17 loo_real K120/panel20: 0.158 and 0.154
E17_TRIAL_SD = float(np.std([0.60, 0.70, 0.80, 0.90], ddof=1))  # e17's fixed trial spread
BLOCK = verdict.BOOTSTRAP_BLOCK_DAYS
DSR_REASONS = {"freeze_deflated_sharpe_below_threshold", "freeze_deflated_sharpe_unavailable"}
# A leg the arm registered as a comparison: named c_*, or a hypothesis that
# opens by calling itself a control or baseline. Printed per arm for audit.
CONTROL_MARKERS = ("对照", "基线", "安慰剂", "control", "baseline", "placebo")


def is_control(name: str, hypothesis: str) -> bool:
    head = hypothesis[:24].lower()
    return name.startswith("c_") or any(marker in head for marker in CONTROL_MARKERS)


def arm_rows(arm: Path) -> tuple[list[dict], dict, list[dict]]:
    ledger = (arm / "ledgers/experiment_ledger.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in ledger if line.strip()]
    research = [r for r in records if r.get("record_type") == "research_session"]
    rows = [dict(row) for record in research for row in record.get("steps") or ()]
    meta: dict = {}
    tree = arm / "steps/tree.json"
    if tree.is_file():
        nodes = json.loads(tree.read_text()).get("nodes") or []
        for node in nodes.values() if isinstance(nodes, dict) else nodes:
            meta[node.get("node_id")] = node.get("metadata") or {}
    for row in rows:
        m = meta.get(row["step_id"], {})
        row["_name"] = str(m.get("candidate") or "")
        row["_hypothesis"] = str(m.get("hypothesis") or "")
    return research, meta, rows


def years_of(arm: Path) -> list[tuple[str, str]]:
    params = json.loads((arm / "hitl/params.json").read_text())
    start, end = str(params.get("research_start") or "20210701"), str(params.get("research_end") or "20250630")
    return [(f"{y}0701", f"{y + 1}0630") for y in range(int(start[:4]), int(end[:4]))]


def style(row: dict) -> dict:
    return json.loads((Path(row["validation_result_ref"]).parent / "style_analysis.json").read_text())


def old_rule(nominee: dict, rows: list[dict]) -> dict | None:
    """The former DSR: N every validated revision, sqrt(V) the s.d. of the
    arm's full-span IRs (recomputed from the sidecars as graded today)."""

    trials = len({row["revision_id"] for row in rows})
    irs = []
    for row in rows:
        if row.get("span") != FULL_SPAN:
            continue
        stats = neutralized(row["validation_result_ref"])
        if stats and stats.get("information_ratio") is not None:
            irs.append(float(stats["information_ratio"]))
    if trials < 2 or len(irs) < 2:
        return None
    graded, _ = verdict._graded(style(nominee))
    statistics, _rows, neutral = verdict._measured(graded, "", "")
    block = verdict.deflated_sharpe(
        observed_sharpe=statistics["information_ratio"],
        effective_trials=trials,
        trial_sharpe_std=float(np.std(irs, ddof=1)),
        returns=neutral,
    )
    block["trials"] = trials
    return block


def new_rule(nominee: dict, rows: list[dict], rules: AcceptanceRules, years, *, flag: bool, offline: int = 0) -> dict:
    flagged = []
    for row in rows:
        item = {k: v for k, v in row.items() if not k.startswith("_")}
        item["control"] = flag and is_control(row["_name"], row["_hypothesis"])
        if offline:
            item["offline_trials"], item["batch_id"] = offline, "declared"
        flagged.append(item)
    nominee_row = next(item for item in flagged if item["step_id"] == nominee["step_id"])
    nominee_row["control"] = False  # a nominated leg is by definition not a control
    rules = replace(rules, min_dsr_probability=THRESHOLD)
    return freeze_gate_for([], flagged, nominee_row, hard_reasons=rules.evaluate(dict(nominee["summary"])), acceptance=rules, years=years)


def fmt(value, digits=3):
    return "–" if value is None else f"{value:.{digits}f}"


def part_a() -> list[str]:
    out = ["## A. Frozen and nominated candidates, re-graded", "",
           "| arm | forward | IR | recorded DSR (era threshold) | former rule: N, √V, DSR | current rule as read (no flags): M, ρ̄, N_eff, bar, DSR, gate | with named controls flagged: controls, M, ρ̄, N_eff, bar, DSR, gate | offline K that flips DSR |",
           "|---|---|---|---|---|---|---|---|"]
    for ledger in sorted(glob.glob("experiments/*/ledgers/experiment_ledger.jsonl")):
        arm = Path(ledger).parents[1]
        research, _meta, rows = arm_rows(arm)
        lines = Path(ledger).read_text().splitlines()
        forward = [r for r in map(json.loads, filter(str.strip, lines)) if r.get("record_type") == "forward"]
        for record in research:
            if record.get("outcome") != "freeze":
                continue
            nominee = next(row for row in rows if row["step_id"] == record["nominated_step_id"])
            recorded = (record.get("freeze_gate") or {}).get("deflated_sharpe") or {}
            era = ((record.get("freeze_gate") or {}).get("thresholds") or {}).get("min_deflated_sharpe_probability")
            rules = AcceptanceRules.from_record(record.get("acceptance_rules") or {})
            years = years_of(arm)
            old = old_rule(nominee, rows)
            plain = new_rule(nominee, rows, rules, years, flag=False)
            named = new_rule(nominee, rows, rules, years, flag=True)
            flips = None
            if named.get("passed"):
                for k in range(1, 201):
                    if new_rule(nominee, rows, rules, years, flag=True, offline=k)["deflated_sharpe"]["deflated_sharpe_probability"] < THRESHOLD:
                        flips = k
                        break
            d0, d1 = plain["deflated_sharpe"], named["deflated_sharpe"]
            controls = [row["_name"] for row in rows if is_control(row["_name"], row["_hypothesis"])]
            out.append(
                f"| `{arm.name}` | {(forward[-1].get('verdict') or {}).get('status') if forward else '–'} | {fmt(plain.get('information_ratio'))} "
                f"| {fmt(recorded.get('deflated_sharpe_probability'))} ({era if era is not None else 0.9}) "
                f"| {old['trials'] if old else '–'}, {fmt(old and old['trial_sharpe_std'])}, {fmt(old and old['deflated_sharpe_probability'])} "
                f"| {d0['trials']}, {fmt(d0['trial_correlation'], 2)}, {fmt(d0['effective_trials'], 2)}, {fmt(d0['information_ratio_bar'], 2)}, {fmt(d0['deflated_sharpe_probability'])}, {'pass' if plain['passed'] else 'fail'} "
                f"| {len(controls)} ({', '.join(controls) or 'none'}), {d1['trials']}, {fmt(d1['trial_correlation'], 2)}, {fmt(d1['effective_trials'], 2)}, {fmt(d1['information_ratio_bar'], 2)}, {fmt(d1['deflated_sharpe_probability'])}, {'pass' if named['passed'] else 'fail: ' + ','.join(r.removeprefix('freeze_') for r in named['reasons'])} "
                f"| {flips if flips is not None else ('–' if not named.get('passed') else '>200')} |"
            )
    return out


def part_b() -> list[str]:
    z = NormalDist().inv_cdf(THRESHOLD)
    out = [f"## B. IR the {THRESHOLD} threshold asks for (normal returns)", "", "| N_eff | 4 years | 6 years | 8 years |", "|---|---|---|---|"]
    for n in (1, 2, 3, 5, 10):
        bars = [verdict.null_sharpe_std(years * TRADING_DAYS_PER_YEAR) * (verdict.expected_max_sharpe(n) + z) for years in (4, 6, 8)]
        out.append(f"| {n} | " + " | ".join(f"{bar:.2f}" for bar in bars) + " |")
    return out


def full_span_trials(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Measurable full-span rows, one per revision: (non-controls, controls)."""

    seen, keep, controls = set(), [], []
    for row in rows:
        if row.get("span") != FULL_SPAN or row["revision_id"] in seen:
            continue
        stats = neutralized(row["validation_result_ref"])
        if not stats or stats.get("information_ratio") is None:
            continue
        seen.add(row["revision_id"])
        (controls if is_control(row["_name"], row["_hypothesis"]) else keep).append(row)
    return keep, controls


def part_c() -> list[str]:
    out = ["## C. Each arm's best non-control full-span candidate", "",
           "| arm | trials (non-control / controls, any span) | best IR | former: N, √V, DSR | current, controls flagged: M, ρ̄, N_eff, bar, DSR | DSR ≥ threshold, former @0.90 → current |",
           "|---|---|---|---|---|---|"]
    for ledger in sorted(glob.glob("experiments/*/ledgers/experiment_ledger.jsonl")):
        arm = Path(ledger).parents[1]
        research, _meta, rows = arm_rows(arm)
        if not rows:
            continue
        keep, _controls = full_span_trials(rows)
        if not keep:
            continue
        best = max(keep, key=lambda row: neutralized(row["validation_result_ref"])["information_ratio"])
        rules = AcceptanceRules.from_record(research[-1].get("acceptance_rules") or {})
        old = old_rule(best, rows)
        new = new_rule(best, rows, rules, years_of(arm), flag=True)
        dsr = new.get("deflated_sharpe")
        if dsr is None:
            continue
        control_revisions = {row["revision_id"] for row in rows if is_control(row["_name"], row["_hypothesis"])}
        before = old is not None and old["deflated_sharpe_probability"] is not None and old["deflated_sharpe_probability"] >= FORMER_THRESHOLD
        after = dsr["deflated_sharpe_probability"] is not None and dsr["deflated_sharpe_probability"] >= THRESHOLD
        out.append(
            f"| `{arm.name}` | {len({r['revision_id'] for r in rows}) - len(control_revisions)} / {len(control_revisions)} | {fmt(new['information_ratio'])} "
            f"| {old['trials'] if old else '–'}, {fmt(old and old['trial_sharpe_std'])}, {fmt(old and old['deflated_sharpe_probability'])} "
            f"| {dsr['trials']}, {fmt(dsr['trial_correlation'], 2)}, {fmt(dsr['effective_trials'], 2)}, {fmt(dsr['information_ratio_bar'], 2)}, {fmt(dsr['deflated_sharpe_probability'])} "
            f"| {'pass' if before else 'fail'} → {'pass' if after else 'fail'}{' **flip**' if before != after else ''} |"
        )
    return out


def _matrix(rows: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Common dates, zero-intercept graded series (T x K), benchmark, size."""

    graded = [verdict._graded(style(row))[0] for row in rows]
    joins = [{d: (v, b, s) for d, v, b, s in verdict._regression_join(g, "", "")} for g in graded]
    dates = sorted(set.intersection(*(set(j) for j in joins)))
    y = np.array([[j[d][0] for j in joins] for d in dates])
    bench = np.array([joins[0][d][1] for d in dates])
    size = np.array([joins[0][d][2] for d in dates])
    design = np.column_stack([np.ones(len(dates)), bench, size])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return dates, y - coefficients[0], bench, size


def _ir(y: np.ndarray, bench: np.ndarray, size: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(bench)), bench, size])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    te = np.sqrt((residual**2).sum(axis=0) / (len(bench) - 3) * TRADING_DAYS_PER_YEAR)
    return coefficients[0] * TRADING_DAYS_PER_YEAR / te


def _analysis(dates, values, bench, size) -> dict:
    return {
        "strategy_daily": [[d, float(v)] for d, v in zip(dates, values)],
        "benchmark_daily": [[d, float(v)] for d, v in zip(dates, bench)],
        "size_factor_daily": [[d, float(v)] for d, v in zip(dates, size)],
    }


def _dsr(analysis: dict, *, trials: float, dispersion: float) -> float:
    graded, _ = verdict._graded(analysis)
    statistics, _rows, neutral = verdict._measured(graded, "", "")
    block = verdict.deflated_sharpe(
        observed_sharpe=statistics["information_ratio"],
        effective_trials=trials,
        trial_sharpe_std=dispersion,
        returns=neutral,
    )
    return block["deflated_sharpe_probability"] or 0.0


# The current rule's rates are also read at stricter DSR thresholds, to show
# what restores the former zero-skill rate and what that costs in power.
CURRENT_THRESHOLDS = (0.90, 0.95, 0.975)
RATE_KEYS = (
    "old", "old_nc", "old1", "oldp", "olda",
    *(f"{kind}@{t}" for kind in ("new", "new1", "newp", "newa") for t in CURRENT_THRESHOLDS),
)


def _arm_rates(task: tuple[str, int, int]) -> dict | None:
    """Zero-skill and power rates of one arm (see the module docstring, D)."""

    arm_name, draws, seed = task
    arm = Path("experiments") / arm_name
    rng = np.random.default_rng(seed)
    _research, _meta, rows = arm_rows(arm)
    keep, controls = full_span_trials(rows)
    if not keep or len(keep) + len(controls) < 2:
        return None
    family = keep + controls
    dates, y, bench, size = _matrix(family)
    if len(dates) < 900:
        return None
    years = years_of(arm)
    k_new, k_all = len(keep), len(family)
    rho, _pairs = verdict.trial_correlation(
        [_analysis(dates, y[:, k], bench, size) for k in range(k_new)]
    )
    n_eff = verdict.effective_trials(k_new, rho)
    te0 = float(np.std(y[:, 0], ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    kwargs = {"active_max_drawdown": AcceptanceRules().active_max_drawdown}
    counts = {key: 0 for key in RATE_KEYS}
    for _ in range(draws):
        starts = rng.integers(0, len(dates) - BLOCK + 1, size=-(-len(dates) // BLOCK))
        idx = (starts[:, None] + np.arange(BLOCK)).reshape(-1)[: len(dates)]
        yb, bb, sb = y[idx], bench[idx], size[idx]
        # Zero skill, then the same draw with a true IR of 1.0 in the first
        # non-control trial: the arm-level selection the Agent makes.
        for arm_drift, (old_key, new_key) in ((0.0, ("old", "new")), (te0 / TRADING_DAYS_PER_YEAR, ("olda", "newa"))):
            ya = yb.copy()
            ya[:, 0] += arm_drift
            irs = _ir(ya, bb, sb)
            best = int(np.argmax(irs[:k_new]))
            nominee = _analysis(dates, ya[:, best], bb, sb)
            gate = verdict.freeze_gate(nominee, trials=k_new, full_span_validations=k_all, years=years, **kwargs)
            if [r for r in gate["reasons"] if r not in DSR_REASONS]:
                continue
            null = gate["deflated_sharpe"]["trial_sharpe_std"]
            current = _dsr(nominee, trials=n_eff, dispersion=null)
            for t in CURRENT_THRESHOLDS:
                counts[f"{new_key}@{t}"] += current >= t
            counts[old_key] += _dsr(nominee, trials=k_all, dispersion=float(np.std(irs, ddof=1))) >= FORMER_THRESHOLD
            if k_new >= 2 and old_key == "old":
                counts["old_nc"] += _dsr(nominee, trials=k_new, dispersion=float(np.std(irs[:k_new], ddof=1))) >= FORMER_THRESHOLD
        for drift, keys in ((0.0, ("old1", "new1")), (te0 / TRADING_DAYS_PER_YEAR, ("oldp", "newp"))):
            single = _analysis(dates, yb[:, 0] + drift, bb, sb)
            gate = verdict.freeze_gate(single, trials=4, full_span_validations=4, years=years, **kwargs)
            if not [r for r in gate["reasons"] if r not in DSR_REASONS]:
                current = gate["deflated_sharpe"]["deflated_sharpe_probability"] or 0.0
                for t in CURRENT_THRESHOLDS:
                    counts[f"{keys[1]}@{t}"] += current >= t
                counts[keys[0]] += _dsr(single, trials=4, dispersion=E17_TRIAL_SD) >= FORMER_THRESHOLD
    return {
        "arm": arm_name,
        "k_new": k_new,
        "controls": len(controls),
        "rho": rho,
        "n_eff": n_eff,
        **{key: value / draws for key, value in counts.items()},
    }


def part_d(draws: int, processes: int) -> tuple[list[str], dict]:
    arms = sorted(Path(p).parents[1].name for p in glob.glob("experiments/*/ledgers/experiment_ledger.jsonl"))
    tasks = [(arm, draws, 20260922 + index) for index, arm in enumerate(arms)]
    with Pool(processes) as pool:
        results = [r for r in pool.map(_arm_rates, tasks) if r is not None]
    out = ["## D. Zero skill and power through the real gate (bootstrap of each arm's own series)", "",
           ("Arm-level columns nominate the best non-control full-span trial of each draw (the selection the Agent makes). "
           "*former* deflates it over every full-span trial, controls included, at their sample IR spread (the rule until now); "
           "*former, no controls* does the same over the non-controls only (V1 alone); *current* deflates over the non-controls "
           "at N_eff from the arm's measured ρ̄ with √V = √(244/T) (V1 + V2; one ρ̄ per arm, which the joint bootstrap keeps). "
           "*single* nominates one fixed trial without selection, as e17 did: former rule at N = 4 with e17's trial spread "
           f"s.d. {E17_TRIAL_SD:.3f}, current rule at N_eff = 4; power adds a true IR of 1.0 to that trial. Arm-level power "
           "adds a true IR of 1.0 to the first non-control trial and again nominates the best non-control. Every other gate "
           "condition (IR ≥ 0.75, 3 of 4 positive years, active drawdown ≤ 0.30, ≥ 2 full-span validations) is the gate's own "
           "and identical under all rules."), "",
           "| arm | non-control / controls | ρ̄ | N_eff | freeze former | former, no controls | current @0.90 / 0.95 / 0.975 | single zero: former; current @0.90 / 0.95 / 0.975 | single power IR 1.0: former; current @0.90 / 0.95 / 0.975 |",
           "|---|---|---|---|---|---|---|---|---|"]

    def trio(r, kind):
        return " / ".join(f"{r[f'{kind}@{t}']:.3f}" for t in CURRENT_THRESHOLDS)

    for r in results:
        out.append(
            f"| `{r['arm']}` | {r['k_new']} / {r['controls']} | {r['rho']:.2f} | {r['n_eff']:.2f} | {r['old']:.3f} | {r['old_nc']:.3f} "
            f"| {trio(r, 'new')} | {r['old1']:.3f}; {trio(r, 'new1')} | {r['oldp']:.3f}; {trio(r, 'newp')} |"
        )
    means = {key: float(np.mean([r[key] for r in results])) for key in RATE_KEYS}
    with_controls = [r for r in results if r["controls"]]
    without = [r for r in results if not r["controls"]]
    out += ["", f"Mean over {len(results)} arms ({draws} draws each; {len(with_controls)} with controls, {len(without)} without):", "",
            "| rate | former | former, no controls | current @0.90 | current @0.95 | current @0.975 |", "|---|---|---|---|---|---|"]
    for label, group in (("freeze with selection, all arms", results), ("  arms with controls", with_controls), ("  arms without controls", without)):
        if group:
            m = {key: float(np.mean([r[key] for r in group])) for key in RATE_KEYS}
            out.append(f"| {label} | {m['old']:.3f} | {m['old_nc']:.3f} | " + " | ".join(f"{m[f'new@{t}']:.3f}" for t in CURRENT_THRESHOLDS) + " |")
    out.append(f"| joint with forward (x {FORWARD_ZERO_SKILL}), all arms | {means['old'] * FORWARD_ZERO_SKILL:.4f} | {means['old_nc'] * FORWARD_ZERO_SKILL:.4f} | "
               + " | ".join(f"{means[f'new@{t}'] * FORWARD_ZERO_SKILL:.4f}" for t in CURRENT_THRESHOLDS) + " |")
    out.append(f"| arm-level power, true IR 1.0 in one non-control trial | {means['olda']:.3f} | – | " + " | ".join(f"{means[f'newa@{t}']:.3f}" for t in CURRENT_THRESHOLDS) + " |")
    out.append(f"| single book N = 4, zero skill | {means['old1']:.3f} | – | " + " | ".join(f"{means[f'new1@{t}']:.3f}" for t in CURRENT_THRESHOLDS) + " |")
    out.append(f"| single book N = 4, power at true IR 1.0 | {means['oldp']:.3f} | – | " + " | ".join(f"{means[f'newp@{t}']:.3f}" for t in CURRENT_THRESHOLDS) + " |")
    return out, means


def main() -> None:
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    started = time.time()
    lines = ["# DSR1 recalibration output", "", f"`python scripts/dev/dsr_recalibration.py {draws}`; former rule at {FORMER_THRESHOLD}, current rule at {THRESHOLD} unless a column names another.", ""]
    lines += part_a() + [""] + part_b() + [""] + part_c() + [""]
    d, _means = part_d(draws, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    lines += d + ["", f"Wall time {time.time() - started:.0f} s."]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
