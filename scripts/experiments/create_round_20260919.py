#!/usr/bin/env python
"""The 2026-09-19 round: books built inside a benchmark, on their own seed.

This round exists for its dataset selection. It is the 2026-09-20 selection
plus `index_weight` — the month-end constituents and weights of the seven core
indices, which the raw lake has carried and audited since 2020 and which no
snapshot could select until now.

Why the arms need it, in one reading (`logs/notes/review_20260920/A7_data_vs_diversity.md`):
a random zero-skill 15-name book drawn from the packs' own pool reads 15.3 %/yr
tracking error against CSI 300 plus size neutralization and does not fall below
11.3 %/yr at any basket size, so the graduation bar demands 12.7–14.6 %/yr of
neutralized excess against a measured ceiling near 8.5. The same random book
drawn from the actual CSI 300 constituents reads 8.9 / 6.9 / 5.8 % at 15 / 30 /
50 names, which puts the bar inside the measured alpha band for the first time.
The lever is the universe the book is built in, never the benchmark constant.

The arms follow in their own reference packs. A round file without arms still
`--dry-run`s: it validates this selection and the seed built for it, which is
what has to be true before a pack can be written against them.

Why a new round rather than an edit of the 2026-09-20 one: a round's dataset
selection IS the snapshot configuration its PIT view seed is built for, and the
four arms of that round hardlink a tree already built for it. Adding a dataset
there would change their contract and make their seed unusable, so the wider
selection gets its own round file and its own tree.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260919.py <port> --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

# Appended, not prepended: the repository root carries directories named `data`,
# `logs` and `results`, which must never shadow an installed package. The
# launcher pins this repo's `src/` itself.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from scripts.experiments._round import Round

# Three of the four domains are the 2026-09-20 round's, imported rather than
# restated so the two selections cannot drift apart by a typo.
from scripts.experiments.create_round_20260920 import (
    EVENTS_DATASETS,
    FUNDAMENTAL_DATASETS,
    TEXT_DATASETS,
)
from scripts.experiments.create_round_20260920 import (
    MACRO_DATASETS as _MACRO_20260920,
)

# The one prebuilt view tree this round's arms hardlink from, built by
# scripts/data/prebuild_pit_views_seed.py for exactly the selection below and
# the default research geometry; gitignored, so an operator precondition.
PIT_VIEWS_SEED = "data/pit_views_seed_research_20260919"

# `index_weight` is opt-in and lands in the macro domain: one row per (index,
# constituent, month-end trade date), stamped 17:30 of that trade date, so a
# decision day reads the latest cross-section dated at or before it and a
# review inside a replay rolls in at the next pre-open.
MACRO_DATASETS = [*_MACRO_20260920, "index_weight"]

# No arms yet: the benchmark-relative packs are written separately. `--dry-run`
# still reads the shared parameters and this round's seed contract.
ARMS: dict[str, dict[str, object]] = {}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=PIT_VIEWS_SEED,
    overrides={
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
