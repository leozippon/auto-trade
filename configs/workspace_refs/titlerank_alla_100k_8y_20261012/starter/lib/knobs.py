"""Every switch of this package, in one place.

Each assignment appears exactly once in the package; no other module restates
a value from here. `refs/README.md` says which legs may run and in which
batch. With every switch at its first value this package is `nn`, the
character-level ranker on seed 1000.
"""

# --- the leg -----------------------------------------------------------------
# "nn": the character-level convolutional ranker (lib/models.py), trained on
# the card. "ridge": its twin, a ridge regression on hashed character
# n-grams, deterministic. "rule": the control c_rule, the incentive-plan
# draft rule of the incdraft lane in this book; nothing is trained.
MODEL = "nn"
# "true", or "shuffled": the control c_shuf, every label permuted across the
# names of its review before anything is fitted.
LABELS = "true"
# "titles", or "blind": the control c_blind, each unit's titles replaced by
# its size decile, board and number of documents (lib/ranker.py).
TEXT = "titles"
# The training seed: weight initialisation, batch order and the c_shuf
# permutation all derive from it. A seed replicate changes this line alone.
SEED = 1000

# --- the book, fixed --------------------------------------------------------
SEATS = 20
# Trading days from the review that bought a name to the first review that
# may sell it, and the horizon of the label.
HOLD = 40

ALLOWED = {
    "MODEL": ("nn", "ridge", "rule"),
    "LABELS": ("true", "shuffled"),
    "TEXT": ("titles", "blind"),
    "SEED": (1000, 2000, 3000, 4000),
}


def leg():
    """The leg name the orders carry, after checking every switch against its registered values."""

    for name, allowed in ALLOWED.items():
        value = globals()[name]
        if not any(type(value) is type(option) and value == option for option in allowed):
            raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    if LABELS == "shuffled" and TEXT == "blind":
        raise ValueError("one control at a time: LABELS = 'shuffled' and TEXT = 'blind' cannot both be set")
    if MODEL == "rule":
        if LABELS != "true" or TEXT != "titles" or SEED != 1000:
            raise ValueError("c_rule trains nothing: keep LABELS, TEXT and SEED at their first values")
        return "c_rule"
    name = "c_shuf" if LABELS == "shuffled" else "c_blind" if TEXT == "blind" else MODEL
    if MODEL == "ridge" and name != MODEL:
        name += "_r"
    return name if SEED == 1000 else f"{name}_s{SEED // 1000}"
