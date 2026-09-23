"""Every default this arm is expected to change, in one place.

Each assignment below appears exactly once in the whole package, so the first
`edit_file` on any of them matches one line. No other module and no docstring
restates a value from this file; they refer to these names only.

Two names live outside this file on purpose:

    CANDIDATE                   the leg this package is; it lives in `main.py`
    index.INDEX_CODE            the benchmark this book is built in and graded
                                against; it belongs beside the read that uses
                                it, and the run facts override it

The account size is not a knob: every seat is sized from the account the
decision sees (`context.account`), never from a constant.
"""

# --- score (registered in `refs/families.md`; the report fixed them in-sample) ---
# Market sessions the amplitude split looks back over, the share of a name's
# valid sessions (lowest amplitude first) whose returns are averaged, and the
# valid sessions a name needs inside the window to be scored at all.
WINDOW = 160
LOW_SHARE = 0.7
MIN_SESSIONS = 140

# --- book ----------------------------------------------------------------
SEATS = 50
# A holding ranked inside SEATS * KEEP_BAND is kept at a review.
KEEP_BAND = 2.0
# At most floor(INDUSTRY_SHARE * SEATS) names of one SW L1 industry, counting
# kept names; it binds only entries and never forces a sale.
INDUSTRY_SHARE = 0.25
CASH_BUFFER = 0.97
