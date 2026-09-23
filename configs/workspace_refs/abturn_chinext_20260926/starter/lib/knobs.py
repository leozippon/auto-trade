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

# --- score (registered in `refs/families.md`, Liu-Stambaugh-Yuan's windows) ---
# Market sessions of the short (numerator) and long (denominator) mean
# turnover rate, and the valid sessions a name needs inside each.
SHORT = 20
LONG = 250
MIN_SHORT = 15
MIN_LONG = 200

# --- book ----------------------------------------------------------------
SEATS = 20
# A holding ranked inside SEATS * KEEP_BAND is kept at a review.
KEEP_BAND = 2.0
# At most floor(INDUSTRY_SHARE * SEATS) names of one SW L1 industry, counting
# kept names; it binds only entries and never forces a sale.
INDUSTRY_SHARE = 0.25
CASH_BUFFER = 0.97
