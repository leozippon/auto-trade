"""Entry module: one book, sized by the account and shaped by one constant.

There is one leg, not three. The zero-skill floor is no longer something a pack
builds: the host draws it from the candidate's own filled trades after every
formal validation and grades the strategy on the difference, so a control leg
inside the artifact would only spend replay years re-measuring a number the
verdict already owns.

What the package does at every decision:

1. reads the account's equity from `context` (cash plus holdings at T-1 close);
2. derives its seat count from that equity by ONE rule (`trade.SEATS`,
   `trade.SEAT_CASH_MIN`, `trade.SEATS_MIN/MAX`) the session may change or pin;
3. builds the book `trade.BOOK` names -- `"pool"`, an equal-cash composite book
   of the affordable whole-market pool, or `"index"`, a benchmark-weight-tilted
   book of the as-of CSI 300 constituents;
4. sizes every buy to the NEAREST whole lot of its target money and reports what
   it actually realised.

`BOOK` defaults to `"pool"` and is a stated choice, never derived: the account
size and whether a tracking mandate is in force are two independent settings of
this arm, both in its run facts, and the strategy context carries neither. An
arm under a tracking mandate sets `BOOK = "index"`.

Nothing here is a thesis. The index book is a near-index tracker that earns ~0
against its own zero-skill panel by construction, and the composite score is a
family this repository has already closed once. Both are here so the first day
has a real book to read, and both are meant to be replaced.

Knobs: trade.BOOK / SEATS / SEAT_CASH_MIN / SEATS_MIN / SEATS_MAX / KEEP_BAND /
MAX_REPLACE / MIN_ADV, composite.LEGS.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
