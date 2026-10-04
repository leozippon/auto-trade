"""Entry module: the industry-cycle book; no fit.

`lib/knobs.py` holds every switch; `refs/README.md` is the authority for which
values may run and in which batch. With every switch at its first value this
package is the candidate `G`: on the first decision of each month, Shenwan L1
industries are ranked by the median acceleration of their members' reported
revenue growth and the median change of their single-quarter core return on
equity (`lib/industry.py`), and the canonical book (`lib/trade.py`, a byte
copy of configs/starter_lib) holds the four largest affordable names of each
of the three first industries (`lib/score.py`).
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
