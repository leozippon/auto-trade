"""Entry: CSI 500 members ranked by how close they are to the CSI 300 entry rule.

What this package is (`refs/families.md` is the authority, `lib/score.py` the one
place the score is defined):

    m1       the candidate: the CSI 500 members with the best entry rank under
             the public CSI 300 selection rule, 30 equal-cash seats, reviewed
             monthly. Not learned: there is no `fit`
    c_shuf   the control: m1's values permuted within the decision day
             (`knobs.SHUFFLE = True`, the only line that differs)

The book (lib/trade.py), the pool and prices (lib/data.py) and the sections
(lib/index.py) are shared by both legs; every default a session is expected to
change lives in lib/knobs.py, each one exactly once.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
