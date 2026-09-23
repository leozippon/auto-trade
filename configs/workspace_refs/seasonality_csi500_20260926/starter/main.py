"""Entry: CSI 500 members ranked by their same-calendar-month history.

What this package is (`refs/families.md` is the authority, `lib/score.py` the one
place the score is defined):

    s1       the candidate: each CSI 500 member's mean excess return over the
             pool in this calendar month one to four years ago; the top 50
             equal-cash seats, re-ranked every month. Not learned: there is no
             `fit`
    c_shuf   the control: s1's values permuted within the decision day
             (`knobs.SHUFFLE = True`, the only line that differs)

The book (lib/trade.py), the pool and prices (lib/data.py) and the sections
(lib/index.py) are shared by both legs; every default a session is expected to
change lives in lib/knobs.py, each one exactly once.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
