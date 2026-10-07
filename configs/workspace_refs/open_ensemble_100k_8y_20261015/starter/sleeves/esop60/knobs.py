"""Event book knobs for the esop60 leg: Employee shareholding plan DRAFT announcements; entry at T0+1, held HOLD_DAYS=60 trading days.

SEATS is a ceiling: the event pool varies from a handful to ~50 active names,
so the book holds up to SEATS names and shrinks when the pool is thin (the
starter's hard 'at least SEATS scored names' check is removed in lib/trade.py;
score.py itself still fails loudly when the title window is corrupt).
"""

INDEX = "000852.SH"
POOL = "alla"
SEATS = 30
CAPITAL = 0.97
KEEP_BAND = 100.0
INDUSTRY_CAP = 10
REVIEW = "week"
SHUFFLE = False
HOLD_DAYS = 60
LATE_SHIFT = 0
