# Starter library

The canonical book mechanics that reference-pack starters copy. A pack's `starter/lib/` takes these files byte for byte, under the same name, and never edits them; a unit test refuses any pack from round 20261001 on whose starter holds a file named like one here with other bytes. A fix or a new knob is made here first and reaches the next packs by copying; a pack whose arm has been created is not touched again, because the copy in that run's workspace is its record.

What belongs here is the part every rule-score book shares and a defect in which would ship in every pack: seat sizing, review and refill, exits, the industry cap and the orders. What does not: the pool, the score and the knobs, which are each arm's own.

`trade.py` is the equal-cash rule-score book. It reads its pack's `lib.data.pool(context, positions)` (a frame indexed by code with `close`, `member`, `tradable`, `weight` and `industry`, the T-1 date and the section date), `lib.score` (`NAME`, `score`, `shuffle`, `describe`) and `lib.knobs` (`SEATS`, `CAPITAL`, `KEEP_BAND`, `INDUSTRY_CAP`, `REVIEW` as `"week"` or `"month"`, `SHUFFLE`). A full book never buys an extra seat, and a holding without a usable close is valued at 0 rather than compared as an empty value.
