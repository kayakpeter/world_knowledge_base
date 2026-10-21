# Stock personality

"Personality" = how *this ticker* has behaved after its previous big days.
Some names run for days, others fade the morning after, every time.
These features are computed **point-in-time**: on date D they use only days
strictly before D, so they can be used as filters without lookahead.

Computed by `scripts/build_day1_universe.py`. "Big day" = close / prev_close − 1
≥ `BIG_DAY_PCT` (default 20%, UNGROUNDED).

| Column | Definition (prior 252 sessions, excluding today) | Why it matters |
|---|---|---|
| `n_big_days` | count of prior big days | Repeat runners attract momentum traders |
| `runner_rate` | of prior big days, share where next day closed above the big day's close | Direct measure of Day-2 continuation history |
| `day2_high_break_rate` | of prior big days, share where next day's high > big day's high | Did it at least offer an exit above Day-1 high? |
| `median_big_close_loc` | median close location on prior big days | Closes-near-high names vs fade-into-close names |
| `atr_pct_20` | 20-day average true range / close | Normal daily range; sizes stops |
| `dollar_vol_20` | 20-day average close × volume | Liquidity, slippage |

`runner_rate` is null when `n_big_days` = 0, which is common. Treat
"first big day in a year" as its own bucket, not as missing data.

## Not yet included (candidates for later)

- Float and shares outstanding (Massive reference endpoints) — low float
  names move further.
- Short interest and short volume — squeeze fuel.
- Dilution history (S-3 / ATM filings) — serial diluters fade.
- Sector/sympathy: did a peer also run on Day 1?

Add each one only as a hypothesis in the ledger, not as a default filter.
