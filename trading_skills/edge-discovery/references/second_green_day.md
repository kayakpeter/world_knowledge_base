# Second Green Day (Day-2 continuation)

## Idea

Day 1: a stock rips on a catalyst (news, earnings, FDA, contract, sector
sympathy) and closes strong. Day 2: buyers who missed Day 1 keep pushing it,
and the stock keeps going up in regular trading hours. The "frontside" Day-2
play buys that continuation. Its mirror, the Day-2 fade, shorts a stock that
fails to hold — keep it separate; it needs locates and borrow.

Every number below is a **starting point, labeled UNGROUNDED** until the base
rate scripts give it a percentile. Replace each with the grounded value and
cite the data span it came from.

## Stage A — Day-1 qualifier (selection, daily data)

| Feature | Starting rule | Status |
|---|---|---|
| Day-1 gain | close / prev_close − 1 ≥ 20% | UNGROUNDED |
| Close location | (close − low) / (high − low) ≥ 0.7 (closed near highs) | UNGROUNDED |
| Liquidity | Day-1 dollar volume ≥ $10M | UNGROUNDED |
| Price | $1 ≤ close ≤ $30 | UNGROUNDED |
| Relative volume | Day-1 volume ≥ 5× 20-day average | UNGROUNDED |
| Catalyst | news on Day 1 (Massive news endpoint) — optional join | not yet wired |

Script: `scripts/build_day1_universe.py`.

## Stage B — Day-2 selection skew (daily data)

Questions to answer from `scripts/day2_base_rates.py` before any intraday work:

1. Base rate: of all Day-1 qualifiers, what % close green on Day 2? What % break
   Day-1 high? Median open→close? Median max-up and max-down from the open?
2. Which single features move those numbers? Check at least:
   - Day-1 close location (strong close vs faded)
   - Day-2 gap (gap up small, gap up big, flat, gap down)
   - Personality: `runner_rate` and `n_big_days` (see stock_personality.md)
   - Day-1 gain bucket (20–40%, 40–100%, 100%+)
   - Price band and dollar volume
   - First big day in a year vs repeat runner
3. Interactions only for features that mattered alone.

**Kill criterion for the selection layer:** if no slice with n ≥ 100 shows a
meaningfully better Day-2 outcome than base in discovery *and* holds in
validation, the daily-data version of this idea is dead. Log it and move to
entry-layer ideas that don't depend on selection skew.

## Stage C — Entry (1-min bars, under backtest-discipline)

Only for slices that survived Stage B. Candidate entries, test one at a time:

- **Opening range break**: break of the first 5-min high after 9:35, stop below
  the opening-range low.
- **Day-1 high reclaim**: first 1-min close above Day-1 high, stop below the
  reclaim bar low.
- **First pullback to VWAP** that holds, after making a new high of day.
- **Pre-market high break** in RTH.

Gate 3 time-of-day histogram is mandatory: a Day-2 entry that is just "buy the
first minute" is not the setup.

## Stage D — Management (1-min / ticks)

No-stop baseline (hold to 15:55) first. Then one mechanic at a time: initial
stop, partial at +1R, trail under higher lows / VWAP, time stop.

## Risk framing for a small account

- Fixed risk per trade (for example 0.5–1% of account) sets the share size:
  shares = risk$ / (entry − stop). Never size by conviction.
- Halts: a halt-down can blow through any stop. Cap position as a % of account
  even when the stop is tight.
- Spread at the entry time from the quotes data is a cost, not noise.
