---
name: edge-discovery
description: Self-directed research loop for finding small-cap momentum day-trading edges in Peter's local Massive.com data (tick, quotes, 1-min bars, daily aggs, 2022+). Use whenever the task is "find a setup", "what should we test next", "why isn't this working", "come up with ideas", second-day / Day-2 / second green day plays, stock personality, runner vs fader, gapper universe, or reviewing the research ledger. Claude generates the hypotheses — Peter should not have to. Works upstream of backtest-discipline (which governs HOW a single test is run); this skill governs WHAT gets tested and remembers what already failed.
---

# Edge Discovery

## Why this skill exists

The old loop was: Peter invents a complete strategy → Claude backtests it →
"doesn't work" → Peter invents another. Each round asked one yes/no question
about a whole strategy (selection + entry + stop + exit all at once), learned
almost nothing when the answer was "no", and forgot it by the next session.

This skill replaces that with **measure first, then hypothesize**:

1. Describe what the universe actually does (base rates) before testing any rule.
2. Find *conditions* where outcomes skew — those are the candidate edges.
3. Only then design an entry/exit around a skew, and test it under
   `backtest-discipline`.
4. Write every result — especially failures, and *why* — to the research
   ledger, and generate the next hypothesis from the ledger.

A strategy has three separable layers. Test them separately, in order:

| Layer | Question | Data | Fails how |
|---|---|---|---|
| **Selection** | Which stocks/days have skewed Day-2 outcomes? | daily aggs | No subset beats the base rate |
| **Entry** | Within a good day, where's the low-risk spot? | 1-min bars | MAE before MFE; entries chase |
| **Exit/management** | How do we harvest the move? | 1-min / ticks | Beats no-stop baseline or not |

A "doesn't work" verdict must say *which layer* failed. A good selection layer
with a bad entry is a fixable problem, not a dead idea.

## Step 0 — Data locations (do once, never hunt again)

Read `references/data_locations.md`. If any path is marked `TODO`, find it
once (ask Peter if not obvious), fill it in, and commit. Every later session
reads the file instead of searching the disk.

## Step 1 — Base rates (the map)

Run `scripts/build_day1_universe.py` then `scripts/day2_base_rates.py`
(paths come from `data_locations.md`). Output: every Day-1 mover since 2022
with point-in-time personality features, and Day-2 outcome tables.

Before anything else, report the **unconditional** Day-2 numbers for the Day-1
universe: % green (close > open), median open→close, % that break Day-1 high,
median max-favorable and max-adverse from the open. This is the bar every
hypothesis must clear. If the base rate is a coin flip, that is fine — edges
live in the conditional slices.

## Step 2 — Hypothesis generation (Claude's job)

Generate hypotheses from four sources, in this priority:

1. **Residuals of the base-rate tables.** Slice Day-2 outcomes by one feature at
   a time (personality, Day-1 close location, Day-1 % gain bucket, gap bucket,
   dollar volume, price band, day of week, market regime). Any bucket with
   n ≥ 100 whose outcome differs from base by a meaningful amount is a candidate.
   Then try two-feature interactions only for features that mattered alone.
2. **The playbook.** SMB-style setups extracted into
   `references/playbook/` (one file per setup, rules made explicit). If the
   folder is thin, extract more from the SMB course material (location in
   `data_locations.md`) — turn discretionary language into testable conditions
   and list which conditions are ambiguous.
3. **The ledger.** For every failed test, ask: which layer failed, and is there
   a neighbouring hypothesis the failure points to? (e.g. "entry failed because
   MAE hit 8% before the move" → test a later, tighter entry on the same days.)
4. **Trade inspection.** The 10 best and 10 worst trades of the last test often
   show the missing condition more clearly than any aggregate.

Write each hypothesis to the ledger *before* testing, with: the layer, the
precise rule, the expected effect size, and what result would kill it.

## Step 3 — Test

Selection-layer tests use daily data and the Day-2 tables — cheap, run many.
Entry/exit tests use 1-min bars and go through **every gate in
`backtest-discipline`** (entry fidelity, no-stop baseline, one mechanic at a
time, time-of-day histogram, 10-trade dump, grounded constants).

Additional rules for this skill:

- **Discovery/validation split.** Discover on 2022-01 → 2024-12. Confirm on
  2025-01 → 2025-12. The Jan–Mar 2026 holdout stays untouched until a declared
  final gate. A hypothesis that only works in discovery is logged as failed.
- **Multiple-testing honesty.** Report how many slices/variants were tried. If
  you tried 40 buckets, expect ~2 to look significant by chance at p<0.05.
- **Measure in R.** Expectancy per trade in units of initial risk, plus the
  distribution, not just win rate. A 35% win rate at 3R average win is an edge.
- **Small-account realism.** Model slippage (at least half the quoted spread
  each side, from the quotes data), skip names under the liquidity floor, and
  remember the US PDT rule (<$25k margin account → 3 day trades per rolling 5
  days) — a setup that needs 10 trades a week can't be run at that size.

## Step 4 — Ledger (this is what makes it self-improving)

Append every hypothesis and result to `references/research_ledger.md` using
the template in that file. Never delete entries. Each session starts by reading
the ledger's **Open threads** and **Dead ends** sections, so work compounds
instead of repeating.

When a hypothesis survives discovery *and* validation: promote it to a playbook
file, then to paper trading on IB with fixed small size, and log live fills vs
backtest expectations in the ledger. Live/backtest divergence is itself a
hypothesis to investigate.

## Session checklist

1. Read `data_locations.md`, then the ledger's Open threads + Dead ends.
2. State which layer today's work targets.
3. Propose 3–5 hypotheses (Claude-generated, with source), pick the best, log it.
4. Test → gates → inspect trades → log the result and which layer failed.
5. End by writing the next 1–3 open threads.

## References

- `references/second_green_day.md` — the Day-2 continuation setup, staged.
- `references/stock_personality.md` — point-in-time personality features.
- `references/research_ledger.md` — the ledger (template + entries).
- `references/data_locations.md` — fixed paths; fill once.
- `references/playbook/` — one explicit spec per setup.
- Conventions: `trading-conventions` (Polars only, absolute paths, `_df`/`_lf`).
