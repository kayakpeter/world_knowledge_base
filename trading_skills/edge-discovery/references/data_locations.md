# Data locations

Fill each TODO **once**, then commit. Every session reads this file instead of
searching the disk. Use absolute paths.

| Data | Path / glob | Format notes |
|---|---|---|
| Massive daily aggs | TODO (e.g. `/media/peter/fast-storage/massive/us_stocks_sip/day_aggs_v1/**/*.csv.gz`) | cols: ticker, volume, open, close, high, low, window_start (ns UTC), transactions |
| Massive 1-min aggs | TODO | same columns, one file per day |
| Massive trades (tick) | TODO | |
| Massive quotes (bid/ask) | TODO | |
| SMB course material | TODO | |
| qbb_trader repo | `/media/peter/fast-storage/projects/qbb_trader` | ratified detector: `src/candidates.py` |
| Day-1 universe output | TODO (e.g. `/media/peter/fast-storage/research/edge_discovery/day1_universe.parquet`) | written by `build_day1_universe.py` |
| IB paper account | TODO (host/port/clientId for paper TWS/Gateway) | |
