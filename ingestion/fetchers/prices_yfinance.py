"""
yfinance daily price fetcher for commodity ETFs.

Reads tickers from company_seeds.parquet (sector == "Commodities"),
fetches daily OHLCV from Yahoo Finance, and maintains a rolling
parquet at data/prices/commodity_etf_prices.parquet.

Incremental by default: only fetches dates after the latest stored date
per ticker. Pass --full to refetch from 2020-01-01.

Schema:
    date         pl.Date
    ticker       str
    open         pl.Float32
    high         pl.Float32
    low          pl.Float32
    close        pl.Float32   (adjusted)
    volume       pl.Int64
    pct_change   pl.Float32   (vs prior close, %)
    fetched_at   str          (ISO UTC timestamp)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import polars as pl
import yfinance as yf

logger = logging.getLogger(__name__)

PRICES_DIR = Path(
    "/media/peter/fast-storage/projects/world_knowledge_base"
    "/global_financial_kb/data/prices"
)
PRICES_PATH = PRICES_DIR / "commodity_etf_prices.parquet"
SEEDS_PATH = Path(
    "/media/peter/fast-storage/projects/world_knowledge_base"
    "/global_financial_kb/data/company_seeds.parquet"
)

FULL_HISTORY_START = date(2020, 1, 1)
# yfinance needs end = day AFTER the last day we want
_FETCH_END_OFFSET = timedelta(days=1)


def load_commodity_tickers() -> list[str]:
    """Read commodity ETF tickers from company_seeds.parquet."""
    seeds_df = pl.read_parquet(SEEDS_PATH)
    tickers = (
        seeds_df
        .filter(pl.col("sector") == "Commodities")
        .get_column("ticker")
        .to_list()
    )
    logger.info("Commodity tickers from seeds: %s", tickers)
    return tickers


def load_existing_prices() -> Optional[pl.DataFrame]:
    """Load existing price parquet, or None if it doesn't exist."""
    if not PRICES_PATH.exists():
        return None
    return pl.read_parquet(PRICES_PATH)


def latest_date_per_ticker(prices_df: pl.DataFrame) -> dict[str, date]:
    """Return the latest stored date for each ticker."""
    return {
        row["ticker"]: row["date"]
        for row in (
            prices_df
            .group_by("ticker")
            .agg(pl.col("date").max())
            .to_dicts()
        )
    }


def fetch_ticker(
    ticker: str,
    start: date,
    end: date,
) -> pl.DataFrame:
    """
    Fetch daily OHLCV for a single ticker from Yahoo Finance.

    Returns an empty DataFrame (correct schema) if no data is available.
    """
    logger.info("  Fetching %s from %s to %s", ticker, start, end)
    try:
        hist = yf.Ticker(ticker).history(
            start=start.isoformat(),
            end=(end + _FETCH_END_OFFSET).isoformat(),
            auto_adjust=True,
            actions=False,
        )
    except Exception as exc:
        logger.error("  yfinance error for %s: %s", ticker, exc)
        return _empty_schema()

    if hist.empty:
        logger.warning("  No data returned for %s", ticker)
        return _empty_schema()

    fetched_at = datetime.now(timezone.utc).isoformat()

    rows = []
    prev_close: Optional[float] = None
    for ts, row in hist.iterrows():
        close = float(row["Close"])
        pct = ((close - prev_close) / prev_close * 100.0) if prev_close is not None else 0.0
        rows.append({
            "date":       ts.date(),
            "ticker":     ticker,
            "open":       float(row["Open"]),
            "high":       float(row["High"]),
            "low":        float(row["Low"]),
            "close":      close,
            "volume":     int(row["Volume"]),
            "pct_change": pct,
            "fetched_at": fetched_at,
        })
        prev_close = close

    if not rows:
        return _empty_schema()

    return (
        pl.DataFrame(rows)
        .with_columns([
            pl.col("date").cast(pl.Date),
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Int64),
            pl.col("pct_change").cast(pl.Float32),
        ])
    )


def _empty_schema() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "date":       pl.Date,
        "ticker":     pl.String,
        "open":       pl.Float32,
        "high":       pl.Float32,
        "low":        pl.Float32,
        "close":      pl.Float32,
        "volume":     pl.Int64,
        "pct_change": pl.Float32,
        "fetched_at": pl.String,
    })


def run_fetch(full: bool = False) -> pl.DataFrame:
    """
    Main entry point. Fetch prices for all commodity ETFs.

    Args:
        full: If True, re-fetch from FULL_HISTORY_START ignoring cache.
              If False (default), only fetch dates after the latest stored date.

    Returns:
        The complete (updated) price DataFrame.
    """
    PRICES_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_commodity_tickers()
    existing_df = load_existing_prices()
    today = date.today()

    if full or existing_df is None:
        latest_dates: dict[str, date] = {}
        logger.info("Full fetch from %s for %d tickers", FULL_HISTORY_START, len(tickers))
    else:
        latest_dates = latest_date_per_ticker(existing_df)
        logger.info("Incremental fetch — existing data up to: %s", latest_dates)

    new_frames: list[pl.DataFrame] = []

    for ticker in tickers:
        if full or existing_df is None:
            start = FULL_HISTORY_START
        else:
            last = latest_dates.get(ticker)
            if last is None:
                start = FULL_HISTORY_START
                logger.info("  %s: no existing data — fetching from %s", ticker, start)
            elif last >= today:
                logger.info("  %s: already up to date (%s)", ticker, last)
                continue
            else:
                start = last + timedelta(days=1)

        df = fetch_ticker(ticker, start=start, end=today)
        if not df.is_empty():
            new_frames.append(df)

    if not new_frames:
        logger.info("No new data fetched — all tickers up to date")
        return existing_df if existing_df is not None else _empty_schema()

    new_df = pl.concat(new_frames)
    logger.info("Fetched %d new rows across %d tickers", len(new_df), len(new_frames))

    if existing_df is not None and not full:
        combined_df = (
            pl.concat([existing_df, new_df])
            .unique(subset=["date", "ticker"], keep="last")
            .sort(["ticker", "date"])
        )
    else:
        combined_df = new_df.sort(["ticker", "date"])

    combined_df.write_parquet(PRICES_PATH, compression="zstd")
    logger.info(
        "Written %d total rows → %s",
        len(combined_df), PRICES_PATH,
    )

    _log_summary(combined_df)
    return combined_df


def _log_summary(df: pl.DataFrame) -> None:
    """Log a per-ticker summary of the price data."""
    summary = (
        df
        .group_by("ticker")
        .agg([
            pl.col("date").min().alias("from"),
            pl.col("date").max().alias("to"),
            pl.len().alias("rows"),
            pl.col("close").last().alias("last_close"),
            pl.col("pct_change").last().alias("last_pct"),
        ])
        .sort("ticker")
    )
    logger.info("Price summary:")
    for row in summary.to_dicts():
        logger.info(
            "  %-6s  %s → %s  (%d rows)  last=%.2f  chg=%.2f%%",
            row["ticker"], row["from"], row["to"],
            row["rows"], row["last_close"], row["last_pct"],
        )
