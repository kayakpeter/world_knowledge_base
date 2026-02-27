"""
Daily commodity ETF price fetch — entry point.

Usage:
    # Incremental (only new dates since last run):
    python run_price_fetch.py

    # Full re-fetch from 2020-01-01:
    python run_price_fetch.py --full

Run from the global_financial_kb directory:
    cd /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb
    python run_price_fetch.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from ingestion.fetchers.prices_yfinance import run_fetch


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch daily commodity ETF prices")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-fetch all history from 2020-01-01 (ignores existing data)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    df = run_fetch(full=args.full)

    if df.is_empty():
        print("No price data available.")
        return

    # Print latest close per ticker to stdout for easy inspection
    latest = (
        df
        .sort("date")
        .group_by("ticker")
        .last()
        .select(["ticker", "date", "close", "pct_change", "volume"])
        .sort("ticker")
    )
    print("\nLatest prices:")
    print(latest)


if __name__ == "__main__":
    main()
