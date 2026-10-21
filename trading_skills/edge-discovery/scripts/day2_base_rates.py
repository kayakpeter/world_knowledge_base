"""Day-2 base rates and one-feature slices for the Day-1 mover universe.

Answers the selection-layer question: after a big Day 1, what does Day 2 do,
and which single features skew that? Every slice is reported against the base
rate so residuals stand out.

Usage:
    python day2_base_rates.py --universe /abs/path/day1_universe.parquet \
        --start 2022-01-01 --end 2024-12-31 [--csv-out /abs/path/slices.csv]

Default span is the discovery window. Re-run with --start 2025-01-01
--end 2025-12-31 to validate. Never point it at the Jan–Mar 2026 holdout
during iteration.
"""

import argparse
import datetime as dt
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

MIN_N = 100  # slices smaller than this are shown but flagged

OUTCOMES = [
    pl.len().alias("n"),
    pl.col("d2_green").mean().alias("green_pct"),
    pl.col("d2_break_hi").mean().alias("break_d1_high_pct"),
    pl.col("d2_oc").median().alias("median_open_to_close"),
    pl.col("d2_cc").median().alias("median_close_to_close"),
    pl.col("d2_mfe").median().alias("median_max_up_from_open"),
    pl.col("d2_mae").median().alias("median_max_down_from_open"),
]


def add_outcomes(universe_df: pl.DataFrame) -> pl.DataFrame:
    return universe_df.filter(pl.col("d2_open").is_not_null()).with_columns(
        (pl.col("d2_open") / pl.col("close") - 1).alias("d2_gap"),
        (pl.col("d2_close") > pl.col("d2_open")).alias("d2_green"),
        (pl.col("d2_high") > pl.col("high")).alias("d2_break_hi"),
        (pl.col("d2_close") / pl.col("d2_open") - 1).alias("d2_oc"),
        (pl.col("d2_close") / pl.col("close") - 1).alias("d2_cc"),
        (pl.col("d2_high") / pl.col("d2_open") - 1).alias("d2_mfe"),
        (pl.col("d2_low") / pl.col("d2_open") - 1).alias("d2_mae"),
    )


def binned(col: str, breaks: list[float], labels: list[str]) -> pl.Expr:
    """Label col by right-closed breaks (works on Polars 1.x and 2.x, unlike cut)."""
    expr = pl.when(pl.col(col).is_null()).then(pl.lit(None, dtype=pl.String))
    for brk, label in zip(breaks, labels):
        expr = expr.when(pl.col(col) <= brk).then(pl.lit(label))
    return expr.otherwise(pl.lit(labels[-1]))


def bucket(col: str, breaks: list[float], labels: list[str]) -> pl.Expr:
    return binned(col, breaks, labels).alias("bucket")


SLICES: dict[str, pl.Expr] = {
    "day1_gain": bucket("day_gain", [0.4, 1.0], ["20-40%", "40-100%", "100%+"]),
    "day1_close_loc": bucket("close_loc", [0.4, 0.7, 0.9], ["<0.4 faded", "0.4-0.7", "0.7-0.9", "0.9+ at highs"]),
    "day2_gap": bucket("d2_gap", [-0.05, 0.0, 0.05, 0.15], ["gap dn >5%", "gap dn 0-5%", "gap up 0-5%", "gap up 5-15%", "gap up 15%+"]),
    "personality_runner_rate": pl.when(pl.col("n_big_days") == 0)
    .then(pl.lit("first big day"))
    .otherwise(binned("runner_rate", [0.34, 0.66], ["fader <0.34", "mixed", "runner >0.66"]))
    .alias("bucket"),
    "personality_n_big_days": bucket("n_big_days", [0, 2, 5], ["0", "1-2", "3-5", "6+"]),
    "rvol": bucket("rvol", [5, 10, 25], ["<5x", "5-10x", "10-25x", "25x+"]),
    "price": bucket("close", [2, 5, 10], ["$1-2", "$2-5", "$5-10", "$10+"]),
    "dollar_volume": bucket("dollar_volume", [25e6, 100e6], ["$10-25M", "$25-100M", "$100M+"]),
    "weekday_d2": pl.col("d2_date").dt.weekday().cast(pl.String).alias("bucket"),
}


def slice_tables(outcomes_df: pl.DataFrame) -> pl.DataFrame:
    base_df = outcomes_df.select(OUTCOMES).with_columns(
        pl.lit("BASE").alias("feature"), pl.lit("all").alias("bucket")
    )
    base_green = base_df["green_pct"][0]
    base_break = base_df["break_d1_high_pct"][0]

    tables = [base_df]
    for feature, expr in SLICES.items():
        tables.append(
            outcomes_df.with_columns(expr)
            .group_by("bucket")
            .agg(OUTCOMES)
            .sort("bucket")
            .with_columns(pl.lit(feature).alias("feature"))
        )
    return (
        pl.concat(tables, how="diagonal")
        .with_columns(
            (pl.col("green_pct") - base_green).alias("green_vs_base"),
            (pl.col("break_d1_high_pct") - base_break).alias("break_vs_base"),
            (pl.col("n") < MIN_N).alias("small_n"),
        )
        .select(
            "feature", "bucket", "n", "small_n", "green_pct", "green_vs_base",
            "break_d1_high_pct", "break_vs_base", "median_open_to_close",
            "median_close_to_close", "median_max_up_from_open", "median_max_down_from_open",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2022, 1, 1))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2024, 12, 31))
    parser.add_argument("--csv-out", type=Path, help="optional CSV of all slices for inspection")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.end >= dt.date(2026, 1, 1):
        raise SystemExit("refusing to read the Jan–Mar 2026 holdout during iteration")

    outcomes_df = add_outcomes(
        pl.read_parquet(args.universe).filter(pl.col("trade_date").is_between(args.start, args.end))
    )
    logger.info("Day-1 candidates %s → %s: %d (%d tickers)", args.start, args.end,
                outcomes_df.height, outcomes_df["ticker"].n_unique())

    slices_df = slice_tables(outcomes_df)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=3):
        logger.info("\n%s", slices_df)

    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        slices_df.write_csv(args.csv_out)
        logger.info("wrote %s", args.csv_out)


if __name__ == "__main__":
    main()
