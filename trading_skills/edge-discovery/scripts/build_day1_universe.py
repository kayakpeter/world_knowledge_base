"""Build the Day-1 mover universe with point-in-time stock personality features.

Reads Massive.com daily aggregate flat files (csv, csv.gz or parquet), computes
per-ticker features using only information available at each day's close, and
writes every "big day" (Day-1 candidate) with its next-session (Day-2) bars.

Usage:
    python build_day1_universe.py \
        --day-aggs "/abs/path/day_aggs_v1/**/*.csv.gz" \
        --out /abs/path/day1_universe.parquet

Thresholds are UNGROUNDED defaults; ground them from the output before relying
on them (see backtest-discipline Gate 5).
"""

import argparse
import glob
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

LOOKBACK = 252  # sessions of history used for personality features
AVG_WINDOW = 20


def load_day_aggs(pattern: str) -> pl.DataFrame:
    """Load daily aggs from a glob of csv / csv.gz / parquet files."""
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern}")
    logger.info("loading %d files", len(paths))

    cols = ["ticker", "volume", "open", "close", "high", "low", "window_start"]
    if all(p.endswith(".parquet") for p in paths):
        raw_df = pl.scan_parquet(paths).select(cols).collect()
    else:
        raw_df = pl.concat(
            [pl.read_csv(p, columns=cols, schema_overrides={"volume": pl.Float64}) for p in paths],
            how="vertical_relaxed",
        )

    # window_start is ns UTC at midnight ET (04:00 or 05:00 UTC), so the UTC
    # calendar date is the trading date. A fixed UTC-5 shift would put DST
    # days on the previous date.
    return (
        raw_df.with_columns(
            pl.from_epoch("window_start", time_unit="ns").dt.date().alias("trade_date"),
            pl.col("ticker").str.strip_chars().str.to_uppercase(),
            pl.col("volume").cast(pl.Int64),
        )
        .drop("window_start")
        # Common stock-like symbols only; drops warrants/units/preferreds with
        # punctuation. 5-letter W/U/R suffixes still slip through.
        .filter(pl.col("ticker").str.contains(r"^[A-Z]{1,5}$"))
        .unique(subset=["ticker", "trade_date"], keep="last")
        .sort(["ticker", "trade_date"])
    )


def add_features(bars_df: pl.DataFrame, big_day_pct: float) -> pl.DataFrame:
    """Add Day-1 features, next-session bars and point-in-time personality."""
    bars_df = bars_df.with_columns(
        pl.col("close").shift(1).over("ticker").alias("prev_close"),
    ).with_columns(
        (pl.col("close") / pl.col("prev_close") - 1).alias("day_gain"),
        (
            (pl.col("close") - pl.col("low"))
            / (pl.col("high") - pl.col("low")).replace(0, None)
        ).alias("close_loc"),
        (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("prev_close")).abs(),
            (pl.col("low") - pl.col("prev_close")).abs(),
        ).alias("true_range"),
        # Next session (Day 2) bars, for outcomes only, never as features.
        pl.col("trade_date").shift(-1).over("ticker").alias("d2_date"),
        pl.col("open").shift(-1).over("ticker").alias("d2_open"),
        pl.col("high").shift(-1).over("ticker").alias("d2_high"),
        pl.col("low").shift(-1).over("ticker").alias("d2_low"),
        pl.col("close").shift(-1).over("ticker").alias("d2_close"),
    )

    is_big = pl.col("day_gain") >= big_day_pct
    bars_df = bars_df.with_columns(
        is_big.fill_null(False).alias("is_big"),
        (is_big & (pl.col("d2_close") > pl.col("close"))).fill_null(False).alias("big_ran"),
        (is_big & (pl.col("d2_high") > pl.col("high"))).fill_null(False).alias("big_hbreak"),
        pl.when(is_big).then(pl.col("close_loc")).alias("big_close_loc"),
    )

    # Personality as of today's close: prior big days only (shift 1). A big day
    # at D-1 has its "ran" outcome resolved by D's close, so this is lookahead-free
    # for a Day-1 decision made after the close of D.
    def prior(col: str) -> pl.Expr:
        return pl.col(col).shift(1).over("ticker")

    def prior_sum(col: str) -> pl.Expr:
        return (
            pl.col(col).cast(pl.Int32).shift(1)
            .rolling_sum(LOOKBACK, min_samples=1)
            .over("ticker")
        )

    bars_df = bars_df.with_columns(
        prior_sum("is_big").fill_null(0).alias("n_big_days"),
        prior_sum("big_ran").alias("_n_ran"),
        prior_sum("big_hbreak").alias("_n_hbreak"),
        pl.col("big_close_loc").shift(1)
        .rolling_median(LOOKBACK, min_samples=1)
        .over("ticker")
        .alias("median_big_close_loc"),
        (
            prior("true_range").rolling_mean(AVG_WINDOW).over("ticker")
            / pl.col("prev_close")
        ).alias("atr_pct_20"),
        prior("dollar_volume").rolling_mean(AVG_WINDOW).over("ticker").alias("dollar_vol_20"),
        prior("volume").rolling_mean(AVG_WINDOW).over("ticker").alias("_vol_20"),
    )

    return bars_df.with_columns(
        pl.when(pl.col("n_big_days") > 0)
        .then(pl.col("_n_ran") / pl.col("n_big_days"))
        .alias("runner_rate"),
        pl.when(pl.col("n_big_days") > 0)
        .then(pl.col("_n_hbreak") / pl.col("n_big_days"))
        .alias("day2_high_break_rate"),
        (pl.col("volume") / pl.col("_vol_20")).alias("rvol"),
    ).drop(["_n_ran", "_n_hbreak", "_vol_20", "big_ran", "big_hbreak", "big_close_loc"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--day-aggs", required=True, help="glob of daily agg files (absolute)")
    parser.add_argument("--out", required=True, type=Path, help="output parquet (absolute)")
    parser.add_argument("--big-day-pct", type=float, default=0.20, help="UNGROUNDED")
    parser.add_argument("--min-price", type=float, default=1.0, help="UNGROUNDED")
    parser.add_argument("--max-price", type=float, default=30.0, help="UNGROUNDED")
    parser.add_argument("--min-dollar-volume", type=float, default=10e6, help="UNGROUNDED")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bars_df = add_features(load_day_aggs(args.day_aggs), args.big_day_pct)
    universe_df = bars_df.filter(
        pl.col("is_big"),
        pl.col("close").is_between(args.min_price, args.max_price),
        pl.col("dollar_volume") >= args.min_dollar_volume,
    ).with_columns(pl.lit(args.big_day_pct).alias("big_day_pct"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    universe_df.write_parquet(args.out, compression="zstd")
    logger.info(
        "wrote %d Day-1 candidates (%d tickers, %s → %s) to %s",
        universe_df.height,
        universe_df["ticker"].n_unique(),
        universe_df["trade_date"].min(),
        universe_df["trade_date"].max(),
        args.out,
    )


if __name__ == "__main__":
    main()
