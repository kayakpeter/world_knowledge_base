"""
collect_interpretations.py — Normalise and merge agent interpretation parquets.

Each agent writes its own interpretations_<agent>_<timestamp>.parquet with a
slightly divergent schema (v1).  This module normalises all three into the
canonical v2 NewsInterpretation schema and concatenates them into a single
unified parquet ready for the Saturday Lambda LLM edge-weight re-inference run.

CLI usage
---------
    python processing/collect_interpretations.py
    python processing/collect_interpretations.py --date 2026-02-20
    python processing/collect_interpretations.py --out custom_output.parquet
    python processing/collect_interpretations.py --dry-run

Output
------
    data/interpretations/interpretations_unified_YYYYMMDD.parquet

Agent v1 → v2 divergences handled
----------------------------------
Prometheus  actor_ids/affected_stats: comma-separated strings
            sentiment: "positive"|"negative"|"mixed"|"neutral"
            cross_country: comma-sep ISO3 codes (combines bool + list)
            urgency: "notable" | "urgent" | "routine" (non-canonical)
            column: "interpretation_note" (→ notes), "agent" (→ interpreted_by)
            missing: headline, published_at, new_actors, stat_direction,
                     estimated_magnitude, causal_chain, confidence

Daedalus    actor_ids/affected_stats: List(String) — already correct
            sentiment: string (→ float)
            stat_direction: JSON dict string — already correct
            column: "affected_countries" (→ cross_country_iso3), "agent" (→ interpreted_by)
            missing: headline already present, published_at, new_actors, notes

Apollo      actor_ids/affected_stats/cross_country_iso3: JSON strings (→ lists)
            new_actors: JSON string — already correct
            sentiment: Float64 (→ Float32)
            urgency: already canonical ("high"|"medium"|"low")
            missing: stat_direction, estimated_magnitude, causal_chain, confidence
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import polars as pl

PROJ            = Path(__file__).parent.parent
INTERP_DIR      = PROJ / "data" / "interpretations"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Sentinel defaults for optional v2 fields ──────────────────────────────────
_DEFAULT_STAT_DIRECTION   = "{}"
_DEFAULT_MAGNITUDE        = ""
_DEFAULT_NEW_ACTORS       = "[]"
_DEFAULT_CONFIDENCE       = 0.5
_DEFAULT_PREDICTION       = "null"    # JSON null for missing predictions

# ── Value maps ────────────────────────────────────────────────────────────────
_SENTIMENT_STR_TO_FLOAT: dict[str, float] = {
    "positive": 0.5,
    "negative": -0.5,
    "mixed":    0.0,
    "neutral":  0.0,
}

_URGENCY_NORMALISE: dict[str, str] = {
    "critical": "critical",
    "urgent":   "high",
    "notable":  "medium",
    "routine":  "low",
    "high":     "high",
    "elevated": "high",
    "medium":   "medium",
    "low":      "low",
}

# ── V2 column order ───────────────────────────────────────────────────────────
V2_COLUMNS = [
    "item_id", "country_iso3", "headline", "published_at",
    "actor_ids", "new_actors", "intent",
    "affected_stats", "stat_direction", "estimated_magnitude", "causal_chain",
    "sentiment", "urgency", "confidence",
    "cross_country", "cross_country_iso3",
    "interpreted_by", "interpreted_at", "notes",
    "risk_reward_horizon", "volatility_risk",
    "operational_risk", "supply_chain_exposure",
    "market_impact", "sector_impact", "ticker_impact",
]


# Vocabulary normalization for the new Hybrid Extension fields.
# gemma4:31b sometimes drifts from H/M/L → high/med/low; tolerate both.
_VOLATILITY_RISK_NORMALISE: dict[str, str] = {
    "h": "H", "high": "H",
    "m": "M", "med": "M", "medium": "M", "moderate": "M",
    "l": "L", "low": "L",
}
_RISK_REWARD_HORIZON_NORMALISE: dict[str, str] = {
    "intraday":      "intraday",
    "1-4_weeks":     "1-4_weeks",
    "1-4 weeks":     "1-4_weeks",
    "1_to_4_weeks":  "1-4_weeks",
    "weeks":         "1-4_weeks",
    "multi_quarter": "multi_quarter",
    "multi-quarter": "multi_quarter",
    "multiquarter":  "multi_quarter",
    "quarters":      "multi_quarter",
    "long_term":     "multi_quarter",
}


def normalize_interpretation_df(df: pl.DataFrame, agent: str) -> pl.DataFrame:
    """
    Normalise a single agent's interpretation DataFrame to the v2 schema.

    Returns a DataFrame with exactly the V2_COLUMNS in order,
    with correct Polars dtypes:
        list[str] columns : List(String)
        float columns     : Float32
        bool columns      : Boolean
        all others        : String

    Handles schema drift across agent versions:
    - List(String) columns that may arrive as native lists or JSON strings
    - Missing optional columns (interpreted_at, interpreted_by, etc.)
    - Prometheus cross_country as List(String) instead of Boolean
    - Daedalus causal_chain as String instead of List(String)
    - Apollo v2 format with native List columns (not JSON strings)
    """
    agent = agent.lower()

    # Guard: reject completely incompatible schemas (e.g. Apollo trading-analysis format)
    if "country_iso3" not in df.columns:
        raise KeyError(
            f"'country_iso3' not found — file appears to use an incompatible schema "
            f"(columns: {df.columns[:6]}...)"
        )

    # Work on a dict of Series for easy manipulation
    out: dict[str, pl.Series] = {}
    n = len(df)

    def col(name: str) -> pl.Series:
        return df[name]

    def str_col(series: pl.Series) -> pl.Series:
        return series.cast(pl.String).fill_null("").alias(series.name)

    def literal_str(value: str, name: str) -> pl.Series:
        return pl.Series(name, [value] * n, dtype=pl.String)

    def literal_list(value: list, name: str) -> pl.Series:
        return pl.Series(name, [value] * n, dtype=pl.List(pl.String))

    def literal_float(value: float, name: str) -> pl.Series:
        return pl.Series(name, [value] * n, dtype=pl.Float32)

    def literal_bool(value: bool, name: str) -> pl.Series:
        return pl.Series(name, [value] * n, dtype=pl.Boolean)

    def _coerce_to_list(series: pl.Series, csv_fallback: bool = False) -> pl.Series:
        """Accept List(String), JSON-string, or comma-CSV string → List(String)."""
        if series.dtype == pl.List(pl.String):
            return series
        if csv_fallback:
            return (
                series.fill_null("")
                .map_elements(
                    lambda s: [x.strip() for x in s.split(",") if x.strip()],
                    return_dtype=pl.List(pl.String),
                )
            )
        return (
            series.fill_null("[]")
            .map_elements(_parse_json_list, return_dtype=pl.List(pl.String))
        )

    # ── item_id, country_iso3 ─────────────────────────────────────────────────
    out["item_id"]      = str_col(col("item_id"))
    out["country_iso3"] = str_col(col("country_iso3"))

    # ── interpreted_at — optional; default to now if absent ───────────────────
    if "interpreted_at" in df.columns:
        out["interpreted_at"] = str_col(col("interpreted_at"))
    else:
        from datetime import datetime, timezone
        out["interpreted_at"] = literal_str(
            datetime.now(timezone.utc).isoformat(), "interpreted_at"
        )

    # ── interpreted_by ────────────────────────────────────────────────────────
    if "interpreted_by" in df.columns:
        out["interpreted_by"] = str_col(col("interpreted_by"))
    elif "agent" in df.columns:
        out["interpreted_by"] = str_col(col("agent"))
    else:
        out["interpreted_by"] = literal_str(agent, "interpreted_by")

    # ── headline ──────────────────────────────────────────────────────────────
    if "headline" in df.columns:
        out["headline"] = str_col(col("headline"))
    else:
        out["headline"] = literal_str("", "headline")

    # ── published_at ──────────────────────────────────────────────────────────
    if "published_at" in df.columns:
        out["published_at"] = str_col(col("published_at"))
    else:
        out["published_at"] = literal_str("", "published_at")

    # ── intent ────────────────────────────────────────────────────────────────
    out["intent"] = str_col(col("intent"))

    # ── actor_ids ─────────────────────────────────────────────────────────────
    # All agents may now write native List(String); _coerce_to_list handles both
    if agent == "prometheus":
        out["actor_ids"] = _coerce_to_list(col("actor_ids"), csv_fallback=True).alias("actor_ids")
    else:
        out["actor_ids"] = _coerce_to_list(col("actor_ids")).alias("actor_ids")

    # ── new_actors (stored as JSON string in v2; agents may write List or String) ─
    if "new_actors" in df.columns:
        raw_na = col("new_actors")
        if raw_na.dtype == pl.List(pl.String):
            # Serialize list back to JSON string (v2 schema requires String).
            # map_elements passes a Polars Series per element — convert to Python list first.
            out["new_actors"] = (
                raw_na
                .map_elements(
                    lambda lst: json.dumps(list(lst) if lst is not None else []),
                    return_dtype=pl.String,
                )
                .alias("new_actors")
            )
        else:
            out["new_actors"] = str_col(raw_na)
    else:
        out["new_actors"] = literal_str(_DEFAULT_NEW_ACTORS, "new_actors")

    # ── affected_stats ────────────────────────────────────────────────────────
    if agent == "prometheus":
        out["affected_stats"] = _coerce_to_list(col("affected_stats"), csv_fallback=True).alias("affected_stats")
    else:
        out["affected_stats"] = _coerce_to_list(col("affected_stats")).alias("affected_stats")

    # ── stat_direction ────────────────────────────────────────────────────────
    if "stat_direction" in df.columns:
        # daedalus: JSON dict string or "{}" — keep as-is
        out["stat_direction"] = str_col(col("stat_direction"))
    else:
        out["stat_direction"] = literal_str(_DEFAULT_STAT_DIRECTION, "stat_direction")

    # ── estimated_magnitude ───────────────────────────────────────────────────
    if "estimated_magnitude" in df.columns:
        out["estimated_magnitude"] = str_col(col("estimated_magnitude"))
    else:
        out["estimated_magnitude"] = literal_str(_DEFAULT_MAGNITUDE, "estimated_magnitude")

    # ── causal_chain ──────────────────────────────────────────────────────────
    if "causal_chain" in df.columns:
        raw_cc = col("causal_chain")
        if raw_cc.dtype == pl.List(pl.String):
            out["causal_chain"] = raw_cc.alias("causal_chain")
        else:
            # Single string (Daedalus) → wrap in a one-element list
            out["causal_chain"] = (
                raw_cc
                .fill_null("")
                .map_elements(lambda s: [s] if s else [], return_dtype=pl.List(pl.String))
                .alias("causal_chain")
            )
    else:
        out["causal_chain"] = literal_list([], "causal_chain")

    # ── sentiment ─────────────────────────────────────────────────────────────
    raw_sent = col("sentiment")
    if raw_sent.dtype == pl.String or raw_sent.dtype == pl.Utf8:
        out["sentiment"] = (
            raw_sent
            .fill_null("neutral")
            .map_elements(
                lambda s: float(_SENTIMENT_STR_TO_FLOAT.get(s.lower().strip(), 0.0)),
                return_dtype=pl.Float32,
            )
            .alias("sentiment")
        )
    else:
        out["sentiment"] = raw_sent.cast(pl.Float32).alias("sentiment")

    # ── urgency ───────────────────────────────────────────────────────────────
    out["urgency"] = (
        col("urgency")
        .fill_null("low")
        .map_elements(
            lambda s: _URGENCY_NORMALISE.get(s.lower().strip(), "low"),
            return_dtype=pl.String,
        )
        .alias("urgency")
    )

    # ── confidence ────────────────────────────────────────────────────────────
    if "confidence" in df.columns:
        out["confidence"] = col("confidence").cast(pl.Float32).alias("confidence")
    else:
        out["confidence"] = literal_float(_DEFAULT_CONFIDENCE, "confidence")

    # ── cross_country + cross_country_iso3 ───────────────────────────────────
    if agent == "prometheus":
        # cross_country may be: comma-sep string, or List(String) (newer format)
        raw_cc = col("cross_country")
        if raw_cc.dtype == pl.List(pl.String):
            iso3_lists = raw_cc.fill_null([])
        else:
            iso3_lists = (
                raw_cc.fill_null("").map_elements(
                    lambda s: [x.strip() for x in s.split(",") if x.strip()],
                    return_dtype=pl.List(pl.String),
                )
            )
        out["cross_country_iso3"] = iso3_lists.alias("cross_country_iso3")
        out["cross_country"] = (
            iso3_lists
            .map_elements(lambda lst: len(lst) > 1, return_dtype=pl.Boolean)
            .alias("cross_country")
        )
    elif agent == "apollo":
        out["cross_country"] = col("cross_country").cast(pl.Boolean).alias("cross_country")
        # cross_country_iso3 may be native List(String) or JSON string
        out["cross_country_iso3"] = _coerce_to_list(col("cross_country_iso3")).alias("cross_country_iso3")
    else:
        # daedalus: cross_country is Boolean, cross_country_iso3 is List(String)
        out["cross_country"] = col("cross_country").cast(pl.Boolean).alias("cross_country")
        cross_col = "affected_countries" if "affected_countries" in df.columns else "cross_country_iso3"
        out["cross_country_iso3"] = _coerce_to_list(col(cross_col)).alias("cross_country_iso3")

    # ── notes ─────────────────────────────────────────────────────────────────
    if "notes" in df.columns:
        out["notes"] = str_col(col("notes"))
    elif "interpretation_note" in df.columns:
        out["notes"] = str_col(col("interpretation_note"))
    else:
        out["notes"] = literal_str("", "notes")

    # ── Hybrid Extension fields (normalized, tolerate vocab drift) ────────────
    if "risk_reward_horizon" in df.columns:
        out["risk_reward_horizon"] = (
            col("risk_reward_horizon")
            .fill_null("")
            .map_elements(
                lambda s: _RISK_REWARD_HORIZON_NORMALISE.get(
                    str(s).lower().strip(), str(s)
                ),
                return_dtype=pl.String,
            )
            .alias("risk_reward_horizon")
        )
    else:
        out["risk_reward_horizon"] = literal_str("", "risk_reward_horizon")

    if "volatility_risk" in df.columns:
        out["volatility_risk"] = (
            col("volatility_risk")
            .fill_null("")
            .map_elements(
                lambda s: _VOLATILITY_RISK_NORMALISE.get(
                    str(s).lower().strip(), str(s).upper()
                ) if s else "",
                return_dtype=pl.String,
            )
            .alias("volatility_risk")
        )
    else:
        out["volatility_risk"] = literal_str("", "volatility_risk")

    if "operational_risk" in df.columns:
        out["operational_risk"] = str_col(col("operational_risk"))
    else:
        out["operational_risk"] = literal_str("", "operational_risk")

    if "supply_chain_exposure" in df.columns:
        out["supply_chain_exposure"] = str_col(col("supply_chain_exposure"))
    else:
        out["supply_chain_exposure"] = literal_str("", "supply_chain_exposure")

    # ── prediction fields (JSON strings, nullable) ─────────────────────────────
    for pred_col in ("market_impact", "sector_impact", "ticker_impact"):
        if pred_col in df.columns:
            out[pred_col] = str_col(col(pred_col))
        else:
            out[pred_col] = literal_str(_DEFAULT_PREDICTION, pred_col)

    # ── Assemble in V2_COLUMNS order ──────────────────────────────────────────
    return pl.DataFrame({k: out[k] for k in V2_COLUMNS})


def _parse_json_list(s: str) -> list[str]:
    """Parse a JSON-encoded list-of-strings, returning [] on any error."""
    if not s:
        return []
    try:
        result = json.loads(s)
        if isinstance(result, list):
            return [str(x) for x in result]
        return []
    except (json.JSONDecodeError, TypeError):
        return []


# ── Collection ────────────────────────────────────────────────────────────────

def collect_interpretations(
    date_str: Optional[str] = None,
    out_path: Optional[Path] = None,
    dry_run: bool = False,
) -> pl.DataFrame:
    """
    Read all interpretations_<agent>_*.parquet files (optionally filtered by date),
    normalise each to v2 schema, concatenate, and write the unified parquet.

    Parameters
    ----------
    date_str : "YYYY-MM-DD" to restrict to parquets whose filename contains the date;
               None = include all available parquets.
    out_path : override output path; default = INTERP_DIR/interpretations_unified_DATE.parquet
    dry_run  : if True, normalise and return df but do NOT write the parquet.

    Returns
    -------
    Unified DataFrame (v2 schema).
    """
    pattern = "interpretations_*.parquet"
    all_files = sorted(INTERP_DIR.glob(pattern))

    # exclude any existing unified parquets to avoid re-reading our own output
    agent_files = [f for f in all_files if "unified" not in f.name]

    if date_str:
        agent_files = [f for f in agent_files if date_str.replace("-", "") in f.name
                       or date_str in f.name]

    if not agent_files:
        logger.warning("No interpretation parquets found (date=%s)", date_str)
        return pl.DataFrame()

    frames: list[pl.DataFrame] = []
    for path in agent_files:
        # derive agent from filename: interpretations_<agent>_<ts>.parquet
        parts = path.stem.split("_")
        if len(parts) < 2:
            logger.warning("Cannot determine agent from filename %s — skipping", path.name)
            continue
        agent = parts[1]

        try:
            raw = pl.read_parquet(path)
            normalised = normalize_interpretation_df(raw, agent)
            frames.append(normalised)
            logger.info("Normalised %s: %d rows (agent=%s)", path.name, len(normalised), agent)
        except Exception as exc:
            logger.error("Failed to normalise %s: %s", path.name, exc)
            continue

    if not frames:
        logger.warning("No frames produced — nothing to write")
        return pl.DataFrame()

    unified = pl.concat(frames, how="vertical_relaxed")

    if date_str:
        date_tag = date_str.replace("-", "")
    else:
        date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")

    default_out = INTERP_DIR / f"interpretations_unified_{date_tag}.parquet"
    out_path = out_path or default_out

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        unified.write_parquet(out_path)
        logger.info(
            "Wrote unified parquet: %s (%d rows, %d cols)",
            out_path.name, len(unified), len(unified.columns),
        )
    else:
        logger.info(
            "DRY RUN — would write %s (%d rows, %d cols)",
            out_path.name, len(unified), len(unified.columns),
        )

    return unified


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Normalise and merge agent interpretation parquets into v2 unified format.",
    )
    p.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Only include parquets containing this date in their filename.",
    )
    p.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Override output path (default: data/interpretations/interpretations_unified_DATE.parquet).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Normalise and print stats without writing the output parquet.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = Path(args.out) if args.out else None
    unified = collect_interpretations(
        date_str=args.date,
        out_path=out,
        dry_run=args.dry_run,
    )
    if unified.is_empty():
        sys.exit(1)

    print(f"\nUnified schema ({len(unified)} rows × {len(unified.columns)} cols):")
    for col_name, dtype in unified.schema.items():
        print(f"  {col_name}: {dtype}")

    print(f"\nAgent breakdown:")
    breakdown = (
        unified
        .group_by("interpreted_by")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    for row in breakdown.iter_rows(named=True):
        print(f"  {row['interpreted_by']}: {row['n']} rows")

    urgency_dist = (
        unified
        .group_by("urgency")
        .agg(pl.len().alias("n"))
        .sort("urgency")
    )
    print(f"\nUrgency distribution:")
    for row in urgency_dist.iter_rows(named=True):
        print(f"  {row['urgency']}: {row['n']}")


if __name__ == "__main__":
    main()
