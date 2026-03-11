"""
Geopolitical Stress Scorer

Aggregates urgency signals from daily interpretation parquets into a
per-country stress score (0.0–1.0) using a 7-day exponential decay window.

Used by geo_fusion.py to blend with HMM state posteriors.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Numeric weight for each urgency level
URGENCY_NUMERIC: dict[str, float] = {
    "low":      0.00,
    "medium":   0.00,   # routine economic items — not a geopolitical stress signal
    "elevated": 0.45,   # currently unused by pipeline (produces low/medium/high/critical only)
    "high":     0.70,
    "critical": 1.00,
}

# Exponential decay half-life in days
DECAY_HALF_LIFE_DAYS: float = 3.0

# Cross-country items count at reduced weight
CROSS_COUNTRY_WEIGHT: float = 0.40

# Rolling window (days of history to load)
WINDOW_DAYS: int = 7

# Root data dir — interpretations live here
_INTERP_DIR = Path(__file__).parent.parent / "data" / "interpretations"


def _recency_weight(days_old: float) -> float:
    """Exponential decay weight; 0 days old = 1.0; DECAY_HALF_LIFE days old = 0.5."""
    return float(np.exp(-days_old * np.log(2) / DECAY_HALF_LIFE_DAYS))


def _load_interpretation_files(
    window_days: int = WINDOW_DAYS,
    interp_dir: Path = _INTERP_DIR,
    reference_date: date | None = None,
) -> list[pl.DataFrame]:
    """Load all unified interpretation parquets from the last window_days."""
    ref = reference_date or date.today()
    dfs = []
    for delta in range(window_days + 1):
        d = ref - timedelta(days=delta)
        fname = interp_dir / f"interpretations_unified_{d.strftime('%Y%m%d')}.parquet"
        if fname.exists():
            try:
                dfs.append(pl.read_parquet(fname))
            except Exception as exc:
                logger.warning("Could not read %s: %s", fname, exc)
    return dfs


def compute_stress_scores(
    dfs: list[pl.DataFrame],
    reference_date: date | None = None,
) -> dict[str, float]:
    """
    Compute per-country stress scores from a list of interpretation DataFrames.

    Args:
        dfs: list of interpretation DataFrames (one per day, oldest first or any order)
        reference_date: the "today" reference for recency decay (default: date.today())

    Returns:
        dict mapping full country name → stress score in [0.0, 1.0]
    """
    # Import here to avoid circular deps at module load
    from config.settings import COUNTRY_CODES
    iso3_to_country: dict[str, str] = {
        v["iso3"]: k for k, v in COUNTRY_CODES.items() if "iso3" in v
    }

    ref = reference_date or date.today()

    # Accumulate (weighted_urgency * recency_weight) per country name
    accum: dict[str, list[float]] = {}

    for df in dfs:
        if df.is_empty():
            continue
        required = {"country_iso3", "urgency", "published_at"}
        if not required.issubset(set(df.columns)):
            continue

        for row in df.iter_rows(named=True):
            urg_val = URGENCY_NUMERIC.get(row.get("urgency", "low"), 0.0)
            if urg_val == 0.0:
                continue  # skip low-urgency items entirely

            # Parse date for recency weight
            published_str = (row.get("published_at") or "")[:10]
            try:
                pub_date = date.fromisoformat(published_str)
                days_old = max(0, (ref - pub_date).days)
            except ValueError:
                days_old = WINDOW_DAYS  # treat unparseable as oldest bucket

            w_recency = _recency_weight(days_old)

            # Direct item: full weight
            iso3 = row.get("country_iso3", "")
            country = iso3_to_country.get(iso3)
            if country:
                accum.setdefault(country, []).append(urg_val * w_recency)

            # Cross-country items: 0.4x weight for referenced countries
            if row.get("cross_country") and row.get("cross_country_iso3"):
                for x_iso3 in (row["cross_country_iso3"] or []):
                    x_country = iso3_to_country.get(x_iso3)
                    if x_country and x_country != country:
                        accum.setdefault(x_country, []).append(
                            urg_val * w_recency * CROSS_COUNTRY_WEIGHT
                        )

    # Average within each country; countries with no data → not included (→ 0.0 via .get())
    scores: dict[str, float] = {}
    for country, weights in accum.items():
        scores[country] = float(np.mean(weights)) if weights else 0.0

    logger.debug("Geo stress scores: %s", {k: f"{v:.3f}" for k, v in sorted(scores.items())})
    return scores


def load_and_score(
    window_days: int = WINDOW_DAYS,
    interp_dir: Path = _INTERP_DIR,
    reference_date: date | None = None,
) -> dict[str, float]:
    """Convenience: load interpretation files and return stress scores."""
    dfs = _load_interpretation_files(window_days, interp_dir, reference_date)
    if not dfs:
        logger.warning("No interpretation files found in last %d days", window_days)
        return {}
    return compute_stress_scores(dfs, reference_date)
