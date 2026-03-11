import polars as pl
import pytest
from datetime import date, timedelta
from pathlib import Path
from processing.geo_stress_scorer import compute_stress_scores, URGENCY_NUMERIC

BASE_DIR = Path(__file__).parent.parent


def _make_interp_df(rows: list[dict]) -> pl.DataFrame:
    """Build minimal interpretation DataFrame for testing."""
    return pl.DataFrame({
        "country_iso3": [r["iso3"] for r in rows],
        "urgency": [r["urgency"] for r in rows],
        "published_at": [r["published_at"] for r in rows],
        "cross_country": [r.get("cross_country", False) for r in rows],
        "cross_country_iso3": [r.get("cross_country_iso3", []) for r in rows],
        "sentiment": [r.get("sentiment", 0.0) for r in rows],
    })


def test_urgency_numeric_keys():
    """All expected urgency levels map to floats in [0, 1]."""
    for k, v in URGENCY_NUMERIC.items():
        assert 0.0 <= v <= 1.0, f"{k} → {v} out of range"


def test_single_critical_item_produces_high_stress():
    """A single critical item for USA from today should produce stress > 0.5."""
    today = date.today().isoformat()
    df = _make_interp_df([
        {"iso3": "USA", "urgency": "critical", "published_at": today}
    ])
    scores = compute_stress_scores([df], reference_date=date.today())
    assert "United States" in scores
    assert scores["United States"] > 0.5


def test_old_items_decay_to_near_zero():
    """Items 14+ days old should contribute near-zero stress."""
    old_date = (date.today() - timedelta(days=14)).isoformat()
    df = _make_interp_df([
        {"iso3": "USA", "urgency": "critical", "published_at": old_date}
    ])
    scores = compute_stress_scores([df], reference_date=date.today())
    usa_score = scores.get("United States", 0.0)
    assert usa_score < 0.05, f"Expected near-zero, got {usa_score}"


def test_cross_country_items_contribute_at_reduced_weight():
    """Cross-country items should contribute at 0.4x weight vs direct items."""
    today = date.today().isoformat()
    # Direct critical item for Japan
    df_direct = _make_interp_df([
        {"iso3": "JPN", "urgency": "critical", "published_at": today}
    ])
    # Cross-country critical item where JPN is in cross_country_iso3
    df_cross = _make_interp_df([
        {
            "iso3": "USA",
            "urgency": "critical",
            "published_at": today,
            "cross_country": True,
            "cross_country_iso3": ["JPN"],
        }
    ])
    scores_direct = compute_stress_scores([df_direct], reference_date=date.today())
    scores_cross = compute_stress_scores([df_cross], reference_date=date.today())
    jpn_direct = scores_direct.get("Japan", 0.0)
    jpn_cross = scores_cross.get("Japan", 0.0)
    assert jpn_direct > jpn_cross, "Direct should produce higher stress than cross-country"
    assert jpn_cross > 0, "Cross-country should still produce non-zero stress"


def test_unknown_iso3_does_not_crash():
    """ISO3 codes not in COUNTRY_CODES should be silently skipped."""
    today = date.today().isoformat()
    df = _make_interp_df([{"iso3": "XYZ", "urgency": "critical", "published_at": today}])
    scores = compute_stress_scores([df], reference_date=date.today())
    assert "XYZ" not in scores


def test_countries_with_no_data_return_zero():
    """Countries with no interpretation data should score 0.0."""
    today = date.today().isoformat()
    df = _make_interp_df([{"iso3": "BRA", "urgency": "low", "published_at": today}])
    scores = compute_stress_scores([df], reference_date=date.today())
    # Canada has no items in this df — should not appear or score 0
    assert scores.get("Canada", 0.0) == 0.0
