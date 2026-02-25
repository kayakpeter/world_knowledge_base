from __future__ import annotations
import pytest
from knowledge_base.company_schema import (
    compute_reality_score,
    derive_risk_tier,
    derive_trading_profile,
    TIER_CONFIG,
)


def test_compute_reality_score_shell():
    """Shell company scores near zero."""
    attrs = dict(
        employee_count=0,
        location_count=1,
        location_type="registered_agent",
        named_customers_count=0,
        has_shipped_product=False,
        auditor_tier="micro",
        ceo_verifiable=False,
    )
    assert compute_reality_score(attrs) == 0.0


def test_compute_reality_score_real():
    """Real company hits max score."""
    attrs = dict(
        employee_count=100,
        location_count=5,
        location_type="multi_site",
        named_customers_count=3,
        has_shipped_product=True,
        auditor_tier="big4",
        ceo_verifiable=True,
    )
    assert compute_reality_score(attrs) == 1.0


def test_compute_reality_score_partial():
    """Partial company scores in between."""
    attrs = dict(
        employee_count=20,       # +0.20
        location_count=1,        # no points (not > 1)
        location_type="real_office",  # +0.10
        named_customers_count=0, # no points
        has_shipped_product=True,# +0.15
        auditor_tier="mid",      # +0.10
        ceo_verifiable=False,    # no points
    )
    assert abs(compute_reality_score(attrs) - 0.55) < 0.001


def test_derive_risk_tier_shell():
    assert derive_risk_tier(0.10) == "SHELL"


def test_derive_risk_tier_micro_op():
    assert derive_risk_tier(0.40) == "MICRO_OP"


def test_derive_risk_tier_real():
    assert derive_risk_tier(0.70) == "REAL"


def test_derive_trading_profile_shell():
    profile = derive_trading_profile("SHELL")
    assert profile["size_multiplier"] == 0.25
    assert profile["target_pct"] == 5.0
    assert profile["reversal_pct"] == 1.5


def test_derive_trading_profile_real():
    profile = derive_trading_profile("REAL")
    assert profile["size_multiplier"] == 1.0
    assert profile["target_pct"] == 10.0


def test_tier_config_has_all_tiers():
    for tier in ("SHELL", "MICRO_OP", "REAL"):
        assert tier in TIER_CONFIG
        cfg = TIER_CONFIG[tier]
        assert "size_multiplier" in cfg
        assert "target_pct" in cfg
        assert "reversal_pct" in cfg
        assert "trading_note" in cfg


# ── Seed loader tests ─────────────────────────────────────────────────────────

from pathlib import Path
import polars as pl
from knowledge_base.company_schema import load_company_seeds


def test_load_company_seeds_returns_dataframe(tmp_path):
    """load_company_seeds returns a Polars DataFrame with required columns."""
    seed = pl.DataFrame({
        "ticker":               ["AAPL"],
        "name":                 ["Apple Inc"],
        "exchange":             ["NASDAQ"],
        "sector":               ["Tech"],
        "country_iso3":         ["USA"],
        "employee_count":       [100_000],
        "location_count":       [50],
        "location_type":        ["multi_site"],
        "named_customers_count":[10],
        "has_shipped_product":  [True],
        "auditor":              ["Ernst & Young"],
        "auditor_tier":         ["big4"],
        "years_operating":      [45.0],
        "ceo_verifiable":       [True],
    })
    seed.write_parquet(tmp_path / "company_seeds.parquet")
    df = load_company_seeds(tmp_path / "company_seeds.parquet")
    assert len(df) == 1
    assert "ticker" in df.columns
    assert "risk_tier" in df.columns   # derived column added by loader


def test_load_company_seeds_missing_file(tmp_path):
    """Returns empty DataFrame when seed file does not exist."""
    df = load_company_seeds(tmp_path / "nonexistent.parquet")
    assert len(df) == 0
