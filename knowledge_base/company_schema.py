"""
Company entity schema for the knowledge graph.

Defines reality scoring, risk tier derivation, and trading profiles
for Company nodes in the NetworkX graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ── Trading profile by risk tier ─────────────────────────────────────────────

TIER_CONFIG: dict[str, dict] = {
    "SHELL": {
        "size_multiplier": 0.25,
        "target_pct":      5.0,
        "reversal_pct":    1.5,
        "trading_note":    "Take the money and run",
    },
    "MICRO_OP": {
        "size_multiplier": 0.75,
        "target_pct":      10.0,
        "reversal_pct":    2.0,
        "trading_note":    "Normal, slight caution",
    },
    "REAL": {
        "size_multiplier": 1.0,
        "target_pct":      10.0,
        "reversal_pct":    2.0,
        "trading_note":    "Hold for full target",
    },
}

# Default when ticker is unknown
_DEFAULT_TIER = "MICRO_OP"

# ── Reality scoring ───────────────────────────────────────────────────────────

def compute_reality_score(attrs: dict) -> float:
    """
    Compute a 0.0–1.0 reality score from 7 binary/graded signals.

    attrs keys required:
        employee_count, location_count, location_type,
        named_customers_count, has_shipped_product,
        auditor_tier, ceo_verifiable
    """
    score = 0.0
    if attrs.get("employee_count", 0) > 10:
        score += 0.20
    if attrs.get("location_count", 0) > 1:
        score += 0.15
    if attrs.get("location_type") in ("real_office", "multi_site"):
        score += 0.10
    if attrs.get("named_customers_count", 0) > 0:
        score += 0.20
    if attrs.get("has_shipped_product", False):
        score += 0.15
    if attrs.get("auditor_tier") in ("big4", "mid"):
        score += 0.10
    if attrs.get("ceo_verifiable", False):
        score += 0.10
    return round(score, 2)


def derive_risk_tier(reality_score: float) -> str:
    """Map reality_score → SHELL / MICRO_OP / REAL."""
    if reality_score >= 0.60:
        return "REAL"
    if reality_score >= 0.25:
        return "MICRO_OP"
    return "SHELL"


def derive_trading_profile(risk_tier: str) -> dict:
    """Return trading profile dict for the given risk tier."""
    return dict(TIER_CONFIG.get(risk_tier, TIER_CONFIG[_DEFAULT_TIER]))


# ── Company node dataclass (for type-safe construction) ───────────────────────

RiskTier = Literal["SHELL", "MICRO_OP", "REAL"]
LocationType = Literal["registered_agent", "real_office", "multi_site"]
AuditorTier = Literal["big4", "mid", "micro", "unknown"]
RealitySource = Literal["seed", "llm", "hybrid"]


@dataclass
class CompanyNode:
    """
    Full attribute set for a Company node in the knowledge graph.
    Pass to graph.add_node(ticker, **company_node.as_dict()).
    """
    # Identity
    ticker: str
    name: str
    exchange: str = "OTC"
    sector: str = "Other"
    country_iso3: str = "USA"

    # Reality attributes
    employee_count: int = 0
    location_count: int = 1
    location_type: LocationType = "registered_agent"
    named_customers: list[str] = field(default_factory=list)
    named_customers_count: int = 0
    has_shipped_product: bool = False
    auditor: str = "unknown"
    auditor_tier: AuditorTier = "unknown"
    years_operating: float = 0.0
    ceo_verifiable: bool = False

    # Derived (computed on construction)
    risk_tier: RiskTier = field(init=False)
    reality_score: float = field(init=False)
    reality_source: RealitySource = "seed"
    reality_updated: str = ""

    # Trading profile (derived from risk_tier)
    size_multiplier: float = field(init=False)
    target_pct: float = field(init=False)
    reversal_pct: float = field(init=False)
    trading_note: str = field(init=False)

    def __post_init__(self) -> None:
        # Sync named_customers_count with list length if list provided
        if self.named_customers:
            self.named_customers_count = len(self.named_customers)

        self.reality_score = compute_reality_score(self.__dict__)
        self.risk_tier = derive_risk_tier(self.reality_score)
        profile = derive_trading_profile(self.risk_tier)
        self.size_multiplier = profile["size_multiplier"]
        self.target_pct      = profile["target_pct"]
        self.reversal_pct    = profile["reversal_pct"]
        self.trading_note    = profile["trading_note"]

    def as_dict(self) -> dict:
        """Return all attributes as a plain dict for graph.add_node()."""
        d = {k: v for k, v in self.__dict__.items()}
        d["node_type"] = "Company"
        return d


# ── Seed loader ───────────────────────────────────────────────────────────────

def load_company_seeds(path: Path) -> "pl.DataFrame":
    """
    Load company seed parquet and add derived columns (reality_score, risk_tier).
    Returns empty DataFrame if file does not exist.
    """
    import polars as pl

    if not path.exists():
        return pl.DataFrame()

    df = pl.read_parquet(path)

    # Compute derived columns row-by-row
    scores = []
    tiers = []
    for row in df.iter_rows(named=True):
        score = compute_reality_score(row)
        scores.append(score)
        tiers.append(derive_risk_tier(score))

    return df.with_columns([
        pl.Series("reality_score", scores, dtype=pl.Float32),
        pl.Series("risk_tier", tiers, dtype=pl.Utf8),
    ])
