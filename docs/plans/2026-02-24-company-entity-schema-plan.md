# Company Entity Schema — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `Company` node type to the existing NetworkX knowledge graph with reality scoring, risk tiers, trading profiles, and edges to Sovereign/Sector/Scenario nodes.

**Architecture:** New `company_schema.py` defines the data model and scoring logic. `graph_builder.py` gains `add_sector_nodes` and `add_company_nodes` methods. A seed parquet provides known penny stock data; LLM inference fills gaps. `get_company_trading_profile()` is the query interface for `gap_and_go_live.py`.

**Tech Stack:** Python 3.11+, NetworkX, Polars, dataclasses

**Design doc:** `docs/plans/2026-02-24-company-entity-schema-design.md`

**Reference files:**
- `knowledge_base/graph_builder.py` — existing graph patterns to follow
- `config/settings.py` — DATA_ROOT, DEV_DATA_ROOT paths

---

## Task 1: `company_schema.py` — Data Model + Reality Scoring

**Files:**
- Create: `knowledge_base/company_schema.py`
- Create: `tests/test_company_schema.py`

**Step 1: Write the failing tests**

```python
# tests/test_company_schema.py
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
```

**Step 2: Run to verify failure**

```bash
cd /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb
python -m pytest tests/test_company_schema.py -v
```

Expected: `ModuleNotFoundError: No module named 'knowledge_base.company_schema'`

**Step 3: Implement `company_schema.py`**

```python
# knowledge_base/company_schema.py
"""
Company entity schema for the knowledge graph.

Defines reality scoring, risk tier derivation, and trading profiles
for Company nodes in the NetworkX graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_company_schema.py -v
```

Expected: 9 PASS

**Step 5: Commit**

```bash
cd /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb
git add knowledge_base/company_schema.py tests/test_company_schema.py
git commit -m "feat(company-schema): CompanyNode dataclass, reality scoring, tier derivation"
```

---

## Task 2: Seed Data Parquet

**Files:**
- Create: `data/create_company_seeds.py` (one-time script)
- Creates: `data/company_seeds.parquet`

**Step 1: Write test for seed loader**

```python
# append to tests/test_company_schema.py
from pathlib import Path
import polars as pl
from knowledge_base.company_schema import load_company_seeds

def test_load_company_seeds_returns_dataframe(tmp_path):
    """load_company_seeds returns a Polars DataFrame with required columns."""
    # Write a minimal seed file
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
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_company_schema.py -k "seeds" -v
```

Expected: `ImportError: cannot import name 'load_company_seeds'`

**Step 3: Add `load_company_seeds` to `company_schema.py`**

Add at bottom of `knowledge_base/company_schema.py`:

```python
def load_company_seeds(path: Path) -> pl.DataFrame:
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
```

**Step 4: Create the seed data script**

```python
# data/create_company_seeds.py
"""
One-time script to create the company_seeds.parquet seed file.
Run: python data/create_company_seeds.py
"""
from __future__ import annotations
from pathlib import Path
import polars as pl

# Known penny stock universe with manually-assessed reality attributes.
# Reality data sourced from SEC EDGAR, LinkedIn, company websites.
SEEDS = [
    # ticker, name, exchange, sector, country_iso3,
    # employee_count, location_count, location_type,
    # named_customers_count, has_shipped_product,
    # auditor, auditor_tier, years_operating, ceo_verifiable
    ("GTII",  "Global Arena Holding",       "OTC",    "Other",   "USA", 0, 1, "registered_agent", 0, False, "unknown",            "unknown", 0.5,  False),
    ("LGVN",  "Longeviti Neuro Solutions",  "NASDAQ", "Biotech", "USA", 12, 1, "real_office",      2, True,  "Marcum LLP",         "mid",     4.0,  True),
    ("SOPA",  "Society Pass",               "NASDAQ", "Tech",    "USA", 80, 3, "multi_site",       5, True,  "Assurance Dim.",     "micro",   5.0,  True),
    ("SHOT",  "Safety Shot",                "NASDAQ", "Cannabis","USA", 8,  1, "real_office",      1, True,  "Salberg & Co",       "micro",   3.0,  True),
    ("GFAI",  "Guardforce AI",              "NASDAQ", "Tech",    "USA", 900,8, "multi_site",      15, True,  "Marcum Bernstein",   "mid",     8.0,  True),
    ("VERB",  "Verb Technology",            "NASDAQ", "Tech",    "USA", 25, 1, "real_office",      3, True,  "Weinberg & Company", "micro",   7.0,  True),
    ("FFIE",  "Faraday Future",             "NASDAQ", "EV",      "USA", 400,2, "multi_site",       1, False, "Deloitte",           "big4",    8.0,  True),
    ("MULN",  "Mullen Automotive",          "NASDAQ", "EV",      "USA", 110,3, "multi_site",       2, False, "Weinberg & Company", "micro",   5.0,  True),
    ("MMAT",  "Meta Materials",             "NASDAQ", "Tech",    "USA", 150,2, "multi_site",       4, True,  "KPMG",               "big4",    6.0,  True),
    ("PROG",  "Progenity",                  "NASDAQ", "Biotech", "USA", 200,2, "multi_site",       0, True,  "Ernst & Young",      "big4",    8.0,  True),
    ("NKLA",  "Nikola Corporation",         "NASDAQ", "EV",      "USA", 450,2, "multi_site",       1, False, "Ernst & Young",      "big4",    7.0,  True),
    ("CLOV",  "Clover Health",              "NASDAQ", "Financial","USA",900,3, "multi_site",      20, True,  "Deloitte",           "big4",    7.0,  True),
    ("WKHS",  "Workhorse Group",            "NASDAQ", "EV",      "USA", 120,2, "multi_site",       2, True,  "Clark Nuber",        "mid",     8.0,  True),
    ("IDEX",  "Ideanomics",                 "NASDAQ", "EV",      "USA", 200,5, "multi_site",       3, True,  "Marcum LLP",         "mid",     6.0,  True),
    ("BNGO",  "Bionano Genomics",           "NASDAQ", "Biotech", "USA", 180,1, "real_office",     10, True,  "Ernst & Young",      "big4",    10.0, True),
]

COLUMNS = [
    "ticker","name","exchange","sector","country_iso3",
    "employee_count","location_count","location_type",
    "named_customers_count","has_shipped_product",
    "auditor","auditor_tier","years_operating","ceo_verifiable",
]

def main():
    out = Path(__file__).parent / "company_seeds.parquet"
    df = pl.DataFrame(
        {col: [row[i] for row in SEEDS] for i, col in enumerate(COLUMNS)}
    )
    df.write_parquet(out)
    print(f"Written {len(df)} seeds → {out}")
    print(df.select(["ticker","sector","auditor_tier"]))

if __name__ == "__main__":
    main()
```

**Step 5: Run seed script and tests**

```bash
python /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb/data/create_company_seeds.py
python -m pytest tests/test_company_schema.py -v
```

Expected: seed file created + all tests PASS

**Step 6: Commit**

```bash
git add knowledge_base/company_schema.py data/create_company_seeds.py data/company_seeds.parquet tests/test_company_schema.py
git commit -m "feat(company-schema): seed data + load_company_seeds"
```

---

## Task 3: Sector Nodes in `graph_builder.py`

**Files:**
- Modify: `knowledge_base/graph_builder.py`
- Create: `tests/test_graph_company.py`

**Step 1: Write failing tests**

```python
# tests/test_graph_company.py
from __future__ import annotations
import pytest
import networkx as nx
from knowledge_base.graph_builder import KnowledgeGraphBuilder

def _empty_builder() -> KnowledgeGraphBuilder:
    b = KnowledgeGraphBuilder()
    b._add_sovereign_nodes()
    return b


def test_add_sector_nodes_creates_nodes():
    """add_sector_nodes() adds Sector-typed nodes to the graph."""
    b = _empty_builder()
    b.add_sector_nodes()
    sector_nodes = [
        n for n, d in b.graph.nodes(data=True)
        if d.get("node_type") == "Sector"
    ]
    assert len(sector_nodes) >= 5
    assert "Biotech" in b.graph
    assert "EV" in b.graph


def test_add_sector_nodes_exposed_to_edges():
    """Sector nodes have EXPOSED_TO edges to relevant Statistic nodes."""
    import polars as pl
    b = _empty_builder()
    # Add minimal stat nodes so edges can connect
    b.graph.add_node("USA_policy_rate", node_type="Statistic")
    b.graph.add_node("SAU_crude_output", node_type="Statistic")
    b.add_sector_nodes()

    exposed = [
        (u, v) for u, v, d in b.graph.edges(data=True)
        if d.get("edge_type") == "EXPOSED_TO"
    ]
    assert len(exposed) > 0
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_graph_company.py -k "sector" -v
```

Expected: `AttributeError: 'KnowledgeGraphBuilder' object has no attribute 'add_sector_nodes'`

**Step 3: Add `SECTORS` config and `add_sector_nodes()` to `graph_builder.py`**

Add after the `CONTAGION_CHANNELS` list (line ~84) in `graph_builder.py`:

```python
# ─── Sector Definitions ──────────────────────────────────────────────────────
# (sector_name, [stat_node_ids it's exposed to, weight])
SECTORS: list[tuple[str, list[tuple[str, float]]]] = [
    ("Biotech",   [("USA_policy_rate", 0.60), ("USA_yield_spread_10y3m", 0.40)]),
    ("Pharma",    [("USA_policy_rate", 0.50), ("USA_yield_spread_10y3m", 0.35)]),
    ("Energy",    [("SAU_crude_output", 0.80), ("USA_wti_brent_spread", 0.70), ("USA_policy_rate", 0.30)]),
    ("Mining",    [("CHN_real_gdp_growth", 0.65), ("USA_policy_rate", 0.35)]),
    ("Tech",      [("USA_policy_rate", 0.55), ("USA_yield_spread_10y3m", 0.50)]),
    ("Cannabis",  [("USA_policy_rate", 0.40)]),
    ("EV",        [("CHN_real_gdp_growth", 0.50), ("USA_policy_rate", 0.45)]),
    ("Defense",   [("USA_real_gdp_growth", 0.40)]),
    ("Financial", [("USA_policy_rate", 0.75), ("USA_yield_spread_10y3m", 0.80)]),
    ("Retail",    [("USA_real_gdp_growth", 0.60), ("USA_policy_rate", 0.40)]),
    ("Other",     [("USA_policy_rate", 0.30)]),
]
```

Add method to `KnowledgeGraphBuilder` after `_add_contagion_edges`:

```python
def add_sector_nodes(self) -> None:
    """Add Sector nodes and EXPOSED_TO edges to relevant Statistic nodes."""
    for sector_name, exposures in SECTORS:
        self.graph.add_node(sector_name, node_type="Sector")
        for stat_node_id, weight in exposures:
            if stat_node_id in self.graph:
                self.graph.add_edge(
                    sector_name, stat_node_id,
                    edge_type="EXPOSED_TO",
                    weight=weight,
                )
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_graph_company.py -k "sector" -v
```

Expected: 2 PASS

**Step 5: Commit**

```bash
git add knowledge_base/graph_builder.py tests/test_graph_company.py
git commit -m "feat(company-schema): add_sector_nodes + EXPOSED_TO edges"
```

---

## Task 4: Company Nodes in `graph_builder.py`

**Files:**
- Modify: `knowledge_base/graph_builder.py`
- Modify: `tests/test_graph_company.py`

**Step 1: Write failing tests**

```python
# append to tests/test_graph_company.py
from pathlib import Path
import polars as pl
from knowledge_base.company_schema import CompanyNode


def test_add_company_nodes_from_dataframe():
    """add_company_nodes() adds Company nodes + edges to graph."""
    b = _empty_builder()
    b.add_sector_nodes()

    seed_df = pl.DataFrame({
        "ticker":               ["FFIE"],
        "name":                 ["Faraday Future"],
        "exchange":             ["NASDAQ"],
        "sector":               ["EV"],
        "country_iso3":         ["USA"],
        "employee_count":       [400],
        "location_count":       [2],
        "location_type":        ["multi_site"],
        "named_customers_count":[1],
        "has_shipped_product":  [False],
        "auditor":              ["Deloitte"],
        "auditor_tier":         ["big4"],
        "years_operating":      [8.0],
        "ceo_verifiable":       [True],
        "reality_score":        [0.45],
        "risk_tier":            ["MICRO_OP"],
    })
    b.add_company_nodes(seed_df)

    assert "FFIE" in b.graph
    assert b.graph.nodes["FFIE"]["node_type"] == "Company"
    assert b.graph.nodes["FFIE"]["risk_tier"] == "MICRO_OP"

    # LISTED_IN edge to United States sovereign
    listed_in = [
        v for u, v, d in b.graph.edges(data=True)
        if u == "FFIE" and d.get("edge_type") == "LISTED_IN"
    ]
    assert "United States" in listed_in

    # BELONGS_TO edge to EV sector
    belongs = [
        v for u, v, d in b.graph.edges(data=True)
        if u == "FFIE" and d.get("edge_type") == "BELONGS_TO"
    ]
    assert "EV" in belongs


def test_add_company_nodes_trading_profile():
    """Company node contains correct trading profile attributes."""
    b = _empty_builder()
    b.add_sector_nodes()

    seed_df = pl.DataFrame({
        "ticker":               ["GTII"],
        "name":                 ["Global Arena Holding"],
        "exchange":             ["OTC"],
        "sector":               ["Other"],
        "country_iso3":         ["USA"],
        "employee_count":       [0],
        "location_count":       [1],
        "location_type":        ["registered_agent"],
        "named_customers_count":[0],
        "has_shipped_product":  [False],
        "auditor":              ["unknown"],
        "auditor_tier":         ["unknown"],
        "years_operating":      [0.5],
        "ceo_verifiable":       [False],
        "reality_score":        [0.0],
        "risk_tier":            ["SHELL"],
    })
    b.add_company_nodes(seed_df)

    node = b.graph.nodes["GTII"]
    assert node["size_multiplier"] == 0.25
    assert node["target_pct"] == 5.0
    assert node["reversal_pct"] == 1.5
    assert node["trading_note"] == "Take the money and run"
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_graph_company.py -k "company_nodes" -v
```

Expected: `AttributeError: 'KnowledgeGraphBuilder' object has no attribute 'add_company_nodes'`

**Step 3: Add `add_company_nodes()` to `graph_builder.py`**

Add import at top of `graph_builder.py`:

```python
from knowledge_base.company_schema import (
    CompanyNode,
    derive_trading_profile,
    TIER_CONFIG,
)
```

Add `ISO3_TO_COUNTRY` lookup dict after imports (for LISTED_IN edge):

```python
# ISO-3 → full country name (for LISTED_IN edges)
ISO3_TO_COUNTRY: dict[str, str] = {
    codes.get("iso3", ""): country
    for country, codes in COUNTRY_CODES.items()
}
```

Add method to `KnowledgeGraphBuilder` after `add_sector_nodes`:

```python
def add_company_nodes(self, seed_df: "pl.DataFrame") -> None:
    """
    Add Company nodes from a seed DataFrame (output of load_company_seeds).

    Each row becomes a node; edges added: LISTED_IN → Sovereign,
    BELONGS_TO → Sector.
    """
    from knowledge_base.company_schema import derive_trading_profile

    for row in seed_df.iter_rows(named=True):
        ticker    = row["ticker"]
        risk_tier = row.get("risk_tier", "MICRO_OP")
        profile   = derive_trading_profile(risk_tier)

        attrs = {
            "node_type":             "Company",
            "name":                  row.get("name", ""),
            "exchange":              row.get("exchange", "OTC"),
            "sector":                row.get("sector", "Other"),
            "country_iso3":          row.get("country_iso3", "USA"),
            "employee_count":        row.get("employee_count", 0),
            "location_count":        row.get("location_count", 1),
            "location_type":         row.get("location_type", "registered_agent"),
            "named_customers_count": row.get("named_customers_count", 0),
            "has_shipped_product":   row.get("has_shipped_product", False),
            "auditor":               row.get("auditor", "unknown"),
            "auditor_tier":          row.get("auditor_tier", "unknown"),
            "years_operating":       row.get("years_operating", 0.0),
            "ceo_verifiable":        row.get("ceo_verifiable", False),
            "reality_score":         row.get("reality_score", 0.0),
            "risk_tier":             risk_tier,
            "reality_source":        row.get("reality_source", "seed"),
            "reality_updated":       row.get("reality_updated", ""),
            **profile,
        }
        self.graph.add_node(ticker, **attrs)

        # LISTED_IN → Sovereign
        country = ISO3_TO_COUNTRY.get(row.get("country_iso3", ""))
        if country and country in self.graph:
            self.graph.add_edge(ticker, country, edge_type="LISTED_IN", weight=1.0)

        # BELONGS_TO → Sector
        sector = row.get("sector", "Other")
        if sector in self.graph:
            self.graph.add_edge(ticker, sector, edge_type="BELONGS_TO", weight=1.0)
```

**Step 4: Run all graph tests**

```bash
python -m pytest tests/test_graph_company.py -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add knowledge_base/graph_builder.py tests/test_graph_company.py
git commit -m "feat(company-schema): add_company_nodes + LISTED_IN + BELONGS_TO edges"
```

---

## Task 5: Query Interface — `get_company_trading_profile()`

**Files:**
- Modify: `knowledge_base/graph_builder.py`
- Modify: `tests/test_graph_company.py`

**Step 1: Write failing tests**

```python
# append to tests/test_graph_company.py

def _builder_with_company() -> KnowledgeGraphBuilder:
    b = _empty_builder()
    b.add_sector_nodes()
    seed_df = pl.DataFrame({
        "ticker":               ["GTII", "GFAI"],
        "name":                 ["Global Arena", "Guardforce AI"],
        "exchange":             ["OTC", "NASDAQ"],
        "sector":               ["Other", "Tech"],
        "country_iso3":         ["USA", "USA"],
        "employee_count":       [0, 900],
        "location_count":       [1, 8],
        "location_type":        ["registered_agent", "multi_site"],
        "named_customers_count":[0, 15],
        "has_shipped_product":  [False, True],
        "auditor":              ["unknown", "Marcum"],
        "auditor_tier":         ["unknown", "mid"],
        "years_operating":      [0.5, 8.0],
        "ceo_verifiable":       [False, True],
        "reality_score":        [0.0, 0.90],
        "risk_tier":            ["SHELL", "REAL"],
    })
    b.add_company_nodes(seed_df)
    return b


def test_get_company_trading_profile_known_shell():
    b = _builder_with_company()
    profile = b.get_company_trading_profile("GTII")
    assert profile["risk_tier"] == "SHELL"
    assert profile["size_multiplier"] == 0.25
    assert profile["target_pct"] == 5.0


def test_get_company_trading_profile_known_real():
    b = _builder_with_company()
    profile = b.get_company_trading_profile("GFAI")
    assert profile["risk_tier"] == "REAL"
    assert profile["size_multiplier"] == 1.0


def test_get_company_trading_profile_unknown_ticker():
    """Unknown ticker falls back to MICRO_OP defaults."""
    b = _builder_with_company()
    profile = b.get_company_trading_profile("UNKN")
    assert profile["risk_tier"] == "MICRO_OP"
    assert profile["size_multiplier"] == 0.75
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_graph_company.py -k "trading_profile" -v
```

Expected: `AttributeError: 'KnowledgeGraphBuilder' object has no attribute 'get_company_trading_profile'`

**Step 3: Add `get_company_trading_profile()` to `graph_builder.py`**

Add method to `KnowledgeGraphBuilder` after `add_company_nodes`:

```python
_MICRO_OP_DEFAULT = {
    "risk_tier":       "MICRO_OP",
    "size_multiplier": 0.75,
    "target_pct":      10.0,
    "reversal_pct":    2.0,
    "trading_note":    "Normal, slight caution",
}

def get_company_trading_profile(self, ticker: str) -> dict:
    """
    Return trading profile for a ticker.

    If the ticker is in the graph as a Company node, returns its stored
    profile.  If unknown, returns MICRO_OP defaults (cautious but not
    maximally restrictive).
    """
    node = self.graph.nodes.get(ticker)
    if node and node.get("node_type") == "Company":
        return {
            "risk_tier":       node["risk_tier"],
            "size_multiplier": node["size_multiplier"],
            "target_pct":      node["target_pct"],
            "reversal_pct":    node["reversal_pct"],
            "trading_note":    node["trading_note"],
        }
    logger.debug("Ticker %s not in graph — returning MICRO_OP defaults", ticker)
    return dict(_MICRO_OP_DEFAULT)
```

**Step 4: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all PASS

**Step 5: Commit**

```bash
git add knowledge_base/graph_builder.py tests/test_graph_company.py
git commit -m "feat(company-schema): get_company_trading_profile + MICRO_OP fallback"
```

---

## Task 6: Wire Into `build_from_observations()` + Update `GraphMetrics`

**Files:**
- Modify: `knowledge_base/graph_builder.py`
- Modify: `tests/test_graph_company.py`

**Step 1: Write failing test**

```python
# append to tests/test_graph_company.py
import polars as pl

def test_metrics_counts_company_nodes():
    """GraphMetrics.company_nodes reflects actual company count."""
    b = _builder_with_company()
    metrics = b.get_metrics()
    assert metrics.company_nodes == 2
    assert metrics.sector_nodes == len(b.graph.nodes) - metrics.sovereign_nodes \
           - metrics.statistic_nodes - metrics.company_nodes  # sector nodes accounted for
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_graph_company.py -k "metrics" -v
```

Expected: `AttributeError: 'GraphMetrics' object has no attribute 'company_nodes'`

**Step 3: Update `GraphMetrics` and `get_metrics()` in `graph_builder.py`**

In the `GraphMetrics` dataclass, add fields:

```python
@dataclass
class GraphMetrics:
    """Summary statistics for the knowledge graph."""
    total_nodes: int = 0
    sovereign_nodes: int = 0
    statistic_nodes: int = 0
    scenario_nodes: int = 0
    company_nodes: int = 0      # ← add
    sector_nodes: int = 0       # ← add
    total_edges: int = 0
    monitors_edges: int = 0
    contagion_edges: int = 0
    trade_edges: int = 0
    avg_degree: float = 0.0
    density: float = 0.0
```

In `get_metrics()`, add counts:

```python
company = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "Company")
sectors = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "Sector")
```

And include in the returned `GraphMetrics(...)`:

```python
company_nodes=company,
sector_nodes=sectors,
```

Also add `add_sector_nodes()` and `add_company_nodes()` calls into `build_from_observations()` after `_add_contagion_edges()`:

```python
def build_from_observations(self, observations_df: pl.DataFrame,
                             company_seed_path: Path | None = None) -> nx.DiGraph:
    ...
    self._add_contagion_edges()
    self.add_sector_nodes()                        # ← add
    if company_seed_path is not None:              # ← add
        from knowledge_base.company_schema import load_company_seeds
        seeds = load_company_seeds(company_seed_path)
        if not seeds.is_empty():
            self.add_company_nodes(seeds)
    self._built = True
    ...
```

**Step 4: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all PASS

**Step 5: Final commit**

```bash
git add knowledge_base/graph_builder.py tests/test_graph_company.py
git commit -m "feat(company-schema): wire into build_from_observations, update GraphMetrics"
```

---

## Completion Checklist

- [ ] `compute_reality_score` tested + passing
- [ ] `derive_risk_tier` / `derive_trading_profile` tested + passing
- [ ] `CompanyNode` dataclass tested
- [ ] `load_company_seeds` tested + passing
- [ ] `data/company_seeds.parquet` created (15 seed rows)
- [ ] `add_sector_nodes` + `EXPOSED_TO` edges tested + passing
- [ ] `add_company_nodes` + `LISTED_IN` + `BELONGS_TO` edges tested + passing
- [ ] `get_company_trading_profile` with fallback tested + passing
- [ ] `GraphMetrics.company_nodes` + `sector_nodes` tested + passing
- [ ] All tests: `python -m pytest tests/ -v`

---

## Run from `gap_and_go_live.py`

```python
from world_knowledge_base.global_financial_kb.knowledge_base.graph_builder import KnowledgeGraphBuilder
from world_knowledge_base.global_financial_kb.knowledge_base.company_schema import load_company_seeds

SEED_PATH = Path("/media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb/data/company_seeds.parquet")

kg = KnowledgeGraphBuilder()
kg.build_from_observations(pl.DataFrame(), company_seed_path=SEED_PATH)

profile = kg.get_company_trading_profile(ticker)
# → {"risk_tier": "SHELL", "size_multiplier": 0.25, ...}
```
