# Company Entity Schema — Design Document

**Date:** 2026-02-24
**Status:** Approved
**Project:** `world_knowledge_base/global_financial_kb`

---

## Goal

Add a `Company` node type to the existing NetworkX knowledge graph. Each company node encodes operational reality (employees, locations, customers, auditor) and a derived risk tier (SHELL / MICRO_OP / REAL) with a trading profile (size_multiplier, target_pct, reversal_pct) consumed by `gap_and_go_live.py` at runtime.

---

## Node Structure

```python
graph.add_node(
    "XYZW",                           # ticker as node ID
    node_type="Company",
    # ── Identity ───────────────────────────────────────────
    name="XYZ Widget Corp",
    exchange="NYSE",                  # NYSE / NASDAQ / OTC
    sector="Biotech",
    country_iso3="USA",               # for LISTED_IN edge
    # ── Reality attributes ─────────────────────────────────
    employee_count=0,                 # 0 = unknown/none
    location_count=1,
    location_type="registered_agent", # registered_agent / real_office / multi_site
    named_customers=[],               # list of strings
    named_customers_count=0,
    has_shipped_product=False,
    auditor="Fruci & Associates",
    auditor_tier="micro",             # big4 / mid / micro / unknown
    years_operating=0.5,
    ceo_verifiable=False,             # LinkedIn-confirmable executive
    # ── Risk classification ────────────────────────────────
    risk_tier="SHELL",                # SHELL / MICRO_OP / REAL
    reality_score=0.05,               # 0.0–1.0 composite
    reality_source="seed",            # seed / llm / hybrid
    reality_updated="2026-02-24",
    # ── Trading profile (derived from risk_tier) ───────────
    size_multiplier=0.25,
    target_pct=5.0,
    reversal_pct=1.5,
    trading_note="Take the money and run",
)
```

---

## Risk Tier Derivation

`reality_score` computed from 7 binary/scored signals:

```python
def compute_reality_score(attrs: dict) -> float:
    score = 0.0
    if attrs["employee_count"] > 10:            score += 0.20
    if attrs["location_count"] > 1:             score += 0.15
    if attrs["location_type"] == "real_office": score += 0.10
    if attrs["named_customers_count"] > 0:      score += 0.20
    if attrs["has_shipped_product"]:            score += 0.15
    if attrs["auditor_tier"] in ("big4","mid"): score += 0.10
    if attrs["ceo_verifiable"]:                 score += 0.10
    return round(score, 2)  # max 1.0
```

Tier thresholds:

| Tier | Score | size_multiplier | target_pct | reversal_pct | note |
|------|-------|-----------------|------------|--------------|------|
| SHELL | < 0.25 | 0.25 | 5.0% | 1.5% | Take the money and run |
| MICRO_OP | 0.25–0.59 | 0.75 | 10.0% | 2.0% | Normal, slight caution |
| REAL | ≥ 0.60 | 1.00 | 10.0% | 2.0% | Hold for full target |

---

## Edges

| Edge | From → To | Weight | Meaning |
|------|-----------|--------|---------|
| `LISTED_IN` | Company → Sovereign | 1.0 | Domicile / listing country |
| `BELONGS_TO` | Company → Sector | 1.0 | Industry classification |
| `TRIGGERS` | Company → Scenario | 0.0–1.0 | Company news can trigger scenario |
| `EXPOSED_TO` | Sector → Statistic | 0.0–1.0 | Sector sensitivity to macro stat |

`EXPOSED_TO` bridges macro shocks to company trading profiles via sector.

---

## Sector Nodes

New `Sector` node type. Initial sectors:
`Biotech`, `Pharma`, `Energy`, `Mining`, `Tech`, `Cannabis`, `EV`, `Defense`, `Financial`, `Retail`, `Other`

Sector nodes carry `EXPOSED_TO` edges to relevant Statistic nodes (e.g. `Biotech → USA_policy_rate`, `Energy → SAU_crude_output`).

---

## Population (Hybrid)

### Seed file
`data/company_seeds.parquet` — manually curated rows for known penny stock universe.
Schema: `ticker, name, exchange, sector, country_iso3, employee_count, location_count, location_type, named_customers_count, has_shipped_product, auditor, auditor_tier, years_operating, ceo_verifiable`

### LLM inference (Apollo)
When scanner discovers a ticker not in the seed file:
1. Queue task to Apollo: *"assess company reality for TICKER — check SEC EDGAR, named customers, auditor"*
2. Apollo writes structured JSON to agent inbox
3. Next graph refresh ingests result, sets `reality_source="llm"`

### Staleness
`reality_updated` field. If > 90 days old → re-queue for LLM reassessment.

---

## Query Interface

`gap_and_go_live.py` calls:

```python
profile = kg.get_company_trading_profile(ticker)
# Returns: {"risk_tier": "SHELL", "size_multiplier": 0.25,
#           "target_pct": 5.0, "reversal_pct": 1.5,
#           "trading_note": "Take the money and run"}
# Falls back to MICRO_OP defaults if ticker unknown
```

---

## Files

| File | Description |
|------|-------------|
| `knowledge_base/company_schema.py` | `compute_reality_score`, `TIER_CONFIG`, `CompanyNode` dataclass |
| `knowledge_base/graph_builder.py` | Extend with `add_company_nodes`, `add_sector_nodes` |
| `data/company_seeds.parquet` | Seed data for known penny stocks |
| `tests/test_company_schema.py` | Unit tests |
