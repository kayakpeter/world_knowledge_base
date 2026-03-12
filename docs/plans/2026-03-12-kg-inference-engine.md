# KG + Open-Brain Inference Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add Neo4j as a persistent graph store, wire Apollo's causal chain triggers into an AgentDirective bridge, and sync graph node summaries to open-brain for semantic search.

**Architecture:** Additive — Neo4j added alongside existing NetworkX. `kg_ingest.py` runs after each news dispatch cycle: upserts Event/Stat nodes → evaluates trigger rules → fires agent inbox messages → posts summaries to open-brain. NetworkX reloads from Neo4j on demand.

**Tech Stack:** `neo4j` Python driver (already installed), NetworkX (existing), `data/interpretations/brain_ingest.py` for open-brain posts, `~/.claude/agent-tools/send.py` for agent messaging, Polars for parquet reading, pytest for tests.

**Design doc:** `docs/plans/2026-03-12-kg-inference-engine-design.md`

**Working directory for all commands:** `/media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb/`

**Run tests with:** `python -m pytest tests/ -v` from the working directory.

**Neo4j connection:** bolt://localhost:7687, no auth (dev). Set `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` env vars or `.env` to override.

---

### Task 1: Neo4j client wrapper

**Files:**
- Create: `knowledge_base/neo4j_client.py`
- Test: `tests/test_neo4j_client.py`

**Step 1: Write the failing tests**

```python
# tests/test_neo4j_client.py
import pytest
from unittest.mock import MagicMock, patch

def test_neo4j_client_upsert_country(monkeypatch):
    """upsert_country runs MERGE with correct parameters."""
    mock_session = MagicMock()
    from knowledge_base.neo4j_client import Neo4jClient
    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = MagicMock()
    client._driver.session.return_value.__enter__ = lambda s: mock_session
    client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    client.upsert_country("USA", "United States", "Americas")
    mock_session.run.assert_called_once()
    call_args = mock_session.run.call_args
    assert "MERGE" in call_args[0][0]
    assert call_args[1]["iso3"] == "USA"

def test_neo4j_client_upsert_event(monkeypatch):
    """upsert_event runs MERGE keyed on item_id."""
    mock_session = MagicMock()
    from knowledge_base.neo4j_client import Neo4jClient
    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = MagicMock()
    client._driver.session.return_value.__enter__ = lambda s: mock_session
    client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    client.upsert_event("evt_001", "Test headline", "2026-03-12T00:00:00Z", "critical", "positive", "test chain", 90)
    mock_session.run.assert_called_once()
    call_args = mock_session.run.call_args
    assert call_args[1]["item_id"] == "evt_001"

def test_neo4j_client_upsert_flag():
    """upsert_flag creates/updates a Flag node."""
    mock_session = MagicMock()
    from knowledge_base.neo4j_client import Neo4jClient
    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = MagicMock()
    client._driver.session.return_value.__enter__ = lambda s: mock_session
    client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    client.upsert_flag("HORMUZ_STATUS", "EFFECTIVE_CLOSURE", "hermes")
    mock_session.run.assert_called_once()
    call_args = mock_session.run.call_args
    assert call_args[1]["name"] == "HORMUZ_STATUS"
    assert call_args[1]["value"] == "EFFECTIVE_CLOSURE"

def test_neo4j_client_get_flag_returns_none_when_missing():
    """get_flag returns None when flag does not exist."""
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.single.return_value = None
    mock_session.run.return_value = mock_result
    from knowledge_base.neo4j_client import Neo4jClient
    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = MagicMock()
    client._driver.session.return_value.__enter__ = lambda s: mock_session
    client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    result = client.get_flag("NONEXISTENT_FLAG")
    assert result is None
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_neo4j_client.py -v
```
Expected: ImportError or ModuleNotFoundError — `knowledge_base.neo4j_client` does not exist yet.

**Step 3: Write the implementation**

```python
# knowledge_base/neo4j_client.py
"""
Neo4j client wrapper — connection management and upsert helpers.

Reads connection config from env vars with fallback to .env file:
  NEO4J_URI       (default: bolt://localhost:7687)
  NEO4J_USER      (default: neo4j)
  NEO4J_PASSWORD  (default: "")

All writes use MERGE for idempotency. The driver is lazy-initialised
on first use so import never fails if Neo4j is unreachable.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_FILE = Path("/media/peter/fast-storage/.env")


def _load_env_var(key: str, default: str = "") -> str:
    val = os.environ.get(key, "")
    if val:
        return val
    if _ENV_FILE.exists():
        for line in _ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return default


class Neo4jClient:
    """Thin wrapper around the neo4j Python driver with upsert helpers."""

    def __init__(self) -> None:
        from neo4j import GraphDatabase  # type: ignore
        uri      = _load_env_var("NEO4J_URI",      "bolt://localhost:7687")
        user     = _load_env_var("NEO4J_USER",     "neo4j")
        password = _load_env_var("NEO4J_PASSWORD", "")
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Neo4j driver connected to %s", uri)

    def close(self) -> None:
        self._driver.close()

    # ── Layer A ────────────────────────────────────────────────────────────────

    def upsert_country(self, iso3: str, name: str, region: str) -> None:
        with self._driver.session() as s:
            s.run(
                "MERGE (c:Country {iso3: $iso3}) "
                "SET c.name=$name, c.region=$region",
                iso3=iso3, name=name, region=region,
            )

    def upsert_infrastructure(self, infra_id: str, name: str,
                               infra_type: str, country_iso3: str,
                               risk_tier: str) -> None:
        with self._driver.session() as s:
            s.run(
                "MERGE (i:Infrastructure {id: $id}) "
                "SET i.name=$name, i.type=$type, "
                "i.country_iso3=$country_iso3, i.risk_tier=$risk_tier",
                id=infra_id, name=name, type=infra_type,
                country_iso3=country_iso3, risk_tier=risk_tier,
            )

    def upsert_trade_route(self, route_id: str, from_iso3: str,
                            to_iso3: str, commodity: str,
                            daily_volume_bbl: float) -> None:
        with self._driver.session() as s:
            s.run(
                "MERGE (r:TradeRoute {id: $id}) "
                "SET r.from_iso3=$from_iso3, r.to_iso3=$to_iso3, "
                "r.commodity=$commodity, r.daily_volume_bbl=$vol",
                id=route_id, from_iso3=from_iso3, to_iso3=to_iso3,
                commodity=commodity, vol=daily_volume_bbl,
            )

    # ── Layer B ────────────────────────────────────────────────────────────────

    def upsert_scenario(self, scenario_id: str, name: str,
                         probability: float, active: bool = False) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._driver.session() as s:
            s.run(
                "MERGE (s:Scenario {id: $id}) "
                "SET s.name=$name, s.probability=$prob, "
                "s.active=$active, s.last_updated=$now",
                id=scenario_id, name=name, prob=probability,
                active=active, now=now,
            )

    def activate_scenario(self, scenario_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._driver.session() as s:
            s.run(
                "MATCH (s:Scenario {id: $id}) "
                "SET s.active=true, s.activated_at=$now",
                id=scenario_id, now=now,
            )

    def upsert_stat(self, country_iso3: str, stat_name: str,
                    value: float | None, direction: str,
                    estimated_at: str) -> None:
        with self._driver.session() as s:
            s.run(
                "MERGE (st:Stat {country_iso3: $iso3, stat_name: $stat}) "
                "SET st.value=$value, st.direction=$dir, "
                "st.estimated_at=$at",
                iso3=country_iso3, stat=stat_name, value=value,
                dir=direction, at=estimated_at,
            )

    # ── Layer C ────────────────────────────────────────────────────────────────

    def upsert_event(self, item_id: str, headline: str, published_at: str,
                     urgency: str, sentiment: str, causal_chain: str,
                     confidence: int) -> None:
        with self._driver.session() as s:
            s.run(
                "MERGE (e:Event {id: $item_id}) "
                "SET e.headline=$headline, e.published_at=$pub, "
                "e.urgency=$urgency, e.sentiment=$sentiment, "
                "e.causal_chain=$chain, e.confidence=$conf",
                item_id=item_id, headline=headline, pub=published_at,
                urgency=urgency, sentiment=sentiment, chain=causal_chain,
                conf=confidence,
            )

    def upsert_flag(self, name: str, value: str, set_by: str) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._driver.session() as s:
            s.run(
                "MERGE (f:Flag {name: $name}) "
                "SET f.value=$value, f.set_by=$set_by, f.set_at=$now",
                name=name, value=value, set_by=set_by, now=now,
            )

    def get_flag(self, name: str) -> str | None:
        with self._driver.session() as s:
            result = s.run("MATCH (f:Flag {name: $name}) RETURN f.value", name=name)
            row = result.single()
            return row["f.value"] if row else None

    def create_affects_edge(self, event_id: str, country_iso3: str,
                             direction: str, magnitude: float | None) -> None:
        with self._driver.session() as s:
            s.run(
                "MATCH (e:Event {id: $eid}), (c:Country {iso3: $iso3}) "
                "MERGE (e)-[r:AFFECTS]->(c) "
                "SET r.direction=$dir, r.magnitude=$mag",
                eid=event_id, iso3=country_iso3, dir=direction,
                mag=magnitude,
            )

    def create_escalates_edge(self, event_id: str, scenario_id: str) -> None:
        with self._driver.session() as s:
            s.run(
                "MATCH (e:Event {id: $eid}), (s:Scenario {id: $sid}) "
                "MERGE (e)-[:ESCALATES]->(s)",
                eid=event_id, sid=scenario_id,
            )

    def get_active_flags(self) -> dict[str, str]:
        """Return all Flag nodes as {name: value} dict."""
        with self._driver.session() as s:
            result = s.run("MATCH (f:Flag) RETURN f.name, f.value")
            return {row["f.name"]: row["f.value"] for row in result}

    def is_scenario_active(self, scenario_id: str) -> bool:
        with self._driver.session() as s:
            result = s.run(
                "MATCH (s:Scenario {id: $id}) RETURN s.active",
                id=scenario_id,
            )
            row = result.single()
            return bool(row["s.active"]) if row else False

    def delete_old_events(self, older_than_hours: int = 72) -> int:
        """Delete Layer C Event nodes older than TTL with no active Scenario links."""
        with self._driver.session() as s:
            result = s.run(
                "MATCH (e:Event) "
                "WHERE datetime(e.published_at) < datetime() - duration({hours: $h}) "
                "  AND NOT (e)-[:ESCALATES]->(:Scenario {active: true}) "
                "WITH e, e.id AS eid DELETE e RETURN count(eid) AS deleted",
                h=older_than_hours,
            )
            row = result.single()
            return int(row["deleted"]) if row else 0
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_neo4j_client.py -v
```
Expected: 4 PASSED.

**Step 5: Commit**

```bash
git add knowledge_base/neo4j_client.py tests/test_neo4j_client.py
git commit -m "feat(kg): Neo4j client wrapper with upsert helpers and flag management"
```

---

### Task 2: Seed Layer A — structural nodes

**Files:**
- Create: `knowledge_base/kg_seed.py`
- Test: `tests/test_kg_seed.py`

**Context:** This seeds Country nodes from `config.settings.COUNTRY_CODES`, Infrastructure from a static registry, and TradeRoute from the existing `TRADE_CORRIDORS` list in `graph_builder.py`. Run once (idempotent via MERGE).

**Step 1: Write the failing test**

```python
# tests/test_kg_seed.py
from unittest.mock import MagicMock, patch, call

def test_seed_countries_calls_upsert_for_each_country():
    mock_client = MagicMock()
    with patch("knowledge_base.kg_seed.Neo4jClient", return_value=mock_client):
        from knowledge_base.kg_seed import seed_layer_a
        seed_layer_a(mock_client)
    # Should have called upsert_country at least once per known country
    assert mock_client.upsert_country.call_count >= 1

def test_seed_trade_routes_calls_upsert():
    mock_client = MagicMock()
    from knowledge_base.kg_seed import seed_layer_a
    seed_layer_a(mock_client)
    assert mock_client.upsert_trade_route.call_count >= 1

def test_seed_scenarios_creates_known_scenarios():
    mock_client = MagicMock()
    from knowledge_base.kg_seed import seed_layer_b_scenarios
    seed_layer_b_scenarios(mock_client)
    called_ids = [c[1]["scenario_id"] if c[1] else c[0][0]
                  for c in mock_client.upsert_scenario.call_args_list]
    # Flatten: call_args is (args, kwargs)
    called_ids = []
    for c in mock_client.upsert_scenario.call_args_list:
        args, kwargs = c
        called_ids.append(kwargs.get("scenario_id") or args[0])
    assert "HORMUZ_CLOSURE" in called_ids
    assert "ABQAIQ_STRIKE" in called_ids
    assert "NUCLEAR_INCIDENT" in called_ids
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_kg_seed.py -v
```
Expected: ImportError — module doesn't exist.

**Step 3: Write the implementation**

```python
# knowledge_base/kg_seed.py
"""
kg_seed.py — One-time seeding of Layer A (structural) and Layer B (scenario) nodes.

Run once against a fresh Neo4j instance. Fully idempotent (MERGE).
Can be re-run safely to add new scenarios or countries.

Usage:
    python -m knowledge_base.kg_seed
"""
from __future__ import annotations

import logging
from knowledge_base.neo4j_client import Neo4jClient
from knowledge_base.graph_builder import TRADE_CORRIDORS
from config.settings import COUNTRY_CODES

logger = logging.getLogger(__name__)

# ── Known infrastructure at risk ──────────────────────────────────────────────
INFRASTRUCTURE_REGISTRY: list[dict] = [
    {"id": "hormuz_strait",    "name": "Strait of Hormuz",      "type": "PORT",          "country_iso3": "IRN", "risk_tier": "critical"},
    {"id": "abqaiq_refinery",  "name": "Abqaiq Processing",     "type": "REFINERY",      "country_iso3": "SAU", "risk_tier": "critical"},
    {"id": "kharg_island",     "name": "Kharg Island Terminal",  "type": "PORT",          "country_iso3": "IRN", "risk_tier": "critical"},
    {"id": "bushehr_npp",      "name": "Bushehr NPP",            "type": "NPP",           "country_iso3": "IRN", "risk_tier": "critical"},
    {"id": "ras_laffan",       "name": "Ras Laffan LNG",        "type": "PIPELINE",      "country_iso3": "QAT", "risk_tier": "high"},
    {"id": "al_udeid",         "name": "Al Udeid Air Base",      "type": "MILITARY_BASE", "country_iso3": "QAT", "risk_tier": "high"},
    {"id": "5th_fleet_bahrain","name": "5th Fleet HQ Bahrain",   "type": "MILITARY_BASE", "country_iso3": "BHR", "risk_tier": "high"},
    {"id": "natanz_facility",  "name": "Natanz Enrichment",      "type": "NPP",           "country_iso3": "IRN", "risk_tier": "critical"},
]

# ── Known scenarios ───────────────────────────────────────────────────────────
SCENARIO_REGISTRY: list[dict] = [
    {"scenario_id": "HORMUZ_CLOSURE",       "name": "Strait of Hormuz Closure",           "probability": 0.85},
    {"scenario_id": "ABQAIQ_STRIKE",        "name": "Abqaiq Refinery Strike",              "probability": 0.45},
    {"scenario_id": "NUCLEAR_INCIDENT",     "name": "Nuclear Facility Incident",           "probability": 0.10},
    {"scenario_id": "GLOBAL_OIL_SHOCK",     "name": "Global Oil Supply Shock",             "probability": 0.90},
    {"scenario_id": "NATO_RUSSIA_PROXY_WAR","name": "NATO-Russia Proxy Confrontation",     "probability": 0.70},
    {"scenario_id": "CEASEFIRE_COLLAPSE",   "name": "All Ceasefire Backchannels Collapsed","probability": 0.80},
    {"scenario_id": "COMMODITY_CASCADE",    "name": "Multi-Commodity Supply Cascade",      "probability": 0.75},
    {"scenario_id": "KHARG_OPERATION",      "name": "Kharg Island Military Operation",     "probability": 0.30},
    {"scenario_id": "UAE_FINANCIAL_SHOCK",  "name": "UAE Financial Hub Disruption",        "probability": 0.25},
    {"scenario_id": "EU_ENERGY_CRISIS",     "name": "EU Gas/Energy Supply Crisis",         "probability": 0.85},
]

# iso3 → region lookup
_ISO3_TO_REGION: dict[str, str] = {
    "USA": "Americas", "CAN": "Americas", "MEX": "Americas", "BRA": "Americas",
    "GBR": "Europe",   "DEU": "Europe",   "FRA": "Europe",   "ITA": "Europe",
    "RUS": "Europe",   "TUR": "Europe",   "POL": "Europe",   "NLD": "Europe",
    "ESP": "Europe",   "HUN": "Europe",   "BEL": "Europe",   "NOR": "Europe",
    "CHN": "Asia",     "JPN": "Asia",     "IND": "Asia",     "KOR": "Asia",
    "IDN": "Asia",     "AUS": "Asia",
    "SAU": "Middle East", "IRN": "Middle East", "QAT": "Middle East",
    "UAE": "Middle East", "BHR": "Middle East", "OMN": "Middle East",
    "ISR": "Middle East", "LBN": "Middle East", "IRQ": "Middle East",
    "KWT": "Middle East",
    "ZAF": "Africa",
    "DPRK": "Asia",
}


def seed_layer_a(client: Neo4jClient) -> None:
    """Seed Country, Infrastructure, and TradeRoute nodes."""
    # Countries from COUNTRY_CODES registry
    seeded_countries = 0
    for country_name, codes in COUNTRY_CODES.items():
        iso3 = codes.get("iso3", "")
        if not iso3:
            continue
        region = _ISO3_TO_REGION.get(iso3, "Other")
        client.upsert_country(iso3, country_name, region)
        seeded_countries += 1
    logger.info("Seeded %d countries", seeded_countries)

    # Infrastructure
    for infra in INFRASTRUCTURE_REGISTRY:
        client.upsert_infrastructure(**infra)
    logger.info("Seeded %d infrastructure nodes", len(INFRASTRUCTURE_REGISTRY))

    # Trade routes from TRADE_CORRIDORS
    # TRADE_CORRIDORS format: (country_name_a, country_name_b, weight)
    # Build name → iso3 lookup
    name_to_iso3: dict[str, str] = {
        name: codes.get("iso3", "")
        for name, codes in COUNTRY_CODES.items()
    }
    seeded_routes = 0
    for from_name, to_name, weight in TRADE_CORRIDORS:
        from_iso3 = name_to_iso3.get(from_name, "")
        to_iso3   = name_to_iso3.get(to_name, "")
        if not from_iso3 or not to_iso3:
            continue
        route_id = f"{from_iso3}_{to_iso3}_TRADE"
        client.upsert_trade_route(route_id, from_iso3, to_iso3, "MIXED", weight * 1_000_000)
        seeded_routes += 1
    logger.info("Seeded %d trade routes", seeded_routes)


def seed_layer_b_scenarios(client: Neo4jClient) -> None:
    """Seed known Scenario nodes."""
    for s in SCENARIO_REGISTRY:
        client.upsert_scenario(**s)
    logger.info("Seeded %d scenarios", len(SCENARIO_REGISTRY))


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    client = Neo4jClient()
    try:
        seed_layer_a(client)
        seed_layer_b_scenarios(client)
        print("Seed complete.")
    finally:
        client.close()
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_kg_seed.py -v
```
Expected: 3 PASSED.

**Step 5: Commit**

```bash
git add knowledge_base/kg_seed.py tests/test_kg_seed.py
git commit -m "feat(kg): Layer A/B seed — countries, infrastructure, trade routes, scenarios"
```

---

### Task 3: Ingestion pipeline (`kg_ingest.py`)

**Files:**
- Create: `processing/kg_ingest.py`
- Test: `tests/test_kg_ingest.py`

**Context:** Reads one interpretation parquet (Polars), upserts Event + Stat nodes, creates AFFECTS and ESCALATES edges using a simple causal_chain → scenario_id keyword map. Called after each dispatch cycle.

**Interpretation parquet schema** (from `data/interpretations/`):
- `item_id`, `country_iso3`, `headline`, `published_at`, `causal_chain`, `sentiment`, `urgency`, `confidence`, `affected_stats`, `stat_direction`, `estimated_magnitude`, `cross_country_iso3`

**Step 1: Write the failing tests**

```python
# tests/test_kg_ingest.py
import polars as pl
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile


def _make_parquet(tmp_path: Path) -> Path:
    """Write a minimal test parquet matching interpretation schema."""
    df = pl.DataFrame({
        "item_id":            ["evt_001", "evt_002"],
        "country_iso3":       ["IRN", "SAU"],
        "headline":           ["Iran closes strait", "Saudi refinery at risk"],
        "published_at":       ["2026-03-12T10:00:00Z", "2026-03-12T11:00:00Z"],
        "causal_chain":       ["hormuz disruption -> oil shock", "abqaiq strike risk"],
        "sentiment":          ["negative", "negative"],
        "urgency":            ["critical", "high"],
        "confidence":         [90, 80],
        "affected_stats":     ["oil_price,gdp_growth", "oil_price"],
        "stat_direction":     ["up,down", "up"],
        "estimated_magnitude":["50.0,2.0", "30.0"],
        "cross_country_iso3": ["USA,DEU", ""],
    })
    p = tmp_path / "test_interp.parquet"
    df.write_parquet(p)
    return p


def test_ingest_parquet_upserts_events(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    mock_client = MagicMock()
    from processing.kg_ingest import ingest_parquet
    n = ingest_parquet(parquet_path, mock_client, dry_run=False)
    assert n == 2
    assert mock_client.upsert_event.call_count == 2
    first_call = mock_client.upsert_event.call_args_list[0]
    assert first_call[1]["item_id"] == "evt_001"
    assert first_call[1]["headline"] == "Iran closes strait"


def test_ingest_parquet_creates_affects_edges(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    mock_client = MagicMock()
    from processing.kg_ingest import ingest_parquet
    ingest_parquet(parquet_path, mock_client, dry_run=False)
    # evt_001: IRN + cross_country USA, DEU = 3 AFFECTS edges
    # evt_002: SAU, no cross = 1 AFFECTS edge
    assert mock_client.create_affects_edge.call_count >= 4


def test_ingest_parquet_dry_run_does_not_call_client(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    mock_client = MagicMock()
    from processing.kg_ingest import ingest_parquet
    n = ingest_parquet(parquet_path, mock_client, dry_run=True)
    assert n == 2  # still counts rows
    mock_client.upsert_event.assert_not_called()


def test_ingest_parquet_escalates_hormuz_scenario(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    mock_client = MagicMock()
    from processing.kg_ingest import ingest_parquet
    ingest_parquet(parquet_path, mock_client, dry_run=False)
    escalate_calls = [c for c in mock_client.create_escalates_edge.call_args_list]
    scenario_ids = [c[1]["scenario_id"] for c in escalate_calls]
    assert "HORMUZ_CLOSURE" in scenario_ids
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_kg_ingest.py -v
```
Expected: ImportError.

**Step 3: Write the implementation**

```python
# processing/kg_ingest.py
"""
kg_ingest.py — Parse interpretation parquets → upsert Neo4j nodes → fire triggers.

Called from run_news_dispatch.py after each dispatch cycle, or standalone for backfill:
    python -m processing.kg_ingest --parquet data/interpretations/foo.parquet
    python -m processing.kg_ingest --pending   # process all unprocessed parquets
    python -m processing.kg_ingest --dry-run --parquet ...
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from knowledge_base.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# ── Causal chain → Scenario keyword map ───────────────────────────────────────
# Each entry: (keyword_in_causal_chain, scenario_id)
# Match is case-insensitive substring search.
_CHAIN_TO_SCENARIO: list[tuple[str, str]] = [
    ("hormuz",          "HORMUZ_CLOSURE"),
    ("oil shock",       "GLOBAL_OIL_SHOCK"),
    ("oil supply",      "GLOBAL_OIL_SHOCK"),
    ("abqaiq",          "ABQAIQ_STRIKE"),
    ("kharg",           "KHARG_OPERATION"),
    ("nuclear",         "NUCLEAR_INCIDENT"),
    ("bushehr",         "NUCLEAR_INCIDENT"),
    ("natanz",          "NUCLEAR_INCIDENT"),
    ("ceasefire",       "CEASEFIRE_COLLAPSE"),
    ("russia",          "NATO_RUSSIA_PROXY_WAR"),
    ("nato",            "NATO_RUSSIA_PROXY_WAR"),
    ("commodity",       "COMMODITY_CASCADE"),
    ("grain",           "COMMODITY_CASCADE"),
    ("lng",             "EU_ENERGY_CRISIS"),
    ("gas crisis",      "EU_ENERGY_CRISIS"),
    ("uae",             "UAE_FINANCIAL_SHOCK"),
    ("dubai",           "UAE_FINANCIAL_SHOCK"),
]


def _chain_to_scenarios(causal_chain: str) -> list[str]:
    """Map causal chain text to matching scenario IDs."""
    lower = (causal_chain or "").lower()
    return [sid for kw, sid in _CHAIN_TO_SCENARIO if kw in lower]


def ingest_parquet(path: Path, client: Neo4jClient,
                   dry_run: bool = False) -> int:
    """
    Parse one interpretation parquet, upsert nodes and edges.

    Returns the number of rows processed.
    Dry-run counts rows but skips all writes.
    """
    df = pl.read_parquet(path)
    rows = df.to_dicts()
    logger.info("Ingesting %d rows from %s (dry_run=%s)", len(rows), path.name, dry_run)

    if dry_run:
        return len(rows)

    for row in rows:
        item_id     = str(row.get("item_id", "") or "")
        headline    = str(row.get("headline", "") or "")
        published   = str(row.get("published_at", "") or "")
        urgency     = str(row.get("urgency", "") or "")
        sentiment   = str(row.get("sentiment", "") or "")
        chain       = str(row.get("causal_chain", "") or "")
        confidence  = int(row.get("confidence", 0) or 0)
        country_iso3 = str(row.get("country_iso3", "") or "")

        if not item_id:
            continue

        # Upsert Event node
        client.upsert_event(
            item_id=item_id,
            headline=headline,
            published_at=published,
            urgency=urgency,
            sentiment=sentiment,
            causal_chain=chain,
            confidence=confidence,
        )

        # AFFECTS edges: primary country + cross-country
        iso3_list = [country_iso3] if country_iso3 else []
        cross = str(row.get("cross_country_iso3", "") or "")
        iso3_list += [x.strip() for x in cross.split(",") if x.strip()]

        # stat_direction and magnitude parallel affected_stats
        stat_dirs = str(row.get("stat_direction", "") or "").split(",")
        magnitudes = str(row.get("estimated_magnitude", "") or "").split(",")

        for i, iso3 in enumerate(iso3_list):
            direction  = stat_dirs[i].strip() if i < len(stat_dirs) else ""
            try:
                magnitude = float(magnitudes[i]) if i < len(magnitudes) else None
            except (ValueError, TypeError):
                magnitude = None
            client.create_affects_edge(item_id, iso3, direction, magnitude)

        # Upsert Stat nodes (affected_stats is comma-separated stat names)
        affected_stats = str(row.get("affected_stats", "") or "")
        for j, stat_name in enumerate(x.strip() for x in affected_stats.split(",") if x.strip()):
            direction = stat_dirs[j].strip() if j < len(stat_dirs) else ""
            try:
                magnitude = float(magnitudes[j]) if j < len(magnitudes) else None
            except (ValueError, TypeError):
                magnitude = None
            client.upsert_stat(
                country_iso3=country_iso3,
                stat_name=stat_name,
                value=magnitude,
                direction=direction,
                estimated_at=published,
            )

        # ESCALATES edges
        for scenario_id in _chain_to_scenarios(chain):
            client.create_escalates_edge(item_id, scenario_id)

    return len(rows)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ingest interpretation parquets into Neo4j")
    parser.add_argument("--parquet", metavar="PATH", help="Single parquet file to ingest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = Neo4jClient()
    try:
        if args.parquet:
            n = ingest_parquet(Path(args.parquet), client, dry_run=args.dry_run)
            print(f"Ingested {n} rows.")
        else:
            print("Specify --parquet PATH", file=sys.stderr)
            sys.exit(1)
    finally:
        client.close()
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_kg_ingest.py -v
```
Expected: 4 PASSED.

**Step 5: Commit**

```bash
git add processing/kg_ingest.py tests/test_kg_ingest.py
git commit -m "feat(kg): ingestion pipeline — Event/Stat upserts, AFFECTS/ESCALATES edges"
```

---

### Task 4: Inference trigger engine (`event_triggers.py`)

**Files:**
- Create: `processing/event_triggers.py`
- Test: `tests/test_event_triggers.py`

**Context:** Rule table with 5 Apollo scenario subgraphs. `evaluate_triggers()` reads all Flag nodes from Neo4j, checks each rule's conditions, and fires AgentDirective messages via `~/.claude/agent-tools/send.py`. Idempotent: each rule fires only if its scenario is not already active.

**Step 1: Write the failing tests**

```python
# tests/test_event_triggers.py
import pytest
from unittest.mock import MagicMock, patch, call


def test_trigger_fires_when_conditions_met():
    """A trigger fires when all its flag conditions are satisfied."""
    mock_client = MagicMock()
    mock_client.get_active_flags.return_value = {
        "HORMUZ_STATUS": "EFFECTIVE_CLOSURE",
        "TANKER_WAR_STATUS": "OPERATIONAL",
    }
    mock_client.is_scenario_active.return_value = False  # not yet active

    with patch("processing.event_triggers._send_directive") as mock_send:
        from processing.event_triggers import evaluate_triggers
        fired = evaluate_triggers(mock_client, dry_run=False)

    assert "OIL_SHOCK_ACTIVATE" in fired
    mock_client.activate_scenario.assert_called()
    mock_send.assert_called()


def test_trigger_does_not_refire_if_scenario_active():
    """A trigger does not fire if its scenario is already active."""
    mock_client = MagicMock()
    mock_client.get_active_flags.return_value = {
        "HORMUZ_STATUS": "EFFECTIVE_CLOSURE",
        "TANKER_WAR_STATUS": "OPERATIONAL",
    }
    mock_client.is_scenario_active.return_value = True  # already active

    with patch("processing.event_triggers._send_directive") as mock_send:
        from processing.event_triggers import evaluate_triggers
        fired = evaluate_triggers(mock_client, dry_run=False)

    assert "OIL_SHOCK_ACTIVATE" not in fired
    mock_send.assert_not_called()


def test_trigger_dry_run_does_not_call_activate():
    """Dry run: conditions checked but no writes or sends."""
    mock_client = MagicMock()
    mock_client.get_active_flags.return_value = {
        "HORMUZ_STATUS": "EFFECTIVE_CLOSURE",
        "TANKER_WAR_STATUS": "OPERATIONAL",
    }
    mock_client.is_scenario_active.return_value = False

    with patch("processing.event_triggers._send_directive") as mock_send:
        from processing.event_triggers import evaluate_triggers
        fired = evaluate_triggers(mock_client, dry_run=True)

    mock_client.activate_scenario.assert_not_called()
    mock_send.assert_not_called()
    assert "OIL_SHOCK_ACTIVATE" in fired  # still reported as "would fire"


def test_trigger_alert_type_does_not_activate_scenario():
    """Alert-type triggers send a message but do not activate a scenario."""
    mock_client = MagicMock()
    mock_client.get_active_flags.return_value = {
        "BUSHEHR_NPP_RISK": "CRITICAL",
        "NUCLEAR_FLAG": "False",
    }
    mock_client.is_scenario_active.return_value = False

    with patch("processing.event_triggers._send_directive") as mock_send:
        from processing.event_triggers import evaluate_triggers
        fired = evaluate_triggers(mock_client, dry_run=False)

    assert "BUSHEHR_WATCH" in fired
    mock_send.assert_called()
    mock_client.activate_scenario.assert_not_called()  # alert, not activate
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_event_triggers.py -v
```
Expected: ImportError.

**Step 3: Write the implementation**

```python
# processing/event_triggers.py
"""
event_triggers.py — Apollo causal chain rule table + evaluator.

After each ingestion cycle, evaluate_triggers() reads all Flag nodes
from Neo4j, checks each rule, and fires AgentDirective messages.

Idempotency: activate_scenario triggers fire only if the scenario is
not already active. Alert triggers fire every time their conditions are
met (they are informational, not state-changing).
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from knowledge_base.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

_SEND_PY = Path.home() / ".claude" / "agent-tools" / "send.py"

# ── Trigger rule table ─────────────────────────────────────────────────────────
# action: "activate_scenario" | "alert"
# conditions: all key-value pairs must match Flag nodes in Neo4j
# For activate_scenario: scenario_id is activated and then the directive is sent.
# For alert: directive is sent; no scenario state change.

TRIGGERS: list[dict] = [
    # Scenario subgraph 1: Global Oil Shock
    {
        "id": "OIL_SHOCK_ACTIVATE",
        "action": "activate_scenario",
        "scenario_id": "GLOBAL_OIL_SHOCK",
        "conditions": {
            "HORMUZ_STATUS": "EFFECTIVE_CLOSURE",
            "TANKER_WAR_STATUS": "OPERATIONAL",
        },
        "directive": {
            "to": "prometheus",
            "type": "status",
            "subject": "KG TRIGGER: GLOBAL_OIL_SHOCK scenario activated",
            "body": (
                "Hormuz closure + active tanker war confirmed in KG graph.\n"
                "Scenario GLOBAL_OIL_SHOCK is now active.\n"
                "Review energy_bonus config — current 1.6x may need upgrade."
            ),
        },
    },
    # Scenario subgraph 2: Nuclear Threshold Watch
    {
        "id": "BUSHEHR_WATCH",
        "action": "alert",
        "conditions": {
            "BUSHEHR_NPP_RISK": "CRITICAL",
            "NUCLEAR_FLAG": "False",
        },
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: Bushehr NPP risk CRITICAL — nuclear threshold watch",
            "body": (
                "NUCLEAR_FLAG is still False but BUSHEHR_NPP_RISK is CRITICAL.\n"
                "Any confirmed strike on Bushehr = set NUCLEAR_FLAG: True immediately.\n"
                "Halt all positions protocol must be ready."
            ),
        },
    },
    # Scenario subgraph 3: NATO-Russia Proxy War
    {
        "id": "RUSSIA_PROXY_WAR",
        "action": "activate_scenario",
        "scenario_id": "NATO_RUSSIA_PROXY_WAR",
        "conditions": {
            "RUSSIA_IRAN_MILITARY_COORDINATION": "CONFIRMED_UK_DEFMIN_LEVEL",
            "NATO_KINETIC_ENGAGEMENT": "True",
        },
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: NATO-Russia proxy war scenario activated",
            "body": (
                "Russia drone tactics confirmed (UK DefSec level) + NATO kinetic engagement.\n"
                "Scenario NATO_RUSSIA_PROXY_WAR is now active.\n"
                "Review CONFLICT_DURATION: PROTRACTED → INDEFINITE upgrade.\n"
                "Ceasefire probability: revise further down."
            ),
        },
    },
    # Scenario subgraph 4: Commodity Cascade at Historic Scale
    {
        "id": "COMMODITY_CASCADE_HISTORIC",
        "action": "activate_scenario",
        "scenario_id": "COMMODITY_CASCADE",
        "conditions": {
            "COMMODITY_CASCADE": "ACTIVE",
            "OIL_SUPPLY_DISRUPTION": "HISTORIC_SCALE",
        },
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: Commodity cascade at historic scale — EM contagion risk",
            "body": (
                "Multi-commodity disruption confirmed at HISTORIC_SCALE (IEA framing).\n"
                "Scenario COMMODITY_CASCADE activated.\n"
                "Food security + metals = second-order EM equity contagion.\n"
                "Review position sizing for EM-exposed names and commodity ETFs."
            ),
        },
    },
    # Scenario subgraph 5: Ceasefire Pathway Collapse
    {
        "id": "CEASEFIRE_COLLAPSE",
        "action": "alert",
        "conditions": {
            "QATAR_UNDER_ATTACK": "True",
            "OMAN_SALALAH_STRUCK": "True",
        },
        "directive": {
            "to": "apollo",
            "type": "task",
            "subject": "KG TRIGGER: Both ceasefire backchannels compromised",
            "body": (
                "Qatar ceasefire channel CLOSED + Oman Salalah struck.\n"
                "No viable ceasefire backchannel remains.\n"
                "Oil long thesis strongly reinforced — exit only at AGREEMENT level.\n"
                "CONFLICT_DURATION likely extends further."
            ),
        },
    },
]


def _conditions_met(conditions: dict[str, str], active_flags: dict[str, str]) -> bool:
    """Return True if all condition key-value pairs match active_flags."""
    return all(active_flags.get(k) == v for k, v in conditions.items())


def _send_directive(directive: dict, dry_run: bool = False) -> None:
    """Send an AgentDirective message via send.py. Non-fatal on failure."""
    if dry_run:
        return
    cmd = [
        sys.executable, str(_SEND_PY),
        "--from", "hermes",
        "--to", directive["to"],
        "--type", directive["type"],
        "--subject", directive["subject"],
        "--body", directive["body"],
        "--no-reply-expected",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=10)
        logger.info("Directive sent to %s: %s", directive["to"], directive["subject"])
    except Exception as exc:
        logger.warning("Failed to send directive to %s: %s", directive["to"], exc)


def evaluate_triggers(client: Neo4jClient, dry_run: bool = False) -> list[str]:
    """
    Evaluate all trigger rules against current Neo4j flag state.

    Returns list of trigger IDs that fired (or would fire in dry_run).
    """
    active_flags = client.get_active_flags()
    fired: list[str] = []

    for trigger in TRIGGERS:
        trigger_id = trigger["id"]
        action = trigger["action"]

        if not _conditions_met(trigger["conditions"], active_flags):
            continue

        # Idempotency check for activate_scenario triggers
        if action == "activate_scenario":
            scenario_id = trigger["scenario_id"]
            if client.is_scenario_active(scenario_id):
                logger.debug("Trigger %s: scenario %s already active, skipping", trigger_id, scenario_id)
                continue

        logger.info("Trigger %s FIRES (dry_run=%s)", trigger_id, dry_run)
        fired.append(trigger_id)

        if not dry_run:
            if action == "activate_scenario":
                client.activate_scenario(trigger["scenario_id"])
            _send_directive(trigger["directive"], dry_run=False)

    return fired
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_event_triggers.py -v
```
Expected: 4 PASSED.

**Step 5: Commit**

```bash
git add processing/event_triggers.py tests/test_event_triggers.py
git commit -m "feat(kg): inference trigger engine — 5 Apollo scenario subgraphs with AgentDirective bridge"
```

---

### Task 5: Open-brain sync (`kg_to_openbrain.py`)

**Files:**
- Create: `processing/kg_to_openbrain.py`
- Test: `tests/test_kg_to_openbrain.py`

**Context:** After each ingestion cycle, post text summaries of new Event and active Scenario nodes to open-brain. Uses `_post_one()` from `data/interpretations/brain_ingest.py`. Import via `sys.path` insert (brain_ingest.py lives in `data/interpretations/`).

**Step 1: Write the failing tests**

```python
# tests/test_kg_to_openbrain.py
from unittest.mock import MagicMock, patch


def test_sync_events_posts_to_brain():
    mock_client = MagicMock()
    mock_client.get_active_flags.return_value = {}
    # Simulate Neo4j returning 2 events
    mock_session = MagicMock()
    mock_client._driver.session.return_value.__enter__ = lambda s: mock_session
    mock_client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = [
        {"e.id": "evt_001", "e.headline": "Test event", "e.urgency": "critical",
         "e.causal_chain": "hormuz", "e.confidence": 90, "e.published_at": "2026-03-12T00:00:00Z",
         "countries": ["IRN", "USA"]},
    ]

    with patch("processing.kg_to_openbrain._post_thought", return_value="thought_123") as mock_post:
        from processing.kg_to_openbrain import sync_events_to_openbrain
        count = sync_events_to_openbrain(mock_client, since_hours=24)

    assert count == 1
    mock_post.assert_called_once()
    call_content = mock_post.call_args[0][0]
    assert "Test event" in call_content


def test_sync_scenarios_posts_active_only():
    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_client._driver.session.return_value.__enter__ = lambda s: mock_session
    mock_client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = [
        {"s.id": "HORMUZ_CLOSURE", "s.name": "Hormuz Closure",
         "s.probability": 0.85, "s.active": True, "infra": ["hormuz_strait"]},
    ]

    with patch("processing.kg_to_openbrain._post_thought", return_value="thought_456") as mock_post:
        from processing.kg_to_openbrain import sync_scenarios_to_openbrain
        count = sync_scenarios_to_openbrain(mock_client)

    assert count == 1
    call_content = mock_post.call_args[0][0]
    assert "HORMUZ_CLOSURE" in call_content or "Hormuz Closure" in call_content
```

**Step 2: Run to verify failure**

```bash
python -m pytest tests/test_kg_to_openbrain.py -v
```
Expected: ImportError.

**Step 3: Write the implementation**

```python
# processing/kg_to_openbrain.py
"""
kg_to_openbrain.py — Sync Neo4j graph node summaries to open-brain (Supabase pgvector).

Posts Event and active Scenario summaries as embedded thoughts.
Called after each ingestion cycle. Non-fatal on brain failure.
"""
from __future__ import annotations

import logging
import sys
import urllib.error
import urllib.request
import json
from datetime import datetime, timezone
from pathlib import Path

from knowledge_base.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# brain_ingest.py lives in data/interpretations/ — add to path
_BRAIN_INGEST_DIR = Path(__file__).parent.parent / "data" / "interpretations"
sys.path.insert(0, str(_BRAIN_INGEST_DIR))

BRAIN_URL = "https://sqhcdhfkinkukduvggtz.supabase.co/functions/v1/ingest-thought"
BRAIN_KEY = "6dc2b398c0fcb2c487ad0c6e27f9ea3f7f97fe80831383888e1307938299a7f4"


def _post_thought(content: str, metadata: dict) -> str | None:
    """POST a single thought to open-brain. Returns thought ID or None."""
    payload = json.dumps({"content": content, "metadata": metadata}).encode()
    req = urllib.request.Request(
        BRAIN_URL,
        data=payload,
        headers={"x-brain-key": BRAIN_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read()).get("id")
    except Exception as exc:
        logger.warning("open-brain post failed: %s", exc)
        return None


def sync_events_to_openbrain(client: Neo4jClient, since_hours: int = 24) -> int:
    """
    Post summaries of recent Event nodes to open-brain.
    Returns count of thoughts posted.
    """
    with client._driver.session() as s:
        rows = list(s.run(
            "MATCH (e:Event) "
            "WHERE datetime(e.published_at) > datetime() - duration({hours: $h}) "
            "OPTIONAL MATCH (e)-[:AFFECTS]->(c:Country) "
            "RETURN e.id, e.headline, e.urgency, e.causal_chain, "
            "       e.confidence, e.published_at, collect(c.iso3) AS countries",
            h=since_hours,
        ))

    count = 0
    for row in rows:
        countries = ", ".join(row["countries"]) if row["countries"] else "unknown"
        content = (
            f"[{row['e.urgency'].upper()}] {row['e.headline']} "
            f"— affects: {countries}, "
            f"chain: {row['e.causal_chain']}, "
            f"confidence: {row['e.confidence']}%"
        )
        metadata = {
            "agent": "apollo",
            "type": "kg_event",
            "subsystem": "geopolitical",
            "tags": ["kg-event", row["e.urgency"]],
            "item_id": row["e.id"],
            "published_at": row["e.published_at"],
        }
        thought_id = _post_thought(content, metadata)
        if thought_id:
            count += 1

    logger.info("Synced %d events to open-brain", count)
    return count


def sync_scenarios_to_openbrain(client: Neo4jClient) -> int:
    """
    Post summaries of active Scenario nodes to open-brain.
    Returns count of thoughts posted.
    """
    with client._driver.session() as s:
        rows = list(s.run(
            "MATCH (s:Scenario {active: true}) "
            "OPTIONAL MATCH (s)-[:THREATENS]->(i:Infrastructure) "
            "RETURN s.id, s.name, s.probability, s.active, collect(i.id) AS infra",
        ))

    count = 0
    for row in rows:
        infra_list = ", ".join(row["infra"]) if row["infra"] else "none"
        content = (
            f"Scenario {row['s.id']} ACTIVE: {row['s.name']} "
            f"— probability={row['s.probability']:.0%}, "
            f"threatens: {infra_list}"
        )
        metadata = {
            "agent": "apollo",
            "type": "kg_scenario",
            "subsystem": "geopolitical",
            "tags": ["kg-scenario", "active"],
            "scenario_id": row["s.id"],
        }
        thought_id = _post_thought(content, metadata)
        if thought_id:
            count += 1

    logger.info("Synced %d active scenarios to open-brain", count)
    return count


def sync_all(client: Neo4jClient, since_hours: int = 24) -> int:
    """Sync events + scenarios. Returns total thoughts posted."""
    n = sync_events_to_openbrain(client, since_hours=since_hours)
    n += sync_scenarios_to_openbrain(client)
    return n
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_kg_to_openbrain.py -v
```
Expected: 2 PASSED.

**Step 5: Commit**

```bash
git add processing/kg_to_openbrain.py tests/test_kg_to_openbrain.py
git commit -m "feat(kg): open-brain sync — Event and Scenario summaries posted as embedded thoughts"
```

---

### Task 6: Morning briefing (`kg_morning_brief.py`)

**Files:**
- Create: `processing/kg_morning_brief.py`

No unit test needed — this is a thin Cypher query runner + send.py wrapper. Integration tested manually.

**Step 1: Write the implementation**

```python
# processing/kg_morning_brief.py
"""
kg_morning_brief.py — Run Apollo's 5 morning Cypher queries → dispatch to Apollo inbox.

Schedule via cron at 06:00 ET:
    0 11 * * 1-5 cd /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb && python -m processing.kg_morning_brief
"""
from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from knowledge_base.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

_SEND_PY = Path.home() / ".claude" / "agent-tools" / "send.py"


def _run_query(client: Neo4jClient, cypher: str, params: dict | None = None) -> list[dict]:
    with client._driver.session() as s:
        result = s.run(cypher, **(params or {}))
        return [dict(row) for row in result]


def build_morning_brief(client: Neo4jClient) -> str:
    """Run 5 Apollo Cypher queries and format as markdown brief."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = [f"# Morning KG Brief — {now_str}\n"]

    # Q1: High/critical events last 24h
    rows = _run_query(client,
        "MATCH (e:Event) WHERE e.published_at > toString(datetime() - duration('PT24H')) "
        "AND e.urgency IN ['critical','high'] "
        "RETURN e.urgency AS urgency, e.headline AS headline, e.causal_chain AS chain "
        "ORDER BY e.urgency, e.published_at DESC LIMIT 20"
    )
    sections.append("## High/Critical Events (last 24h)")
    for r in rows:
        sections.append(f"- [{r['urgency'].upper()}] {r['headline']}")
        if r.get("chain"):
            sections.append(f"  → chain: {r['chain']}")
    if not rows:
        sections.append("- None.")

    # Q2: Active scenarios
    rows = _run_query(client,
        "MATCH (s:Scenario {active: true}) "
        "RETURN s.name AS name, s.probability AS prob, s.last_updated AS updated "
        "ORDER BY s.probability DESC"
    )
    sections.append("\n## Active Scenarios")
    for r in rows:
        sections.append(f"- {r['name']}: {float(r['prob']):.0%} (updated {r['updated'][:10]})")
    if not rows:
        sections.append("- None.")

    # Q3: Infrastructure at risk
    rows = _run_query(client,
        "MATCH (s:Scenario {active: true})-[:THREATENS]->(i:Infrastructure) "
        "RETURN s.name AS scenario, i.name AS infra, i.type AS type, i.country_iso3 AS iso3"
    )
    sections.append("\n## Infrastructure at Risk")
    for r in rows:
        sections.append(f"- {r['infra']} ({r['type']}, {r['iso3']}) ← {r['scenario']}")
    if not rows:
        sections.append("- None.")

    # Q4: Causal chain paths (Event → Scenario → Infra)
    rows = _run_query(client,
        "MATCH (e:Event)-[:ESCALATES]->(s:Scenario)-[:THREATENS]->(i:Infrastructure) "
        "WHERE e.published_at > toString(datetime() - duration('PT24H')) "
        "RETURN e.headline AS event, s.name AS scenario, i.name AS infra LIMIT 10"
    )
    sections.append("\n## Causal Chain Paths (24h)")
    for r in rows:
        sections.append(f"- {r['event'][:80]}")
        sections.append(f"  → {r['scenario']} → {r['infra']}")
    if not rows:
        sections.append("- None.")

    # Q5: Cross-country contagion
    rows = _run_query(client,
        "MATCH (c1:Country)<-[:AFFECTS]-(e:Event)-[:AFFECTS]->(c2:Country) "
        "WHERE e.published_at > toString(datetime() - duration('PT24H')) AND c1 <> c2 "
        "RETURN e.headline AS event, c1.iso3 AS from_c, c2.iso3 AS to_c LIMIT 15"
    )
    sections.append("\n## Cross-Country Contagion (24h)")
    for r in rows:
        sections.append(f"- {r['from_c']} → {r['to_c']}: {r['event'][:70]}")
    if not rows:
        sections.append("- None.")

    return "\n".join(sections)


def dispatch_brief(brief: str, dry_run: bool = False) -> None:
    """Send morning brief to Apollo's inbox."""
    if dry_run:
        print(brief)
        return
    cmd = [
        sys.executable, str(_SEND_PY),
        "--from", "hermes",
        "--to", "apollo",
        "--type", "status",
        "--subject", f"KG Morning Brief — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "--body", brief,
        "--no-reply-expected",
    ]
    try:
        subprocess.run(cmd, check=True, timeout=15)
        logger.info("Morning brief dispatched to Apollo.")
    except Exception as exc:
        logger.error("Failed to dispatch morning brief: %s", exc)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = Neo4jClient()
    try:
        brief = build_morning_brief(client)
        dispatch_brief(brief, dry_run=args.dry_run)
    finally:
        client.close()
```

**Step 2: Commit**

```bash
git add processing/kg_morning_brief.py
git commit -m "feat(kg): morning briefing — 5 Apollo Cypher queries dispatched to Apollo inbox at 06:00 ET"
```

---

### Task 7: TTL cleanup (`kg_ttl_cleanup.py`)

**Files:**
- Create: `processing/kg_ttl_cleanup.py`

**Step 1: Write the implementation**

```python
# processing/kg_ttl_cleanup.py
"""
kg_ttl_cleanup.py — Nightly cleanup of expired Layer C (Signal) nodes.

Deletes Event nodes older than 72h with no active Scenario links.
Run nightly via cron:
    0 4 * * * cd /media/peter/.../global_financial_kb && python -m processing.kg_ttl_cleanup
"""
from __future__ import annotations

import argparse
import logging

from knowledge_base.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


def cleanup(client: Neo4jClient, ttl_hours: int = 72, dry_run: bool = False) -> int:
    """Delete expired Layer C nodes. Returns count deleted."""
    if dry_run:
        with client._driver.session() as s:
            result = s.run(
                "MATCH (e:Event) "
                "WHERE datetime(e.published_at) < datetime() - duration({hours: $h}) "
                "  AND NOT (e)-[:ESCALATES]->(:Scenario {active: true}) "
                "RETURN count(e) AS would_delete",
                h=ttl_hours,
            )
            row = result.single()
            n = int(row["would_delete"]) if row else 0
            logger.info("DRY RUN: would delete %d expired Event nodes", n)
            return n

    deleted = client.delete_old_events(older_than_hours=ttl_hours)
    logger.info("Deleted %d expired Event nodes (TTL=%dh)", deleted, ttl_hours)
    return deleted


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl-hours", type=int, default=72)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = Neo4jClient()
    try:
        n = cleanup(client, ttl_hours=args.ttl_hours, dry_run=args.dry_run)
        print(f"{'Would delete' if args.dry_run else 'Deleted'} {n} Event nodes.")
    finally:
        client.close()
```

**Step 2: Commit**

```bash
git add processing/kg_ttl_cleanup.py
git commit -m "feat(kg): nightly TTL cleanup — delete expired Layer C Event nodes"
```

---

### Task 8: Wire Neo4j into dispatch pipeline

**Files:**
- Modify: `run_news_dispatch.py` (add `kg_ingest` + `event_triggers` + `kg_to_openbrain` calls after dispatch)

**Step 1: Read the current file before editing**

Already read at the start of this session. Lines 89–104 contain the dispatch logic. Insert after line 104 (after `dispatcher.dispatch_*` call and before the `if n == 0:` check).

**Step 2: Add kg_ingest call**

In `run_news_dispatch.py`, after the existing imports at line 53, add:

```python
# ── KG ingestion (optional — skipped if Neo4j unavailable) ───────────────────
try:
    from processing.kg_ingest import ingest_parquet as _kg_ingest_parquet
    from processing.event_triggers import evaluate_triggers as _evaluate_triggers
    from processing.kg_to_openbrain import sync_all as _kg_sync_openbrain
    from knowledge_base.neo4j_client import Neo4jClient as _Neo4jClient
    _KG_AVAILABLE = True
except ImportError:
    _KG_AVAILABLE = False
```

Then in `main()`, after the existing `n = dispatcher.dispatch_*` block (around line 99), add:

```python
    # ── KG ingestion ─────────────────────────────────────────────────────────
    if _KG_AVAILABLE and not args.dry_run:
        try:
            _neo4j = _Neo4jClient()
            # Ingest whichever parquet was just dispatched
            if args.force:
                _kg_ingest_parquet(Path(args.force), _neo4j)
            elif args.parquet:
                _kg_ingest_parquet(Path(args.parquet), _neo4j)
            else:
                # Full sweep: ingest all parquets that were dispatched this run
                interp_dir = Path(__file__).parent / "data" / "interpretations"
                import glob as _glob
                for pq in sorted(_glob.glob(str(interp_dir / "interpretations_*.parquet")))[-5:]:
                    _kg_ingest_parquet(Path(pq), _neo4j)
            _evaluate_triggers(_neo4j)
            _kg_sync_openbrain(_neo4j)
            _neo4j.close()
        except Exception as _kg_exc:
            logger.warning("KG ingestion skipped (Neo4j unavailable?): %s", _kg_exc)
```

**Step 3: Apply the edit**

Make the two edits described above to `run_news_dispatch.py` using the Edit tool.

**Step 4: Smoke test (dry run)**

```bash
cd /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb
python run_news_dispatch.py --dry-run
```
Expected: runs cleanly, logs show "Nothing dispatched." or dispatch count. No KG writes in dry-run mode.

**Step 5: Commit**

```bash
git add run_news_dispatch.py
git commit -m "feat(kg): wire kg_ingest + triggers + openbrain sync into dispatch pipeline"
```

---

### Task 9: Add `load_from_neo4j()` to `graph_builder.py`

**Files:**
- Modify: `knowledge_base/graph_builder.py`

**Context:** The existing `KnowledgeGraphBuilder` builds a NetworkX graph from in-memory seeds. Add a `load_from_neo4j(driver)` class method that rebuilds the NetworkX graph from Neo4j nodes — called at startup to sync NetworkX with the persistent store.

**Step 1: Read `graph_builder.py` lines 60–120 for context**

```bash
sed -n '60,120p' knowledge_base/graph_builder.py
```

**Step 2: Add the method**

Locate the `KnowledgeGraphBuilder` class definition and add:

```python
@classmethod
def load_from_neo4j(cls, driver) -> "KnowledgeGraphBuilder":
    """
    Build a KnowledgeGraphBuilder instance seeded from Neo4j Sovereign + Statistic nodes.
    Falls back to the standard in-memory seed if Neo4j is unreachable.
    """
    builder = cls()  # creates standard in-memory graph
    try:
        with driver.session() as s:
            # Sovereign nodes → existing graph node format
            countries = list(s.run("MATCH (c:Country) RETURN c.iso3, c.name"))
            for row in countries:
                iso3 = row["c.iso3"]
                if iso3 and iso3 not in builder.graph:
                    builder.graph.add_node(iso3, node_type="Sovereign", country=row["c.name"])
            # Stat nodes → "{iso3}_{stat_name}" format (matches existing MONITORS edges)
            stats = list(s.run("MATCH (st:Stat) RETURN st.country_iso3, st.stat_name, st.value, st.direction"))
            for row in stats:
                node_id = f"{row['st.country_iso3']}_{row['st.stat_name']}"
                builder.graph.add_node(node_id, node_type="Statistic",
                                       country=row["st.country_iso3"],
                                       stat=row["st.stat_name"],
                                       value=row["st.value"])
        logger.info("NetworkX graph loaded from Neo4j: %d nodes", builder.graph.number_of_nodes())
    except Exception as exc:
        logger.warning("Neo4j unavailable, using in-memory seed: %s", exc)
    return builder
```

**Step 3: Run existing graph tests to confirm nothing broke**

```bash
python -m pytest tests/test_graph_company.py tests/test_geo_financial_bridge.py -v
```
Expected: all PASSED.

**Step 4: Commit**

```bash
git add knowledge_base/graph_builder.py
git commit -m "feat(kg): add load_from_neo4j() to KnowledgeGraphBuilder — syncs NetworkX from Neo4j at startup"
```

---

### Task 10: Cron entries for morning brief + TTL cleanup

**Files:**
- Modify: `run_daily_ingestion.sh` (or create if absent)

**Step 1: Check if file exists**

```bash
ls /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb/run_daily_ingestion.sh
```

**Step 2: Add cron entries**

Run `crontab -e` and add:

```cron
# KG morning brief — Apollo's 5 Cypher queries dispatched at 06:00 ET (11:00 UTC weekdays)
0 11 * * 1-5 cd /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb && python -m processing.kg_morning_brief >> /tmp/kg_morning_brief.log 2>&1

# KG TTL cleanup — delete expired Layer C Event nodes nightly at 04:00 UTC
0 4 * * * cd /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb && python -m processing.kg_ttl_cleanup >> /tmp/kg_ttl_cleanup.log 2>&1
```

**Step 3: Verify crontab**

```bash
crontab -l | grep kg_
```
Expected: 2 lines shown.

**Step 4: Commit**

```bash
git commit --allow-empty -m "ops: add KG morning brief and TTL cleanup cron entries"
```

---

### Task 11: Full integration smoke test

**Pre-condition:** Neo4j server is running at bolt://localhost:7687.

**Step 1: Seed the database**

```bash
cd /media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb
python -m knowledge_base.kg_seed
```
Expected: logs show seeded N countries, N infrastructure, N trade routes, N scenarios.

**Step 2: Ingest a recent interpretation parquet**

```bash
ls data/interpretations/interpretations_*.parquet | tail -1
python -m processing.kg_ingest --parquet data/interpretations/interpretations_<latest>.parquet
```
Expected: logs show "Ingesting N rows".

**Step 3: Evaluate triggers**

```bash
python -c "
from knowledge_base.neo4j_client import Neo4jClient
from processing.event_triggers import evaluate_triggers
c = Neo4jClient()
fired = evaluate_triggers(c, dry_run=True)
print('Would fire:', fired)
c.close()
"
```
Expected: prints trigger IDs that match current flag state.

**Step 4: Run morning brief dry run**

```bash
python -m processing.kg_morning_brief --dry-run
```
Expected: formatted markdown brief printed to stdout with 5 sections.

**Step 5: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short
```
Expected: all tests PASS. Any Neo4j-dependent tests (neo4j_client) pass via mocks.

---

## Out of Scope

- Neo4j server installation (infrastructure — install separately via `apt install neo4j` or Docker)
- Historical backfill of pre-existing interpretation parquets (separate one-off task)
- GraphQL/REST API over Neo4j
- Removing NetworkX

## First Run Checklist

Before running Task 11, ensure:
- [ ] Neo4j is running: `systemctl status neo4j` or `docker ps | grep neo4j`
- [ ] Python driver installed: `pip show neo4j`
- [ ] `.env` or env vars set if auth is needed (`NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`)
- [ ] `~/.claude/agent-tools/send.py` exists and works: `python3 ~/.claude/agent-tools/send.py --help`
