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
    {"infra_id": "hormuz_strait",    "name": "Strait of Hormuz",      "infra_type": "PORT",          "country_iso3": "IRN", "risk_tier": "critical"},
    {"infra_id": "abqaiq_refinery",  "name": "Abqaiq Processing",     "infra_type": "REFINERY",      "country_iso3": "SAU", "risk_tier": "critical"},
    {"infra_id": "kharg_island",     "name": "Kharg Island Terminal",  "infra_type": "PORT",          "country_iso3": "IRN", "risk_tier": "critical"},
    {"infra_id": "bushehr_npp",      "name": "Bushehr NPP",            "infra_type": "NPP",           "country_iso3": "IRN", "risk_tier": "critical"},
    {"infra_id": "ras_laffan",       "name": "Ras Laffan LNG",        "infra_type": "LNG_TERMINAL",  "country_iso3": "QAT", "risk_tier": "high"},
    {"infra_id": "al_udeid",         "name": "Al Udeid Air Base",      "infra_type": "MILITARY_BASE", "country_iso3": "QAT", "risk_tier": "high"},
    {"infra_id": "5th_fleet_bahrain","name": "5th Fleet HQ Bahrain",   "infra_type": "MILITARY_BASE", "country_iso3": "BHR", "risk_tier": "high"},
    {"infra_id": "natanz_facility",  "name": "Natanz Enrichment",      "infra_type": "ENRICHMENT",    "country_iso3": "IRN", "risk_tier": "critical"},
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
