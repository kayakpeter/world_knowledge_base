"""
kg_ingest.py — Parse interpretation parquets → upsert Neo4j nodes → fire triggers.

Called from run_news_dispatch.py after each dispatch cycle, or standalone for backfill:
    python -m processing.kg_ingest --parquet data/interpretations/foo.parquet
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
    """Map causal chain text to matching scenario IDs (deduplicated)."""
    lower = (causal_chain or "").lower()
    seen: set[str] = set()
    result: list[str] = []
    for kw, sid in _CHAIN_TO_SCENARIO:
        if kw in lower and sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


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
        item_id      = str(row.get("item_id", "") or "")
        headline     = str(row.get("headline", "") or "")
        published    = str(row.get("published_at", "") or "")
        urgency      = str(row.get("urgency", "") or "")
        sentiment    = str(row.get("sentiment", "") or "")
        chain        = str(row.get("causal_chain", "") or "")
        confidence   = int(row.get("confidence", 0) or 0)
        country_iso3 = str(row.get("country_iso3", "") or "")

        if not item_id:
            continue

        # Upsert Event node — all kwargs so tests can introspect by name
        client.upsert_event(
            item_id=item_id,
            headline=headline,
            published_at=published,
            urgency=urgency,
            sentiment=sentiment,
            causal_chain=chain,
            confidence=confidence,
        )

        # AFFECTS edges: primary country + cross-country iso3 list
        iso3_list = [country_iso3] if country_iso3 else []
        cross = str(row.get("cross_country_iso3", "") or "")
        iso3_list += [x.strip() for x in cross.split(",") if x.strip()]

        # stat_direction and magnitude parallel affected_stats
        stat_dirs  = str(row.get("stat_direction", "") or "").split(",")
        magnitudes = str(row.get("estimated_magnitude", "") or "").split(",")

        for i, iso3 in enumerate(iso3_list):
            direction = stat_dirs[i].strip() if i < len(stat_dirs) else ""
            try:
                magnitude = float(magnitudes[i]) if i < len(magnitudes) and magnitudes[i].strip() else None
            except (ValueError, TypeError):
                magnitude = None
            client.create_affects_edge(item_id, iso3, direction, magnitude)

        # Upsert Stat nodes — affected_stats is comma-separated stat names
        affected_stats_raw = str(row.get("affected_stats", "") or "")
        stat_names = [x.strip() for x in affected_stats_raw.split(",") if x.strip()]
        for j, stat_name in enumerate(stat_names):
            direction = stat_dirs[j].strip() if j < len(stat_dirs) else ""
            try:
                magnitude = float(magnitudes[j]) if j < len(magnitudes) and magnitudes[j].strip() else None
            except (ValueError, TypeError):
                magnitude = None
            client.upsert_stat(
                country_iso3=country_iso3,
                stat_name=stat_name,
                value=magnitude,
                direction=direction,
                estimated_at=published,
            )

        # ESCALATES edges from causal chain keyword matching
        for scenario_id in _chain_to_scenarios(chain):
            client.create_escalates_edge(item_id, scenario_id=scenario_id)

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
