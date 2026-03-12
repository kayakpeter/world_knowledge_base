"""
kg_to_openbrain.py — Sync Neo4j graph node summaries to open-brain (Supabase pgvector).

Posts Event and active Scenario summaries as embedded thoughts.
Called after each ingestion cycle. Non-fatal on brain failure.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from knowledge_base.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

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
    """Post summaries of recent Event nodes to open-brain. Returns count posted."""
    with client._driver.session() as s:
        rows = list(s.run(
            "MATCH (e:Event) "
            "WHERE e.published_at > toString(datetime() - duration({hours: $h})) "
            "OPTIONAL MATCH (e)-[:AFFECTS]->(c:Country) "
            "RETURN e.id, e.headline, e.urgency, e.causal_chain, "
            "       e.confidence, e.published_at, collect(c.iso3) AS countries",
            h=since_hours,
        ))

    count = 0
    for row in rows:
        countries = ", ".join(row["countries"]) if row["countries"] else "unknown"
        content = (
            f"[{(row['e.urgency'] or 'unknown').upper()}] {row['e.headline']} "
            f"— affects: {countries}, "
            f"chain: {row['e.causal_chain']}, "
            f"confidence: {row['e.confidence']}%"
        )
        metadata = {
            "agent": "apollo",
            "type": "kg_event",
            "subsystem": "geopolitical",
            "tags": ["kg-event", row["e.urgency"] or "unknown"],
            "item_id": row["e.id"],
            "published_at": row["e.published_at"],
        }
        thought_id = _post_thought(content, metadata)
        if thought_id:
            count += 1

    logger.info("Synced %d events to open-brain", count)
    return count


def sync_scenarios_to_openbrain(client: Neo4jClient) -> int:
    """Post summaries of active Scenario nodes to open-brain. Returns count posted."""
    with client._driver.session() as s:
        rows = list(s.run(
            "MATCH (s:Scenario {active: true}) "
            "OPTIONAL MATCH (s)-[:THREATENS]->(i:Infrastructure) "
            "RETURN s.id, s.name, s.probability, s.active, collect(i.id) AS infra",
        ))

    count = 0
    for row in rows:
        infra_list = ", ".join(row["infra"]) if row["infra"] else "none"
        prob = row["s.probability"] or 0.0
        content = (
            f"Scenario {row['s.id']} ACTIVE: {row['s.name']} "
            f"— probability={float(prob):.0%}, "
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
