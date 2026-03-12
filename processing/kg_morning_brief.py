"""
kg_morning_brief.py — Run Apollo's 5 morning Cypher queries → dispatch to Apollo inbox.

Schedule via cron at 06:00 ET (11:00 UTC weekdays):
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
