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
    """Delete expired Layer C nodes. Returns count deleted (or would delete in dry-run)."""
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
