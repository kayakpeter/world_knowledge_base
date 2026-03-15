#!/usr/bin/env python3
# strategist/run_strategist.py
"""
CLI entry point for the Strategist scenario expansion engine.

Usage:
    python -m strategist.run_strategist --event "US strikes Kharg Island" \
        --severity CRITICAL --confirmed

    python -m strategist.run_strategist --event "Iran threatens Hormuz" \
        --severity HIGH
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("strategist.run")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strategist — probabilistic geopolitical scenario expansion engine"
    )
    parser.add_argument("--event", required=True, help="Trigger event description")
    parser.add_argument(
        "--severity",
        choices=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default="HIGH",
        help="Event severity (default: HIGH)",
    )
    parser.add_argument(
        "--confirmed",
        action="store_true",
        help="Mark event as confirmed (not just a threat)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing to Neo4j (mock KG writer)",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> int:
    from strategist.config import StrategistConfig
    from strategist.expander import ScenarioExpander
    from strategist.infra_state import InfraStateRegistry
    from strategist.prometheus_writer import PrometheusDeltaWriter
    from strategist.kg_writer import KGWriter
    from strategist.runner import ScenarioRunner

    cfg = StrategistConfig()

    logger.info("=== STRATEGIST ===")
    logger.info("Event:    %s", args.event)
    logger.info("Severity: %s", args.severity)
    logger.info("Confirmed: %s", args.confirmed)

    # Build expander (uses LocalModelProvider → Ollama)
    expander = ScenarioExpander(cfg)

    # Build infra registry
    if args.dry_run:
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.upsert_flag = MagicMock()
        infra_reg = InfraStateRegistry(mock_client)
    else:
        from knowledge_base.neo4j_client import Neo4jClient
        import os
        neo4j_client = Neo4jClient(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "palantir-2026"),
        )
        infra_reg = InfraStateRegistry(neo4j_client)

    infra_reg.seed_from_canonical()

    # Run scenario expansion
    runner = ScenarioRunner(cfg, expander=expander, infra_registry=infra_reg)
    logger.info("Expanding scenario...")
    tree = await runner.run_async(
        trigger_event=args.event,
        severity=args.severity,
        confirmed=args.confirmed,
    )

    logger.info("Expansion complete: %d nodes, %d tombstones",
                tree.node_count(), len(tree.tombstones))

    # Write Prometheus delta
    prom_writer = PrometheusDeltaWriter(cfg)
    delta = prom_writer.write(tree)
    logger.info("Prometheus delta: %s", {k: v for k, v in delta.items()
                                          if k.endswith("_bonus")})

    # Write to KG
    if not args.dry_run:
        from knowledge_base.neo4j_client import Neo4jClient
        import os
        kg_client = Neo4jClient(
            uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "palantir-2026"),
        )
        kg_writer = KGWriter(cfg, neo4j_client=kg_client)
        kg_writer.write(tree)
        logger.info("KG write complete")
    else:
        logger.info("[dry-run] skipping KG write")

    logger.info("Scenario saved: %s", tree.scenario_id)
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(asyncio.run(main(args)))
