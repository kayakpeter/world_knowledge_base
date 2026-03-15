# strategist/kg_writer.py
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional

from strategist.config import StrategistConfig
from strategist.schema import ScenarioTree

logger = logging.getLogger(__name__)


class KGWriter:
    """
    Writes a completed ScenarioTree into the Neo4j knowledge graph.

    Uses existing Neo4jClient methods:
    - upsert_scenario(scenario_id, name, probability, active)
    - activate_scenario(scenario_id)
    - upsert_flag(name, value, set_by)
    """

    def __init__(self, config: StrategistConfig, *, neo4j_client=None) -> None:
        self._cfg = config
        if neo4j_client is not None:
            self._client = neo4j_client
        else:
            from knowledge_base.neo4j_client import Neo4jClient
            import os
            self._client = Neo4jClient(
                uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                user=os.environ.get("NEO4J_USER", "neo4j"),
                password=os.environ.get("NEO4J_PASSWORD", ""),
            )

    def write(self, tree: ScenarioTree) -> None:
        """
        Persist the scenario tree to Neo4j:
        1. Upsert the scenario root node
        2. Activate it
        3. Store sector impacts and infra effects as flags (prefixed with scenario_id)
        """
        # 1. Upsert scenario root
        try:
            self._client.upsert_scenario(
                tree.scenario_id,
                tree.trigger_event,
                probability=1.0,
                active=True,
            )
        except Exception:
            logger.exception("Failed to upsert scenario: %s", tree.scenario_id)
            return

        # 2. Activate
        try:
            self._client.activate_scenario(tree.scenario_id)
        except Exception:
            logger.exception("Failed to activate scenario: %s", tree.scenario_id)

        # 3. Store flags for each kept node (tombstones are excluded — they are
        #    not in tree.nodes, only in tree.tombstones)
        prefix = f"SCENARIO_{tree.scenario_id[:16]}"
        for node in tree.nodes.values():
            node_prefix = f"{prefix}_NODE_{node.node_id[:8]}"

            # Store sector impacts
            for impact in node.sector_impacts:
                flag_name = f"{node_prefix}_SECTOR_{impact.sector.upper()}"
                flag_value = (
                    f"dir={impact.direction}|mag={impact.magnitude}"
                    f"|jp={node.joint_probability:.4f}"
                    f"|tickers={','.join(impact.tickers)}"
                )
                self._safe_upsert_flag(flag_name, flag_value)

            # Store infrastructure effects
            for effect in node.infrastructure_effects:
                infra_id = effect.get("infra_id", "UNKNOWN")
                flag_name = f"{node_prefix}_INFRA_{infra_id}"
                flag_value = (
                    f"new_status={effect.get('new_status', '')}|"
                    f"conf={effect.get('confidence', 0.0):.2f}|"
                    f"jp={node.joint_probability:.4f}"
                )
                self._safe_upsert_flag(flag_name, flag_value)

        logger.info("KG write complete for scenario: %s (%d nodes)",
                    tree.scenario_id, tree.node_count())

    def _safe_upsert_flag(self, name: str, value: str) -> None:
        try:
            self._client.upsert_flag(name, value, set_by="strategist")
        except Exception:
            logger.warning("Failed to upsert flag: %s", name)
