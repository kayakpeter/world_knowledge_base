# strategist/infra_state.py
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class InfraStatus(str, Enum):
    OPERATIONAL       = "OPERATIONAL"
    AT_RISK           = "AT_RISK"
    DEGRADED          = "DEGRADED"
    STRUCK_MILITARY   = "STRUCK_MILITARY"
    STRUCK_OIL_INFRA  = "STRUCK_OIL_INFRA"
    CLOSED            = "CLOSED"
    EFFECTIVE_CLOSURE = "EFFECTIVE_CLOSURE"
    SMOKE_REPORTED    = "SMOKE_REPORTED"
    UNKNOWN           = "UNKNOWN"


CANONICAL_INFRA: list[dict] = [
    dict(infra_id="SUEZ_CANAL",      name="Suez Canal",           status=InfraStatus.OPERATIONAL,
         affected_sectors=["energy", "container", "agriculture", "tanker", "bulk"],
         country_iso3="EGY", risk_level=0.30),
    dict(infra_id="HORMUZ_STRAIT",   name="Strait of Hormuz",     status=InfraStatus.EFFECTIVE_CLOSURE,
         affected_sectors=["energy", "tanker", "insurance"],
         country_iso3="IRN", risk_level=0.95),
    dict(infra_id="KHARG_ISLAND",    name="Kharg Island",         status=InfraStatus.STRUCK_MILITARY,
         affected_sectors=["energy", "tanker"],
         country_iso3="IRN", risk_level=0.90),
    dict(infra_id="ABQAIQ",          name="Abqaiq Processing",    status=InfraStatus.AT_RISK,
         affected_sectors=["energy"],
         country_iso3="SAU", risk_level=0.55),
    dict(infra_id="FUJAIRAH",        name="Fujairah Terminal",    status=InfraStatus.SMOKE_REPORTED,
         affected_sectors=["energy", "tanker"],
         country_iso3="ARE", risk_level=0.70),
    dict(infra_id="RAS_LAFFAN",      name="Ras Laffan LNG",       status=InfraStatus.OPERATIONAL,
         affected_sectors=["energy"],
         country_iso3="QAT", risk_level=0.30),
    dict(infra_id="JEBEL_ALI",       name="Jebel Ali Port",       status=InfraStatus.AT_RISK,
         affected_sectors=["container", "bulk"],
         country_iso3="ARE", risk_level=0.55),
    dict(infra_id="BAGHDAD_EMBASSY", name="Baghdad US Embassy",   status=InfraStatus.STRUCK_MILITARY,
         affected_sectors=["sovereign", "currency"],
         country_iso3="IRQ", risk_level=0.80),
]


@dataclass
class InfraNode:
    infra_id: str
    name: str
    status: InfraStatus
    affected_sectors: list[str]
    country_iso3: str = "UNK"
    risk_level: float = 0.0
    confidence: float = 1.0
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = ""


class InfraStateRegistry:
    """
    Manages persistent infrastructure status nodes.
    Neo4j is authoritative; in-memory cache for fast reads during scenario expansion.

    Contract:
    - Always call seed_from_canonical() or load_from_neo4j() before get()/update_status()
    - update_status() writes to both cache and Neo4j
    """

    def __init__(self, neo4j_client) -> None:
        self._client = neo4j_client
        self._cache: dict[str, InfraNode] = {}

    def seed_from_canonical(self) -> None:
        """Populate cache from CANONICAL_INFRA list (no Neo4j call)."""
        for spec in CANONICAL_INFRA:
            node = InfraNode(
                infra_id=spec["infra_id"],
                name=spec["name"],
                status=spec["status"],
                affected_sectors=spec["affected_sectors"],
                country_iso3=spec.get("country_iso3", "UNK"),
                risk_level=spec.get("risk_level", 0.0),
            )
            self._cache[node.infra_id] = node

    def get(self, infra_id: str) -> Optional[InfraNode]:
        """Return cached node or None if not loaded."""
        return self._cache.get(infra_id)

    def get_affected_sectors(self, infra_id: str) -> list[str]:
        """Return affected sectors for an infra node, or empty list."""
        node = self._cache.get(infra_id)
        return node.affected_sectors if node else []

    def update_status(
        self,
        infra_id: str,
        new_status: InfraStatus,
        *,
        confidence: float = 0.8,
        source: str = "",
    ) -> None:
        """Update status in cache and persist to Neo4j."""
        node = self._cache.get(infra_id)
        if node is None:
            logger.warning("update_status called for unknown infra_id=%s", infra_id)
            return
        node.status = new_status
        node.confidence = confidence
        node.source = source
        node.last_updated = datetime.now(timezone.utc).isoformat()
        # Persist to Neo4j as a flag for cross-scenario visibility
        flag_name = f"INFRA_{infra_id}_STATUS"
        flag_value = f"{new_status.value}|conf={confidence:.2f}|src={source}"
        try:
            self._client.upsert_flag(flag_name, flag_value, set_by="strategist")
        except Exception:
            logger.exception("Failed to persist infra status to Neo4j: %s", infra_id)
