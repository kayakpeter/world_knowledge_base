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
                "DELETE e RETURN count(e) AS deleted",
                h=older_than_hours,
            )
            row = result.single()
            return int(row["deleted"]) if row else 0
