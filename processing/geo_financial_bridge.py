"""
Geopolitical↔Financial Bridge

Reads recent strategic analysis thoughts from Open Brain and injects them
as CONTAGION edges into the knowledge graph.

Each OrderEffect stored in Open Brain has metadata fields:
  affected_countries: ["United States", "Germany"]
  affected_stats: ["policy_rate", "yield_spread_10y3m"]
  probability: 0.65

This bridge converts those into directed CONTAGION edges between the matching
Statistic nodes in the NetworkX graph. Node IDs use the real graph format:
'{iso3}_{stat_name}' (e.g. 'USA_policy_rate', 'DEU_yield_spread_10y3m').
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx

from config.settings import COUNTRY_CODES

logger = logging.getLogger(__name__)

# Build a full-name → iso3 lookup from the shared registry
_COUNTRY_TO_ISO3: dict[str, str] = {
    country: codes.get("iso3", "")
    for country, codes in COUNTRY_CODES.items()
}


@dataclass
class GeoEdge:
    source_country: str
    source_stat: str
    target_country: str
    target_stat: str
    weight: float          # = probability from the OrderEffect
    mechanism: str         # human-readable transmission channel
    event_title: str       # the originating geopolitical event


class GeoFinancialBridge:
    """
    Converts Open Brain geopolitical thoughts → CONTAGION graph edges.

    Call parse_thoughts() to convert raw thought dicts from the MCP.
    Call inject_into_graph() to add edges to a NetworkX graph.
    """

    def parse_thoughts(self, thoughts: list[dict]) -> list[GeoEdge]:
        """
        Convert Open Brain thought records into GeoEdge objects.

        Each thought must have metadata with:
          - affected_countries: list[str] (min 2 entries: source, target)
          - affected_stats: list[str] (min 1 entry)
          - probability: float
        """
        edges: list[GeoEdge] = []
        for t in thoughts:
            meta = t.get("metadata", {})
            countries = meta.get("affected_countries", [])
            stats = meta.get("affected_stats", [])
            prob = float(meta.get("probability", 0.0))
            title = meta.get("event_title", t.get("content", "unknown")[:60])

            if len(countries) < 2 or not stats or prob <= 0.0:
                continue

            source_country = countries[0]
            for target_country in countries[1:]:
                for stat in stats:
                    edges.append(GeoEdge(
                        source_country=source_country,
                        source_stat=stat,
                        target_country=target_country,
                        target_stat=stat,  # same stat, cross-border contagion
                        weight=min(prob, 1.0),
                        mechanism=meta.get("mechanism", "geopolitical contagion"),
                        event_title=title,
                    ))
        logger.info(f"GeoFinancialBridge: parsed {len(edges)} edges from {len(thoughts)} thoughts")
        return edges

    def inject_into_graph(
        self,
        G: nx.DiGraph,
        edges: list[GeoEdge],
        country_codes: dict[str, str] | None = None,
    ) -> int:
        """
        Add GeoEdge objects to the graph as CONTAGION edges.

        Node IDs are built as '{iso3}_{stat_name}' to match the real graph format.
        country_codes: optional override mapping full country name → iso3.
                       Defaults to COUNTRY_CODES from config.settings.
        Skips edges where source or target node doesn't exist in graph.
        Returns count of edges injected.
        """
        lookup = country_codes if country_codes is not None else _COUNTRY_TO_ISO3
        injected = 0
        for e in edges:
            src_iso3 = lookup.get(e.source_country, "")
            tgt_iso3 = lookup.get(e.target_country, "")
            src_node = f"{src_iso3}_{e.source_stat}"
            tgt_node = f"{tgt_iso3}_{e.target_stat}"
            if src_node not in G or tgt_node not in G:
                logger.warning(
                    "GeoFinancialBridge: skipping edge — node not in graph "
                    "(%s → %s)", src_node, tgt_node
                )
                continue
            # If edge already exists, take the max weight (most pessimistic)
            if G.has_edge(src_node, tgt_node):
                existing = G[src_node][tgt_node].get("weight", 0.0)
                G[src_node][tgt_node]["weight"] = max(existing, e.weight)
                G[src_node][tgt_node]["mechanism"] += f"; {e.mechanism}"
            else:
                G.add_edge(
                    src_node, tgt_node,
                    edge_type="CONTAGION",
                    weight=e.weight,
                    mechanism=e.mechanism,
                    event_title=e.event_title,
                    source="geopolitical",
                )
            injected += 1
        logger.info(f"GeoFinancialBridge: injected {injected} edges into graph")
        return injected
