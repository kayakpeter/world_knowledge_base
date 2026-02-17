"""
Knowledge Graph — Schema definition and graph builder.

Builds a typed, weighted directed graph from ingested observations.
Node types: Sovereign, Statistic, Scenario, MarkovState
Edge types: MONITORS, CONTAGION, TRIGGERS, TRANSITIONS, TRADE_DEPENDS
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import polars as pl

from config.settings import (
    COUNTRIES,
    COUNTRY_CODES,
    STAT_REGISTRY,
    FULL_STAT_REGISTRY,
    StatDefinition,
)

logger = logging.getLogger(__name__)


# ─── Trade Dependency Weights ────────────────────────────────────────────────
# Approximate bilateral trade intensity (exports + imports as share of total)
# Source: IMF Direction of Trade Statistics, simplified to major corridors
# These are seed values — the LLM processing phase refines them
TRADE_CORRIDORS: list[tuple[str, str, float]] = [
    ("United States", "China", 0.65),
    ("United States", "Canada", 0.75),
    ("United States", "Mexico", 0.72),
    ("United States", "Japan", 0.40),
    ("United States", "Germany", 0.38),
    ("United States", "United Kingdom", 0.35),
    ("United States", "South Korea", 0.33),
    ("United States", "India", 0.28),
    ("China", "Japan", 0.55),
    ("China", "South Korea", 0.52),
    ("China", "Germany", 0.42),
    ("China", "Australia", 0.58),
    ("China", "Brazil", 0.35),
    ("China", "Russia", 0.30),
    ("China", "Indonesia", 0.32),
    ("Germany", "France", 0.55),
    ("Germany", "Netherlands", 0.52),
    ("Germany", "Italy", 0.42),
    ("Germany", "Poland", 0.45),
    ("Germany", "Spain", 0.30),
    ("Japan", "South Korea", 0.35),
    ("Japan", "Australia", 0.30),
    ("United Kingdom", "France", 0.30),
    ("United Kingdom", "Netherlands", 0.28),
    ("Canada", "Mexico", 0.20),
    ("Saudi Arabia", "India", 0.40),
    ("Saudi Arabia", "China", 0.45),
    ("Russia", "Turkey", 0.25),
    ("Brazil", "Mexico", 0.18),
    ("Indonesia", "India", 0.22),
    ("Indonesia", "Japan", 0.30),
]

# ─── Cross-Border Contagion Channels ────────────────────────────────────────
# Maps (source_stat, target_stat) → base_weight
# These capture the economic transmission mechanisms
CONTAGION_CHANNELS: list[tuple[str, str, str, str, float, str]] = [
    # (source_country, source_stat, target_country, target_stat, weight, mechanism)
    ("United States", "policy_rate", "*", "policy_rate", 0.85, "Dollar hegemony forces CB responses"),
    ("United States", "yield_spread_10y3m", "*", "yield_spread_10y3m", 0.70, "Global risk-free rate benchmark"),
    ("China", "real_gdp_growth", "*", "export_velocity", 0.60, "Demand channel for commodity exporters"),
    ("China", "private_credit_npls", "*", "bank_capital_adequacy", 0.45, "Shadow banking contagion"),
    ("United States", "m2_supply", "*", "currency_volatility", 0.55, "Dollar liquidity drives EM FX"),
    ("Saudi Arabia", "crude_output", "*", "wti_brent_spread", 0.75, "OPEC supply-side dominance"),
    ("Germany", "real_gdp_growth", "France", "real_gdp_growth", 0.50, "Eurozone core linkage"),
    ("Germany", "real_gdp_growth", "Italy", "real_gdp_growth", 0.45, "Eurozone core linkage"),
    ("Germany", "real_gdp_growth", "Poland", "real_gdp_growth", 0.55, "Supply chain integration"),
    ("Japan", "policy_rate", "*", "currency_volatility", 0.40, "Yen carry trade unwind channel"),
]


@dataclass
class GraphMetrics:
    """Summary statistics for the knowledge graph."""
    total_nodes: int = 0
    sovereign_nodes: int = 0
    statistic_nodes: int = 0
    scenario_nodes: int = 0
    total_edges: int = 0
    monitors_edges: int = 0
    contagion_edges: int = 0
    trade_edges: int = 0
    avg_degree: float = 0.0
    density: float = 0.0


class KnowledgeGraphBuilder:
    """
    Builds and manages the global financial knowledge graph.

    The graph has three layers:
    1. Sovereign layer — country nodes with trade dependency edges
    2. Statistic layer — metric nodes linked to countries via MONITORS edges
    3. Contagion layer — cross-border transmission channels between stats
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._built = False

    def build_from_observations(self, observations_df: pl.DataFrame) -> nx.DiGraph:
        """
        Build the full knowledge graph from ingested observations.

        Args:
            observations_df: Polars DataFrame from the ingestion pipeline
                Expected columns: country, country_iso3, stat_name, category,
                value, period, node_id

        Returns:
            The populated NetworkX DiGraph
        """
        logger.info("Building knowledge graph from %d observations...", len(observations_df))

        self._add_sovereign_nodes()
        self._add_trade_edges()
        self._add_statistic_nodes(observations_df)
        self._add_contagion_edges()
        self._built = True

        metrics = self.get_metrics()
        logger.info(
            "Graph built: %d nodes (%d sovereign, %d stats), %d edges (density=%.4f)",
            metrics.total_nodes, metrics.sovereign_nodes, metrics.statistic_nodes,
            metrics.total_edges, metrics.density,
        )

        return self.graph

    def _add_sovereign_nodes(self) -> None:
        """Add the 20 country nodes with metadata."""
        for rank, country in enumerate(COUNTRIES, 1):
            codes = COUNTRY_CODES.get(country, {})
            self.graph.add_node(
                country,
                node_type="Sovereign",
                iso3=codes.get("iso3", ""),
                iso2=codes.get("iso2", ""),
                gdp_rank=rank,
                markov_state="S0_Tranquil",  # initial state
            )

    def _add_trade_edges(self) -> None:
        """Add bilateral trade dependency edges between sovereigns."""
        for src, tgt, weight in TRADE_CORRIDORS:
            if src in self.graph and tgt in self.graph:
                # Bidirectional trade dependency
                self.graph.add_edge(
                    src, tgt,
                    edge_type="TRADE_DEPENDS",
                    weight=weight,
                    mechanism="bilateral_trade",
                )
                self.graph.add_edge(
                    tgt, src,
                    edge_type="TRADE_DEPENDS",
                    weight=weight * 0.9,  # slight asymmetry — larger partner has more pull
                    mechanism="bilateral_trade",
                )

    def _add_statistic_nodes(self, observations_df: pl.DataFrame) -> None:
        """
        Add statistic nodes from observations and link to countries.

        Creates one node per (country, stat_name) pair with the latest value.
        """
        if observations_df.is_empty():
            logger.warning("Empty observations — adding stat nodes from registry only")
            self._add_stat_nodes_from_registry()
            return

        # Get the latest observation per (country_iso3, stat_name)
        latest_df = (
            observations_df
            .sort("period", descending=True)
            .group_by(["country_iso3", "stat_name"])
            .first()
        )

        for row in latest_df.iter_rows(named=True):
            node_id = row["node_id"]
            country = row["country"]

            self.graph.add_node(
                node_id,
                node_type="Statistic",
                stat_name=row["stat_name"],
                category=row["category"],
                value=row["value"],
                period=row["period"],
                country_iso3=row["country_iso3"],
            )

            # Link country → stat via MONITORS edge
            if country in self.graph:
                self.graph.add_edge(
                    country, node_id,
                    edge_type="MONITORS",
                    weight=1.0,
                )

        # Also add nodes for stats not yet in the data (LLM-deferred, etc.)
        self._add_stat_nodes_from_registry()

    def _add_stat_nodes_from_registry(self) -> None:
        """Add placeholder nodes for stats not yet populated by data."""
        for stat in FULL_STAT_REGISTRY:
            for country in COUNTRIES:
                codes = COUNTRY_CODES.get(country, {})
                iso3 = codes.get("iso3", "")
                node_id = f"{iso3}_{stat.name}"

                if node_id not in self.graph:
                    self.graph.add_node(
                        node_id,
                        node_type="Statistic",
                        stat_name=stat.name,
                        category=stat.category,
                        value=None,  # awaiting data
                        period=None,
                        country_iso3=iso3,
                        source_type=stat.source_type,
                    )
                    if country in self.graph:
                        self.graph.add_edge(
                            country, node_id,
                            edge_type="MONITORS",
                            weight=1.0,
                        )

    def _add_contagion_edges(self) -> None:
        """
        Add cross-border contagion channels.

        Wildcard '*' in target_country means the channel applies to all countries.
        """
        for src_country, src_stat, tgt_country, tgt_stat, weight, mechanism in CONTAGION_CHANNELS:
            src_iso3 = COUNTRY_CODES.get(src_country, {}).get("iso3", "")
            src_node = f"{src_iso3}_{src_stat}"

            if tgt_country == "*":
                # Apply to all countries except source
                target_countries = [c for c in COUNTRIES if c != src_country]
            else:
                target_countries = [tgt_country]

            for tc in target_countries:
                tgt_iso3 = COUNTRY_CODES.get(tc, {}).get("iso3", "")
                tgt_node = f"{tgt_iso3}_{tgt_stat}"

                if src_node in self.graph and tgt_node in self.graph:
                    self.graph.add_edge(
                        src_node, tgt_node,
                        edge_type="CONTAGION",
                        weight=weight,
                        mechanism=mechanism,
                        source_country=src_country,
                        target_country=tc,
                    )

    def get_metrics(self) -> GraphMetrics:
        """Calculate summary metrics for the graph."""
        sovereign = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "Sovereign")
        stats = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "Statistic")
        scenarios = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "Scenario")

        monitors = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("edge_type") == "MONITORS")
        contagion = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("edge_type") == "CONTAGION")
        trade = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("edge_type") == "TRADE_DEPENDS")

        n = self.graph.number_of_nodes()
        return GraphMetrics(
            total_nodes=n,
            sovereign_nodes=sovereign,
            statistic_nodes=stats,
            scenario_nodes=scenarios,
            total_edges=self.graph.number_of_edges(),
            monitors_edges=monitors,
            contagion_edges=contagion,
            trade_edges=trade,
            avg_degree=sum(dict(self.graph.degree()).values()) / max(n, 1),
            density=nx.density(self.graph),
        )

    def propagate_shock(
        self,
        source_node: str,
        shock_magnitude: float,
        decay_rate: float = 0.6,
        min_impact: float = 0.01,
    ) -> dict[str, float]:
        """
        Propagate a shock through the graph using BFS with decay.

        Args:
            source_node: The node where the shock originates
            shock_magnitude: Initial shock size (e.g., 1.5 for a 150bp rate move)
            decay_rate: How much the shock decays at each hop (0-1)
            min_impact: Stop propagating when impact falls below this

        Returns:
            Dict of node_id → impact magnitude for all affected nodes
        """
        if source_node not in self.graph:
            logger.warning("Source node %s not in graph", source_node)
            return {}

        impacts: dict[str, float] = {source_node: shock_magnitude}
        queue: list[tuple[str, float]] = [(source_node, shock_magnitude)]
        visited: set[str] = {source_node}

        while queue:
            current_node, current_impact = queue.pop(0)

            for neighbor in self.graph.successors(current_node):
                if neighbor in visited:
                    continue

                edge_data = self.graph.edges[current_node, neighbor]
                edge_weight = edge_data.get("weight", 0.5)

                # Impact = parent_impact × edge_weight × decay
                neighbor_impact = current_impact * edge_weight * decay_rate

                if abs(neighbor_impact) >= min_impact:
                    impacts[neighbor] = neighbor_impact
                    queue.append((neighbor, neighbor_impact))
                    visited.add(neighbor)

        logger.info(
            "Shock propagation from %s: %d nodes affected (max depth impact: %.4f)",
            source_node, len(impacts), min(impacts.values(), default=0),
        )
        return impacts

    def export_graph(self, output_path: Path) -> None:
        """Export the graph to a JSON file for visualization or transfer."""
        data = nx.node_link_data(self.graph)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Graph exported to %s", output_path)

    def get_country_dashboard(self, country: str) -> dict:
        """
        Get the full 50-stat dashboard for a single country.

        Returns a dict keyed by category, each containing stat values.
        """
        codes = COUNTRY_CODES.get(country, {})
        iso3 = codes.get("iso3", "")

        dashboard: dict[str, list[dict]] = {}

        for stat in FULL_STAT_REGISTRY:
            node_id = f"{iso3}_{stat.name}"
            node_data = self.graph.nodes.get(node_id, {})

            entry = {
                "stat_id": stat.stat_id,
                "name": stat.name,
                "value": node_data.get("value"),
                "period": node_data.get("period"),
                "source_type": stat.source_type,
                "unit": stat.unit,
            }

            category = stat.category
            if category not in dashboard:
                dashboard[category] = []
            dashboard[category].append(entry)

        return dashboard
