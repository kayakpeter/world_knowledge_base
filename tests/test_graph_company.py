from __future__ import annotations
import pytest
import networkx as nx
from knowledge_base.graph_builder import KnowledgeGraphBuilder


def _empty_builder() -> KnowledgeGraphBuilder:
    b = KnowledgeGraphBuilder()
    b._add_sovereign_nodes()
    return b


def test_add_sector_nodes_creates_nodes():
    """add_sector_nodes() adds Sector-typed nodes to the graph."""
    b = _empty_builder()
    b.add_sector_nodes()
    sector_nodes = [
        n for n, d in b.graph.nodes(data=True)
        if d.get("node_type") == "Sector"
    ]
    assert len(sector_nodes) >= 5
    assert "Biotech" in b.graph
    assert "EV" in b.graph


def test_add_sector_nodes_exposed_to_edges():
    """Sector nodes have EXPOSED_TO edges to relevant Statistic nodes."""
    b = _empty_builder()
    # Add minimal stat nodes so edges can connect
    b.graph.add_node("USA_policy_rate", node_type="Statistic")
    b.graph.add_node("SAU_crude_output", node_type="Statistic")
    b.add_sector_nodes()

    exposed = [
        (u, v) for u, v, d in b.graph.edges(data=True)
        if d.get("edge_type") == "EXPOSED_TO"
    ]
    assert len(exposed) > 0
