import networkx as nx
import pytest
from processing.geo_financial_bridge import GeoFinancialBridge, GeoEdge

# Minimal country_codes override matching real graph node format: {iso3}_{stat}
_TEST_COUNTRY_CODES = {
    "United States": "USA",
    "Germany": "DEU",
    "Brazil": "BRA",
}


def make_graph():
    """Minimal graph with two stat nodes using real graph node ID format."""
    G = nx.DiGraph()
    G.add_node("USA_policy_rate", node_type="Statistic",
               country="United States", stat="policy_rate")
    G.add_node("DEU_yield_spread_10y3m", node_type="Statistic",
               country="Germany", stat="yield_spread_10y3m")
    return G


def test_inject_edge_into_graph():
    G = make_graph()
    bridge = GeoFinancialBridge()
    edges = [
        GeoEdge(
            source_country="United States",
            source_stat="policy_rate",
            target_country="Germany",
            target_stat="yield_spread_10y3m",
            weight=0.65,
            mechanism="Fed rate decision forces ECB response",
            event_title="Fed holds rates Q1 2026",
        )
    ]
    bridge.inject_into_graph(G, edges, country_codes=_TEST_COUNTRY_CODES)
    assert G.has_edge("USA_policy_rate", "DEU_yield_spread_10y3m")
    data = G["USA_policy_rate"]["DEU_yield_spread_10y3m"]
    assert abs(data["weight"] - 0.65) < 1e-6
    assert data["edge_type"] == "CONTAGION"
    assert data["source"] == "geopolitical"


def test_inject_skips_missing_nodes():
    G = make_graph()
    bridge = GeoFinancialBridge()
    edges = [
        GeoEdge(
            source_country="Brazil",     # not in graph
            source_stat="real_gdp_growth",
            target_country="Germany",
            target_stat="yield_spread_10y3m",
            weight=0.40,
            mechanism="commodity export",
            event_title="Brazil slowdown",
        )
    ]
    bridge.inject_into_graph(G, edges, country_codes=_TEST_COUNTRY_CODES)
    assert G.number_of_edges() == 0  # nothing was injected
