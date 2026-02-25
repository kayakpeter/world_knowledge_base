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


# ── Company node tests ────────────────────────────────────────────────────────

from pathlib import Path
import polars as pl
from knowledge_base.company_schema import CompanyNode


def test_add_company_nodes_from_dataframe():
    """add_company_nodes() adds Company nodes + edges to graph."""
    b = _empty_builder()
    b.add_sector_nodes()

    seed_df = pl.DataFrame({
        "ticker":               ["FFIE"],
        "name":                 ["Faraday Future"],
        "exchange":             ["NASDAQ"],
        "sector":               ["EV"],
        "country_iso3":         ["USA"],
        "employee_count":       [400],
        "location_count":       [2],
        "location_type":        ["multi_site"],
        "named_customers_count":[1],
        "has_shipped_product":  [False],
        "auditor":              ["Deloitte"],
        "auditor_tier":         ["big4"],
        "years_operating":      [8.0],
        "ceo_verifiable":       [True],
        "reality_score":        [0.45],
        "risk_tier":            ["MICRO_OP"],
    })
    b.add_company_nodes(seed_df)

    assert "FFIE" in b.graph
    assert b.graph.nodes["FFIE"]["node_type"] == "Company"
    assert b.graph.nodes["FFIE"]["risk_tier"] == "MICRO_OP"

    # LISTED_IN edge to United States sovereign
    listed_in = [
        v for u, v, d in b.graph.edges(data=True)
        if u == "FFIE" and d.get("edge_type") == "LISTED_IN"
    ]
    assert "United States" in listed_in

    # BELONGS_TO edge to EV sector
    belongs = [
        v for u, v, d in b.graph.edges(data=True)
        if u == "FFIE" and d.get("edge_type") == "BELONGS_TO"
    ]
    assert "EV" in belongs


def test_add_company_nodes_trading_profile():
    """Company node contains correct trading profile attributes."""
    b = _empty_builder()
    b.add_sector_nodes()

    seed_df = pl.DataFrame({
        "ticker":               ["GTII"],
        "name":                 ["Global Arena Holding"],
        "exchange":             ["OTC"],
        "sector":               ["Other"],
        "country_iso3":         ["USA"],
        "employee_count":       [0],
        "location_count":       [1],
        "location_type":        ["registered_agent"],
        "named_customers_count":[0],
        "has_shipped_product":  [False],
        "auditor":              ["unknown"],
        "auditor_tier":         ["unknown"],
        "years_operating":      [0.5],
        "ceo_verifiable":       [False],
        "reality_score":        [0.0],
        "risk_tier":            ["SHELL"],
    })
    b.add_company_nodes(seed_df)

    node = b.graph.nodes["GTII"]
    assert node["size_multiplier"] == 0.25
    assert node["target_pct"] == 5.0
    assert node["reversal_pct"] == 1.5
    assert node["trading_note"] == "Take the money and run"


# ── Trading profile query tests ───────────────────────────────────────────────

def _builder_with_company() -> KnowledgeGraphBuilder:
    b = _empty_builder()
    b.add_sector_nodes()
    seed_df = pl.DataFrame({
        "ticker":               ["GTII", "GFAI"],
        "name":                 ["Global Arena", "Guardforce AI"],
        "exchange":             ["OTC", "NASDAQ"],
        "sector":               ["Other", "Tech"],
        "country_iso3":         ["USA", "USA"],
        "employee_count":       [0, 900],
        "location_count":       [1, 8],
        "location_type":        ["registered_agent", "multi_site"],
        "named_customers_count":[0, 15],
        "has_shipped_product":  [False, True],
        "auditor":              ["unknown", "Marcum"],
        "auditor_tier":         ["unknown", "mid"],
        "years_operating":      [0.5, 8.0],
        "ceo_verifiable":       [False, True],
        "reality_score":        [0.0, 0.90],
        "risk_tier":            ["SHELL", "REAL"],
    })
    b.add_company_nodes(seed_df)
    return b


def test_get_company_trading_profile_known_shell():
    b = _builder_with_company()
    profile = b.get_company_trading_profile("GTII")
    assert profile["risk_tier"] == "SHELL"
    assert profile["size_multiplier"] == 0.25
    assert profile["target_pct"] == 5.0


def test_get_company_trading_profile_known_real():
    b = _builder_with_company()
    profile = b.get_company_trading_profile("GFAI")
    assert profile["risk_tier"] == "REAL"
    assert profile["size_multiplier"] == 1.0


def test_get_company_trading_profile_unknown_ticker():
    """Unknown ticker falls back to MICRO_OP defaults."""
    b = _builder_with_company()
    profile = b.get_company_trading_profile("UNKN")
    assert profile["risk_tier"] == "MICRO_OP"
    assert profile["size_multiplier"] == 0.75
