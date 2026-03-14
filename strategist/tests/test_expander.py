# strategist/tests/test_expander.py
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from strategist.expander import ScenarioExpander
from strategist.schema import ScenarioNode, ScenarioTree
from strategist.config import StrategistConfig


def _make_llm_response(branches: list[dict]) -> str:
    """Build a JSON string the expander would parse from Ollama."""
    return json.dumps({"branches": branches})


def _make_expander(llm_response: str) -> ScenarioExpander:
    cfg = StrategistConfig()
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value=MagicMock(content=llm_response))
    return ScenarioExpander(cfg, llm_provider=mock_provider)


def test_expand_returns_child_nodes():
    expander = _make_expander(_make_llm_response([
        {
            "description": "Iran retaliates by closing Hormuz",
            "probability": 0.65,
            "sector_impacts": [{"sector": "energy", "direction": "UP", "magnitude": "CRITICAL", "magnitude_pct": 40.0, "tickers": ["FRO"], "notes": ""}],
            "infrastructure_effects": [{"infra_id": "HORMUZ_STRAIT", "new_status": "CLOSED", "confidence": 0.9}],
            "time_offset_hours": 6.0,
        },
        {
            "description": "Iran threatens Abqaiq strike",
            "probability": 0.40,
            "sector_impacts": [],
            "infrastructure_effects": [],
            "time_offset_hours": 12.0,
        },
    ]))

    tree = ScenarioTree(scenario_id="t1", trigger_event="US strikes Kharg Island", severity="CRITICAL", confirmed=True)
    root = ScenarioNode(node_id="root", description="US strikes Kharg Island", branch_probability=1.0, parent_id=None, depth=0)
    tree.add_node(root)

    children = asyncio.get_event_loop().run_until_complete(
        expander.expand(root, tree, severity="CRITICAL")
    )

    assert len(children) == 2
    assert all(isinstance(c, ScenarioNode) for c in children)
    assert children[0].branch_probability == 0.65
    assert children[1].branch_probability == 0.40


def test_expand_normalizes_probabilities_over_one():
    """If LLM returns branch probs summing > 1.0, they get scaled down."""
    expander = _make_expander(_make_llm_response([
        {"description": "A", "probability": 0.80, "sector_impacts": [], "infrastructure_effects": [], "time_offset_hours": 0},
        {"description": "B", "probability": 0.80, "sector_impacts": [], "infrastructure_effects": [], "time_offset_hours": 0},
    ]))

    tree = ScenarioTree(scenario_id="t2", trigger_event="Test", severity="HIGH", confirmed=True)
    root = ScenarioNode(node_id="root", description="root", branch_probability=1.0, parent_id=None, depth=0)
    tree.add_node(root)

    children = asyncio.get_event_loop().run_until_complete(
        expander.expand(root, tree, severity="HIGH")
    )

    total = sum(c.branch_probability for c in children)
    assert total <= 1.0 + 1e-6  # normalized


def test_expand_sets_parent_joint_probability():
    """Children inherit the parent's joint_probability as their parent_joint_probability."""
    expander = _make_expander(_make_llm_response([
        {"description": "Child", "probability": 0.50, "sector_impacts": [], "infrastructure_effects": [], "time_offset_hours": 0},
    ]))

    tree = ScenarioTree(scenario_id="t3", trigger_event="Test", severity="HIGH", confirmed=True)
    root = ScenarioNode(node_id="root", description="root", branch_probability=0.80, parent_id=None, depth=0, parent_joint_probability=1.0)
    tree.add_node(root)

    children = asyncio.get_event_loop().run_until_complete(
        expander.expand(root, tree, severity="HIGH")
    )

    # child.parent_joint_probability == root.joint_probability == 0.80
    assert abs(children[0].parent_joint_probability - 0.80) < 1e-9


def test_expand_handles_malformed_llm_response():
    """If LLM returns non-JSON, expander returns empty list (graceful degradation)."""
    cfg = StrategistConfig()
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value=MagicMock(content="not json at all"))
    expander = ScenarioExpander(cfg, llm_provider=mock_provider)

    tree = ScenarioTree(scenario_id="t4", trigger_event="Test", severity="HIGH", confirmed=True)
    root = ScenarioNode(node_id="root", description="root", branch_probability=1.0, parent_id=None, depth=0)
    tree.add_node(root)

    children = asyncio.get_event_loop().run_until_complete(
        expander.expand(root, tree, severity="HIGH")
    )
    assert children == []


def test_expand_sets_depth():
    expander = _make_expander(_make_llm_response([
        {"description": "Child", "probability": 0.50, "sector_impacts": [], "infrastructure_effects": [], "time_offset_hours": 0},
    ]))

    tree = ScenarioTree(scenario_id="t5", trigger_event="Test", severity="HIGH", confirmed=True)
    root = ScenarioNode(node_id="root", description="root", branch_probability=1.0, parent_id=None, depth=2)
    tree.add_node(root)

    children = asyncio.get_event_loop().run_until_complete(
        expander.expand(root, tree, severity="HIGH")
    )
    assert children[0].depth == 3  # parent depth + 1


def test_expand_all_zero_probabilities_returns_empty():
    """If all branch probabilities are 0, return [] (graceful degradation)."""
    expander = _make_expander(_make_llm_response([
        {"description": "A", "probability": 0.0, "sector_impacts": [], "infrastructure_effects": [], "time_offset_hours": 0},
        {"description": "B", "probability": 0.0, "sector_impacts": [], "infrastructure_effects": [], "time_offset_hours": 0},
    ]))

    tree = ScenarioTree(scenario_id="t6", trigger_event="Test", severity="HIGH", confirmed=True)
    root = ScenarioNode(node_id="root", description="root", branch_probability=1.0, parent_id=None, depth=0)
    tree.add_node(root)

    children = asyncio.get_event_loop().run_until_complete(
        expander.expand(root, tree, severity="HIGH")
    )
    assert children == []


def test_expand_filters_invalid_infra_effects():
    """Infrastructure effects with invalid new_status are silently dropped."""
    expander = _make_expander(_make_llm_response([
        {
            "description": "Branch with bad infra",
            "probability": 0.50,
            "sector_impacts": [],
            "infrastructure_effects": [
                {"infra_id": "HORMUZ_STRAIT", "new_status": "CLOSED", "confidence": 0.9},
                {"infra_id": "UNKNOWN_NODE", "new_status": "INVALID_STATUS", "confidence": 0.5},
            ],
            "time_offset_hours": 0,
        },
    ]))

    tree = ScenarioTree(scenario_id="t7", trigger_event="Test", severity="HIGH", confirmed=True)
    root = ScenarioNode(node_id="root", description="root", branch_probability=1.0, parent_id=None, depth=0)
    tree.add_node(root)

    children = asyncio.get_event_loop().run_until_complete(
        expander.expand(root, tree, severity="HIGH")
    )
    assert len(children) == 1
    # Only the valid CLOSED effect should remain
    assert len(children[0].infrastructure_effects) == 1
    assert children[0].infrastructure_effects[0]["new_status"] == "CLOSED"
