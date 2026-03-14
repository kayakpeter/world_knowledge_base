# strategist/tests/test_schema.py
import json
from strategist.schema import (
    ScenarioNode, AttemptModel, SectorImpact, ScenarioTree, NodeStatus
)


def test_attempt_model_cumulative():
    m = AttemptModel(p_attempt_72h=0.30, p_success_per_attempt=0.18, expected_attempts=3)
    # P(at least 1 success) = 1 - (1-0.18)^3
    expected = 1 - (1 - 0.18) ** 3
    assert abs(m.p_cumulative_success - expected) < 1e-6


def test_node_joint_probability():
    child = ScenarioNode(
        node_id="c1",
        description="Iran threatens UAE ports",
        branch_probability=0.70,
        parent_id="root",
        depth=1,
        parent_joint_probability=1.0,
    )
    assert abs(child.joint_probability - 0.70) < 1e-6


def test_node_joint_probability_chained():
    """joint_prob = parent_joint * branch_prob"""
    child = ScenarioNode(
        node_id="c2",
        description="Suez missile strike",
        branch_probability=0.40,
        parent_id="c1",
        depth=2,
        parent_joint_probability=0.70,
    )
    assert abs(child.joint_probability - 0.28) < 1e-6


def test_node_serialization_roundtrip():
    node = ScenarioNode(
        node_id="n1",
        description="Test event",
        branch_probability=0.5,
        parent_id=None,
        depth=0,
    )
    d = node.to_dict()
    restored = ScenarioNode.from_dict(d)
    assert restored.node_id == node.node_id
    assert restored.branch_probability == node.branch_probability
    assert restored.status == node.status
    assert restored.depth == node.depth


def test_sector_impact_roundtrip():
    si = SectorImpact(
        sector="energy",
        direction="UP",
        magnitude="CRITICAL",
        magnitude_pct=35.0,
        tickers=["FRO", "STNG"],
        notes="Brent spike",
    )
    d = si.to_dict()
    restored = SectorImpact.from_dict(d)
    assert restored.sector == "energy"
    assert restored.tickers == ["FRO", "STNG"]


def test_scenario_tree_node_count():
    tree = ScenarioTree(
        scenario_id="test-001",
        trigger_event="Test trigger",
        severity="HIGH",
        confirmed=True,
    )
    root = ScenarioNode(node_id="root", description="root", branch_probability=1.0,
                        parent_id=None, depth=0)
    tree.add_node(root)
    assert tree.node_count() == 1


def test_scenario_tree_add_child_updates_parent():
    tree = ScenarioTree(
        scenario_id="test-002",
        trigger_event="Test",
        severity="HIGH",
        confirmed=True,
    )
    root = ScenarioNode(node_id="root", description="root", branch_probability=1.0,
                        parent_id=None, depth=0)
    child = ScenarioNode(node_id="child1", description="child", branch_probability=0.5,
                         parent_id="root", depth=1)
    tree.add_node(root)
    tree.add_node(child)
    assert "child1" in tree.nodes["root"].children


def test_scenario_tree_tombstone():
    tree = ScenarioTree(scenario_id="t3", trigger_event="e", severity="LOW", confirmed=False)
    tree.add_tombstone("pruned1", "Low prob branch", 0.02, "below floor")
    assert len(tree.tombstones) == 1
    assert tree.tombstones[0]["node_id"] == "pruned1"


def test_scenario_tree_make_id():
    sid = ScenarioTree.make_id("US strikes Kharg Island")
    assert "us-strikes-kharg-island" in sid
    assert len(sid) > 10


def test_add_node_no_duplicate_children():
    tree = ScenarioTree(scenario_id="dup-test", trigger_event="e", severity="HIGH", confirmed=True)
    root = ScenarioNode(node_id="root", description="root", branch_probability=1.0, parent_id=None, depth=0)
    child = ScenarioNode(node_id="c1", description="c1", branch_probability=0.5, parent_id="root", depth=1)
    tree.add_node(root)
    tree.add_node(child)
    tree.add_node(child)  # duplicate — should not add again
    assert tree.nodes["root"].children.count("c1") == 1


def test_sector_impact_from_dict_ignores_extra_keys():
    d = {
        "sector": "energy",
        "direction": "UP",
        "magnitude": "HIGH",
        "magnitude_pct": 20.0,
        "tickers": ["FRO"],
        "notes": "test",
        "FUTURE_KEY": "should be ignored",
    }
    # Should not raise
    si = SectorImpact.from_dict(d)
    assert si.sector == "energy"
