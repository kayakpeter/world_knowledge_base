# strategist/tests/test_pruner.py
from strategist.pruner import PruningEngine, PruneDecision
from strategist.config import StrategistConfig
from strategist.schema import ScenarioNode, NodeStatus


def _node(node_id, joint_prob, depth=1, status=NodeStatus.PENDING):
    """Helper: creates node where joint_probability == joint_prob (parent_joint=1.0 * branch=joint_prob)."""
    return ScenarioNode(
        node_id=node_id,
        description="test",
        branch_probability=joint_prob,
        parent_id="root",
        depth=depth,
        parent_joint_probability=1.0,
    )


def test_keep_above_floor():
    cfg = StrategistConfig()
    engine = PruningEngine(cfg)
    node = _node("n1", 0.10)
    decision, reason = engine.evaluate(node, severity="HIGH", current_node_count=5)
    assert decision == PruneDecision.KEEP


def test_prune_below_floor():
    cfg = StrategistConfig()
    engine = PruningEngine(cfg)
    # HIGH floor = 0.05; joint_prob = 0.04 → should prune
    node = _node("n2", 0.04)
    decision, reason = engine.evaluate(node, severity="HIGH", current_node_count=5)
    assert decision == PruneDecision.PRUNE_PROBABILITY
    assert "0.04" in reason or "probability" in reason.lower()


def test_exactly_at_floor_is_kept():
    cfg = StrategistConfig()
    engine = PruningEngine(cfg)
    # HIGH floor = 0.05; exactly at floor → keep (inclusive)
    node = _node("n3", 0.05)
    decision, reason = engine.evaluate(node, severity="HIGH", current_node_count=5)
    assert decision == PruneDecision.KEEP


def test_prune_at_max_depth():
    cfg = StrategistConfig()
    engine = PruningEngine(cfg)
    # max_depth = 4; depth 4 is leaf (no further expansion)
    node = _node("n4", 0.50, depth=4)
    decision, reason = engine.evaluate(node, severity="HIGH", current_node_count=5)
    assert decision == PruneDecision.PRUNE_DEPTH


def test_prune_budget_exceeded():
    cfg = StrategistConfig()
    engine = PruningEngine(cfg)
    node = _node("n5", 0.50, depth=1)
    # max_nodes = 50; pass current_node_count = 51
    decision, reason = engine.evaluate(node, severity="HIGH", current_node_count=51)
    assert decision == PruneDecision.PRUNE_BUDGET


def test_prune_budget_exactly_at_limit():
    cfg = StrategistConfig()
    engine = PruningEngine(cfg)
    node = _node("n_budget", 0.50, depth=1)
    # Exactly at max_nodes — should now prune (strict ceiling)
    decision, reason = engine.evaluate(node, severity="HIGH", current_node_count=50)
    assert decision == PruneDecision.PRUNE_BUDGET


def test_severity_critical_has_lower_floor():
    cfg = StrategistConfig()
    engine = PruningEngine(cfg)
    # CRITICAL floor = 0.03; joint=0.04 should KEEP for CRITICAL but PRUNE for HIGH
    node = _node("n6", 0.04)
    decision_critical, _ = engine.evaluate(node, severity="CRITICAL", current_node_count=5)
    decision_high, _ = engine.evaluate(node, severity="HIGH", current_node_count=5)
    assert decision_critical == PruneDecision.KEEP
    assert decision_high == PruneDecision.PRUNE_PROBABILITY


def test_severity_medium_higher_floor():
    cfg = StrategistConfig()
    engine = PruningEngine(cfg)
    # MEDIUM floor = 0.10; joint=0.08 should PRUNE
    node = _node("n7", 0.08)
    decision, reason = engine.evaluate(node, severity="MEDIUM", current_node_count=5)
    assert decision == PruneDecision.PRUNE_PROBABILITY
