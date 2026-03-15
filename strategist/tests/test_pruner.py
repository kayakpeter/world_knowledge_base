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


def test_leaf_at_main_bfs_max_depth():
    cfg = StrategistConfig()
    engine = PruningEngine(cfg)
    # main BFS: depth == max_depth (4) → LEAF_MAX_DEPTH (kept in tree, eligible for extension)
    node = _node("n4", 0.50, depth=4)
    decision, reason = engine.evaluate(node, severity="HIGH", current_node_count=5)
    assert decision == PruneDecision.LEAF_MAX_DEPTH
    assert "extension" in reason


def test_prune_beyond_extension_depth():
    from strategist.config import PruningConfig
    cfg = StrategistConfig(pruning=PruningConfig(max_depth=4, extension_depth=2))
    engine = PruningEngine(cfg)
    # extension pass: effective_max_depth = 6; depth 6 → PRUNE_DEPTH (hard stop)
    node = _node("n4b", 0.50, depth=6)
    decision, reason = engine.evaluate(
        node, severity="HIGH", current_node_count=5, effective_max_depth=6
    )
    assert decision == PruneDecision.PRUNE_DEPTH


def test_extension_pass_keep_within_limit():
    from strategist.config import PruningConfig
    cfg = StrategistConfig(pruning=PruningConfig(max_depth=4, extension_depth=2))
    engine = PruningEngine(cfg)
    # extension pass: effective_max_depth = 6; depth 5 → KEEP
    node = _node("n4c", 0.50, depth=5)
    decision, _ = engine.evaluate(
        node, severity="HIGH", current_node_count=5, effective_max_depth=6
    )
    assert decision == PruneDecision.KEEP


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
