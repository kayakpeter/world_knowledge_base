# strategist/tests/test_runner.py
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from strategist.runner import ScenarioRunner
from strategist.schema import ScenarioNode, ScenarioTree, NodeStatus
from strategist.config import StrategistConfig


def _mock_expander_with_children(children_per_node: int = 2) -> MagicMock:
    """Expander that always returns `children_per_node` child branches."""
    expander = MagicMock()

    async def fake_expand(node, tree, *, severity):
        if node.depth >= 2:  # Stop at depth 2 to keep trees small
            return []
        return [
            ScenarioNode(
                node_id=f"{node.node_id}_c{i}",
                description=f"Child {i} of {node.node_id}",
                branch_probability=0.40,
                parent_id=node.node_id,
                depth=node.depth + 1,
                parent_joint_probability=node.joint_probability,
            )
            for i in range(children_per_node)
        ]

    expander.expand = fake_expand
    return expander


def _mock_infra_registry() -> MagicMock:
    reg = MagicMock()
    reg.update_status = MagicMock()
    return reg


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_runner_builds_tree():
    cfg = StrategistConfig()
    expander = _mock_expander_with_children(2)
    infra_reg = _mock_infra_registry()

    runner = ScenarioRunner(cfg, expander=expander, infra_registry=infra_reg)
    tree = _run(runner.run_async(
        trigger_event="US strikes Kharg Island",
        severity="CRITICAL",
        confirmed=True,
    ))

    assert isinstance(tree, ScenarioTree)
    # Root + 2 depth-1 children + 4 depth-2 children = 7 nodes
    assert tree.node_count() >= 3


def test_runner_prunes_low_probability():
    """Nodes with joint_prob below floor should be pruned (added to tombstones)."""
    cfg = StrategistConfig()
    expander = MagicMock()

    async def fake_expand(node, tree, *, severity):
        if node.depth >= 1:
            return []
        # Return one very-low-prob child that should be pruned (HIGH floor = 0.05)
        return [
            ScenarioNode(
                node_id=f"{node.node_id}_low",
                description="Low prob child",
                branch_probability=0.01,  # 0.01 < HIGH floor 0.05 → pruned
                parent_id=node.node_id,
                depth=node.depth + 1,
                parent_joint_probability=node.joint_probability,
            ),
            ScenarioNode(
                node_id=f"{node.node_id}_high",
                description="High prob child",
                branch_probability=0.80,  # Above floor → kept
                parent_id=node.node_id,
                depth=node.depth + 1,
                parent_joint_probability=node.joint_probability,
            ),
        ]

    expander.expand = fake_expand
    infra_reg = _mock_infra_registry()

    runner = ScenarioRunner(cfg, expander=expander, infra_registry=infra_reg)
    tree = _run(runner.run_async(
        trigger_event="Test",
        severity="HIGH",
        confirmed=True,
    ))

    # Low prob child should be in tombstones, not nodes
    assert any(t["node_id"].endswith("_low") for t in tree.tombstones)
    assert all(t["node_id"] != f for t in tree.tombstones
               for f in tree.nodes if f.endswith("_high"))


def test_runner_saves_checkpoint(tmp_path):
    """Runner should save tree JSON to scenarios_dir after completion."""
    cfg = StrategistConfig()
    cfg.scenarios_dir = tmp_path  # Override to temp dir

    expander = _mock_expander_with_children(1)
    infra_reg = _mock_infra_registry()

    runner = ScenarioRunner(cfg, expander=expander, infra_registry=infra_reg)
    tree = _run(runner.run_async(
        trigger_event="Test event",
        severity="HIGH",
        confirmed=True,
    ))

    # Check file was written
    saved_files = list(tmp_path.glob("active/*.json"))
    assert len(saved_files) == 1

    # Check it's valid JSON with the right scenario_id
    data = json.loads(saved_files[0].read_text())
    assert data["scenario_id"] == tree.scenario_id


def test_runner_applies_infra_effects():
    """When a node has infrastructure_effects, InfraStateRegistry.update_status() is called."""
    cfg = StrategistConfig()
    expander = MagicMock()

    async def fake_expand(node, tree, *, severity):
        if node.depth >= 1:
            return []
        child = ScenarioNode(
            node_id="child_with_infra",
            description="Iran closes Hormuz",
            branch_probability=0.70,
            parent_id=node.node_id,
            depth=1,
            parent_joint_probability=node.joint_probability,
            infrastructure_effects=[
                {"infra_id": "HORMUZ_STRAIT", "new_status": "CLOSED", "confidence": 0.9}
            ],
        )
        return [child]

    expander.expand = fake_expand
    infra_reg = _mock_infra_registry()

    runner = ScenarioRunner(cfg, expander=expander, infra_registry=infra_reg)
    _run(runner.run_async(trigger_event="Test", severity="HIGH", confirmed=True))

    # Should have called update_status for the HORMUZ_STRAIT effect
    infra_reg.update_status.assert_called()
    call_args_list = infra_reg.update_status.call_args_list
    infra_ids = [call[0][0] for call in call_args_list]
    assert "HORMUZ_STRAIT" in infra_ids
