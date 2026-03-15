# strategist/tests/test_prometheus_writer.py
import json
import tempfile
from pathlib import Path

from strategist.prometheus_writer import PrometheusDeltaWriter
from strategist.schema import ScenarioTree, ScenarioNode, SectorImpact, NodeStatus
from strategist.config import StrategistConfig


def _make_tree_with_impacts() -> ScenarioTree:
    """Build a small tree with known sector impacts for deterministic testing."""
    tree = ScenarioTree(
        scenario_id="test-001",
        trigger_event="US strikes Kharg Island",
        severity="CRITICAL",
        confirmed=True,
    )
    root = ScenarioNode(
        node_id="root",
        description="US strikes Kharg Island",
        branch_probability=1.0,
        parent_id=None,
        depth=0,
        sector_impacts=[
            SectorImpact(sector="energy", direction="UP", magnitude="CRITICAL",
                         magnitude_pct=40.0, tickers=["FRO"]),
        ],
    )
    child1 = ScenarioNode(
        node_id="c1",
        description="Iran closes Hormuz",
        branch_probability=0.65,
        parent_id="root",
        depth=1,
        parent_joint_probability=1.0,
        sector_impacts=[
            SectorImpact(sector="tanker", direction="UP", magnitude="HIGH",
                         magnitude_pct=25.0, tickers=["STNG", "FRO"]),
            SectorImpact(sector="energy", direction="UP", magnitude="HIGH",
                         magnitude_pct=20.0, tickers=["XOM"]),
        ],
        status=NodeStatus.EXPANDED,
    )
    child2 = ScenarioNode(
        node_id="c2",
        description="Iran threatens Abqaiq",
        branch_probability=0.40,
        parent_id="root",
        depth=1,
        parent_joint_probability=1.0,
        sector_impacts=[
            SectorImpact(sector="energy", direction="UP", magnitude="MODERATE",
                         magnitude_pct=10.0, tickers=[]),
        ],
        status=NodeStatus.EXPANDED,
    )
    tree.add_node(root)
    tree.add_node(child1)
    tree.add_node(child2)
    return tree


def test_write_creates_file(tmp_path):
    cfg = StrategistConfig()
    cfg.regime_config_dir = tmp_path

    writer = PrometheusDeltaWriter(cfg)
    tree = _make_tree_with_impacts()
    writer.write(tree)

    files = list(tmp_path.glob("delta_*.json"))
    assert len(files) == 1


def test_energy_bonus_present(tmp_path):
    cfg = StrategistConfig()
    cfg.regime_config_dir = tmp_path

    writer = PrometheusDeltaWriter(cfg)
    tree = _make_tree_with_impacts()
    delta = writer.write(tree)

    assert "energy_bonus" in delta
    assert delta["energy_bonus"] > 1.0  # must be > 1 (multiplier)


def test_tanker_bonus_from_high_impact(tmp_path):
    cfg = StrategistConfig()
    cfg.regime_config_dir = tmp_path

    writer = PrometheusDeltaWriter(cfg)
    tree = _make_tree_with_impacts()
    delta = writer.write(tree)

    # c1 has HIGH tanker impact at joint_prob=0.65
    assert "shipping_bonus" in delta
    assert delta["shipping_bonus"] > 1.0


def test_delta_json_valid(tmp_path):
    cfg = StrategistConfig()
    cfg.regime_config_dir = tmp_path

    writer = PrometheusDeltaWriter(cfg)
    tree = _make_tree_with_impacts()
    writer.write(tree)

    files = list(tmp_path.glob("delta_*.json"))
    data = json.loads(files[0].read_text())
    assert "scenario_id" in data
    assert "energy_bonus" in data
    assert "generated_at" in data


def test_no_impacts_produces_neutral_delta(tmp_path):
    cfg = StrategistConfig()
    cfg.regime_config_dir = tmp_path

    writer = PrometheusDeltaWriter(cfg)
    tree = ScenarioTree(scenario_id="empty-001", trigger_event="No impacts", severity="LOW", confirmed=False)
    root = ScenarioNode(node_id="root", description="root", branch_probability=1.0, parent_id=None, depth=0)
    tree.add_node(root)

    delta = writer.write(tree)
    # No impacts — all bonuses should be at or near 1.0 (neutral)
    assert delta.get("energy_bonus", 1.0) == 1.0
    assert delta.get("shipping_bonus", 1.0) == 1.0
