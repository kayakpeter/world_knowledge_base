# strategist/tests/test_config.py
from strategist.config import StrategistConfig, PruningConfig, SectorConfig


def test_default_config():
    cfg = StrategistConfig()
    assert cfg.pruning.joint_probability_floor == 0.05
    assert cfg.pruning.max_nodes_per_scenario == 50
    assert cfg.window_hours == 72
    assert "energy" in cfg.sectors.sector_names


def test_severity_thresholds():
    cfg = StrategistConfig()
    assert cfg.pruning_floor_for_severity("CRITICAL") == 0.03
    assert cfg.pruning_floor_for_severity("HIGH") == 0.05
    assert cfg.pruning_floor_for_severity("MEDIUM") == 0.10


def test_unknown_severity_returns_default():
    cfg = StrategistConfig()
    assert cfg.pruning_floor_for_severity("BOGUS") == cfg.pruning.joint_probability_floor


def test_sector_names_complete():
    cfg = StrategistConfig()
    for s in ["energy", "tanker", "bulk", "agriculture", "container", "defense", "currency", "sovereign", "insurance"]:
        assert s in cfg.sectors.sector_names
