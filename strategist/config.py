# strategist/config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PruningConfig:
    joint_probability_floor: float = 0.05
    max_nodes_per_scenario: int = 50
    tombstone_enabled: bool = True
    max_depth: int = 4  # main BFS stops here; leaves exist at depth max_depth-1
    max_branching_factor: int = 4
    # Deep extension: after main BFS, re-expand high-probability leaf nodes
    extension_enabled: bool = True
    extension_depth: int = 2          # extra levels beyond max_depth
    extension_probability_floor: float = 0.15  # only extend nodes above this joint probability


@dataclass
class SectorConfig:
    sector_names: list[str] = field(default_factory=lambda: [
        "energy",
        "tanker",
        "bulk",
        "agriculture",
        "container",
        "defense",
        "currency",
        "sovereign",
        "insurance",
    ])


@dataclass
class StrategistConfig:
    pruning: PruningConfig = field(default_factory=PruningConfig)
    sectors: SectorConfig = field(default_factory=SectorConfig)
    window_hours: int = 72
    ollama_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen2.5:32b"
    scenarios_dir: Path = field(
        default_factory=lambda: Path("/media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb/data/scenarios")
    )
    regime_config_dir: Path = field(
        default_factory=lambda: Path("/media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb/data/regime_config")
    )

    _SEVERITY_FLOORS: dict[str, float] = field(
        default_factory=lambda: {
            "CRITICAL": 0.03,
            "HIGH": 0.05,
            "MEDIUM": 0.10,
            "LOW": 0.20,
        },
        init=False,
        repr=False,
    )

    def pruning_floor_for_severity(self, severity: str) -> float:
        return self._SEVERITY_FLOORS.get(severity, self.pruning.joint_probability_floor)
