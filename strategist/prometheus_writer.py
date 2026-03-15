# strategist/prometheus_writer.py
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from strategist.config import StrategistConfig
from strategist.schema import ScenarioTree, NodeStatus

logger = logging.getLogger(__name__)

# Maps sector name → Prometheus config key
SECTOR_TO_CONFIG_KEY: dict[str, str] = {
    "energy":      "energy_bonus",
    "tanker":      "shipping_bonus",
    "bulk":        "shipping_bonus",
    "agriculture": "agriculture_bonus",
    "container":   "shipping_bonus",
    "defense":     "defense_bonus",
    "currency":    "currency_bonus",
    "sovereign":   "sovereign_bonus",
    "insurance":   "insurance_bonus",
}

# Maps magnitude string → weight for weighted aggregation
MAGNITUDE_TO_WEIGHT: dict[str, float] = {
    "LOW":      0.25,
    "MODERATE": 0.50,
    "HIGH":     0.75,
    "CRITICAL": 1.00,
}


class PrometheusDeltaWriter:
    """
    Aggregates sector impacts from a ScenarioTree into a Prometheus config delta.

    Weighted aggregation:
        raw_score[sector] = sum over all KEPT nodes:
            node.joint_probability * MAGNITUDE_TO_WEIGHT[impact.magnitude]
            where impact.sector == sector

    Output config key:
        bonus[config_key] = 1.0 + raw_score[sector] * SECTOR_SCALE
    """

    SECTOR_SCALE = 2.0  # scales raw weighted score to a multiplier

    def __init__(self, config: StrategistConfig) -> None:
        self._cfg = config

    def write(self, tree: ScenarioTree) -> dict:
        """
        Compute delta, write to regime_config_dir/delta_{scenario_id}.json,
        return the delta dict.
        """
        raw_scores: dict[str, float] = {}

        for node in tree.nodes.values():
            if node.status == NodeStatus.PRUNED:
                continue  # skip pruned nodes (shouldn't be in tree.nodes, but guard anyway)
            weight_jp = node.joint_probability
            for impact in node.sector_impacts:
                config_key = SECTOR_TO_CONFIG_KEY.get(impact.sector)
                if config_key is None:
                    continue
                mag_weight = MAGNITUDE_TO_WEIGHT.get(impact.magnitude, 0.0)
                raw_scores[config_key] = raw_scores.get(config_key, 0.0) + weight_jp * mag_weight

        # Build delta: 1.0 + scaled raw score, rounded to 3 decimal places
        delta: dict = {
            "scenario_id": tree.scenario_id,
            "trigger_event": tree.trigger_event,
            "severity": tree.severity,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        for config_key, raw in raw_scores.items():
            delta[config_key] = round(1.0 + raw * self.SECTOR_SCALE, 3)

        # Write to disk
        out_dir = Path(self._cfg.regime_config_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"delta_{tree.scenario_id}.json"
        out_path.write_text(json.dumps(delta, indent=2))
        logger.info("Prometheus delta written: %s", out_path)

        return delta
