# strategist/pruner.py
from __future__ import annotations
from enum import Enum

from strategist.config import StrategistConfig
from strategist.schema import ScenarioNode


class PruneDecision(str, Enum):
    KEEP             = "KEEP"
    PRUNE_PROBABILITY = "PRUNE_PROBABILITY"
    PRUNE_BUDGET     = "PRUNE_BUDGET"
    PRUNE_DEPTH      = "PRUNE_DEPTH"


class PruningEngine:
    """
    Decides whether a ScenarioNode should be expanded or pruned.

    Priority order:
    1. PRUNE_DEPTH   — node is at max depth
    2. PRUNE_BUDGET  — tree node budget exceeded
    3. PRUNE_PROBABILITY — joint probability below severity-adaptive floor
    4. KEEP
    """

    def __init__(self, config: StrategistConfig) -> None:
        self._cfg = config

    def evaluate(
        self,
        node: ScenarioNode,
        *,
        severity: str,
        current_node_count: int,
    ) -> tuple[PruneDecision, str]:
        """
        Returns (decision, reason_string).
        reason_string is human-readable for tombstone audit.

        current_node_count: number of nodes ALREADY in the tree (before adding this candidate).
        Budget fires when current_node_count > max_nodes_per_scenario, i.e., the tree is
        already full before this node is evaluated.
        """
        # 1. Depth check
        if node.depth >= self._cfg.pruning.max_depth:
            return (
                PruneDecision.PRUNE_DEPTH,
                f"depth={node.depth} >= max_depth={self._cfg.pruning.max_depth}",
            )

        # 2. Budget check
        if current_node_count > self._cfg.pruning.max_nodes_per_scenario:
            return (
                PruneDecision.PRUNE_BUDGET,
                f"node_count={current_node_count} > budget={self._cfg.pruning.max_nodes_per_scenario}",
            )

        # 3. Probability floor check
        floor = self._cfg.pruning_floor_for_severity(severity)
        joint_prob = node.joint_probability
        if joint_prob < floor:
            return (
                PruneDecision.PRUNE_PROBABILITY,
                f"joint_probability={joint_prob:.4f} < floor={floor:.4f} (severity={severity})",
            )

        return (PruneDecision.KEEP, "")
