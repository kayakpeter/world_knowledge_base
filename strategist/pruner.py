# strategist/pruner.py
from __future__ import annotations
from enum import Enum

from strategist.config import StrategistConfig
from strategist.schema import ScenarioNode


class PruneDecision(str, Enum):
    KEEP              = "KEEP"
    PRUNE_PROBABILITY = "PRUNE_PROBABILITY"
    PRUNE_BUDGET      = "PRUNE_BUDGET"
    PRUNE_DEPTH       = "PRUNE_DEPTH"     # beyond extension limit — hard stop, not added to tree
    LEAF_MAX_DEPTH    = "LEAF_MAX_DEPTH"  # at main BFS depth cap — added to tree, eligible for extension


class PruningEngine:
    """
    Decides whether a ScenarioNode should be expanded or pruned.

    Priority order:
    1. LEAF_MAX_DEPTH — node is at main BFS depth cap (kept in tree, not expanded further)
    2. PRUNE_DEPTH    — node is beyond extension depth limit (hard stop, not added)
    3. PRUNE_BUDGET   — tree node budget exceeded
    4. PRUNE_PROBABILITY — joint probability below severity-adaptive floor
    5. KEEP

    The distinction between LEAF_MAX_DEPTH and PRUNE_DEPTH enables the extension pass:
    after the main BFS, nodes tagged LEAF_MAX_DEPTH with joint_probability above
    extension_probability_floor are re-queued for deeper expansion.
    """

    def __init__(self, config: StrategistConfig) -> None:
        self._cfg = config

    def evaluate(
        self,
        node: ScenarioNode,
        *,
        severity: str,
        current_node_count: int,
        effective_max_depth: int | None = None,
    ) -> tuple[PruneDecision, str]:
        """
        Returns (decision, reason_string).

        effective_max_depth: override for the extension pass (max_depth + extension_depth).
            If None, uses config.pruning.max_depth for the main BFS.

        LEAF_MAX_DEPTH: node is at main BFS cap — add to tree, do not enqueue.
        PRUNE_DEPTH:    node is beyond effective limit — discard entirely.
        """
        main_depth = self._cfg.pruning.max_depth
        ext_depth  = main_depth + self._cfg.pruning.extension_depth
        limit      = effective_max_depth if effective_max_depth is not None else main_depth

        # 1. Depth check
        if node.depth >= limit:
            if effective_max_depth is None and node.depth == main_depth:
                # At main BFS cap — keep in tree but do not expand further
                return (
                    PruneDecision.LEAF_MAX_DEPTH,
                    f"depth={node.depth} == max_depth={main_depth} — leaf, eligible for extension",
                )
            return (
                PruneDecision.PRUNE_DEPTH,
                f"depth={node.depth} >= limit={limit}",
            )

        # 2. Budget check
        if current_node_count >= self._cfg.pruning.max_nodes_per_scenario:
            return (
                PruneDecision.PRUNE_BUDGET,
                f"node_count={current_node_count} >= budget={self._cfg.pruning.max_nodes_per_scenario}",
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
