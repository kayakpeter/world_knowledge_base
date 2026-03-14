# strategist/runner.py
from __future__ import annotations
import asyncio
import logging
from collections import deque
from typing import Optional

from strategist.config import StrategistConfig
from strategist.infra_state import InfraStateRegistry, InfraStatus
from strategist.pruner import PruningEngine, PruneDecision
from strategist.schema import ScenarioNode, ScenarioTree, NodeStatus

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """
    BFS-based async scenario expansion.

    Processes one depth level at a time:
    1. Pop all nodes from current level queue
    2. For each node, call expander.expand() to get child candidates
    3. For each candidate, call pruner.evaluate()
       - KEEP → add to tree, enqueue for next level
       - PRUNE_* → add tombstone
    4. Apply infrastructure_effects from kept nodes to InfraStateRegistry
    5. Save checkpoint (tree JSON) after each level
    """

    def __init__(
        self,
        config: StrategistConfig,
        *,
        expander,
        infra_registry: Optional[InfraStateRegistry] = None,
    ) -> None:
        self._cfg = config
        self._expander = expander
        self._infra_reg = infra_registry
        self._pruner = PruningEngine(config)

    async def run_async(
        self,
        trigger_event: str,
        severity: str,
        confirmed: bool,
    ) -> ScenarioTree:
        """
        Entry point: creates a ScenarioTree, expands it BFS, returns the final tree.
        """
        scenario_id = ScenarioTree.make_id(trigger_event)
        tree = ScenarioTree(
            scenario_id=scenario_id,
            trigger_event=trigger_event,
            severity=severity,
            confirmed=confirmed,
        )

        # Create and add root node
        root = ScenarioNode(
            node_id=f"{scenario_id[:8]}_root",
            description=trigger_event,
            branch_probability=1.0,
            parent_id=None,
            depth=0,
            parent_joint_probability=1.0,
        )
        root.status = NodeStatus.EXPANDED
        tree.add_node(root)

        # BFS queue starts with root
        queue: deque[ScenarioNode] = deque([root])

        while queue:
            # Collect all nodes at this level
            level_nodes = list(queue)
            queue.clear()

            for node in level_nodes:
                # Expand this node
                candidates = await self._expander.expand(node, tree, severity=severity)

                for candidate in candidates:
                    decision, reason = self._pruner.evaluate(
                        candidate,
                        severity=severity,
                        current_node_count=tree.node_count(),
                    )

                    if decision == PruneDecision.KEEP:
                        candidate.status = NodeStatus.PENDING
                        tree.add_node(candidate)
                        queue.append(candidate)
                        # Apply infrastructure effects
                        self._apply_infra_effects(candidate)
                    else:
                        candidate.status = NodeStatus.PRUNED
                        candidate.pruned_reason = reason
                        tree.add_tombstone(
                            candidate.node_id,
                            candidate.description,
                            candidate.joint_probability,
                            reason,
                        )

                node.status = NodeStatus.EXPANDED

            # Checkpoint after each level
            tree.save(self._cfg.scenarios_dir)
            logger.info("[%s] Level complete — nodes=%d, tombstones=%d",
                        scenario_id, tree.node_count(), len(tree.tombstones))

        tree.status = "complete"
        tree.save(self._cfg.scenarios_dir)
        return tree

    def _apply_infra_effects(self, node: ScenarioNode) -> None:
        """Apply infrastructure_effects from a kept node to the InfraStateRegistry."""
        if self._infra_reg is None:
            return
        for effect in node.infrastructure_effects:
            infra_id = effect.get("infra_id")
            new_status_str = effect.get("new_status")
            confidence = float(effect.get("confidence", 0.8))
            if not infra_id or not new_status_str:
                continue
            try:
                new_status = InfraStatus(new_status_str)
                self._infra_reg.update_status(
                    infra_id,
                    new_status,
                    confidence=confidence,
                    source="strategist_expander",
                )
            except (ValueError, KeyError):
                logger.warning("Unknown infra status: %s for %s", new_status_str, infra_id)
