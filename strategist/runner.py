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
    BFS-based async scenario expansion with optional deep extension pass.

    Main BFS:
    1. Pop all nodes at current level
    2. For each node, call expander.expand() to get child candidates
    3. For each candidate, call pruner.evaluate():
       - KEEP          → add to tree, enqueue for next level
       - LEAF_MAX_DEPTH → add to tree, do NOT enqueue (but eligible for extension)
       - PRUNE_*       → add tombstone, discard
    4. Apply infrastructure_effects from kept/leaf nodes to InfraStateRegistry
    5. Checkpoint after each level

    Extension pass (if extension_enabled):
    After main BFS, collect all LEAF_MAX_DEPTH nodes with joint_probability >=
    extension_probability_floor. Re-run BFS from those nodes with effective_max_depth
    = max_depth + extension_depth.
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

        # ── Main BFS ────────────────────────────────────────────────────────
        leaf_max_depth_nodes: list[ScenarioNode] = []
        await self._bfs(
            tree=tree,
            seed_nodes=[root],
            severity=severity,
            effective_max_depth=None,  # use config.pruning.max_depth
            leaf_collector=leaf_max_depth_nodes,
        )

        # ── Extension pass ───────────────────────────────────────────────────
        p = self._cfg.pruning
        if p.extension_enabled and leaf_max_depth_nodes:
            ext_depth   = p.max_depth + p.extension_depth
            ext_floor   = p.extension_probability_floor
            candidates  = [
                n for n in leaf_max_depth_nodes
                if n.joint_probability >= ext_floor
            ]
            logger.info(
                "[%s] Extension pass: %d/%d LEAF_MAX_DEPTH nodes qualify "
                "(joint_prob >= %.2f), extending to depth %d",
                scenario_id, len(candidates), len(leaf_max_depth_nodes),
                ext_floor, ext_depth,
            )
            if candidates:
                await self._bfs(
                    tree=tree,
                    seed_nodes=candidates,
                    severity=severity,
                    effective_max_depth=ext_depth,
                    leaf_collector=None,  # no further extension
                )
        elif p.extension_enabled:
            logger.info("[%s] Extension pass: no LEAF_MAX_DEPTH nodes to extend", scenario_id)

        tree.status = "complete"
        tree.save(self._cfg.scenarios_dir)
        return tree

    async def _bfs(
        self,
        *,
        tree: ScenarioTree,
        seed_nodes: list[ScenarioNode],
        severity: str,
        effective_max_depth: int | None,
        leaf_collector: list[ScenarioNode] | None,
    ) -> None:
        """
        Generic BFS loop. Expands seed_nodes and their descendants.

        effective_max_depth: passed to pruner.evaluate(); None = use config.pruning.max_depth.
        leaf_collector: if provided, LEAF_MAX_DEPTH nodes are appended here.
        """
        queue: deque[ScenarioNode] = deque(seed_nodes)

        while queue:
            level_nodes = list(queue)
            queue.clear()

            for node in level_nodes:
                candidates = await self._expander.expand(node, tree, severity=severity)

                for candidate in candidates:
                    decision, reason = self._pruner.evaluate(
                        candidate,
                        severity=severity,
                        current_node_count=tree.node_count(),
                        effective_max_depth=effective_max_depth,
                    )

                    if decision == PruneDecision.KEEP:
                        candidate.status = NodeStatus.PENDING
                        tree.add_node(candidate)
                        queue.append(candidate)
                        self._apply_infra_effects(candidate)

                    elif decision == PruneDecision.LEAF_MAX_DEPTH:
                        candidate.status = NodeStatus.PENDING
                        tree.add_node(candidate)
                        self._apply_infra_effects(candidate)
                        if leaf_collector is not None:
                            leaf_collector.append(candidate)

                    else:  # PRUNE_PROBABILITY, PRUNE_BUDGET, PRUNE_DEPTH
                        candidate.status = NodeStatus.PRUNED
                        candidate.pruned_reason = reason
                        tree.add_tombstone(
                            candidate.node_id,
                            candidate.description,
                            candidate.joint_probability,
                            reason,
                        )

                node.status = NodeStatus.EXPANDED

            tree.save(self._cfg.scenarios_dir)
            logger.info(
                "[%s] Level complete — nodes=%d, tombstones=%d",
                tree.scenario_id, tree.node_count(), len(tree.tombstones),
            )

    def _apply_infra_effects(self, node: ScenarioNode) -> None:
        """Apply infrastructure_effects from a kept node to the InfraStateRegistry."""
        if self._infra_reg is None:
            return
        for effect in node.infrastructure_effects:
            infra_id       = effect.get("infra_id")
            new_status_str = effect.get("new_status")
            confidence     = float(effect.get("confidence", 0.8))
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
