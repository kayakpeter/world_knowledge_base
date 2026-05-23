"""Latent-state & rollout contract. DESIGN.md §5, §6.

The single external contract of the co-trained encoder+forward-model block.
Swapping *within* the block (a new encoder/predictor from the field) is a
deliberate larger operation; everything else only sees this contract.

The latent `z` fuses the multi-rate observation streams (vision+proprioception
share `z`, which is WHY the skill-#1 visuomotor map needs no new module).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LatentState:
    """Encoder output. `z` is opaque (its geometry drifts during co-training — that
    is fine; the gate never uses raw latent distance as a competence claim, see §7).
    `reliability` is the per-region forward-model trust used to govern critic
    imagination horizon (§10) and as a gate signal (§7)."""
    z: object                    # array-like; opaque latent
    reliability: float | None = None


@runtime_checkable
class RolloutContract(Protocol):
    """What the world-model block exposes to the actor and the critic."""

    def predict(self, state: LatentState, action: object) -> LatentState:
        """Action-conditioned one-step prediction f(z, efference) -> z'. §6."""
        ...

    def rollout(self, state: LatentState, actions: object) -> list[LatentState]:
        """Multi-step imagined rollout for MPC (§6) and model-based credit
        assignment (§10). Callers MUST respect `reliability` / horizon governor."""
        ...
