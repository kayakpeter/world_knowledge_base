"""Encoder + forward model — the single co-trained block. DESIGN.md §5, §6.

Exposes exactly one external contract (RolloutContract over LatentState). Swapping
within (a new JEPA-class encoder/predictor) is a deliberate larger operation.

Locked decisions:
  - action-conditioned JEPA predictor f(z, efference) -> z'
  - encoder fuses multi-rate streams into a shared z (vision+proprioception together)
  - anti-collapse lives HERE: EMA target encoder + stop-gradient
  - self/world tag = prediction error conditioned on efference (the self-model)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..contracts.latent import LatentState, RolloutContract


@runtime_checkable
class Encoder(Protocol):
    def encode(self, observation: dict) -> LatentState:
        """Fuse multi-rate `o` (+ efference copy) into the shared latent `z`."""
        ...

    def ema_target(self, observation: dict) -> LatentState:
        """Slow EMA target representation. Anti-collapse + the L0 training target /
        stable reference for the gate (§7)."""
        ...


@runtime_checkable
class ForwardModel(RolloutContract, Protocol):
    """Action-conditioned predictor. Inherits predict()/rollout() from RolloutContract."""

    def self_world_tag(self, observation: dict, efference: object) -> dict:
        """Per-sensory-dimension self(predictable-from-efference) vs world(residual).
        Operational self-model: 'the body is whatever responds predictably to my
        efference.' Passive motion -> tagged world-caused (still training data). §6."""
        ...

    def region_reliability(self, state: LatentState) -> float:
        """Validated predictive reliability near `state` — governs critic imagination
        horizon (§10) and feeds the gate (§7)."""
        ...
