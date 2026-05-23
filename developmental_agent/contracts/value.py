"""Value contract. DESIGN.md §9, §10, §11.

What flows across the drive -> actor seam: a scalar value field V(z, h) conditioned
on need state. Goal-proximity is just a special case (V(z) = -||z - z*||). The drive
system is deliberately LOW-volatility (it defines the agent's identity).

Key locked decisions encoded here:
  - reward == homeostatic drive-reduction: r = D(h_t) - D(h_{t+1})   (gating is a *consequence*)
  - value is a property of (state, need): V(z, h), never V(z) alone
  - per-drive modular critics composed here: V(z,h) = sum_d g_d(h) * V_d(z)
  - the objective/drive-weighting is held UNCERTAIN & externally authored (corrigibility)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .latent import LatentState


@dataclass(frozen=True)
class NeedState:
    """Interoceptive / homeostatic state `h` over NORMALIZED abstract need-variables
    (energy, integrity, ...). Embodiment adapters map real hardware onto these, so the
    drive function is embodiment-agnostic (§11 P3). Must be viability-coupled (§11 P1)."""
    needs: dict[str, float]


@runtime_checkable
class Drive(Protocol):
    """One drive. Reward is drive-reduction over its regulated need; the interoceptive
    gate g_d(h) and the auto-decay (for the epistemic/novelty drive) are properties of
    the drive itself, so combination needs no manual schedule (§10)."""

    name: str

    def discomfort(self, h: NeedState) -> float:
        """D(h) for this drive. Reward across a step is D(h_t) - D(h_{t+1})."""
        ...

    def gate(self, h: NeedState) -> float:
        """g_d(h): how much this drive weighs given current need state."""
        ...


@runtime_checkable
class ValueContract(Protocol):
    """Composed value field handed to the actor. DESIGN §10."""

    def value(self, state: LatentState, h: NeedState) -> float:
        """V(z, h) = sum_d g_d(h) * V_d(z). Maximized by the actor's MPC."""
        ...

    def reward(self, h_prev: NeedState, h: NeedState) -> float:
        """Realized reward = total homeostatic drive-reduction across a step.
        NOTE: this is computed on the EMBODIMENT side, never inside the agent (§11 P1).
        The agent's critic only *predicts* it."""
        ...
