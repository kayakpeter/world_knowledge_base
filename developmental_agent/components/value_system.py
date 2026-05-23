"""Drive / value system. DESIGN.md §10, §11.

Hosts the per-drive critics and composes them at the value contract. The mouth-value
terminal and novelty/learning-progress are both Drives; combination needs no schedule
(novelty auto-decays, homeostatic auto-gates).

The drive-weighting D is held UNCERTAIN and externally authored (corrigibility, §11):
the agent maintains a distribution over D and updates it ONLY from the privileged
correction channel, never treating its current D as ground truth.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..contracts.latent import LatentState
from ..contracts.value import Drive, NeedState, ValueContract
from .world_model import ForwardModel


@runtime_checkable
class Critic(Protocol):
    """Per-drive learned value V_d(z, h): expected future drive-reduction for drive d.

    Trained by TD bootstrapping, SEEDED by reflex-driven completions and accelerated by
    MODEL-BASED imagined rollouts (Dyna). Imagination horizon is GOVERNED by the world
    model's validated reliability to prevent model exploitation (§10)."""

    drive: Drive

    def value(self, state: LatentState, h: NeedState) -> float: ...

    def td_update(self, transition: object, world_model: ForwardModel) -> None:
        """Bootstrap; horizon bounded by world_model.region_reliability()."""
        ...


@runtime_checkable
class ValueSystem(ValueContract, Protocol):
    """Composes critics: V(z,h) = sum_d g_d(h) * V_d(z). Adding a drive == registering a
    new Critic (drop-in; no retraining of others)."""

    def register_drive(self, critic: Critic) -> None: ...

    def objective_uncertainty(self) -> object:
        """Current distribution over the drive-weighting D. Reduced only by the
        privileged correction channel (§11 P2)."""
        ...
