"""Policy / actor. DESIGN.md §6, §9.

Selects actions by planning through the differentiable forward model (MPC /
back-prop, V-JEPA-2-AC style). Also hosts:
  - the babbling source (prime mover before agency exists): temporally-correlated
    noise + motor synergies, modulated by learning-progress (-> goal babbling)
  - the visuomotor map: visual goal -> body-state target z*. NOT a new module —
    representation alignment inside the shared latent (vision+proprioception co-fused).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..contracts.latent import LatentState
from ..contracts.value import NeedState, ValueContract
from .world_model import ForwardModel


@runtime_checkable
class Actor(Protocol):
    def babble(self) -> object:
        """Spontaneous motor activity. Generates (action, sensory-change) pairs that
        feed the forward model so agency can be discovered (§6). Safe because early
        morphology is weak (§4)."""
        ...

    def reach_to_target(self, state: LatentState, z_star: LatentState,
                        world_model: ForwardModel) -> object:
        """Controllability via MPC through the forward model. Skill #0 validates this
        on arbitrary self-manifold targets; skill #1 supplies drive-selected targets."""
        ...

    def visual_goal_to_target(self, state: LatentState, visual_goal: object) -> LatentState:
        """Visuomotor map: a visually-specified goal -> a body-state target z*. §9."""
        ...

    def plan(self, state: LatentState, h: NeedState, value: ValueContract,
            world_model: ForwardModel) -> object:
        """Mode-2 planning: maximize predicted cumulative V(z,h) through the world model.
        The exploration/exploitation balance self-regulates via the drive gates (§10)."""
        ...
