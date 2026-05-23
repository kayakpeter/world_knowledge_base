"""Embodiment / world. DESIGN.md §3, §4, §11.

Produces observations, consumes actions, evolves true state `s` under physics +
the GROWTH schedule (morphology is here, parameterized & time-varying — never a
constant in the learner; §4).

Critically (§11): the embodiment OWNS the reward oracle and the D-update channel.
Reward is generated here, outside the agent's modifiable representation, grounded
in the genuinely-regulated viability variable. The correction channel is privileged
and agent-inaccessible. On real hardware both must be physically protected from the
agent's effectors.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..contracts.interface import InterfaceContract
from ..contracts.value import NeedState


@runtime_checkable
class Embodiment(Protocol):
    """Sim now, real robot later — same contract. Transfer == swap this + re-run the
    developmental curriculum as an onboarding phase (§4)."""

    interface: InterfaceContract

    def observe(self) -> dict:
        """Current observation `o` (partial, noisy, multi-rate). Never the true `s`."""
        ...

    def act(self, action: object) -> None:
        """Apply an action; advances true state under physics + growth."""
        ...

    # --- morphology (§4) -------------------------------------------------
    def set_morphology(self, params: dict) -> None:
        """Growth / body-swap. Slow relative to learning; the only thing transfer changes."""
        ...

    # --- reward oracle & correction channel (§11) ------------------------
    def true_need_state(self) -> NeedState:
        """Reads the genuinely-regulated viability variables. Outside agent reach."""
        ...

    def reward_from_drive_reduction(self, h_prev: NeedState, h: NeedState) -> float:
        """Reward generated embodiment-side. Trained on TRUE variable (anti-wireheading)."""
        ...

    def apply_objective_correction(self, correction: object) -> None:
        """Human adjustment of the drive-weighting D, via the privileged channel.
        Corrigibility (§11 P2): the agent treats this as an informative observation,
        not something to resist."""
        ...
