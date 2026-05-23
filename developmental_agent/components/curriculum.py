"""Curriculum / morphology scheduler. DESIGN.md §4.

The developmental clock. Drives growth-from-t0 (the chosen "option b": two moving
targets) and gates which skills are exercised. Growth is BOTH a free curriculum
(small/weak body -> smaller control search space) AND the safety mechanism (low early
strength == safe babbling) AND a rehearsal for body transfer.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .embodiment import Embodiment


@runtime_checkable
class Curriculum(Protocol):
    def step_morphology(self, embodiment: Embodiment, developmental_time: float) -> None:
        """Advance growth. Slow relative to learning so it stays within-distribution for
        the online body estimator. A discrete large change == a synthetic body-swap
        (transfer-readiness rehearsal)."""
        ...

    def active_skill(self, developmental_time: float) -> str:
        """Which skill the curriculum is currently exercising (respects the dependency
        order: skill #0 must pass its gate before skill #1, etc.)."""
        ...
