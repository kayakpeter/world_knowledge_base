"""Skill #0 — agency discovery. DESIGN.md §6, §8.

The keystone: nothing above can be validated until this exists. Spans two boxes
(world-model block: forward model + self/world tagging; actor: babbling +
learning-progress modulation). The drive/value box stays DARK here.

Three nested acquisitions, in order:
  1. a forward model with above-chance predictive power exists
  2. self/world segmentation (self = predictable-from-efference)
  3. controllability (inverse / MPC to arbitrary self-manifold targets)

Handoff to skill #1: a forward model with a quantified controllability index, a
self/world tag over sensory dims, and a working MPC/inverse that hits arbitrary
self-manifold targets.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..gates.milestones import SKILL0_GATE


@dataclass(frozen=True)
class Skill0Spec:
    name: str = "agency_discovery"
    prime_mover: str = "spontaneous babbling (temporally-correlated + synergies, learning-progress modulated)"
    forward_model: str = "action-conditioned JEPA predictor f(z, efference) -> z'"
    self_model: str = "the body is whatever responds predictably to my efference"
    gate: tuple[str, ...] = SKILL0_GATE
    handoff: str = "controllability index + self/world tags + working inverse/MPC"


SPEC = Skill0Spec()
