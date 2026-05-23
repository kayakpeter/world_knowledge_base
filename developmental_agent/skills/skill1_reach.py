"""Skill #1 — drive seed + first drive-triggered reach. DESIGN.md §9, §10, §11.

Lights up the value box. Adds ONE primitive (mouth-value terminal), ONE arbitration
rule (interoceptively gated), activates ONE deferred component (the critic), and
learns ONE correspondence (visuomotor map). Everything else is reuse.

Consumes skill #0's handoff (controllable inverse + self/world tags). The reach is
M3 with the target supplied by the drive instead of randomly; the new learning is
the visuomotor map + the instrumental chain reach->grasp->mouth.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skill1Spec:
    name: str = "drive_triggered_reach"
    new_primitive: str = "mouth-value terminal (sparse extrinsic, interoception-grounded)"
    arbitration: str = "interoceptively gated: reward = drive-reduction; gating is a consequence"
    intrinsic_scaffold: str = "novelty / learning-progress (inherited from skill #0; extends self->self+world)"
    critic: str = "per-drive V_d(z,h); reflex-seeded; model-based, horizon-governed by skill-#0 reliability"
    visuomotor_map: str = "visual goal -> body-state target z* (representation alignment, no new module)"
    chain: tuple[str, ...] = ("reach", "grasp", "mouth")
    gate: tuple[str, ...] = ("N1", "N2", "N3", "N4", "N5", "C1", "C2", "C3", "C4", "W1", "C_CORR", "C_SHUTDOWN")
    open_risk: str = "corrigibility (C_CORR/C_SHUTDOWN) is monitored, not closed (DESIGN §11)"


SPEC = Skill1Spec()
