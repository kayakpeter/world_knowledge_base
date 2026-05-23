"""Interface contract — the cross-embodiment invariant. DESIGN.md §2, §3.

The ONE thing fixed across every embodiment (sim now, unknown humanoid later):
the *categories* of sensing/actuation, their units, semantics, and rates. The
numbers (DOF count, ranges, loop rates) are runtime config discovered by
self-calibration at load — NOT part of this contract.

Invariant-kernel obligations encoded here:
  - closed channel set == "complete initial state" (not omniscient world state)
  - egocentric-only at t0 (vestibular gives gravity-down + angular velocity,
    never position; no allocentric frame is an input)
  - multi-rate clock is explicit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Modality(Enum):
    PROPRIOCEPTION = "proprioception"   # per-DOF angle + angular velocity, stretch/tension
    TACTILE = "tactile"                 # pressure field, temperature, nociception (contact-only)
    VISION = "vision"                   # binocular; low-acuity / no stereopsis at t0
    AUDITION = "audition"               # binaural waveform
    VESTIBULAR = "vestibular"           # gravity vector + angular velocity (NOT position)
    INTEROCEPTION = "interoception"     # homeostatic need state == the drive substrate
    EFFERENCE_COPY = "efference_copy"   # feedback copy of motor command (enables agency discovery)


@dataclass(frozen=True)
class ChannelSpec:
    """One typed sensory channel. Units/ranges are physical and invariant; the
    physical layout (e.g. #DOF, image size) is per-embodiment config, set at load."""
    modality: Modality
    units: str
    rate_hz: float
    egocentric: bool = True
    notes: str = ""


@dataclass(frozen=True)
class ActionSpec:
    """Per-DOF actuator command space. Limits & slew caps are reflex-level safety
    (and the early-growth weakness that makes babbling non-destructive). DESIGN §4, §6."""
    units: str = "torque|activation"
    has_slew_limit: bool = True
    notes: str = ""


@dataclass(frozen=True)
class InterfaceContract:
    """The full I/O surface. Closure of this set is the definition of a complete
    initial state. Any embodiment must expose an adapter satisfying this contract."""
    channels: tuple[ChannelSpec, ...]
    action: ActionSpec
    base_clock_hz: float = field(default=100.0)

    def modalities(self) -> set[Modality]:
        return {c.modality for c in self.channels}

    def is_complete(self) -> bool:
        """All seven channel categories present. Raises nothing — design-stage check."""
        return self.modalities() == set(Modality)


# Reference t0 surface (categories only; numbers are config). DESIGN §3.
T0_REFERENCE = InterfaceContract(
    channels=(
        ChannelSpec(Modality.PROPRIOCEPTION, "rad, rad/s", 500.0),
        ChannelSpec(Modality.TACTILE, "N/cm^2, degC", 200.0, notes="contact-only; palm+mouth dense"),
        ChannelSpec(Modality.VISION, "intensity", 30.0, notes="low acuity; withhold clean depth if faithful"),
        ChannelSpec(Modality.AUDITION, "pressure", 16000.0),
        ChannelSpec(Modality.VESTIBULAR, "m/s^2, rad/s", 1000.0, notes="down + rotation, not position"),
        ChannelSpec(Modality.INTEROCEPTION, "normalized need", 1.0, notes="drive substrate; viability-coupled"),
        ChannelSpec(Modality.EFFERENCE_COPY, "= action dim", 500.0),
    ),
    action=ActionSpec(notes="mostly uncontrolled at t0; reflex arcs pre-wired"),
)
