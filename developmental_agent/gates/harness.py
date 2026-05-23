"""Measurement harness — the moving-metric resolution. DESIGN.md §7.

Separates the LEARNING yardstick (latent, moving, fine) from the GATE yardstick
(physical units, invariant). Layers:
  L0  training target            -> EMA target encoder (drift is fine)
  L1  intra-snapshot stability    -> freeze encoder+EMA+probe per eval
  L2  pass/fail at a snapshot      -> scale-invariant, null-normalized stats
  L3  cross-snapshot trajectory    -> frozen capacity-limited LINEAR probe to fixed
                                      physical units; report R^2; gate worst-stratum

Under growth-from-t0 the system is non-stationary, so every metric is a TRAJECTORY,
each point a contrast against a null computed on the SAME instantaneous morphology.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Snapshot:
    """A frozen eval point: weights + growth frozen, battery run on held-out data."""
    developmental_time: float
    morphology: dict


@dataclass(frozen=True)
class ProbeReport:
    """Result of the L3 linear probe to fixed physical observables.

    R^2 is the cross-age-comparable (scale-invariant) quantity. We gate on
    worst-stratum, not mean (prevents pass-by-avoidance). A collapsed latent makes
    these crater -> doubles as the collapse sentinel."""
    r2_per_target: dict[str, float]
    worst_stratum_r2: float
    linear_mlp_gap: float          # nonlinearly-entangled physical info (diagnostic)
    latent_info: float             # collapse sentinel (effective rank / variance of z)


class GateHarness:
    """Design-stage stub. Each method is a *measurement contract*, not an
    implementation. Thresholds are statistical separation from a same-condition null
    (one pre-registered effect size) — never magic numbers."""

    def snapshot_and_freeze(self, developmental_time: float) -> Snapshot:
        raise NotImplementedError("design-stage scaffold")

    def fit_linear_probe(self, snapshot: Snapshot) -> ProbeReport:
        """Refit EVERY eval (cheap, convex) and report the OPTIMALLY-refit probe's
        decodability -> removes the staleness confound. Fit-set (current, stratified)
        is separated from an in-distribution held-out test-set."""
        raise NotImplementedError("design-stage scaffold")

    def null_normalized_gain(self, metric: float, null: float) -> float:
        """L2 dimensionless contrast (e.g. (E_null - E_model)/E_null). Latent
        rescaling cancels, so pass/fail needs no cross-snapshot comparability."""
        raise NotImplementedError("design-stage scaffold")

    def passes(self, contrast: float, effect_size: float) -> bool:
        """Pass == CI of the contrast separates from the null by the pre-registered
        effect size. Embodiment/scale-invariant -> identical on sim and real robot."""
        raise NotImplementedError("design-stage scaffold")
