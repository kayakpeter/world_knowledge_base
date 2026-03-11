"""
Geopolitical Fusion — blend HMM state posteriors with geo stress signal.

fuse_states() merges:
  - hmm_states: dict from hmm_states.json (per-country state + state_probs)
  - stress_scores: dict from geo_stress_scorer.compute_stress_scores()

Output: enriched dict with fused_state_probs, geo_stress_score,
        geo_stress_weight added to each country entry.
Written to: fused_hmm_states.json in the snapshot dir.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Stress score → state probs thresholds
STRESS_THRESHOLDS = [
    (0.05, {"Tranquil": 1.0, "Turbulent": 0.0, "Crisis": 0.0}),
    (0.12, {"Tranquil": 0.0, "Turbulent": 1.0, "Crisis": 0.0}),
    (0.20, {"Tranquil": 0.0, "Turbulent": 0.30, "Crisis": 0.70}),
    (1.01, {"Tranquil": 0.0, "Turbulent": 0.0, "Crisis": 1.0}),
]

# News signal caps at 90% — HMM always has at least 10% weight
MAX_GEO_WEIGHT: float = 0.90

# Geo stress × scale factor → news weight (clamped to MAX_GEO_WEIGHT)
GEO_WEIGHT_SCALE: float = 2.5


def stress_to_state_probs(stress: float) -> dict[str, float]:
    """Map a scalar stress score to state probability distribution."""
    for threshold, probs in STRESS_THRESHOLDS:
        if stress < threshold:
            return dict(probs)
    return {"Tranquil": 0.0, "Turbulent": 0.0, "Crisis": 1.0}


def fuse_states(
    hmm_states: dict[str, dict],
    stress_scores: dict[str, float],
) -> dict[str, dict]:
    """
    Fuse HMM state posteriors with geopolitical stress scores.

    For each country:
      w_news = min(stress_score * GEO_WEIGHT_SCALE, MAX_GEO_WEIGHT)
      w_hmm  = 1.0 - w_news
      fused_probs = w_hmm * hmm_probs + w_news * stress_to_state_probs(stress)

    Returns enriched copy of hmm_states with added fields:
      fused_state_probs: {Tranquil, Turbulent, Crisis}
      fused_state: most probable fused state name
      geo_stress_score: raw stress input
      geo_stress_weight: w_news used
    """
    result = {}
    for country, entry in hmm_states.items():
        out = dict(entry)

        stress = stress_scores.get(country, 0.0)
        w_news = min(stress * GEO_WEIGHT_SCALE, MAX_GEO_WEIGHT)
        w_hmm = 1.0 - w_news

        hmm_probs = entry.get("state_probs", {})
        t_hmm  = float(hmm_probs.get("Tranquil",  0.60))
        tu_hmm = float(hmm_probs.get("Turbulent", 0.30))
        c_hmm  = float(hmm_probs.get("Crisis",    0.10))

        news_probs = stress_to_state_probs(stress)
        t_fused  = w_hmm * t_hmm  + w_news * news_probs["Tranquil"]
        tu_fused = w_hmm * tu_hmm + w_news * news_probs["Turbulent"]
        c_fused  = w_hmm * c_hmm  + w_news * news_probs["Crisis"]

        # Normalise to handle floating-point drift
        total = t_fused + tu_fused + c_fused
        if total > 0:
            t_fused, tu_fused, c_fused = t_fused / total, tu_fused / total, c_fused / total

        if c_fused >= 0.5:
            fused_state = "S2_Crisis"
        elif tu_fused >= 0.5:
            fused_state = "S1_Turbulent"
        else:
            fused_state = "S0_Tranquil"

        out["fused_state_probs"] = {
            "Tranquil":  round(t_fused, 6),
            "Turbulent": round(tu_fused, 6),
            "Crisis":    round(c_fused, 6),
        }
        out["fused_state"] = fused_state
        out["geo_stress_score"] = round(stress, 4)
        out["geo_stress_weight"] = round(w_news, 4)

        result[country] = out
        logger.debug(
            "  %-20s stress=%.3f w_news=%.2f → fused_state=%s (C=%.3f T=%.3f Tu=%.3f)",
            country, stress, w_news, fused_state, c_fused, t_fused, tu_fused,
        )

    return result


def write_fused_states(fused: dict[str, dict], output_path: Path) -> None:
    """Write fused_hmm_states.json to the given path."""
    output_path.write_text(json.dumps(fused, indent=2))
    logger.info("Fused HMM states written: %s (%d countries)", output_path, len(fused))
