"""
HMM Prior Updater — Bayesian-blends static transition priors with historical crack data.

Algorithm:
1. Read last N crack reports from Open Brain (subsystem: crack-detection)
2. Map regimes to states: thriving→0, cracks_appearing→1, crisis_imminent→2
3. Count observed transitions per country
4. Bayesian blend: new_prior = (1-α)*static_prior + α*empirical_freq
5. Write updated priors to data/hmm_priors/YYYY-MM-DD_priors.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REGIME_TO_STATE = {"thriving": 0, "cracks_appearing": 1, "crisis_imminent": 2}
STATE_NAMES = ["Tranquil", "Turbulent", "Crisis"]

STATIC_PRIOR = np.array([
    [0.70, 0.25, 0.05],
    [0.15, 0.55, 0.30],
    [0.05, 0.25, 0.70],
], dtype=np.float32)

PRIORS_DIR = Path(__file__).parent.parent / "data" / "hmm_priors"


@dataclass
class RegimeHistory:
    country: str
    regime_sequence: list[str]  # ordered chronologically


class HMMPriorUpdater:
    def __init__(self, blend_weight: float = 0.30, n_states: int = 3) -> None:
        """
        blend_weight: how much weight to give empirical data (0=ignore, 1=use only empirical).
        0.30 means 70% static prior, 30% empirical — conservative to avoid overfitting.
        """
        self.blend_weight = blend_weight
        self.n_states = n_states

    def count_transitions(self, regime_sequence: list[str]) -> np.ndarray:
        """Count observed state transitions from a regime sequence."""
        counts = np.zeros((self.n_states, self.n_states), dtype=np.float32)
        for i in range(len(regime_sequence) - 1):
            s_from = REGIME_TO_STATE.get(regime_sequence[i])
            s_to = REGIME_TO_STATE.get(regime_sequence[i + 1])
            if s_from is not None and s_to is not None:
                counts[s_from, s_to] += 1
        return counts

    def blend_with_prior(
        self,
        counts: np.ndarray,
        static_prior: np.ndarray = STATIC_PRIOR,
    ) -> np.ndarray:
        """Blend empirical transition counts with static prior."""
        row_sums = counts.sum(axis=1, keepdims=True)
        # Avoid divide-by-zero: if no transitions observed, use static prior
        empirical = np.where(row_sums > 0, counts / np.maximum(row_sums, 1e-9), static_prior)
        blended = (1.0 - self.blend_weight) * static_prior + self.blend_weight * empirical
        # Renormalize rows
        row_sums = blended.sum(axis=1, keepdims=True)
        return (blended / row_sums).astype(np.float32)

    def compute_updated_priors(
        self, histories: list[RegimeHistory]
    ) -> dict[str, np.ndarray]:
        """Return per-country updated transition matrices."""
        updated: dict[str, np.ndarray] = {}
        for h in histories:
            counts = self.count_transitions(h.regime_sequence)
            updated[h.country] = self.blend_with_prior(counts)
        return updated

    def save(self, priors: dict[str, np.ndarray]) -> Path:
        """Save updated priors to JSON for today."""
        PRIORS_DIR.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat().replace("-", "")
        out_path = PRIORS_DIR / f"{today}_priors.json"
        serializable = {
            country: matrix.tolist()
            for country, matrix in priors.items()
        }
        out_path.write_text(json.dumps(serializable, indent=2))
        logger.info(f"HMM priors saved: {out_path}")
        return out_path

    def load_latest(self) -> dict[str, np.ndarray] | None:
        """Load most recent priors file, or None if none exists."""
        if not PRIORS_DIR.exists():
            return None
        files = sorted(PRIORS_DIR.glob("*_priors.json"))
        if not files:
            return None
        data = json.loads(files[-1].read_text())
        return {country: np.array(matrix, dtype=np.float32) for country, matrix in data.items()}
