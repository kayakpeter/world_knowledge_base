import numpy as np
import pytest
from processing.hmm_prior_updater import HMMPriorUpdater, RegimeHistory


def test_transition_count_from_history():
    history = [
        RegimeHistory(country="United States", regime_sequence=["thriving", "thriving", "cracks_appearing"]),
    ]
    updater = HMMPriorUpdater()
    counts = updater.count_transitions(history[0].regime_sequence)
    # thriving→thriving: 1, thriving→cracks_appearing: 1
    assert counts[0][0] == 1   # Tranquil→Tranquil
    assert counts[0][1] == 1   # Tranquil→Turbulent


def test_blend_priors_is_valid():
    updater = HMMPriorUpdater(blend_weight=0.3)
    counts = np.array([[10, 2, 0], [1, 5, 2], [0, 1, 4]], dtype=float)
    original = np.array([[0.70, 0.25, 0.05], [0.15, 0.55, 0.30], [0.05, 0.25, 0.70]])
    blended = updater.blend_with_prior(counts, original)
    # Each row must sum to 1.0
    assert np.allclose(blended.sum(axis=1), 1.0)
    # Blended must be between prior and empirical
    assert blended.shape == (3, 3)
