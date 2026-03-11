import json
import pytest
from processing.geo_fusion import fuse_states, stress_to_state_probs, STRESS_THRESHOLDS


def test_stress_to_state_probs_buckets():
    """Verify correct state classification at threshold boundaries (recalibrated to actual data range)."""
    p = stress_to_state_probs(0.03)
    assert p == {"Tranquil": 1.0, "Turbulent": 0.0, "Crisis": 0.0}

    p = stress_to_state_probs(0.08)
    assert p["Turbulent"] == 1.0

    p = stress_to_state_probs(0.16)
    assert p["Crisis"] > 0.5

    p = stress_to_state_probs(1.0)
    assert p == {"Tranquil": 0.0, "Turbulent": 0.0, "Crisis": 1.0}


def test_fuse_zero_stress_returns_hmm_unchanged():
    """With zero stress, fused probs should be identical to HMM probs."""
    states = {
        "Germany": {
            "state": "S0_Tranquil",
            "state_probs": {"Tranquil": 0.9, "Turbulent": 0.08, "Crisis": 0.02},
        }
    }
    stress_scores = {"Germany": 0.0}
    result = fuse_states(states, stress_scores)
    fused_probs = result["Germany"]["fused_state_probs"]
    assert abs(fused_probs["Tranquil"] - 0.9) < 0.01
    assert result["Germany"]["geo_stress_score"] == 0.0


def test_fuse_high_stress_shifts_toward_crisis():
    """High stress should push fused probs toward crisis."""
    states = {
        "Germany": {
            "state": "S0_Tranquil",
            "state_probs": {"Tranquil": 0.99, "Turbulent": 0.01, "Crisis": 0.0},
        }
    }
    stress_scores = {"Germany": 0.80}
    result = fuse_states(states, stress_scores)
    fused_probs = result["Germany"]["fused_state_probs"]
    # With stress=0.80, news weight = min(0.80 * 2.5, 0.90) = 0.90
    # stress_to_state_probs(0.80) = Crisis=1.0
    # fused = 0.10 * {T:0.99, Tu:0.01, C:0} + 0.90 * {T:0, Tu:0, C:1}
    assert fused_probs["Crisis"] > 0.88


def test_fuse_news_weight_caps_at_90_percent():
    """News signal should never fully evict HMM — capped at 90% weight."""
    states = {
        "United States": {
            "state": "S0_Tranquil",
            "state_probs": {"Tranquil": 1.0, "Turbulent": 0.0, "Crisis": 0.0},
        }
    }
    stress_scores = {"United States": 1.0}  # Maximum possible stress
    result = fuse_states(states, stress_scores)
    geo_weight = result["United States"]["geo_stress_weight"]
    assert geo_weight <= 0.90 + 1e-6


def test_fuse_countries_not_in_stress_use_hmm_only():
    """Countries absent from stress_scores fall back to HMM probs."""
    states = {
        "Canada": {
            "state": "S0_Tranquil",
            "state_probs": {"Tranquil": 1.0, "Turbulent": 0.0, "Crisis": 0.0},
        }
    }
    stress_scores = {}  # Canada has no stress data
    result = fuse_states(states, stress_scores)
    fused_probs = result["Canada"]["fused_state_probs"]
    assert abs(fused_probs["Tranquil"] - 1.0) < 0.001
    assert result["Canada"]["geo_stress_weight"] == 0.0


def test_fused_probs_sum_to_one():
    """Fused probabilities must sum to 1.0 for all countries."""
    states = {
        "Japan": {
            "state": "S2_Crisis",
            "state_probs": {"Tranquil": 0.0, "Turbulent": 0.0, "Crisis": 1.0},
        },
        "France": {
            "state": "S0_Tranquil",
            "state_probs": {"Tranquil": 0.95, "Turbulent": 0.05, "Crisis": 0.0},
        },
    }
    stress_scores = {"Japan": 0.90, "France": 0.30}
    result = fuse_states(states, stress_scores)
    for country, entry in result.items():
        probs = entry["fused_state_probs"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6, f"{country}: probs sum to {total}"
