# strategist/tests/test_attempt_modeler.py
import pytest
from strategist.attempt_modeler import AttemptModeler
from strategist.schema import AttemptModel


def test_single_attempt_returns_p_success():
    """P(at least 1 success | 1 attempt) = p_success_per_attempt."""
    m = AttemptModeler.compute(p_attempt=0.60, p_success_per=0.30, n_attempts=1)
    assert isinstance(m, AttemptModel)
    assert abs(m.p_attempt_72h - 0.60) < 1e-9
    assert abs(m.p_cumulative_success - 0.30) < 1e-9


def test_multiple_attempts_cumulative():
    """P(at least 1 success | 3 attempts) = 1 - (1-0.18)^3"""
    m = AttemptModeler.compute(p_attempt=0.50, p_success_per=0.18, n_attempts=3)
    expected = 1 - (1 - 0.18) ** 3
    assert abs(m.p_cumulative_success - expected) < 1e-9


def test_high_n_approaches_certainty():
    """Many attempts with decent success rate → near certain"""
    m = AttemptModeler.compute(p_attempt=1.0, p_success_per=0.5, n_attempts=20)
    assert m.p_cumulative_success > 0.999


def test_zero_success_rate_returns_zero():
    """If p_success_per = 0, cumulative is always 0."""
    m = AttemptModeler.compute(p_attempt=1.0, p_success_per=0.0, n_attempts=5)
    assert m.p_cumulative_success == 0.0


def test_invalid_probability_raises():
    """p_attempt must be 0-1."""
    with pytest.raises(ValueError, match="p_attempt"):
        AttemptModeler.compute(p_attempt=1.5, p_success_per=0.3, n_attempts=1)


def test_invalid_success_rate_raises():
    with pytest.raises(ValueError, match="p_success_per"):
        AttemptModeler.compute(p_attempt=0.5, p_success_per=-0.1, n_attempts=1)


def test_invalid_n_attempts_raises():
    with pytest.raises(ValueError, match="n_attempts"):
        AttemptModeler.compute(p_attempt=0.5, p_success_per=0.3, n_attempts=0)


def test_attempt_modeler_stores_inputs():
    m = AttemptModeler.compute(p_attempt=0.40, p_success_per=0.25, n_attempts=2)
    assert m.p_attempt_72h == 0.40
    assert m.p_success_per_attempt == 0.25
    assert m.expected_attempts == 2
