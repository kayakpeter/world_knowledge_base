"""
Hidden Markov Model for sovereign economic state estimation.

States: {Tranquil, Turbulent, Crisis} per country
Observations: Discretized statistics → {Low, Normal, High}

Implements:
- Forward-Backward algorithm for state posterior probabilities
- Viterbi algorithm for most-likely state sequence
- Baum-Welch for parameter estimation from historical data
- Monte Carlo simulation over the transition matrix
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class HMMParams:
    """Parameters for a single country's Hidden Markov Model."""
    country: str
    country_iso3: str
    n_states: int = 3        # Tranquil, Turbulent, Crisis
    n_obs_levels: int = 3    # Low, Normal, High

    # Initial state distribution π
    initial_probs: np.ndarray = field(default_factory=lambda: np.array([0.60, 0.30, 0.10]))

    # Transition matrix A[i,j] = P(state_j at t+1 | state_i at t)
    transition_matrix: np.ndarray = field(default_factory=lambda: np.array([
        [0.70, 0.25, 0.05],  # From Tranquil
        [0.15, 0.55, 0.30],  # From Turbulent
        [0.05, 0.25, 0.70],  # From Crisis
    ]))

    # Emission matrix B[i,j] = P(obs_j | state_i)
    # Rows = states, Cols = observation levels {Low=0, Normal=1, High=2}
    emission_matrix: np.ndarray = field(default_factory=lambda: np.array([
        [0.10, 0.70, 0.20],  # Tranquil: mostly Normal observations
        [0.25, 0.45, 0.30],  # Turbulent: more spread
        [0.50, 0.25, 0.25],  # Crisis: many Low observations
    ]))


STATE_NAMES = ["Tranquil", "Turbulent", "Crisis"]
OBS_NAMES = ["Low", "Normal", "High"]


class SovereignHMM:
    """
    Per-country Hidden Markov Model for economic state estimation.

    Each country has its own HMM parameters (transition + emission matrices),
    but the state space is shared across all countries.
    """

    def __init__(self, params: HMMParams):
        self.params = params
        self._validate_params()

    def _validate_params(self) -> None:
        """Ensure probability matrices are valid."""
        p = self.params
        assert p.transition_matrix.shape == (p.n_states, p.n_states)
        assert p.emission_matrix.shape == (p.n_states, p.n_obs_levels)
        assert np.allclose(p.transition_matrix.sum(axis=1), 1.0), "Transition rows must sum to 1"
        assert np.allclose(p.emission_matrix.sum(axis=1), 1.0), "Emission rows must sum to 1"
        assert np.isclose(p.initial_probs.sum(), 1.0), "Initial probs must sum to 1"

    def forward(self, observations: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Forward algorithm — compute P(observations) and forward variables.

        Args:
            observations: 1D array of observation indices (0=Low, 1=Normal, 2=High)

        Returns:
            (alpha, log_likelihood) where alpha[t, i] = P(o_1...o_t, q_t=i)
        """
        T = len(observations)
        N = self.params.n_states
        A = self.params.transition_matrix
        B = self.params.emission_matrix
        pi = self.params.initial_probs

        alpha = np.zeros((T, N))

        # Initialization
        alpha[0] = pi * B[:, observations[0]]

        # Induction
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t - 1] * A[:, j]) * B[j, observations[t]]

        # Scaling to avoid underflow
        log_likelihood = np.sum(np.log(alpha[-1].sum()))

        return alpha, log_likelihood

    def backward(self, observations: np.ndarray) -> np.ndarray:
        """
        Backward algorithm — compute backward variables.

        Args:
            observations: 1D array of observation indices

        Returns:
            beta[t, i] = P(o_{t+1}...o_T | q_t=i)
        """
        T = len(observations)
        N = self.params.n_states
        A = self.params.transition_matrix
        B = self.params.emission_matrix

        beta = np.zeros((T, N))
        beta[-1] = 1.0  # Initialization

        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(A[i, :] * B[:, observations[t + 1]] * beta[t + 1])

        return beta

    def viterbi(self, observations: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Viterbi algorithm — find the most likely state sequence.

        Args:
            observations: 1D array of observation indices

        Returns:
            (state_sequence, log_prob) — the ML state path and its log probability
        """
        T = len(observations)
        N = self.params.n_states
        A = self.params.transition_matrix
        B = self.params.emission_matrix
        pi = self.params.initial_probs

        # Work in log space to avoid underflow
        log_A = np.log(A + 1e-300)
        log_B = np.log(B + 1e-300)
        log_pi = np.log(pi + 1e-300)

        # Viterbi variables
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        # Initialization
        delta[0] = log_pi + log_B[:, observations[0]]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                candidates = delta[t - 1] + log_A[:, j]
                psi[t, j] = np.argmax(candidates)
                delta[t, j] = candidates[psi[t, j]] + log_B[j, observations[t]]

        # Termination
        state_sequence = np.zeros(T, dtype=int)
        state_sequence[-1] = np.argmax(delta[-1])
        log_prob = delta[-1, state_sequence[-1]]

        # Backtrack
        for t in range(T - 2, -1, -1):
            state_sequence[t] = psi[t + 1, state_sequence[t + 1]]

        return state_sequence, log_prob

    def state_posteriors(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute posterior state probabilities P(q_t=i | O) using Forward-Backward.

        Args:
            observations: 1D array of observation indices

        Returns:
            gamma[t, i] = P(q_t=i | all observations)
        """
        alpha, _ = self.forward(observations)
        beta = self.backward(observations)

        gamma = alpha * beta
        # Normalize each time step
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-300  # avoid division by zero
        gamma = gamma / row_sums

        return gamma

    def baum_welch(
        self,
        observations: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> float:
        """
        Baum-Welch (EM) algorithm to estimate HMM parameters from data.

        Updates transition_matrix, emission_matrix, and initial_probs in place.

        Args:
            observations: 1D array of observation indices
            max_iterations: Maximum EM iterations
            tolerance: Convergence threshold for log-likelihood change

        Returns:
            Final log-likelihood
        """
        T = len(observations)
        N = self.params.n_states
        M = self.params.n_obs_levels

        prev_ll = -np.inf

        for iteration in range(max_iterations):
            # E-step
            alpha, log_ll = self.forward(observations)
            beta = self.backward(observations)

            # Check convergence
            if abs(log_ll - prev_ll) < tolerance:
                logger.info(
                    "Baum-Welch converged at iteration %d (LL=%.6f)", iteration, log_ll
                )
                break
            prev_ll = log_ll

            # Compute gamma and xi
            gamma = alpha * beta
            gamma_sums = gamma.sum(axis=1, keepdims=True)
            gamma_sums[gamma_sums == 0] = 1e-300
            gamma = gamma / gamma_sums

            xi = np.zeros((T - 1, N, N))
            for t in range(T - 1):
                for i in range(N):
                    for j in range(N):
                        xi[t, i, j] = (
                            alpha[t, i]
                            * self.params.transition_matrix[i, j]
                            * self.params.emission_matrix[j, observations[t + 1]]
                            * beta[t + 1, j]
                        )
                denom = xi[t].sum()
                if denom > 0:
                    xi[t] /= denom

            # M-step: update parameters
            self.params.initial_probs = gamma[0]

            for i in range(N):
                gamma_sum_i = gamma[:-1, i].sum()
                if gamma_sum_i > 0:
                    for j in range(N):
                        self.params.transition_matrix[i, j] = xi[:, i, j].sum() / gamma_sum_i

                gamma_sum_all = gamma[:, i].sum()
                if gamma_sum_all > 0:
                    for k in range(M):
                        mask = (observations == k).astype(float)
                        self.params.emission_matrix[i, k] = (gamma[:, i] * mask).sum() / gamma_sum_all

        return prev_ll

    def current_state_estimate(self, observations: np.ndarray) -> dict:
        """
        Estimate the current economic state from the most recent observations.

        Returns a dict with state probabilities and the most likely state.
        """
        if len(observations) == 0:
            return {
                "state": "Unknown",
                "probabilities": dict(zip(STATE_NAMES, self.params.initial_probs.tolist())),
            }

        gamma = self.state_posteriors(observations)
        latest_probs = gamma[-1]

        ml_state_idx = np.argmax(latest_probs)

        return {
            "country": self.params.country,
            "state": STATE_NAMES[ml_state_idx],
            "probabilities": dict(zip(STATE_NAMES, latest_probs.tolist())),
            "n_observations": len(observations),
        }

    def monte_carlo_forecast(
        self,
        current_state_idx: int,
        n_steps: int = 12,
        n_simulations: int = 10000,
    ) -> np.ndarray:
        """
        Monte Carlo simulation of future state paths.

        Args:
            current_state_idx: Starting state (0=Tranquil, 1=Turbulent, 2=Crisis)
            n_steps: Number of time steps to simulate (e.g., 12 months)
            n_simulations: Number of Monte Carlo paths

        Returns:
            Array of shape (n_simulations, n_steps) with state indices
        """
        A = self.params.transition_matrix
        N = self.params.n_states

        paths = np.zeros((n_simulations, n_steps), dtype=int)

        for sim in range(n_simulations):
            state = current_state_idx
            for t in range(n_steps):
                state = np.random.choice(N, p=A[state])
                paths[sim, t] = state

        return paths

    def forecast_summary(
        self,
        current_state_idx: int,
        n_steps: int = 12,
        n_simulations: int = 10000,
    ) -> dict:
        """
        Summarize Monte Carlo forecast into probability distributions per step.

        Returns dict with per-step state probabilities and crisis probability trajectory.
        """
        paths = self.monte_carlo_forecast(current_state_idx, n_steps, n_simulations)

        step_probs = []
        crisis_trajectory = []

        for t in range(n_steps):
            counts = np.bincount(paths[:, t], minlength=self.params.n_states)
            probs = counts / n_simulations
            step_probs.append(dict(zip(STATE_NAMES, probs.tolist())))
            crisis_trajectory.append(probs[2])  # Crisis probability

        return {
            "country": self.params.country,
            "starting_state": STATE_NAMES[current_state_idx],
            "n_steps": n_steps,
            "n_simulations": n_simulations,
            "step_probabilities": step_probs,
            "crisis_trajectory": crisis_trajectory,
            "crisis_prob_12m": crisis_trajectory[-1] if crisis_trajectory else 0.0,
        }


def discretize_observations(
    values: np.ndarray,
    thresholds: tuple[float, float] = (33.0, 66.0),
) -> np.ndarray:
    """
    Convert continuous stat values to discrete {Low=0, Normal=1, High=2}.

    Uses percentile-based thresholds by default.
    """
    if len(values) == 0:
        return np.array([], dtype=int)

    low_thresh = np.percentile(values, thresholds[0])
    high_thresh = np.percentile(values, thresholds[1])

    discrete = np.ones(len(values), dtype=int)  # Default: Normal
    discrete[values <= low_thresh] = 0   # Low
    discrete[values >= high_thresh] = 2  # High

    return discrete
