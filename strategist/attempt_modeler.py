# strategist/attempt_modeler.py
from __future__ import annotations
from strategist.schema import AttemptModel


class AttemptModeler:
    """
    Factory for AttemptModel instances.
    Handles validation and wraps AttemptModel construction.

    The cumulative probability formula:
        P(at least one success in window) = 1 - (1 - p_success_per_attempt)^expected_attempts

    This models scenarios like "only one missile needs to get through":
        p_attempt   = P(Iran attempts a Suez strike in the 72h window)
        p_success   = P(strike succeeds | one attempt)
        n_attempts  = how many launch attempts expected if they try

    The resulting AttemptModel stores these as inputs;
    p_cumulative_success is a derived property on AttemptModel.
    """

    @staticmethod
    def compute(
        p_attempt: float,
        p_success_per: float,
        n_attempts: int,
    ) -> AttemptModel:
        """
        Validate inputs and return an AttemptModel.

        Args:
            p_attempt: P(event is attempted in the 72h window). Must be [0, 1].
            p_success_per: P(success | one attempt). Must be [0, 1].
            n_attempts: Expected number of attempts if event occurs. Must be >= 1.

        Returns:
            AttemptModel with the given parameters.

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if not 0.0 <= p_attempt <= 1.0:
            raise ValueError(f"p_attempt must be between 0 and 1, got {p_attempt}")
        if not 0.0 <= p_success_per <= 1.0:
            raise ValueError(f"p_success_per must be between 0 and 1, got {p_success_per}")
        if n_attempts < 1:
            raise ValueError(f"n_attempts must be >= 1, got {n_attempts}")

        return AttemptModel(
            p_attempt_72h=p_attempt,
            p_success_per_attempt=p_success_per,
            expected_attempts=n_attempts,
        )
