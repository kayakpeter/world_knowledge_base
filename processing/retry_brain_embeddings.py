"""
Retry job — re-posts any Open Brain thoughts with skipped embeddings.

Run: python -m processing.retry_brain_embeddings
Intended to be called at start of each daily KB refresh (from main.py).

This script logs what needs retrying. The actual re-ingest is performed
by the calling Claude agent via the open-brain MCP ingest-thought tool,
since embedding requires MCP tool access not available to subprocesses.
"""
from __future__ import annotations

import logging

from processing.brain_retry_queue import BrainRetryQueue

logger = logging.getLogger(__name__)


def run_retry_pass() -> tuple[int, int]:
    """
    Log all queued thoughts pending re-embedding.
    Returns (attempted, succeeded).

    Note: actual MCP re-ingest must be triggered by the Claude agent
    reading the queue via BrainRetryQueue.load_pending().
    """
    queue = BrainRetryQueue()
    pending = queue.load_pending()
    if not pending:
        logger.info("Brain retry queue: empty — nothing to retry")
        return 0, 0

    logger.info(f"Brain retry queue: {len(pending)} thoughts pending re-embedding")
    for thought in pending:
        logger.info(f"  → Needs retry: id={thought['id']}, enqueued={thought['enqueued_at']}")

    return len(pending), 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    attempted, succeeded = run_retry_pass()
    print(f"Retry pass complete: {succeeded}/{attempted} re-embedded")
