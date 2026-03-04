"""
Brain retry queue — persists thoughts that failed embedding for later retry.

When Open Brain's embedding service (OpenRouter) is unavailable, the MCP server
stores thoughts without vector embeddings (embedding_skipped: true). This means
they won't surface in semantic search. This queue re-posts them when the service
recovers.

Queue format: JSONL file at data/brain_retry_queue.jsonl
Each line: {"id": str, "content": str, "metadata": dict, "enqueued_at": str}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = Path(__file__).parent.parent / "data" / "brain_retry_queue.jsonl"


class BrainRetryQueue:
    def __init__(self, path: Path = DEFAULT_QUEUE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(self, thought: dict) -> None:
        """Add a thought to the retry queue (idempotent by id)."""
        existing_ids = {item["id"] for item in self.load_pending()}
        if thought["id"] in existing_ids:
            return
        entry = {
            "id": thought["id"],
            "content": thought["content"],
            "metadata": thought.get("metadata", {}),
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"BrainRetryQueue: enqueued thought id={thought['id']}")

    def load_pending(self) -> list[dict]:
        """Return all pending thoughts (not yet successfully re-embedded)."""
        if not self.path.exists():
            return []
        items = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def mark_done(self, thought_id: str) -> None:
        """Remove a successfully re-embedded thought from the queue."""
        items = [i for i in self.load_pending() if i["id"] != thought_id]
        with self.path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        logger.info(f"BrainRetryQueue: marked done id={thought_id}")
