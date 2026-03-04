from pathlib import Path
import json
import pytest
from processing.brain_retry_queue import BrainRetryQueue

QUEUE_FILE = Path("/tmp/test_brain_retry_queue.jsonl")


def teardown_function():
    QUEUE_FILE.unlink(missing_ok=True)


def test_enqueue_and_load():
    q = BrainRetryQueue(QUEUE_FILE)
    q.enqueue({"id": "abc123", "content": "test thought", "metadata": {"agent": "apollo"}})
    items = q.load_pending()
    assert len(items) == 1
    assert items[0]["id"] == "abc123"


def test_mark_done_removes_item():
    q = BrainRetryQueue(QUEUE_FILE)
    q.enqueue({"id": "abc123", "content": "test", "metadata": {}})
    q.mark_done("abc123")
    assert len(q.load_pending()) == 0


def test_enqueue_idempotent():
    q = BrainRetryQueue(QUEUE_FILE)
    q.enqueue({"id": "abc123", "content": "test", "metadata": {}})
    q.enqueue({"id": "abc123", "content": "test", "metadata": {}})
    assert len(q.load_pending()) == 1
