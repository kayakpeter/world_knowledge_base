# tests/test_kg_to_openbrain.py
from unittest.mock import MagicMock, patch


def test_sync_events_posts_to_brain():
    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_client._driver.session.return_value.__enter__ = lambda s: mock_session
    mock_client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = [
        {"e.id": "evt_001", "e.headline": "Test event", "e.urgency": "critical",
         "e.causal_chain": "hormuz", "e.confidence": 90,
         "e.published_at": "2026-03-12T00:00:00Z",
         "countries": ["IRN", "USA"]},
    ]

    with patch("processing.kg_to_openbrain._post_thought", return_value="thought_123") as mock_post:
        from processing.kg_to_openbrain import sync_events_to_openbrain
        count = sync_events_to_openbrain(mock_client, since_hours=24)

    assert count == 1
    mock_post.assert_called_once()
    call_content = mock_post.call_args[0][0]
    assert "Test event" in call_content


def test_sync_scenarios_posts_active_only():
    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_client._driver.session.return_value.__enter__ = lambda s: mock_session
    mock_client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    mock_session.run.return_value = [
        {"s.id": "HORMUZ_CLOSURE", "s.name": "Hormuz Closure",
         "s.probability": 0.85, "s.active": True, "infra": ["hormuz_strait"]},
    ]

    with patch("processing.kg_to_openbrain._post_thought", return_value="thought_456") as mock_post:
        from processing.kg_to_openbrain import sync_scenarios_to_openbrain
        count = sync_scenarios_to_openbrain(mock_client)

    assert count == 1
    call_content = mock_post.call_args[0][0]
    assert "HORMUZ_CLOSURE" in call_content or "Hormuz Closure" in call_content
