# tests/test_neo4j_client.py
import pytest
from unittest.mock import MagicMock, patch

def test_neo4j_client_upsert_country(monkeypatch):
    """upsert_country runs MERGE with correct parameters."""
    mock_session = MagicMock()
    from knowledge_base.neo4j_client import Neo4jClient
    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = MagicMock()
    client._driver.session.return_value.__enter__ = lambda s: mock_session
    client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    client.upsert_country("USA", "United States", "Americas")
    mock_session.run.assert_called_once()
    call_args = mock_session.run.call_args
    assert "MERGE" in call_args[0][0]
    assert call_args[1]["iso3"] == "USA"

def test_neo4j_client_upsert_event(monkeypatch):
    """upsert_event runs MERGE keyed on item_id."""
    mock_session = MagicMock()
    from knowledge_base.neo4j_client import Neo4jClient
    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = MagicMock()
    client._driver.session.return_value.__enter__ = lambda s: mock_session
    client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    client.upsert_event("evt_001", "Test headline", "2026-03-12T00:00:00Z", "critical", "positive", "test chain", 90)
    mock_session.run.assert_called_once()
    call_args = mock_session.run.call_args
    assert call_args[1]["item_id"] == "evt_001"

def test_neo4j_client_upsert_flag():
    """upsert_flag creates/updates a Flag node."""
    mock_session = MagicMock()
    from knowledge_base.neo4j_client import Neo4jClient
    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = MagicMock()
    client._driver.session.return_value.__enter__ = lambda s: mock_session
    client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    client.upsert_flag("HORMUZ_STATUS", "EFFECTIVE_CLOSURE", "hermes")
    mock_session.run.assert_called_once()
    call_args = mock_session.run.call_args
    assert call_args[1]["name"] == "HORMUZ_STATUS"
    assert call_args[1]["value"] == "EFFECTIVE_CLOSURE"

def test_neo4j_client_get_flag_returns_none_when_missing():
    """get_flag returns None when flag does not exist."""
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.single.return_value = None
    mock_session.run.return_value = mock_result
    from knowledge_base.neo4j_client import Neo4jClient
    client = Neo4jClient.__new__(Neo4jClient)
    client._driver = MagicMock()
    client._driver.session.return_value.__enter__ = lambda s: mock_session
    client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
    result = client.get_flag("NONEXISTENT_FLAG")
    assert result is None
