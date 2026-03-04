"""
Tests for open_brain_client — thought text parsing and wiring logic.

MCP HTTP call is mocked; tests cover:
1. _parse_thought_text() — reconstructing dicts from MCP text response
2. search_thoughts() end-to-end with mocked httpx
3. RegimeHistory construction logic (mirrors what main.py will do)
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from processing.open_brain_client import search_thoughts, _parse_thought_text
from processing.hmm_prior_updater import RegimeHistory


# ─── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_MCP_TEXT = """Found 2 thought(s):

[1] similarity=0.820 | 2026-03-03T14:09:37.307667+00:00
meta: {"subsystem":"geopolitical","affected_countries":["United States","Germany"],"affected_stats":["policy_rate","yield_spread_10y3m"],"probability":0.65,"mechanism":"Fed rate decision forces ECB response"}
US Fed holds rates — ECB forced to recalibrate.

---

[2] similarity=0.711 | 2026-03-02T10:00:00.000000+00:00
meta: {"subsystem":"crack-detection","country":"United States","overall_regime":"cracks_appearing","session_date":"2026-03-02"}
US credit conditions tightening; crack detection flagged cracks_appearing."""

SAMPLE_CRACK_TEXT = """Found 3 thought(s):

[1] similarity=0.900 | 2026-03-01T09:00:00.000000+00:00
meta: {"subsystem":"crack-detection","country":"United States","overall_regime":"thriving","session_date":"2026-03-01"}
US economy in thriving state.

---

[2] similarity=0.880 | 2026-03-02T09:00:00.000000+00:00
meta: {"subsystem":"crack-detection","country":"United States","overall_regime":"cracks_appearing","session_date":"2026-03-02"}
US showing early cracks.

---

[3] similarity=0.850 | 2026-03-03T09:00:00.000000+00:00
meta: {"subsystem":"crack-detection","country":"Germany","overall_regime":"thriving","session_date":"2026-03-03"}
Germany economy healthy."""


def _make_mock_response(text: str) -> MagicMock:
    """Build a mock httpx response returning MCP tool output."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "content": [{"type": "text", "text": text}],
            "isError": False,
        },
    }
    return mock


# ─── _parse_thought_text tests ───────────────────────────────────────────────

def test_parse_extracts_two_thoughts():
    thoughts = _parse_thought_text(SAMPLE_MCP_TEXT)
    assert len(thoughts) == 2


def test_parse_extracts_similarity():
    thoughts = _parse_thought_text(SAMPLE_MCP_TEXT)
    assert abs(thoughts[0]["similarity"] - 0.820) < 1e-6
    assert abs(thoughts[1]["similarity"] - 0.711) < 1e-6


def test_parse_extracts_metadata():
    thoughts = _parse_thought_text(SAMPLE_MCP_TEXT)
    meta = thoughts[0]["metadata"]
    assert meta["affected_countries"] == ["United States", "Germany"]
    assert abs(meta["probability"] - 0.65) < 1e-6


def test_parse_extracts_content():
    thoughts = _parse_thought_text(SAMPLE_MCP_TEXT)
    assert "Fed holds rates" in thoughts[0]["content"]


def test_parse_handles_no_matching_thoughts():
    thoughts = _parse_thought_text("No matching thoughts found.")
    assert thoughts == []


def test_parse_handles_bad_meta_json():
    bad_text = "[1] similarity=0.5 | 2026-03-01T00:00:00+00:00\nmeta: {broken json\nsome content"
    thoughts = _parse_thought_text(bad_text)
    # Should still return the thought with empty metadata rather than crashing
    assert thoughts[0]["metadata"] == {}


# ─── search_thoughts (mocked HTTP) ───────────────────────────────────────────

def test_search_returns_parsed_thoughts():
    with patch("processing.open_brain_client.httpx") as mock_httpx:
        mock_httpx.post.return_value = _make_mock_response(SAMPLE_MCP_TEXT)
        results = search_thoughts(
            query="test query",
            filter_subsystem="geopolitical",
            match_count=10,
        )
    assert len(results) == 2
    assert results[0]["metadata"]["subsystem"] == "geopolitical"


def test_search_returns_empty_on_http_error():
    with patch("processing.open_brain_client.httpx") as mock_httpx:
        mock_httpx.post.side_effect = Exception("connection refused")
        results = search_thoughts(query="test")
    assert results == []


def test_search_returns_empty_when_no_brain_key(monkeypatch):
    monkeypatch.setenv("BRAIN_KEY", "")
    # Patch the env file path to a non-existent file
    with patch("processing.open_brain_client._BRAIN_KEY_ENV_FILE", "/nonexistent/.env"):
        results = search_thoughts(query="test")
    assert results == []


# ─── RegimeHistory construction logic ────────────────────────────────────────

def _build_regime_histories(thoughts: list[dict]) -> list[RegimeHistory]:
    """Mirror of the main.py wiring logic for CRACK_HISTORIES."""
    valid_regimes = {"thriving", "cracks_appearing", "crisis_imminent"}
    country_seqs: dict[str, list[tuple[str, str]]] = {}
    for t in thoughts:
        meta = t.get("metadata", {})
        country = meta.get("country") or (meta.get("affected_countries") or [None])[0]
        regime = meta.get("overall_regime", "")
        ts = t.get("created_at", "")
        if country and regime in valid_regimes:
            country_seqs.setdefault(country, []).append((ts, regime))
    return [
        RegimeHistory(
            country=country,
            regime_sequence=[r for _, r in sorted(seqs)],
        )
        for country, seqs in country_seqs.items()
        if len(seqs) >= 1
    ]


def test_regime_history_groups_by_country():
    thoughts = _parse_thought_text(SAMPLE_CRACK_TEXT)
    histories = _build_regime_histories(thoughts)
    countries = {h.country for h in histories}
    assert "United States" in countries
    assert "Germany" in countries


def test_regime_history_orders_chronologically():
    thoughts = _parse_thought_text(SAMPLE_CRACK_TEXT)
    histories = _build_regime_histories(thoughts)
    us = next(h for h in histories if h.country == "United States")
    assert us.regime_sequence == ["thriving", "cracks_appearing"]


def test_regime_history_ignores_unknown_regimes():
    bad_thoughts = [{"metadata": {"country": "France", "overall_regime": "unknown_state"}, "created_at": "2026-03-01T00:00:00+00:00", "content": ""}]
    histories = _build_regime_histories(bad_thoughts)
    assert len(histories) == 0
