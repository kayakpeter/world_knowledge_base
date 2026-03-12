# tests/test_kg_ingest.py
import polars as pl
import pytest
from pathlib import Path
from unittest.mock import MagicMock


def _make_parquet(tmp_path: Path) -> Path:
    """Write a minimal test parquet matching interpretation schema."""
    df = pl.DataFrame({
        "item_id":            ["evt_001", "evt_002"],
        "country_iso3":       ["IRN", "SAU"],
        "headline":           ["Iran closes strait", "Saudi refinery at risk"],
        "published_at":       ["2026-03-12T10:00:00Z", "2026-03-12T11:00:00Z"],
        "causal_chain":       ["hormuz disruption -> oil shock", "abqaiq strike risk"],
        "sentiment":          ["negative", "negative"],
        "urgency":            ["critical", "high"],
        "confidence":         [90, 80],
        "affected_stats":     ["oil_price,gdp_growth", "oil_price"],
        "stat_direction":     ["up,down", "up"],
        "estimated_magnitude":["50.0,2.0", "30.0"],
        "cross_country_iso3": ["USA,DEU", ""],
    })
    p = tmp_path / "test_interp.parquet"
    df.write_parquet(p)
    return p


def test_ingest_parquet_upserts_events(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    mock_client = MagicMock()
    from processing.kg_ingest import ingest_parquet
    n = ingest_parquet(parquet_path, mock_client, dry_run=False)
    assert n == 2
    assert mock_client.upsert_event.call_count == 2
    first_call = mock_client.upsert_event.call_args_list[0]
    assert first_call[1]["item_id"] == "evt_001"
    assert first_call[1]["headline"] == "Iran closes strait"


def test_ingest_parquet_creates_affects_edges(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    mock_client = MagicMock()
    from processing.kg_ingest import ingest_parquet
    ingest_parquet(parquet_path, mock_client, dry_run=False)
    # evt_001: IRN + cross_country USA, DEU = 3 AFFECTS edges
    # evt_002: SAU, no cross = 1 AFFECTS edge
    assert mock_client.create_affects_edge.call_count >= 4
    # Verify the correct countries were targeted
    called_iso3s = [c[0][1] for c in mock_client.create_affects_edge.call_args_list]
    assert set(called_iso3s) == {"IRN", "USA", "DEU", "SAU"}


def test_ingest_parquet_dry_run_does_not_call_client(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    mock_client = MagicMock()
    from processing.kg_ingest import ingest_parquet
    n = ingest_parquet(parquet_path, mock_client, dry_run=True)
    assert n == 2  # still counts rows
    mock_client.upsert_event.assert_not_called()


def test_ingest_parquet_escalates_hormuz_scenario(tmp_path):
    parquet_path = _make_parquet(tmp_path)
    mock_client = MagicMock()
    from processing.kg_ingest import ingest_parquet
    ingest_parquet(parquet_path, mock_client, dry_run=False)
    escalate_calls = mock_client.create_escalates_edge.call_args_list
    scenario_ids = [c[1]["scenario_id"] for c in escalate_calls]
    assert "HORMUZ_CLOSURE" in scenario_ids
