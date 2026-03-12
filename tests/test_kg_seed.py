# tests/test_kg_seed.py
from unittest.mock import MagicMock, call


def test_seed_countries_calls_upsert_for_each_country():
    mock_client = MagicMock()
    from knowledge_base.kg_seed import seed_layer_a
    seed_layer_a(mock_client)
    assert mock_client.upsert_country.call_count >= 1


def test_seed_trade_routes_calls_upsert():
    mock_client = MagicMock()
    from knowledge_base.kg_seed import seed_layer_a
    seed_layer_a(mock_client)
    assert mock_client.upsert_trade_route.call_count >= 1


def test_seed_scenarios_creates_known_scenarios():
    mock_client = MagicMock()
    from knowledge_base.kg_seed import seed_layer_b_scenarios
    seed_layer_b_scenarios(mock_client)
    # Collect all scenario_id args from calls
    called_ids = []
    for c in mock_client.upsert_scenario.call_args_list:
        args, kwargs = c
        called_ids.append(kwargs.get("scenario_id") or args[0])
    assert "HORMUZ_CLOSURE" in called_ids
    assert "ABQAIQ_STRIKE" in called_ids
    assert "NUCLEAR_INCIDENT" in called_ids
