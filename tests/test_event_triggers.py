# tests/test_event_triggers.py
import pytest
from unittest.mock import MagicMock, patch


def test_trigger_fires_when_conditions_met():
    """A trigger fires when all its flag conditions are satisfied."""
    mock_client = MagicMock()
    mock_client.get_active_flags.return_value = {
        "HORMUZ_STATUS": "EFFECTIVE_CLOSURE",
        "TANKER_WAR_STATUS": "OPERATIONAL",
    }
    mock_client.is_scenario_active.return_value = False

    with patch("processing.event_triggers._send_directive") as mock_send:
        from processing.event_triggers import evaluate_triggers
        fired = evaluate_triggers(mock_client, dry_run=False)

    assert "OIL_SHOCK_ACTIVATE" in fired
    mock_client.activate_scenario.assert_called()
    mock_send.assert_called()


def test_trigger_does_not_refire_if_scenario_active():
    """A trigger does not fire if its scenario is already active."""
    mock_client = MagicMock()
    mock_client.get_active_flags.return_value = {
        "HORMUZ_STATUS": "EFFECTIVE_CLOSURE",
        "TANKER_WAR_STATUS": "OPERATIONAL",
    }
    mock_client.is_scenario_active.return_value = True

    with patch("processing.event_triggers._send_directive") as mock_send:
        from processing.event_triggers import evaluate_triggers
        fired = evaluate_triggers(mock_client, dry_run=False)

    assert "OIL_SHOCK_ACTIVATE" not in fired
    mock_send.assert_not_called()


def test_trigger_dry_run_does_not_call_activate():
    """Dry run: conditions checked but no writes or sends."""
    mock_client = MagicMock()
    mock_client.get_active_flags.return_value = {
        "HORMUZ_STATUS": "EFFECTIVE_CLOSURE",
        "TANKER_WAR_STATUS": "OPERATIONAL",
    }
    mock_client.is_scenario_active.return_value = False

    with patch("processing.event_triggers._send_directive") as mock_send:
        from processing.event_triggers import evaluate_triggers
        fired = evaluate_triggers(mock_client, dry_run=True)

    mock_client.activate_scenario.assert_not_called()
    mock_send.assert_not_called()
    assert "OIL_SHOCK_ACTIVATE" in fired  # still reported as "would fire"


def test_trigger_alert_type_does_not_activate_scenario():
    """Alert-type triggers send a message but do not activate a scenario."""
    mock_client = MagicMock()
    mock_client.get_active_flags.return_value = {
        "BUSHEHR_NPP_RISK": "CRITICAL",
        "NUCLEAR_FLAG": "False",
    }
    mock_client.is_scenario_active.return_value = False

    with patch("processing.event_triggers._send_directive") as mock_send:
        from processing.event_triggers import evaluate_triggers
        fired = evaluate_triggers(mock_client, dry_run=False)

    assert "BUSHEHR_WATCH" in fired
    mock_send.assert_called()
    mock_client.activate_scenario.assert_not_called()
