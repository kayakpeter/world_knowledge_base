import json
import pytest
from unittest.mock import patch, MagicMock
from processing.dynamic_scenario_generator import (
    DynamicScenarioGenerator,
    parse_scenario_json,
)

SAMPLE_LLM_RESPONSE = json.dumps([
    {
        "title": "US Consumer Credit Cliff",
        "category": "moderate",
        "probability_12m": 0.25,
        "description": "Credit card delinquencies exceed 4% triggering lending pullback",
        "primary_channel": "financial",
        "affected_countries": ["United States", "Canada"],
        "affected_stats": ["credit_card_delinquencies", "consumer_credit_growth"],
        "shock_magnitude": -2.5,
        "causal_chain": ["delinquencies_spike", "banks_tighten", "consumption_falls"],
        "severity": "high"
    }
])


def test_parse_scenario_json():
    scenarios = parse_scenario_json(SAMPLE_LLM_RESPONSE)
    assert len(scenarios) == 1
    assert scenarios[0].title == "US Consumer Credit Cliff"
    assert scenarios[0].probability_12m == 0.25


def test_parse_handles_bad_json():
    scenarios = parse_scenario_json("not valid json {{ broken")
    assert scenarios == []


def test_generator_returns_list():
    gen = DynamicScenarioGenerator(max_scenarios=3)
    with patch("processing.dynamic_scenario_generator.openai") as mock_openai:
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = SAMPLE_LLM_RESPONSE
        results = gen.generate(active_crack_summaries=["US: consumer credit stress active"])
    assert isinstance(results, list)
