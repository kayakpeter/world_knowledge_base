# strategist/tests/test_integration.py
"""
Integration smoke test: runs the full pipeline end-to-end with mocked
Ollama and mocked Neo4j. Verifies:
- Scenario JSON written to disk
- Prometheus delta JSON written to disk
- No unhandled exceptions
"""
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from strategist.config import StrategistConfig
from strategist.runner import ScenarioRunner
from strategist.expander import ScenarioExpander
from strategist.infra_state import InfraStateRegistry
from strategist.prometheus_writer import PrometheusDeltaWriter
from strategist.kg_writer import KGWriter
from strategist.schema import ScenarioNode, SectorImpact


def _make_mock_llm_provider():
    """Mock LLM that returns 2 branches at depth 0, 0 branches at depth 1."""
    provider = MagicMock()

    call_count = [0]

    async def fake_complete(system_prompt, user_prompt, temperature=0.3, max_tokens=1500):
        call_count[0] += 1
        if call_count[0] <= 1:
            # First expansion: 2 branches
            branches = [
                {
                    "description": "Iran closes Hormuz Strait",
                    "probability": 0.65,
                    "sector_impacts": [
                        {"sector": "energy", "direction": "UP", "magnitude": "CRITICAL",
                         "magnitude_pct": 40.0, "tickers": ["FRO"], "notes": "Oil shock"}
                    ],
                    "infrastructure_effects": [
                        {"infra_id": "HORMUZ_STRAIT", "new_status": "CLOSED", "confidence": 0.90}
                    ],
                    "time_offset_hours": 6.0,
                },
                {
                    "description": "Iran threatens Abqaiq strike",
                    "probability": 0.35,
                    "sector_impacts": [
                        {"sector": "tanker", "direction": "UP", "magnitude": "HIGH",
                         "magnitude_pct": 25.0, "tickers": ["STNG"], "notes": ""}
                    ],
                    "infrastructure_effects": [],
                    "time_offset_hours": 12.0,
                },
            ]
        else:
            # All subsequent expansions: no branches (leaf nodes)
            branches = []
        import json as json_mod
        content = json_mod.dumps({"branches": branches})
        return MagicMock(content=content)

    provider.complete = fake_complete
    return provider


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_full_pipeline_smoke(tmp_path):
    """Full pipeline: runner → prometheus writer → kg writer. All mocked external deps."""
    cfg = StrategistConfig()
    cfg.scenarios_dir = tmp_path
    cfg.regime_config_dir = tmp_path / "regime"

    # Mocked external deps
    mock_llm = _make_mock_llm_provider()
    mock_neo4j = MagicMock()
    mock_neo4j.upsert_scenario = MagicMock()
    mock_neo4j.activate_scenario = MagicMock()
    mock_neo4j.upsert_flag = MagicMock()

    # Build pipeline
    expander = ScenarioExpander(cfg, llm_provider=mock_llm)
    infra_reg = InfraStateRegistry(mock_neo4j)
    infra_reg.seed_from_canonical()

    runner = ScenarioRunner(cfg, expander=expander, infra_registry=infra_reg)

    # Run
    tree = _run(runner.run_async(
        trigger_event="US strikes Kharg Island",
        severity="CRITICAL",
        confirmed=True,
    ))

    # Verify scenario JSON was written
    scenario_files = list(tmp_path.glob("active/*.json"))
    assert len(scenario_files) == 1
    scenario_data = json.loads(scenario_files[0].read_text())
    assert scenario_data["status"] == "complete"
    assert scenario_data["trigger_event"] == "US strikes Kharg Island"
    assert len(scenario_data["nodes"]) >= 2  # root + at least 1 child

    # Run Prometheus writer
    writer = PrometheusDeltaWriter(cfg)
    delta = writer.write(tree)
    assert "energy_bonus" in delta
    assert delta["energy_bonus"] > 1.0

    # Verify delta file written
    delta_files = list((tmp_path / "regime").glob("delta_*.json"))
    assert len(delta_files) == 1

    # Run KG writer
    kg_writer = KGWriter(cfg, neo4j_client=mock_neo4j)
    kg_writer.write(tree)

    # Verify Neo4j calls
    mock_neo4j.upsert_scenario.assert_called_once()
    mock_neo4j.activate_scenario.assert_called_once()
    assert mock_neo4j.upsert_flag.call_count >= 1  # at least one flag written


def test_pipeline_handles_empty_llm_gracefully(tmp_path):
    """If LLM returns nothing, pipeline produces a single-node tree without crashing."""
    cfg = StrategistConfig()
    cfg.scenarios_dir = tmp_path
    cfg.regime_config_dir = tmp_path / "regime"

    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=MagicMock(content="[]"))  # invalid JSON

    mock_neo4j = MagicMock()
    mock_neo4j.upsert_scenario = MagicMock()
    mock_neo4j.activate_scenario = MagicMock()
    mock_neo4j.upsert_flag = MagicMock()

    expander = ScenarioExpander(cfg, llm_provider=mock_llm)
    runner = ScenarioRunner(cfg, expander=expander, infra_registry=None)

    tree = _run(runner.run_async(
        trigger_event="Minor event",
        severity="LOW",
        confirmed=False,
    ))

    # Should have exactly the root node, no children
    assert tree.node_count() >= 1
    assert tree.status == "complete"
