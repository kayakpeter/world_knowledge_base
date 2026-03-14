# strategist/tests/test_infra_state.py
from unittest.mock import MagicMock
from strategist.infra_state import InfraStateRegistry, InfraNode, InfraStatus, CANONICAL_INFRA


def test_infra_node_defaults():
    node = InfraNode(
        infra_id="SUEZ_CANAL",
        name="Suez Canal",
        status=InfraStatus.OPERATIONAL,
        affected_sectors=["energy", "container", "agriculture"],
    )
    assert node.risk_level == 0.0
    assert node.confidence == 1.0
    assert node.country_iso3 == "UNK"


def _make_registry(mock_client=None):
    """Create InfraStateRegistry with a mock Neo4j client — no live DB needed."""
    if mock_client is None:
        mock_client = MagicMock()
    reg = InfraStateRegistry.__new__(InfraStateRegistry)
    reg._client = mock_client
    reg._cache = {}
    return reg


def test_get_status_from_cache():
    reg = _make_registry()
    reg._cache["SUEZ_CANAL"] = InfraNode(
        infra_id="SUEZ_CANAL",
        name="Suez Canal",
        status=InfraStatus.OPERATIONAL,
        affected_sectors=["energy"],
    )
    node = reg.get("SUEZ_CANAL")
    assert node.status == InfraStatus.OPERATIONAL
    # Should not call any Neo4j methods since we hit cache
    reg._client.upsert_flag.assert_not_called()


def test_update_status_writes_to_cache():
    mock_client = MagicMock()
    reg = _make_registry(mock_client)
    reg._cache["SUEZ_CANAL"] = InfraNode(
        infra_id="SUEZ_CANAL",
        name="Suez Canal",
        status=InfraStatus.OPERATIONAL,
        affected_sectors=["energy"],
    )
    reg.update_status("SUEZ_CANAL", InfraStatus.DEGRADED, confidence=0.8, source="AP")
    assert reg._cache["SUEZ_CANAL"].status == InfraStatus.DEGRADED
    assert reg._cache["SUEZ_CANAL"].confidence == 0.8


def test_update_status_calls_neo4j():
    mock_client = MagicMock()
    reg = _make_registry(mock_client)
    reg._cache["HORMUZ_STRAIT"] = InfraNode(
        infra_id="HORMUZ_STRAIT",
        name="Strait of Hormuz",
        status=InfraStatus.EFFECTIVE_CLOSURE,
        affected_sectors=["energy", "tanker"],
    )
    reg.update_status("HORMUZ_STRAIT", InfraStatus.CLOSED, confidence=0.95, source="USN")
    # Should persist to Neo4j
    mock_client.upsert_flag.assert_called_once()
    call_args = mock_client.upsert_flag.call_args
    # First positional arg should be flag name containing the infra_id
    assert "HORMUZ_STRAIT" in str(call_args)


def test_get_affected_sectors():
    reg = _make_registry()
    reg._cache["SUEZ_CANAL"] = InfraNode(
        infra_id="SUEZ_CANAL",
        name="Suez Canal",
        status=InfraStatus.OPERATIONAL,
        affected_sectors=["energy", "container", "agriculture"],
    )
    sectors = reg.get_affected_sectors("SUEZ_CANAL")
    assert "energy" in sectors
    assert "container" in sectors
    assert "agriculture" in sectors


def test_get_unknown_returns_none():
    reg = _make_registry()
    result = reg.get("NONEXISTENT_INFRA")
    assert result is None


def test_canonical_infra_has_eight_nodes():
    assert len(CANONICAL_INFRA) == 8


def test_canonical_infra_includes_key_nodes():
    ids = {d["infra_id"] for d in CANONICAL_INFRA}
    for expected_id in ["SUEZ_CANAL", "HORMUZ_STRAIT", "KHARG_ISLAND", "ABQAIQ",
                        "FUJAIRAH", "RAS_LAFFAN", "JEBEL_ALI", "BAGHDAD_EMBASSY"]:
        assert expected_id in ids


def test_seed_from_canonical():
    """seed_from_canonical() populates cache from CANONICAL_INFRA without Neo4j."""
    mock_client = MagicMock()
    reg = _make_registry(mock_client)
    reg.seed_from_canonical()
    assert len(reg._cache) == 8
    assert "HORMUZ_STRAIT" in reg._cache
    assert reg._cache["HORMUZ_STRAIT"].status == InfraStatus.EFFECTIVE_CLOSURE
