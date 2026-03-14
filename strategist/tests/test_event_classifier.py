# strategist/tests/test_event_classifier.py
from strategist.event_classifier import EventClassifier, TriggerEvent, INFRA_KEYWORD_MAP


def test_confirmed_kharg_strike_is_critical():
    clf = EventClassifier()
    result = clf.classify("US strikes Kharg Island oil terminal", confirmed=True)
    assert isinstance(result, TriggerEvent)
    assert result.severity == "CRITICAL"
    assert result.confirmed is True


def test_threat_only_is_high():
    clf = EventClassifier()
    result = clf.classify("Iran threatens to close Hormuz Strait", confirmed=False)
    assert result.severity == "HIGH"
    assert result.confirmed is False


def test_kharg_maps_to_infra():
    clf = EventClassifier()
    result = clf.classify("Kharg Island attacked", confirmed=True)
    assert "KHARG_ISLAND" in result.infra_ids


def test_suez_maps_to_infra():
    clf = EventClassifier()
    result = clf.classify("Missile strike on Suez Canal zone", confirmed=True)
    assert "SUEZ_CANAL" in result.infra_ids


def test_hormuz_maps_to_infra():
    clf = EventClassifier()
    result = clf.classify("Hormuz closure announced by IRGC", confirmed=True)
    assert "HORMUZ_STRAIT" in result.infra_ids


def test_fujairah_maps_to_infra():
    clf = EventClassifier()
    result = clf.classify("Smoke seen over Fujairah terminal", confirmed=False)
    assert "FUJAIRAH" in result.infra_ids


def test_unrelated_event_has_no_infra():
    clf = EventClassifier()
    result = clf.classify("Federal Reserve raises interest rates", confirmed=False)
    assert result.infra_ids == []
    assert result.severity == "LOW"


def test_event_text_stored():
    clf = EventClassifier()
    text = "Iran fires ballistic missiles at US forces in Bahrain"
    result = clf.classify(text, confirmed=True)
    assert result.event_text == text


def test_keyword_map_has_required_nodes():
    for key in ["SUEZ_CANAL", "HORMUZ_STRAIT", "KHARG_ISLAND", "ABQAIQ",
                "FUJAIRAH", "RAS_LAFFAN", "JEBEL_ALI"]:
        assert key in INFRA_KEYWORD_MAP


def test_multiple_infra_detected():
    clf = EventClassifier()
    result = clf.classify("Strike on Abqaiq and Kharg Island simultaneously", confirmed=True)
    assert "ABQAIQ" in result.infra_ids
    assert "KHARG_ISLAND" in result.infra_ids
