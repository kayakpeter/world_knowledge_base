import json
import tempfile
from pathlib import Path

import pytest

from production.regime_multiplier_exporter import RegimeMultiplierExporter, RegimeProbs


def test_compute_multiplier():
    r = RegimeProbs(country="United States", tranquil=0.6, turbulent=0.3, crisis=0.1)
    mult = r.position_size_multiplier()
    expected = 0.6 * 1.0 + 0.3 * 0.6 + 0.1 * 0.2
    assert abs(mult - expected) < 1e-6


def test_multiplier_bounded():
    # Extreme crisis: all weight on crisis state
    r = RegimeProbs(country="Turkey", tranquil=0.0, turbulent=0.0, crisis=1.0)
    assert r.position_size_multiplier() == pytest.approx(0.2)
    # Full tranquility
    r2 = RegimeProbs(country="Germany", tranquil=1.0, turbulent=0.0, crisis=0.0)
    assert r2.position_size_multiplier() == pytest.approx(1.0)


def test_export_creates_json():
    with tempfile.TemporaryDirectory() as td:
        exporter = RegimeMultiplierExporter(output_dir=Path(td))
        probs = [
            RegimeProbs("United States", 0.7, 0.2, 0.1),
            RegimeProbs("China", 0.4, 0.4, 0.2),
        ]
        path = exporter.export(probs)
        data = json.loads(path.read_text())
        assert "generated_at" in data
        assert "countries" in data
        us = data["countries"]["United States"]
        assert "position_size_multiplier" in us
        assert 0.0 <= us["position_size_multiplier"] <= 1.0
