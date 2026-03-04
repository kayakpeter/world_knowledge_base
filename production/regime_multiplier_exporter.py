"""
Regime Multiplier Exporter

Converts HMM posterior state probabilities into position-size multipliers
for the trading system. Written to:
  data/regime_multipliers/YYYYMMDD_multipliers.json

Trading system integration (ib_momentum_trader_v30b.py):
  Read this file at session start. For each candidate stock, look up the
  country of its primary market. Apply the multiplier to max_position_size.

  Example (US stock):
    multiplier = data["countries"]["United States"]["position_size_multiplier"]
    adjusted_size = base_size * multiplier

Multiplier formula:
  tranquil_prob * 1.0 + turbulent_prob * 0.6 + crisis_prob * 0.2

Rationale: in crisis, we still trade at 20% size (not zero) because
momentum stocks can gap up even in bad markets. But we size down hard.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "regime_multipliers"

# Weights for computing position-size multiplier from state probs
TRANQUIL_WEIGHT = 1.0   # Full size in calm markets
TURBULENT_WEIGHT = 0.6  # 60% size in turbulent markets
CRISIS_WEIGHT = 0.2     # 20% size in crisis — still trade but small


@dataclass
class RegimeProbs:
    country: str
    tranquil: float   # P(state=Tranquil | observations)
    turbulent: float  # P(state=Turbulent | observations)
    crisis: float     # P(state=Crisis | observations)

    def position_size_multiplier(self) -> float:
        """Weighted blend of state probabilities → position size scalar in [0.2, 1.0]."""
        return (
            self.tranquil * TRANQUIL_WEIGHT
            + self.turbulent * TURBULENT_WEIGHT
            + self.crisis * CRISIS_WEIGHT
        )


class RegimeMultiplierExporter:
    def __init__(self, output_dir: Path = OUTPUT_DIR) -> None:
        self.output_dir = output_dir

    def export(self, probs: list[RegimeProbs]) -> Path:
        """
        Write multipliers JSON and return the output path.

        Output schema:
        {
          "generated_at": "2026-03-04T...",
          "schema_version": "1.0",
          "multiplier_weights": {"tranquil": 1.0, "turbulent": 0.6, "crisis": 0.2},
          "countries": {
            "United States": {
              "tranquil_prob": 0.70,
              "turbulent_prob": 0.20,
              "crisis_prob": 0.10,
              "position_size_multiplier": 0.90,
              "regime": "Tranquil"
            }, ...
          }
        }
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        today = now.date().isoformat().replace("-", "")
        out_path = self.output_dir / f"{today}_multipliers.json"

        country_data: dict[str, dict] = {}
        for p in probs:
            mult = p.position_size_multiplier()
            if p.tranquil >= 0.5:
                regime = "Tranquil"
            elif p.crisis >= 0.5:
                regime = "Crisis"
            else:
                regime = "Turbulent"
            country_data[p.country] = {
                "tranquil_prob": round(p.tranquil, 4),
                "turbulent_prob": round(p.turbulent, 4),
                "crisis_prob": round(p.crisis, 4),
                "position_size_multiplier": round(mult, 4),
                "regime": regime,
            }

        output = {
            "generated_at": now.isoformat(),
            "schema_version": "1.0",
            "multiplier_weights": {
                "tranquil": TRANQUIL_WEIGHT,
                "turbulent": TURBULENT_WEIGHT,
                "crisis": CRISIS_WEIGHT,
            },
            "countries": country_data,
        }
        out_path.write_text(json.dumps(output, indent=2))
        logger.info("Regime multipliers exported: %s (%d countries)", out_path, len(probs))
        return out_path

    @staticmethod
    def load_latest(output_dir: Path = OUTPUT_DIR) -> dict | None:
        """Load the most recently exported multipliers file. Returns None if none exist."""
        if not output_dir.exists():
            return None
        files = sorted(output_dir.glob("*_multipliers.json"))
        return json.loads(files[-1].read_text()) if files else None
