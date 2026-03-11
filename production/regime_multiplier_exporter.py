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
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Override state → (tranquil_prob, turbulent_prob, crisis_prob)
_OVERRIDE_PROBS = {
    "S0_Tranquil": (1.0, 0.0, 0.0),
    "S1_Turbulent": (0.0, 1.0, 0.0),
    "S2_Crisis":    (0.0, 0.0, 1.0),
}

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

    @staticmethod
    def probs_from_hmm_states(states: dict[str, dict]) -> list["RegimeProbs"]:
        """
        Build RegimeProbs list from an hmm_states dict (as loaded from hmm_states.json).

        If a country entry has a non-null override_state that hasn't expired,
        the override probabilities (1.0 for the overridden state) are used instead
        of the HMM posteriors. The final entry records which source was used.
        """
        try:
            # Import lazily to avoid circular deps when used standalone
            _override_path = Path(__file__).parent.parent / "processing" / "geo_override.py"
            sys.path.insert(0, str(_override_path.parent.parent))
            from processing.geo_override import get_active_overrides, is_expired
            active_overrides = get_active_overrides()
        except Exception as exc:
            logger.warning("Could not load geo overrides: %s — using HMM states only", exc)
            active_overrides = {}

        probs: list[RegimeProbs] = []
        for country, entry in states.items():
            override_state = entry.get("override_state")
            override_active = (
                override_state is not None
                and override_state in _OVERRIDE_PROBS
                and country in active_overrides
            )

            if override_active:
                t, tu, c = _OVERRIDE_PROBS[override_state]
                logger.info(
                    "  %-20s → %s [OVERRIDE by %s, expires %s]",
                    country, override_state,
                    entry.get("override_set_by", "?"),
                    (entry.get("override_expires") or "?")[:10],
                )
            else:
                # Use HMM state posteriors
                state_probs = entry.get("state_probs", {})
                t = float(state_probs.get("Tranquil", 0.60))
                tu = float(state_probs.get("Turbulent", 0.30))
                c = float(state_probs.get("Crisis", 0.10))

            probs.append(RegimeProbs(country=country, tranquil=t, turbulent=tu, crisis=c))
        return probs

    def export_from_states(self, states_file: Path) -> Path:
        """Load hmm_states.json, apply overrides, and export multipliers."""
        states = json.loads(states_file.read_text())
        probs = self.probs_from_hmm_states(states)
        return self.export(probs)

    def export_from_snapshot(self, snapshot_dir: Path) -> Path:
        """
        Export multipliers from a snapshot directory.

        Prefers fused_hmm_states.json (geo-fused) over hmm_states.json (raw HMM).
        Manual overrides (geo_override.py) still take priority over both.
        """
        fused_path = snapshot_dir / "fused_hmm_states.json"
        raw_path   = snapshot_dir / "hmm_states.json"

        if fused_path.exists():
            logger.info("Using fused HMM states: %s", fused_path)
            states = json.loads(fused_path.read_text())
            # Promote fused_state_probs into state_probs so probs_from_hmm_states uses them
            for entry in states.values():
                if "fused_state_probs" in entry:
                    entry["state_probs"] = entry["fused_state_probs"]
        elif raw_path.exists():
            logger.warning(
                "fused_hmm_states.json not found in %s — using raw HMM states", snapshot_dir
            )
            states = json.loads(raw_path.read_text())
        else:
            raise FileNotFoundError(f"No HMM states file found in {snapshot_dir}")

        probs = self.probs_from_hmm_states(states)
        return self.export(probs)

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
