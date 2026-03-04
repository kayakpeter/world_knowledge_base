"""
Dynamic Scenario Generator — Ollama-powered open-ended scenario expansion.

Takes current crack detector output and asks local Ollama to generate new scenarios
beyond the 50 pre-coded ones. These are stored as JSON for inspection and used
in the current run's scenario analysis.

Model: qwen2.5-coder:32b at http://localhost:11434/v1
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import openai

from processing.scenario_engine import Scenario

logger = logging.getLogger(__name__)

DYNAMIC_DIR = Path(__file__).parent.parent / "data" / "dynamic_scenarios"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5-coder:32b"

SYSTEM_PROMPT = """\
You are a sovereign economic analyst. Given a list of active crack patterns,
generate new economic scenarios that are NOT already covered by standard
baseline/recession scenarios.

Output ONLY a JSON array of scenario objects. No preamble. No explanation.

Each object must have exactly these fields:
  title: str
  category: one of ["baseline","likely","moderate","unlikely","black_swan"]
  probability_12m: float (0.0-1.0)
  description: str (one sentence)
  primary_channel: one of ["trade","financial","commodity","political","technology"]
  affected_countries: list[str]
  affected_stats: list[str]
  shock_magnitude: float (% impact, negative = contractionary)
  causal_chain: list[str] (3-5 ordered events)
  severity: one of ["low","medium","high","critical"]
"""


def parse_scenario_json(content: str) -> list[Scenario]:
    """Parse LLM JSON output into Scenario objects. Returns [] on failure."""
    try:
        raw = json.loads(content)
        scenarios = []
        for i, item in enumerate(raw, start=51):  # start IDs after the 50 static ones
            try:
                scenarios.append(Scenario(
                    scenario_id=i,
                    title=item["title"],
                    category=item.get("category", "moderate"),
                    probability_12m=float(item.get("probability_12m", 0.1)),
                    description=item.get("description", ""),
                    primary_channel=item.get("primary_channel", "financial"),
                    affected_countries=item.get("affected_countries", []),
                    affected_stats=item.get("affected_stats", []),
                    shock_magnitude=float(item.get("shock_magnitude", 0.0)),
                    causal_chain=item.get("causal_chain", []),
                    severity=item.get("severity", "medium"),
                ))
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Skipping malformed scenario item {i}: {e}")
        return scenarios
    except json.JSONDecodeError as e:
        logger.warning(f"LLM response was not valid JSON: {e}")
        return []


class DynamicScenarioGenerator:
    def __init__(
        self,
        max_scenarios: int = 10,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.max_scenarios = max_scenarios
        self.model = model
        self.base_url = base_url

    def generate(self, active_crack_summaries: list[str]) -> list[Scenario]:
        """
        Generate new scenarios from active crack pattern summaries.
        Returns list of Scenario objects (IDs starting at 51).
        """
        if not active_crack_summaries:
            logger.info("No active crack patterns — skipping dynamic scenario generation")
            return []

        user_content = (
            f"Generate up to {self.max_scenarios} new economic scenarios based on "
            f"these active crack patterns:\n\n"
            + "\n".join(f"- {s}" for s in active_crack_summaries)
        )

        client = openai.OpenAI(base_url=self.base_url, api_key="ollama")
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            content = response.choices[0].message.content
            scenarios = parse_scenario_json(content)
            logger.info(f"Dynamic scenario generator: produced {len(scenarios)} scenarios")
            return scenarios
        except Exception as e:
            logger.error(f"Dynamic scenario generation failed: {e}")
            return []

    def save(self, scenarios: list[Scenario]) -> Path:
        """Save generated scenarios to dated JSON file."""
        DYNAMIC_DIR.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat().replace("-", "")
        out_path = DYNAMIC_DIR / f"{today}_scenarios.json"
        data = [
            {
                "scenario_id": s.scenario_id,
                "title": s.title,
                "category": s.category,
                "probability_12m": s.probability_12m,
                "description": s.description,
                "primary_channel": s.primary_channel,
                "affected_countries": s.affected_countries,
                "affected_stats": s.affected_stats,
                "shock_magnitude": s.shock_magnitude,
                "causal_chain": s.causal_chain,
                "severity": s.severity,
                "source": "dynamic_ollama",
            }
            for s in scenarios
        ]
        out_path.write_text(json.dumps(data, indent=2))
        logger.info(f"Dynamic scenarios saved: {out_path} ({len(scenarios)} scenarios)")
        return out_path
