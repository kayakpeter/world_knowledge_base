# strategist/expander.py
from __future__ import annotations
import json
import logging
import uuid
from typing import Optional

from strategist.config import StrategistConfig
from strategist.schema import ScenarioNode, ScenarioTree, SectorImpact, NodeStatus

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a geopolitical scenario analyst. Given a triggering event and its immediate context, generate plausible near-term (72-hour window) consequences as a scenario tree.

For each branch you generate:
- description: brief description of the consequence (one sentence)
- probability: float 0-1 (probability this branch occurs given parent)
- sector_impacts: list of market sector effects (may be empty)
- infrastructure_effects: list of infrastructure status changes (may be empty)
- time_offset_hours: estimated hours after trigger when this branch materializes

Rules:
- Generate 2-4 branches maximum
- Probabilities do NOT need to sum to 1 (branches are not mutually exclusive)
- Focus on high-impact, plausible outcomes
- Sector impacts: sector must be one of: energy, tanker, bulk, agriculture, container, defense, currency, sovereign, insurance
- Infrastructure effects: new_status must be one of: OPERATIONAL, AT_RISK, DEGRADED, STRUCK_MILITARY, STRUCK_OIL_INFRA, CLOSED, EFFECTIVE_CLOSURE, SMOKE_REPORTED

Respond ONLY with valid JSON in this exact format:
{
  "branches": [
    {
      "description": "...",
      "probability": 0.XX,
      "sector_impacts": [
        {"sector": "energy", "direction": "UP", "magnitude": "CRITICAL", "magnitude_pct": 40.0, "tickers": ["FRO"], "notes": "..."}
      ],
      "infrastructure_effects": [
        {"infra_id": "HORMUZ_STRAIT", "new_status": "CLOSED", "confidence": 0.9}
      ],
      "time_offset_hours": 6.0
    }
  ]
}"""

USER_PROMPT_TEMPLATE = """Trigger event: {trigger_event}
Severity: {severity}
Confirmed: {confirmed}
Current node description: {node_description}
Current joint probability: {joint_probability:.4f}
Depth in tree: {depth}

Generate the next branches from this node."""


class ScenarioExpander:
    """
    Expands a ScenarioNode by calling Ollama to generate child branches.
    """

    def __init__(self, config: StrategistConfig, llm_provider=None) -> None:
        self._cfg = config
        if llm_provider is not None:
            self._llm = llm_provider
        else:
            # Import here to avoid hard dep at module level
            from processing.llm_interface import LocalModelProvider
            self._llm = LocalModelProvider(
                base_url=config.ollama_url,
                model=config.ollama_model,
            )

    _VALID_STATUSES = {
        "OPERATIONAL", "AT_RISK", "DEGRADED", "STRUCK_MILITARY",
        "STRUCK_OIL_INFRA", "CLOSED", "EFFECTIVE_CLOSURE", "SMOKE_REPORTED", "UNKNOWN",
    }

    def _validate_infra_effects(self, raw_effects: list) -> list[dict]:
        """Keep only well-formed infrastructure effect dicts."""
        valid = []
        for effect in raw_effects:
            if not isinstance(effect, dict):
                continue
            if "infra_id" not in effect or "new_status" not in effect:
                continue
            if effect["new_status"] not in self._VALID_STATUSES:
                logger.warning("Skipping infra effect with unknown status: %s", effect.get("new_status"))
                continue
            valid.append(effect)
        return valid

    async def expand(
        self,
        node: ScenarioNode,
        tree: ScenarioTree,
        *,
        severity: str,
    ) -> list[ScenarioNode]:
        """
        Call Ollama to generate child branches for `node`.
        Returns list of child ScenarioNodes (not yet added to tree).
        Returns [] on any LLM or parse error.
        """
        user_prompt = USER_PROMPT_TEMPLATE.format(
            trigger_event=tree.trigger_event,
            severity=severity,
            confirmed=tree.confirmed,
            node_description=node.description,
            joint_probability=node.joint_probability,
            depth=node.depth,
        )

        try:
            response = await self._llm.complete(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.3,
                max_tokens=1500,
            )
            raw_content = response.content
        except Exception:
            logger.exception("LLM call failed during expand for node=%s", node.node_id)
            return []

        return self._parse_response(raw_content, node, severity)

    def _parse_response(
        self,
        raw: str,
        parent: ScenarioNode,
        severity: str,
    ) -> list[ScenarioNode]:
        """Parse LLM JSON response into child ScenarioNodes."""
        try:
            data = json.loads(raw)
            items = data.get("branches", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Failed to parse LLM response as JSON for node=%s", parent.node_id)
            return []

        # Normalize probabilities if they sum significantly > 1.0.
        # Branches are not mutually exclusive, so a sum slightly above 1.0
        # (e.g. 1.05) is acceptable and preserved as-is. Only normalize when
        # the sum materially exceeds 1.0 (threshold: 1.10).
        total = sum(max(0.0, float(i.get("probability", 0.0))) for i in items)
        if total == 0.0 and items:
            logger.warning("All branch probabilities are zero for node=%s — treating as parse error", parent.node_id)
            return []
        _NORMALIZATION_THRESHOLD = 1.10
        scale = 1.0 / total if total > _NORMALIZATION_THRESHOLD else 1.0

        children: list[ScenarioNode] = []
        for item in items:
            try:
                prob = max(0.0, float(item.get("probability", 0.0))) * scale

                # Parse sector impacts
                sector_impacts = []
                for si_dict in item.get("sector_impacts", []):
                    try:
                        sector_impacts.append(SectorImpact.from_dict(si_dict))
                    except Exception:
                        pass  # skip malformed sector impacts

                child = ScenarioNode(
                    node_id=uuid.uuid4().hex[:8],
                    description=str(item.get("description", "Unknown")),
                    branch_probability=prob,
                    parent_id=parent.node_id,
                    depth=parent.depth + 1,
                    parent_joint_probability=parent.joint_probability,
                    sector_impacts=sector_impacts,
                    infrastructure_effects=self._validate_infra_effects(item.get("infrastructure_effects", [])),
                    time_offset_hours=float(item.get("time_offset_hours", 0.0)),
                )
                children.append(child)
            except Exception:
                logger.warning("Skipping malformed branch item: %s", item)
                continue

        return children
