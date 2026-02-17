"""
News Intelligence Processor — Converts raw headlines and news events into
structured knowledge graph updates.

This is where the qualitative meets the quantitative. Every headline contains:
- Actors (who is doing/saying something)
- Intent signals (what they want to achieve)
- Affected statistics (which numbers will move)
- Relationship changes (who is aligning/opposing whom)
- Scenario probability shifts (which scenarios became more/less likely)

The LLM processes each headline and outputs structured JSON that the graph
builder can ingest. This is the "ears and eyes" of the system — the stats
are the body, the actors are the brain, and the news feed is the nervous system.

Pipeline:
  Raw headline → LLM extraction → Structured event → Graph update
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config.actors import ACTOR_REGISTRY, Actor

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEvent:
    """Structured extraction from a single news headline or article."""
    event_id: str
    headline: str
    timestamp: str
    source: str

    # Actor extraction
    actors_mentioned: list[str]         # actor_ids from registry
    new_actors: list[dict]              # actors not yet in registry

    # Intent and sentiment
    primary_intent: str                 # "negotiate", "threaten", "signal", "announce", "escalate", "de-escalate"
    sentiment: str                      # "positive", "negative", "neutral", "mixed"
    urgency: str                        # "routine", "notable", "urgent", "critical"

    # Economic impact mapping
    affected_countries: list[str]       # country ISO3 codes
    affected_stats: list[str]           # stat_name references
    stat_direction: dict[str, str]      # stat_name → "up", "down", "uncertain"
    estimated_magnitude: str            # "negligible", "minor", "moderate", "major", "severe"

    # Scenario linkage
    scenario_ids_affected: list[int]    # which of the 50 scenarios does this affect
    probability_shift: dict[int, float] # scenario_id → delta probability (-1.0 to +1.0)

    # Relationship updates
    relationship_updates: list[dict]    # [{source, target, new_type, new_strength, context}]

    # Graph integration
    causal_chain: list[str]             # predicted sequence of consequences
    confidence: float                   # 0.0 to 1.0 in the extraction quality


@dataclass
class CommunicationEdge:
    """
    An edge in the graph representing a communication or signal between actors.

    Unlike trade or contagion edges (which are structural), communication edges
    are temporal — they have a timestamp and a decay rate. A threat made yesterday
    has more weight than one made six months ago.
    """
    source_actor_id: str
    target_actor_id: str
    event_id: str
    communication_type: str     # "threat", "offer", "demand", "concession", "signal", "announcement"
    topic: str                  # brief description
    sentiment: float            # -1.0 (hostile) to +1.0 (cooperative)
    timestamp: str              # ISO datetime
    decay_half_life_days: int = 30  # how quickly the signal fades
    affected_stats: list[str] = field(default_factory=list)


# ─── LLM Prompt Templates ───────────────────────────────────────────────────

NEWS_EXTRACTION_SYSTEM = """You are a geopolitical intelligence analyst for a sovereign risk knowledge graph.
Given a news headline and optional context, extract structured intelligence.

You have access to these known actors (by actor_id):
{actor_list}

You have access to these economic statistics (by stat_name):
{stat_list}

You have access to these scenarios (by scenario_id):
{scenario_list}

Respond with JSON matching this schema:
{{
    "actors_mentioned": ["actor_id_1", "actor_id_2"],
    "new_actors": [
        {{"name": "...", "role": "...", "country_iso3": "...", "context": "..."}}
    ],
    "primary_intent": "negotiate|threaten|signal|announce|escalate|de-escalate",
    "sentiment": "positive|negative|neutral|mixed",
    "urgency": "routine|notable|urgent|critical",
    "affected_countries": ["USA", "CAN"],
    "affected_stats": ["effective_tariff", "fdi_inflow"],
    "stat_direction": {{"effective_tariff": "up", "fdi_inflow": "down"}},
    "estimated_magnitude": "negligible|minor|moderate|major|severe",
    "scenario_ids_affected": [4, 20],
    "probability_shift": {{4: 0.05, 20: -0.02}},
    "relationship_updates": [
        {{"source": "USA_POTUS", "target": "CAN_PM", "type": "negotiating", "strength": 0.6, "context": "..."}}
    ],
    "causal_chain": ["trade_negotiator_appointed", "usmca_talks_accelerate", "tariff_resolution_closer"],
    "confidence": 0.85
}}"""


DAILY_BRIEFING_SYSTEM = """You are preparing a daily intelligence briefing for {persona}.
Synthesize the following extracted events into a cohesive narrative that highlights:
1. The most consequential developments and their economic implications
2. Relationship shifts between key actors
3. Scenario probability changes that require attention
4. Recommended actions or monitoring priorities

Write in the voice appropriate for the persona. Be specific about which statistics
and scenarios are affected. Do not hedge — give your best assessment.

Events to synthesize:
{events_json}

Current actor relationships:
{relationships_json}

Current scenario probabilities:
{scenario_probs_json}"""


# ─── Sample Headlines (from Peter's input) ──────────────────────────────────

SAMPLE_HEADLINES: list[dict] = [
    {
        "headline": "Prime Minister Carney announces new Chief Trade Negotiator to the United States",
        "source": "pm.gc.ca",
        "timestamp": "2026-02-15",
        "context": "Canada preparing for USMCA mandatory review in July 2026.",
    },
    {
        "headline": "The US Secretary of State is on a two-day trip to Slovakia and Hungary, whose leaders have close ties with Trump. Both countries are being courted to cut their reliance on Russia for energy in favor of US alternatives.",
        "source": "Reuters",
        "timestamp": "2026-02-15",
        "context": "Rubio visiting CEE allies to expand US energy exports and reduce Russian influence.",
    },
    {
        "headline": "Iran ready to discuss compromises to reach nuclear deal, minister tells BBC in Tehran",
        "source": "BBC",
        "timestamp": "2026-02-15",
        "context": "Araghchi signaling willingness to negotiate amid maximum pressure campaign.",
    },
    {
        "headline": "Was Navalny poisoning by frog toxin meant to send a message?",
        "source": "Investigative journalism",
        "timestamp": "2026-02-15",
        "context": "Russia internal politics. Signals about Putin regime's methods and stability.",
    },
    {
        "headline": "Ottawans in Cuba fly home early as fuel shortages worsen",
        "source": "CBC",
        "timestamp": "2026-02-15",
        "context": "Cuba energy crisis. Caribbean instability. Venezuelan oil supply disruption effects.",
    },
    {
        "headline": "Foreign Minister Anand leads Canada at Munich conference, focuses on Arctic security",
        "source": "Global News",
        "timestamp": "2026-02-15",
        "context": "Munich Security Conference. Arctic sovereignty. NATO coordination.",
    },
    {
        "headline": "US forces board tanker in Indian Ocean that fled Trump's Venezuela blockade",
        "source": "AP",
        "timestamp": "2026-02-15",
        "context": "Military enforcement of Venezuela energy operation. Shadow fleet interdiction.",
    },
    {
        "headline": "Trump and Netanyahu align on Iran pressure but split on endgame",
        "source": "Reuters",
        "timestamp": "2026-02-15",
        "context": "US-Israel strategic alignment with divergent goals on Iran's nuclear program.",
    },
    {
        "headline": "Europe bashing: EU's top diplomat rejects US talk of 'civilisational erasure'",
        "source": "EUObserver",
        "timestamp": "2026-02-15",
        "context": "Kallas pushing back on US rhetoric. Transatlantic tension.",
    },
    {
        "headline": "'Constant rain has made farming more difficult'",
        "source": "BBC",
        "timestamp": "2026-02-15",
        "context": "Climate impact on agriculture. Food price implications.",
    },
]


class NewsIntelligenceProcessor:
    """
    Processes news headlines into structured graph updates.

    In dev mode: Uses pre-analyzed events or Claude API.
    In production: Processes real-time news feeds continuously.
    """

    def __init__(self, llm_provider=None):
        self._llm = llm_provider
        self._events: list[ExtractedEvent] = []
        self._actor_lookup = {a.actor_id: a for a in ACTOR_REGISTRY}

    def _build_actor_list_for_prompt(self) -> str:
        """Format actor registry for LLM context."""
        lines = []
        for a in ACTOR_REGISTRY:
            lines.append(f"  {a.actor_id}: {a.name} — {a.role} ({a.country})")
        return "\n".join(lines)

    async def process_headline(
        self,
        headline: str,
        source: str = "",
        timestamp: str = "",
        context: str = "",
    ) -> ExtractedEvent:
        """
        Process a single headline through LLM extraction.

        Returns structured event data for graph integration.
        """
        from config.settings import FULL_STAT_REGISTRY
        from processing.scenario_engine import SCENARIO_REGISTRY

        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()

        event_id = f"evt_{hash(headline) % 10**8:08d}"

        if self._llm is None:
            logger.warning("No LLM provider — returning skeleton event for: %s", headline[:60])
            return self._skeleton_event(event_id, headline, source, timestamp)

        # Build context for the LLM
        actor_list = self._build_actor_list_for_prompt()
        stat_list = ", ".join(s.name for s in FULL_STAT_REGISTRY[:50])  # truncate for token budget
        scenario_list = "\n".join(
            f"  {s.scenario_id}: {s.title} (P={s.probability_12m:.0%})"
            for s in SCENARIO_REGISTRY[:20]
        )

        system = NEWS_EXTRACTION_SYSTEM.format(
            actor_list=actor_list,
            stat_list=stat_list,
            scenario_list=scenario_list,
        )

        user_prompt = f"Headline: {headline}"
        if context:
            user_prompt += f"\nContext: {context}"
        if source:
            user_prompt += f"\nSource: {source}"

        result = await self._llm.complete_json(system, user_prompt)

        if "error" in result:
            logger.error("LLM extraction failed: %s", result["error"])
            return self._skeleton_event(event_id, headline, source, timestamp)

        event = ExtractedEvent(
            event_id=event_id,
            headline=headline,
            timestamp=timestamp,
            source=source,
            actors_mentioned=result.get("actors_mentioned", []),
            new_actors=result.get("new_actors", []),
            primary_intent=result.get("primary_intent", "signal"),
            sentiment=result.get("sentiment", "neutral"),
            urgency=result.get("urgency", "routine"),
            affected_countries=result.get("affected_countries", []),
            affected_stats=result.get("affected_stats", []),
            stat_direction=result.get("stat_direction", {}),
            estimated_magnitude=result.get("estimated_magnitude", "minor"),
            scenario_ids_affected=result.get("scenario_ids_affected", []),
            probability_shift=result.get("probability_shift", {}),
            relationship_updates=result.get("relationship_updates", []),
            causal_chain=result.get("causal_chain", []),
            confidence=result.get("confidence", 0.5),
        )

        self._events.append(event)
        return event

    async def process_batch(self, headlines: list[dict]) -> list[ExtractedEvent]:
        """Process multiple headlines."""
        events = []
        for h in headlines:
            event = await self.process_headline(
                headline=h["headline"],
                source=h.get("source", ""),
                timestamp=h.get("timestamp", ""),
                context=h.get("context", ""),
            )
            events.append(event)
        return events

    def _skeleton_event(
        self,
        event_id: str,
        headline: str,
        source: str,
        timestamp: str,
    ) -> ExtractedEvent:
        """Create a minimal event when LLM is not available."""
        return ExtractedEvent(
            event_id=event_id,
            headline=headline,
            timestamp=timestamp,
            source=source,
            actors_mentioned=[],
            new_actors=[],
            primary_intent="signal",
            sentiment="neutral",
            urgency="routine",
            affected_countries=[],
            affected_stats=[],
            stat_direction={},
            estimated_magnitude="minor",
            scenario_ids_affected=[],
            probability_shift={},
            relationship_updates=[],
            causal_chain=[],
            confidence=0.0,
        )

    def generate_graph_updates(self, events: list[ExtractedEvent]) -> dict:
        """
        Convert extracted events into graph operations.

        Returns a dict of operations the graph builder can apply:
        - node_updates: stat nodes to update
        - edge_updates: relationship edges to add/modify
        - scenario_shifts: probability adjustments
        - actor_additions: new actors to register
        """
        node_updates = []
        edge_updates = []
        scenario_shifts = {}
        actor_additions = []

        for event in events:
            # Accumulate scenario probability shifts
            for sid_str, delta in event.probability_shift.items():
                sid = int(sid_str) if isinstance(sid_str, str) else sid_str
                scenario_shifts[sid] = scenario_shifts.get(sid, 0.0) + delta

            # Communication edges between mentioned actors
            if len(event.actors_mentioned) >= 2:
                for i in range(len(event.actors_mentioned)):
                    for j in range(i + 1, len(event.actors_mentioned)):
                        edge_updates.append({
                            "source": event.actors_mentioned[i],
                            "target": event.actors_mentioned[j],
                            "edge_type": "COMMUNICATES",
                            "event_id": event.event_id,
                            "sentiment": event.sentiment,
                            "topic": event.headline[:100],
                            "timestamp": event.timestamp,
                        })

            # Relationship updates from LLM extraction
            for rel in event.relationship_updates:
                edge_updates.append({
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "edge_type": "RELATIONSHIP",
                    "relationship_type": rel.get("type", ""),
                    "strength": rel.get("strength", 0.5),
                    "context": rel.get("context", ""),
                    "timestamp": event.timestamp,
                })

            # New actors discovered in news
            actor_additions.extend(event.new_actors)

            # Stat direction signals (for crack detection weighting)
            for stat_name, direction in event.stat_direction.items():
                for country_iso3 in event.affected_countries:
                    node_updates.append({
                        "node_id": f"{country_iso3}_{stat_name}",
                        "signal_direction": direction,
                        "signal_source": event.event_id,
                        "signal_confidence": event.confidence,
                        "timestamp": event.timestamp,
                    })

        return {
            "node_updates": node_updates,
            "edge_updates": edge_updates,
            "scenario_shifts": scenario_shifts,
            "actor_additions": actor_additions,
            "total_events_processed": len(events),
        }

    def print_event_summary(self, event: ExtractedEvent) -> None:
        """Print a formatted summary of an extracted event."""
        urgency_map = {"routine": "○", "notable": "◐", "urgent": "●", "critical": "◉"}
        sentiment_map = {"positive": "↑", "negative": "↓", "neutral": "→", "mixed": "↕"}

        logger.info(
            "  %s %s [%s] %s",
            urgency_map.get(event.urgency, "?"),
            sentiment_map.get(event.sentiment, "?"),
            event.primary_intent.upper(),
            event.headline[:80],
        )

        if event.actors_mentioned:
            names = []
            for aid in event.actors_mentioned:
                actor = self._actor_lookup.get(aid)
                names.append(f"{actor.name} ({actor.role})" if actor else aid)
            logger.info("    Actors: %s", ", ".join(names))

        if event.affected_stats:
            directions = [f"{s} {event.stat_direction.get(s, '?')}" for s in event.affected_stats]
            logger.info("    Stats: %s", ", ".join(directions))

        if event.scenario_ids_affected:
            from processing.scenario_engine import SCENARIO_REGISTRY
            scenario_lookup = {s.scenario_id: s for s in SCENARIO_REGISTRY}
            shifts = []
            for sid_key, delta in event.probability_shift.items():
                sid = int(sid_key) if isinstance(sid_key, str) else sid_key
                sc = scenario_lookup.get(sid)
                name = sc.title if sc else f"#{sid}"
                sign = "+" if delta > 0 else ""
                shifts.append(f"{name} ({sign}{delta:.0%})")
            logger.info("    Scenarios: %s", ", ".join(shifts))

        if event.causal_chain:
            logger.info("    Chain: %s", " → ".join(event.causal_chain))


# ─── Pre-analyzed events for demo without LLM ───────────────────────────────

DEMO_EVENTS: list[ExtractedEvent] = [
    ExtractedEvent(
        event_id="evt_00000001",
        headline="Prime Minister Carney announces new Chief Trade Negotiator to the United States",
        timestamp="2026-02-15", source="pm.gc.ca",
        actors_mentioned=["CAN_PM", "CAN_TRADE_NEG"],
        new_actors=[],
        primary_intent="announce",
        sentiment="neutral", urgency="notable",
        affected_countries=["CAN", "USA"],
        affected_stats=["effective_tariff", "fdi_inflow", "rule_of_origin_pct"],
        stat_direction={"effective_tariff": "uncertain", "fdi_inflow": "uncertain", "rule_of_origin_pct": "uncertain"},
        estimated_magnitude="moderate",
        scenario_ids_affected=[4, 20],
        probability_shift={4: 0.05, 20: -0.03},
        relationship_updates=[
            {"source": "CAN_PM", "target": "USA_USTR", "type": "negotiating", "strength": 0.7,
             "context": "Carney signaling seriousness about USMCA by dedicating senior negotiator."}
        ],
        causal_chain=["negotiator_appointed", "usmca_talks_formalize", "bilateral_positions_clarify"],
        confidence=0.90,
    ),
    ExtractedEvent(
        event_id="evt_00000002",
        headline="US Secretary of State on two-day trip to Slovakia and Hungary — courting energy switch from Russia",
        timestamp="2026-02-15", source="Reuters",
        actors_mentioned=["USA_SEC_STATE"],
        new_actors=[
            {"name": "Robert Fico", "role": "Prime Minister of Slovakia", "country_iso3": "SVK",
             "context": "Close ties with Trump. Russian energy dependent."},
            {"name": "Viktor Orbán", "role": "Prime Minister of Hungary", "country_iso3": "HUN",
             "context": "Close ties with Trump. Russian energy dependent."},
        ],
        primary_intent="negotiate",
        sentiment="positive", urgency="notable",
        affected_countries=["USA", "RUS"],
        affected_stats=["crude_output", "export_velocity", "energy_cpi"],
        stat_direction={"crude_output": "up", "export_velocity": "up", "energy_cpi": "down"},
        estimated_magnitude="moderate",
        scenario_ids_affected=[9, 25],
        probability_shift={9: 0.02, 25: -0.02},
        relationship_updates=[
            {"source": "USA_SEC_STATE", "target": "RUS_PRES", "type": "adversary", "strength": 0.85,
             "context": "Rubio actively undermining Russian energy influence in CEE."}
        ],
        causal_chain=["rubio_visits_cee", "energy_deals_proposed", "russian_gas_share_drops", "us_lng_exports_rise"],
        confidence=0.85,
    ),
    ExtractedEvent(
        event_id="evt_00000003",
        headline="Iran ready to discuss compromises to reach nuclear deal, minister tells BBC in Tehran",
        timestamp="2026-02-15", source="BBC",
        actors_mentioned=["IRN_FM"],
        new_actors=[],
        primary_intent="de-escalate",
        sentiment="positive", urgency="urgent",
        affected_countries=["IRN", "USA", "ISR", "SAU"],
        affected_stats=["crude_output", "wti_brent_spread", "political_instability", "currency_volatility"],
        stat_direction={"crude_output": "up", "wti_brent_spread": "down", "political_instability": "down"},
        estimated_magnitude="major",
        scenario_ids_affected=[9],
        probability_shift={9: 0.05},
        relationship_updates=[
            {"source": "IRN_FM", "target": "USA_POTUS", "type": "negotiating", "strength": 0.3,
             "context": "Iran opening door to talks. Shifts from adversarial to potentially negotiating."},
        ],
        causal_chain=["iran_signals_compromise", "back_channel_opens", "sanctions_relief_possible", "oil_supply_expands"],
        confidence=0.80,
    ),
    ExtractedEvent(
        event_id="evt_00000004",
        headline="US forces board tanker in Indian Ocean that fled Trump's Venezuela blockade",
        timestamp="2026-02-15", source="AP",
        actors_mentioned=["USA_POTUS", "USA_SEC_DEF"],
        new_actors=[],
        primary_intent="escalate",
        sentiment="negative", urgency="urgent",
        affected_countries=["USA"],
        affected_stats=["crude_output", "wti_brent_spread", "shipping_cost_index", "political_instability"],
        stat_direction={"crude_output": "uncertain", "wti_brent_spread": "down", "shipping_cost_index": "up"},
        estimated_magnitude="moderate",
        scenario_ids_affected=[9, 22],
        probability_shift={9: 0.03, 22: 0.01},
        relationship_updates=[],
        causal_chain=["tanker_boarded", "shadow_fleet_deterred", "venezuela_oil_redirected_to_us", "global_supply_tightens_others"],
        confidence=0.90,
    ),
    ExtractedEvent(
        event_id="evt_00000005",
        headline="Trump and Netanyahu align on Iran pressure but split on endgame",
        timestamp="2026-02-15", source="Reuters",
        actors_mentioned=["USA_POTUS", "ISR_PM"],
        new_actors=[],
        primary_intent="signal",
        sentiment="mixed", urgency="notable",
        affected_countries=["USA", "ISR", "IRN", "SAU"],
        affected_stats=["political_instability", "crude_output"],
        stat_direction={"political_instability": "up", "crude_output": "uncertain"},
        estimated_magnitude="moderate",
        scenario_ids_affected=[],
        probability_shift={},
        relationship_updates=[
            {"source": "USA_POTUS", "target": "ISR_PM", "type": "ally", "strength": 0.80,
             "context": "Aligned on pressure but divergent on objectives. Fragile consensus."},
        ],
        causal_chain=["alignment_on_pressure", "divergence_on_endgame", "potential_policy_split_later"],
        confidence=0.85,
    ),
    ExtractedEvent(
        event_id="evt_00000006",
        headline="EU's top diplomat rejects US talk of 'civilisational erasure'",
        timestamp="2026-02-15", source="EUObserver",
        actors_mentioned=["EU_DIPLOMAT", "USA_SEC_STATE"],
        new_actors=[],
        primary_intent="signal",
        sentiment="negative", urgency="notable",
        affected_countries=["USA"],
        affected_stats=["political_instability", "fdi_inflow"],
        stat_direction={"political_instability": "up", "fdi_inflow": "down"},
        estimated_magnitude="minor",
        scenario_ids_affected=[11],
        probability_shift={11: 0.01},
        relationship_updates=[
            {"source": "EU_DIPLOMAT", "target": "USA_SEC_STATE", "type": "adversary", "strength": 0.5,
             "context": "Kallas publicly rejecting US rhetoric. Transatlantic relationship cooling."},
        ],
        causal_chain=["us_rhetoric_escalates", "eu_pushes_back", "trade_cooperation_weakens"],
        confidence=0.80,
    ),
    ExtractedEvent(
        event_id="evt_00000007",
        headline="Foreign Minister Anand leads Canada at Munich conference, focuses on Arctic security",
        timestamp="2026-02-15", source="Global News",
        actors_mentioned=["CAN_ANAND"],
        new_actors=[],
        primary_intent="signal",
        sentiment="neutral", urgency="routine",
        affected_countries=["CAN", "RUS"],
        affected_stats=["infrastructure_spend_pct_gdp", "political_instability"],
        stat_direction={"infrastructure_spend_pct_gdp": "up"},
        estimated_magnitude="minor",
        scenario_ids_affected=[32],
        probability_shift={32: 0.005},
        relationship_updates=[],
        causal_chain=["arctic_security_prioritized", "defense_spend_pressure", "nato_coordination"],
        confidence=0.75,
    ),
    ExtractedEvent(
        event_id="evt_00000008",
        headline="Ottawans in Cuba fly home early as fuel shortages worsen",
        timestamp="2026-02-15", source="CBC",
        actors_mentioned=[],
        new_actors=[],
        primary_intent="signal",
        sentiment="negative", urgency="routine",
        affected_countries=["CAN"],
        affected_stats=["crude_output", "airline_passenger_volume"],
        stat_direction={"crude_output": "down"},
        estimated_magnitude="negligible",
        scenario_ids_affected=[9],
        probability_shift={9: 0.005},
        relationship_updates=[],
        causal_chain=["venezuela_blockade_effects", "caribbean_fuel_shortage", "tourism_disrupted"],
        confidence=0.70,
    ),
    ExtractedEvent(
        event_id="evt_00000009",
        headline="'Constant rain has made farming more difficult'",
        timestamp="2026-02-15", source="BBC",
        actors_mentioned=[],
        new_actors=[],
        primary_intent="signal",
        sentiment="negative", urgency="routine",
        affected_countries=["GBR"],
        affected_stats=["food_cpi", "food_to_income_ratio"],
        stat_direction={"food_cpi": "up", "food_to_income_ratio": "up"},
        estimated_magnitude="minor",
        scenario_ids_affected=[],
        probability_shift={},
        relationship_updates=[],
        causal_chain=["climate_impact_agriculture", "yields_drop", "food_prices_rise", "inflation_pressure"],
        confidence=0.65,
    ),
]
