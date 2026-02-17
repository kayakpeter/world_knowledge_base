"""
Strategic Analysis Engine — Deep geopolitical intelligence processing.

This is NOT a headline parser. This is a strategic analyst that thinks in:
  - Multi-order effects (1st → 2nd → 3rd → nth consequences)
  - Hidden intent (what's the real agenda behind the stated action)
  - Adversarial response modeling (how will each major power react)
  - Resource competition mapping (who controls what, who needs what)
  - Generational strategic plans (what's the 10/20/50-year play)
  - Debt/influence mapping (soft power projection through economics)

The key insight: every action by a sovereign actor is simultaneously:
  1. What it appears to be (stated intent)
  2. A move in a larger strategic game (hidden intent)
  3. A signal to allies (reassurance or coordination)
  4. A signal to adversaries (deterrence or provocation)
  5. A precedent for future actions (pattern establishment)

The LLM must reason through ALL of these layers for every event.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Multi-Order Effect Analysis ─────────────────────────────────────────────

@dataclass
class OrderEffect:
    """A single consequence at a specific depth from the original event."""
    order: int                  # 1 = direct, 2 = indirect, 3+ = cascading
    effect: str                 # description of the effect
    affected_countries: list[str]
    affected_stats: list[str]
    probability: float          # how likely this effect materializes
    timeframe: str              # "days", "weeks", "months", "years", "decades"
    reversible: bool            # can this be undone?


@dataclass
class StrategicIntent:
    """The multi-layered intent behind a sovereign action."""
    stated_intent: str          # what they say they're doing
    probable_intent: str        # what they're actually trying to achieve
    hidden_agendas: list[str]   # deeper strategic objectives
    beneficiaries: list[str]    # who really benefits (may not be obvious)
    losers: list[str]           # who is disadvantaged
    precedent_set: str          # what future actions does this enable


@dataclass
class AdversarialResponse:
    """How a specific actor/country is likely to respond to an event."""
    responder: str              # country or actor_id
    concern_level: str          # "unconcerned", "watching", "concerned", "alarmed", "threatened"
    likely_response: str        # what they'll probably do
    response_timeframe: str     # when
    escalation_risk: float      # 0-1, chance this triggers an escalatory cycle
    counter_moves: list[str]    # specific actions they might take


@dataclass
class StrategicAnalysis:
    """Complete strategic analysis of a single event or development."""
    event_summary: str
    timestamp: str

    # Multi-order effects
    effects: list[OrderEffect]

    # Intent analysis
    intent: StrategicIntent

    # How each major power views and responds to this
    adversarial_responses: list[AdversarialResponse]

    # Resource/influence implications
    resource_implications: list[str]
    influence_shift: dict[str, str]     # country → "gaining" or "losing" influence

    # Scenario linkage
    scenario_probability_shifts: dict[int, float]

    # Confidence and sourcing
    analysis_confidence: float
    key_uncertainties: list[str]


# ─── Generational Strategic Plans ────────────────────────────────────────────
# Every major power is executing a multi-decade strategy.
# Individual events only make sense in context of these plans.

@dataclass
class StrategicObjective:
    """A single objective within a country's generational plan."""
    objective_id: str
    title: str
    description: str
    category: str           # "economic", "military", "technological", "demographic", "resource", "diplomatic"
    timeframe: str          # "near" (1-3y), "medium" (3-10y), "long" (10-30y), "generational" (30+y)
    current_status: str     # "on_track", "behind", "ahead", "stalled", "pivoting"
    key_metrics: list[str]  # stat_names that track progress
    dependencies: list[str] # other objective_ids this depends on
    vulnerabilities: list[str]  # what could derail this objective


@dataclass
class GenerationalPlan:
    """A country's multi-decade strategic framework."""
    country: str
    country_iso3: str
    plan_name: str
    core_thesis: str            # the fundamental strategic bet
    objectives: list[StrategicObjective]
    key_actors: list[str]       # actor_ids driving this plan
    primary_competitors: list[str]  # countries competing for same objectives
    critical_resources: list[str]   # what they need to execute
    biggest_risk: str


GENERATIONAL_PLANS: list[GenerationalPlan] = [

    GenerationalPlan(
        country="United States", country_iso3="USA",
        plan_name="American Primacy Renewal",
        core_thesis="Maintain global hegemony through energy dominance, technological supremacy, "
                     "dollar reserve status, and military reach — while reshoring critical supply chains.",
        objectives=[
            StrategicObjective(
                "USA_OBJ_1", "Energy Dominance",
                "Become the world's swing producer and primary LNG exporter. "
                "Use energy as diplomatic leverage (Slovakia/Hungary courtship). "
                "Control Venezuela oil as strategic reserve.",
                "resource", "medium", "on_track",
                ["crude_output", "wti_brent_spread", "export_velocity"],
                [], ["OPEC discipline", "renewable transition pace"],
            ),
            StrategicObjective(
                "USA_OBJ_2", "Technology Supremacy",
                "Maintain 2-generation lead in semiconductors, AI, and quantum. "
                "Deny China access through export controls. Build domestic fab capacity.",
                "technological", "long", "on_track",
                ["rd_to_gdp", "capex_growth", "power_consumption"],
                ["USA_OBJ_1"],  # AI needs energy
                ["Talent retention", "Allied cooperation (NLD, JPN, KOR)", "China workarounds"],
            ),
            StrategicObjective(
                "USA_OBJ_3", "Dollar Hegemony Defense",
                "Preserve USD as global reserve currency. Counter BRICS alternatives. "
                "Leverage SWIFT access as strategic weapon.",
                "economic", "generational", "behind",
                ["reserve_adequacy", "debt_to_gdp", "net_interest_to_revenue"],
                [],
                ["Debt trajectory unsustainable", "BRICS payment systems", "Weaponization backlash"],
            ),
            StrategicObjective(
                "USA_OBJ_4", "Supply Chain Reshoring",
                "Reduce dependence on China for critical goods. USMCA as nearshoring vehicle. "
                "Rare earth processing capacity.",
                "economic", "medium", "behind",
                ["effective_tariff", "rule_of_origin_pct", "fdi_inflow"],
                ["USA_OBJ_2"],
                ["Cost inflation", "Mexico capacity limits", "Political will"],
            ),
            StrategicObjective(
                "USA_OBJ_5", "Fiscal Sustainability",
                "Address $1.9T annual deficit trajectory before debt spiral becomes self-reinforcing. "
                "OBBBA adds $3.7T — betting on growth to outrun debt.",
                "economic", "near", "behind",
                ["debt_to_gdp", "net_interest_to_revenue", "primary_deficit"],
                [],
                ["Interest rate sensitivity", "Entitlement growth", "Political gridlock"],
            ),
        ],
        key_actors=["USA_POTUS", "USA_TREASURY", "USA_FED", "USA_USTR", "USA_SEC_STATE"],
        primary_competitors=["China", "Russia"],
        critical_resources=["Semiconductors (domestic fab)", "Rare earths (diversification)",
                            "AI talent", "Energy infrastructure", "Allied cooperation"],
        biggest_risk="Debt trajectory causes forced austerity, undermining all other objectives simultaneously.",
    ),

    GenerationalPlan(
        country="China", country_iso3="CHN",
        plan_name="National Rejuvenation (中华民族伟大复兴)",
        core_thesis="Achieve technological self-sufficiency, regional military dominance, "
                     "and economic co-dependence with enough nations to make isolation impossible. "
                     "The Belt and Road is the scaffolding; debt is the glue.",
        objectives=[
            StrategicObjective(
                "CHN_OBJ_1", "Technological Self-Sufficiency",
                "Break dependence on Western semiconductors, software, and IP. "
                "Huawei as national champion. Domestic EUV lithography by 2030.",
                "technological", "long", "behind",
                ["rd_to_gdp", "export_velocity"],
                [],
                ["US export controls effective", "Talent gap", "ASML lock-out"],
            ),
            StrategicObjective(
                "CHN_OBJ_2", "Belt and Road Influence Network",
                "Build infrastructure in 140+ countries to create economic dependency, "
                "secure resource access, and expand diplomatic influence. "
                "Hospitals, ports, railways, telecom — all create leverage.",
                "diplomatic", "generational", "on_track",
                ["fdi_inflow", "export_velocity", "trade_concentration"],
                ["CHN_OBJ_3"],
                ["Debt backlash from recipients", "Quality concerns", "Western counter-offers"],
            ),
            StrategicObjective(
                "CHN_OBJ_3", "Resource Security",
                "Lock up long-term supply of rare earths, lithium, cobalt, copper, and food. "
                "African mining concessions. Australian iron ore alternatives. "
                "Arctic shipping route development.",
                "resource", "long", "on_track",
                ["mineral_reserves", "trade_concentration", "shipping_cost_index"],
                [],
                ["Resource nationalism in Africa", "Substitution technology", "Arctic geopolitics"],
            ),
            StrategicObjective(
                "CHN_OBJ_4", "RMB Internationalization",
                "Gradually expand RMB use in bilateral trade. Digital yuan as SWIFT alternative. "
                "Oil purchases in RMB from Saudi/Russia.",
                "economic", "generational", "behind",
                ["currency_volatility", "reserve_adequacy"],
                ["CHN_OBJ_2"],
                ["Capital controls incompatible with reserve currency", "Trust deficit"],
            ),
            StrategicObjective(
                "CHN_OBJ_5", "Demographic Transition Management",
                "Navigate the most dramatic aging in human history. 400M fewer working-age by 2050. "
                "Automation, AI, robotics to replace shrinking workforce.",
                "demographic", "generational", "stalled",
                ["demographic_drag", "labor_participation", "dependency_ratio"],
                ["CHN_OBJ_1"],
                ["Irreversible demographic trajectory", "Social stability", "Pension burden"],
            ),
        ],
        key_actors=["CHN_PRES", "CHN_PBOC", "CHN_FM"],
        primary_competitors=["United States", "India"],
        critical_resources=["Semiconductors (until self-sufficient)", "Food (imports 30% of calories)",
                            "Energy (oil/LNG imports)", "Talent (reverse brain drain)"],
        biggest_risk="Property sector collapse triggers banking crisis before tech self-sufficiency achieved.",
    ),

    GenerationalPlan(
        country="Canada", country_iso3="CAN",
        plan_name="Arctic/Resource Sovereignty + Trade Diversification",
        core_thesis="Leverage vast natural resources and strategic Arctic position while reducing "
                     "existential dependence on US trade relationship. Carney's appointment signals "
                     "a shift from passive resource economy to active strategic player.",
        objectives=[
            StrategicObjective(
                "CAN_OBJ_1", "USMCA Survival/Optimization",
                "Secure favorable USMCA renewal. Carney's trade negotiator appointment is the opening move. "
                "Maintain auto sector, protect dairy, expand energy provisions.",
                "economic", "near", "on_track",
                ["effective_tariff", "rule_of_origin_pct", "fdi_inflow", "export_velocity"],
                [],
                ["Trump's unpredictability", "Rules of origin tightening", "Mexico competition"],
            ),
            StrategicObjective(
                "CAN_OBJ_2", "Arctic Sovereignty & Security",
                "Establish effective control over Northwest Passage. Build northern infrastructure. "
                "Anand at Munich signals NATO coordination. Submarine procurement from Korea = capability gap closure.",
                "military", "long", "behind",
                ["infrastructure_spend_pct_gdp", "political_instability"],
                ["CAN_OBJ_4"],
                ["Russia's Arctic military buildup", "Climate change accelerating access", "Funding gaps"],
            ),
            StrategicObjective(
                "CAN_OBJ_3", "Critical Mineral Superpower",
                "Canada has world-class deposits of nickel, cobalt, lithium, rare earths. "
                "Position as democratic-aligned alternative to Chinese mineral dominance. "
                "Processing capacity is the bottleneck.",
                "resource", "medium", "on_track",
                ["mineral_reserves", "export_velocity", "fdi_inflow"],
                [],
                ["Processing capacity gap", "Indigenous consultation timelines", "Capital flight to cheaper jurisdictions"],
            ),
            StrategicObjective(
                "CAN_OBJ_4", "Defense Industrial Buildup",
                "Korea submarine deal with domestic manufacturing is the template: "
                "buy capability, build industry, create jobs. Apply to shipbuilding, drones, cybersecurity.",
                "military", "medium", "on_track",
                ["capex_growth", "labor_participation", "rd_to_gdp"],
                ["CAN_OBJ_1"],  # needs stable trade for economic base
                ["Brain drain to US", "Scale disadvantages", "Procurement bureaucracy"],
            ),
        ],
        key_actors=["CAN_PM", "CAN_ANAND", "CAN_TRADE_NEG", "CAN_BOC"],
        primary_competitors=["United States (trade)", "Russia (Arctic)", "Australia (minerals)"],
        critical_resources=["Defense technology (importing from KOR)", "Processing capacity for minerals",
                            "Arctic infrastructure", "Immigration (population growth engine)"],
        biggest_risk="US trade relationship deterioration forces economic contraction before diversification completes.",
    ),

    GenerationalPlan(
        country="Russia", country_iso3="RUS",
        plan_name="Fortress Russia + Eurasian Pivot",
        core_thesis="Survive Western sanctions by pivoting trade to China/India/Global South. "
                     "Use energy as weapon. Maintain military threat to prevent NATO encirclement. "
                     "Wait for Western resolve to fracture.",
        objectives=[
            StrategicObjective(
                "RUS_OBJ_1", "Energy Revenue Resilience",
                "Redirect oil/gas from Europe to Asia. Shadow fleet for sanctions evasion. "
                "Rubio's CEE tour is a direct attack on this objective.",
                "resource", "near", "behind",
                ["crude_output", "export_velocity", "current_account_gdp"],
                [],
                ["Price cap enforcement", "Shadow fleet interdiction", "Asian demand limits"],
            ),
            StrategicObjective(
                "RUS_OBJ_2", "Arctic Dominance",
                "Militarize Northern Sea Route. Icebreaker fleet (40+ vs 2 for US). "
                "Arctic resource extraction. Counter Canadian/NATO Arctic claims.",
                "military", "long", "on_track",
                ["infrastructure_spend_pct_gdp", "crude_output"],
                ["RUS_OBJ_1"],
                ["Economic constraints", "Climate unpredictability", "Technology sanctions"],
            ),
            StrategicObjective(
                "RUS_OBJ_3", "BRICS Alternative Financial Architecture",
                "Co-build with China a SWIFT alternative. Ruble/RMB bilateral trade. "
                "Gold reserves as sanctions hedge.",
                "economic", "medium", "behind",
                ["reserve_adequacy", "currency_volatility"],
                [],
                ["System adoption is slow", "China doesn't want to be too visible", "Technical challenges"],
            ),
        ],
        key_actors=["RUS_PRES"],
        primary_competitors=["United States", "NATO bloc"],
        critical_resources=["Technology (sanctions limited)", "Semiconductors (acute shortage)",
                            "Demographic replacement (war losses + emigration)"],
        biggest_risk="Prolonged low oil prices collapse fiscal position before Eurasian pivot completes.",
    ),

    GenerationalPlan(
        country="India", country_iso3="IND",
        plan_name="Viksit Bharat (Developed India) 2047",
        core_thesis="Leverage demographic dividend (youngest major economy) to become $30T GDP by 2047. "
                     "Multi-alignment: friend to everyone, dependent on no one. "
                     "Manufacturing hub as China+1 beneficiary.",
        objectives=[
            StrategicObjective(
                "IND_OBJ_1", "Manufacturing Scale-Up",
                "Capture China+1 supply chain shift. PLI schemes for semiconductors, pharma, defense. "
                "Compete with Vietnam and Mexico for factory floor.",
                "economic", "medium", "on_track",
                ["fdi_inflow", "export_velocity", "capex_growth", "pmi_manufacturing"],
                ["IND_OBJ_3"],
                ["Infrastructure gaps", "Land acquisition", "Bureaucracy", "Skills mismatch"],
            ),
            StrategicObjective(
                "IND_OBJ_2", "Energy Security Through Diversification",
                "Buy from everyone — Russia, Saudi, US, renewables. Solar manufacturing. "
                "Nuclear expansion. Never be held hostage by a single supplier.",
                "resource", "long", "on_track",
                ["crude_output", "power_consumption", "renewable_share_generation", "energy_cpi"],
                [],
                ["Oil import dependence (85%)", "Coal transition pace", "Grid modernization"],
            ),
            StrategicObjective(
                "IND_OBJ_3", "Infrastructure Leap",
                "Build the roads, ports, rail, digital backbone for a $30T economy. "
                "National Infrastructure Pipeline. Digital India.",
                "economic", "long", "on_track",
                ["infrastructure_spend_pct_gdp", "broadband_penetration", "freight_volume_index"],
                [],
                ["Fiscal space", "State-level execution variance", "Land acquisition"],
            ),
            StrategicObjective(
                "IND_OBJ_4", "Demographic Dividend Capture",
                "Convert 1.4B population into productive workforce. Education quality, "
                "female labor participation, skills training at scale.",
                "demographic", "generational", "behind",
                ["labor_participation", "youth_unemployment", "education_expenditure_gdp", "tertiary_enrollment_rate"],
                [],
                ["Education quality crisis", "Gender gap", "Informal economy dominance"],
            ),
        ],
        key_actors=["IND_PM", "IND_RBI"],
        primary_competitors=["China (manufacturing)", "Vietnam (FDI)", "Indonesia (demographic)"],
        critical_resources=["Semiconductors (importing)", "Oil/gas (85% import dependent)",
                            "Skilled labor (training gap)", "Water (climate vulnerability)"],
        biggest_risk="Youth unemployment remains high despite growth — demographic dividend becomes demographic disaster.",
    ),

    GenerationalPlan(
        country="Germany", country_iso3="DEU",
        plan_name="Zeitenwende (Turning Point) Industrial Renewal",
        core_thesis="Rebuild from energy shock, rearm, and reindustrialize. The €500B infrastructure "
                     "program is the opening move. Abandon fiscal conservatism for strategic investment. "
                     "Lead European defense autonomy.",
        objectives=[
            StrategicObjective(
                "DEU_OBJ_1", "Energy System Reconstruction",
                "Replace Russian gas with LNG, renewables, hydrogen. €150B+ required. "
                "Energy costs must come down for industrial competitiveness.",
                "resource", "medium", "behind",
                ["power_consumption", "energy_cpi", "renewable_share_generation", "carbon_price"],
                [],
                ["Industrial electricity prices 3x US", "NIMBYism on renewables", "Nuclear exit irreversible"],
            ),
            StrategicObjective(
                "DEU_OBJ_2", "Defense Rearmament",
                "€100B special fund + sustained 2% GDP. Bundeswehr modernization. "
                "European defense industrial base coordination with France.",
                "military", "medium", "on_track",
                ["infrastructure_spend_pct_gdp", "debt_to_gdp"],
                ["DEU_OBJ_3"],
                ["Procurement delays", "Skilled labor shortage", "Industrial capacity"],
            ),
            StrategicObjective(
                "DEU_OBJ_3", "Fiscal Paradigm Shift",
                "Merz's €500B infrastructure program breaks the debt brake taboo. "
                "Bet: public investment drives growth that keeps debt/GDP stable.",
                "economic", "near", "on_track",
                ["debt_to_gdp", "public_investment_ratio", "real_gdp_growth"],
                [],
                ["Bond market tolerance", "Constitutional court challenges", "Growth disappoints"],
            ),
        ],
        key_actors=["DEU_CHAN", "DEU_FIN"],
        primary_competitors=["China (industrial)", "United States (energy costs)"],
        critical_resources=["Skilled labor (demographic crunch)", "Energy (transitioning)",
                            "Semiconductors (ASML partnership)", "Capital (fiscal expansion)"],
        biggest_risk="Energy costs remain structurally high, driving deindustrialization before transition completes.",
    ),

    GenerationalPlan(
        country="Saudi Arabia", country_iso3="SAU",
        plan_name="Vision 2030",
        core_thesis="Transform from oil-dependent rentier state to diversified economy before "
                     "oil demand peaks. NEOM, tourism, entertainment, financial services. "
                     "But the clock is ticking — oil revenue funds the transition.",
        objectives=[
            StrategicObjective(
                "SAU_OBJ_1", "Economic Diversification",
                "Non-oil GDP to 65% by 2030. NEOM, Red Sea tourism, entertainment. "
                "FDI attraction. Financial center ambitions.",
                "economic", "medium", "behind",
                ["real_gdp_growth", "fdi_inflow", "export_velocity"],
                ["SAU_OBJ_2"],
                ["Oil revenue declining", "Execution risk on mega-projects", "Social reform pace"],
            ),
            StrategicObjective(
                "SAU_OBJ_2", "Oil Revenue Maximization",
                "Maximize oil income while demand persists. OPEC+ coordination. "
                "Aramco value. Strategic petroleum reserves.",
                "resource", "near", "on_track",
                ["crude_output", "wti_brent_spread", "current_account_gdp"],
                [],
                ["Venezuela supply returning", "Iran deal lifts sanctions", "US shale discipline breaks"],
            ),
            StrategicObjective(
                "SAU_OBJ_3", "Geopolitical Realignment",
                "Multi-alignment: maintain US security umbrella while expanding China/Russia ties. "
                "Oil pricing in non-dollar currencies as leverage. BRICS membership.",
                "diplomatic", "long", "on_track",
                ["reserve_adequacy", "political_instability"],
                [],
                ["US-China forces binary choice", "Iran threat requires US protection"],
            ),
        ],
        key_actors=["SAU_MBS"],
        primary_competitors=["UAE (diversification)", "Qatar (LNG)"],
        critical_resources=["Water (desalination dependent)", "Labor (expat workforce)",
                            "Technology (importing everything)", "Time (oil demand peak approaching)"],
        biggest_risk="Oil revenue drops before diversification revenue replaces it — fiscal crisis forces austerity on Vision 2030.",
    ),
]


# ─── Strategic Analysis LLM Prompts ──────────────────────────────────────────

STRATEGIC_ANALYSIS_SYSTEM = """You are a senior intelligence analyst at a sovereign wealth fund.
You think like George Kennan, Ray Dalio, and Henry Kissinger combined.

For every event, you MUST analyze:

1. MULTI-ORDER EFFECTS (at least 5 orders deep):
   - 1st order: What directly happens
   - 2nd order: What that causes
   - 3rd order: The reaction to the reaction
   - 4th order: How adversaries adapt
   - 5th order: The new equilibrium

2. HIDDEN INTENT (who really benefits):
   - Stated intent vs probable intent vs hidden agendas
   - Follow the resources: who gets access to what?
   - Follow the debt: who becomes obligated to whom?
   - Follow the precedent: what future actions does this enable?

3. ADVERSARIAL RESPONSE MODELING (for each major power):
   - How does the US view this? What will they do?
   - How does China view this? What will they do?
   - How does Russia view this? What will they do?
   - How does the EU view this? What will they do?
   - Who else is affected and how will they respond?

4. RESOURCE COMPETITION:
   - Does this change who controls critical resources?
   - Rare earths, energy, food, water, semiconductors, shipping lanes
   - Is this a move in the Great Resource Game?

5. GENERATIONAL PLAN CONTEXT:
   - Which country's long-term plan does this advance or threaten?
   - Is this a one-off event or part of a pattern?
   - What's the 10-year view of this development?

You are known in the actor and scenario context below.

Known actors:
{actor_context}

Known generational plans (key objectives):
{plan_context}

Known scenarios with current probabilities:
{scenario_context}

Respond ONLY with JSON matching this schema:
{{
    "effects": [
        {{
            "order": 1,
            "effect": "Direct consequence description",
            "affected_countries": ["ISO3"],
            "affected_stats": ["stat_name"],
            "probability": 0.9,
            "timeframe": "weeks",
            "reversible": false
        }}
    ],
    "intent": {{
        "stated_intent": "What was announced/claimed",
        "probable_intent": "What they're actually trying to achieve",
        "hidden_agendas": ["deeper strategic objective 1", "objective 2"],
        "beneficiaries": ["who really benefits"],
        "losers": ["who is disadvantaged"],
        "precedent_set": "what future actions this enables"
    }},
    "adversarial_responses": [
        {{
            "responder": "country name or actor_id",
            "concern_level": "watching|concerned|alarmed|threatened",
            "likely_response": "what they'll do",
            "response_timeframe": "months",
            "escalation_risk": 0.3,
            "counter_moves": ["specific action 1", "specific action 2"]
        }}
    ],
    "resource_implications": ["resource shift description"],
    "influence_shift": {{"country": "gaining|losing"}},
    "generational_plan_impacts": [
        {{
            "country": "country name",
            "objective_affected": "objective title",
            "impact": "advances|threatens|neutral",
            "explanation": "how and why"
        }}
    ],
    "scenario_probability_shifts": {{4: 0.05, 20: -0.03}},
    "analysis_confidence": 0.85,
    "key_uncertainties": ["what we don't know that matters"]
}}"""


PATTERN_RECOGNITION_SYSTEM = """You are tracking patterns across multiple events over time.
Given a sequence of recent events and their strategic analyses, identify:

1. CONVERGENT PATTERNS: Multiple events pointing in the same direction
2. CONTRADICTORY SIGNALS: Events that suggest different outcomes
3. INFLECTION POINTS: Is a major shift about to happen?
4. COORDINATION SIGNALS: Are actors coordinating (even if not openly)?
5. VULNERABILITY WINDOWS: Temporary openings for action

Events and analyses:
{events_json}

Country generational plans:
{plans_json}

Respond with JSON:
{{
    "convergent_patterns": [
        {{
            "pattern": "description",
            "supporting_events": ["event_id_1", "event_id_2"],
            "implication": "what this means",
            "confidence": 0.8
        }}
    ],
    "contradictory_signals": [...],
    "inflection_points": [...],
    "coordination_signals": [...],
    "vulnerability_windows": [...],
    "overall_assessment": "the big picture in 2-3 sentences"
}}"""


# ─── Example: Deep Analysis of "China Builds Hospital in Africa" ─────────────

EXAMPLE_DEEP_ANALYSIS = StrategicAnalysis(
    event_summary="China funds and builds a 500-bed hospital in Zambia through Belt and Road Initiative",
    timestamp="2026-02-15",
    effects=[
        OrderEffect(1, "Zambia gets a modern hospital. Healthcare capacity increases.",
                    ["ZMB"], ["medical_expenditure_per_capita"], 0.95, "months", False),
        OrderEffect(2, "Zambia takes on $200M in Chinese debt. Repayment tied to copper mining concessions.",
                    ["ZMB", "CHN"], ["net_external_debt", "mineral_reserves"], 0.90, "years", False),
        OrderEffect(3, "China gains preferential access to Zambian copper reserves (critical for EVs/electronics). "
                    "Chinese mining companies expand operations.",
                    ["CHN", "ZMB"], ["trade_concentration", "fdi_inflow"], 0.85, "years", False),
        OrderEffect(4, "US and EU launch competing 'Global Gateway' health infrastructure offers to neighboring countries. "
                    "France increases aid to Francophone West Africa to counter.",
                    ["USA", "FRA"], ["infrastructure_spend_pct_gdp"], 0.60, "years", True),
        OrderEffect(5, "African countries learn to play major powers against each other for better terms. "
                    "Resource nationalism grows. Some countries renegotiate or default on Chinese loans.",
                    ["ZMB"], ["political_instability", "net_external_debt"], 0.40, "decades", False),
    ],
    intent=StrategicIntent(
        stated_intent="Improve healthcare access in Zambia as part of South-South cooperation",
        probable_intent="Secure long-term copper supply through debt-for-resource exchange. "
                        "Build political goodwill for UN General Assembly votes.",
        hidden_agendas=[
            "Lock up copper supply critical for China's EV manufacturing dominance",
            "Establish logistics footprint for potential naval base of supply",
            "Create dependency that ensures Zambia votes with China in UN",
            "Demonstrate BRI model to neighboring countries to expand deals",
            "Chinese construction firms get contracts — recycling domestic overcapacity",
        ],
        beneficiaries=["Chinese mining companies", "Chinese construction SOEs",
                        "Zambian political elite", "Zambian healthcare workers"],
        losers=["Zambian fiscal sovereignty", "Western mining companies",
                "Zambian taxpayers (long-term debt burden)"],
        precedent_set="Health-for-resources template replicable across 140+ BRI countries",
    ),
    adversarial_responses=[
        AdversarialResponse(
            "United States", "concerned",
            "Announce competing DFC (Development Finance Corporation) health project in region. "
            "Pressure Zambia through IMF conditionality on Chinese debt terms.",
            "months", 0.2,
            ["DFC counter-offer", "IMF debt sustainability review", "Congressional hearing on BRI"],
        ),
        AdversarialResponse(
            "Russia", "unconcerned",
            "Russia has no competing offer. May quietly support China in diplomatic forums. "
            "Focuses own Africa strategy on military (Wagner/Africa Corps) not health.",
            "years", 0.05,
            ["Symbolic BRICS solidarity statement"],
        ),
        AdversarialResponse(
            "EU/France", "concerned",
            "France sees this as encroachment on traditional Francophone sphere of influence. "
            "EU Global Gateway health corridor acceleration. "
            "Paris Club coordination on debt sustainability.",
            "months", 0.15,
            ["Global Gateway counter-project", "Paris Club debt review", "EU-AU summit agenda item"],
        ),
        AdversarialResponse(
            "India", "watching",
            "India's Africa strategy focuses on pharma exports and IT training. "
            "May accelerate generic drug supply deals as counter-narrative to Chinese hardware.",
            "years", 0.1,
            ["Generic drug supply agreements", "IT training center offers"],
        ),
    ],
    resource_implications=[
        "China gains copper access in a market where supply is tightening",
        "Copper is critical for EV motors, wiring, grid infrastructure — demand growing 5% annually",
        "Zambia holds 6% of global copper reserves — strategic asset",
        "This reduces Western companies' ability to source copper at market rates",
    ],
    influence_shift={
        "China": "gaining",
        "United States": "losing",
        "Zambia": "losing (sovereignty)",
        "France": "losing (traditional influence)",
    },
    scenario_probability_shifts={
        43: 0.005,  # Global Rare Earth Supply Crisis (slightly less likely — China diversifying supply)
        33: 0.01,   # Dollar Hegemony Challenge (BRI expands non-dollar trade)
    },
    analysis_confidence=0.80,
    key_uncertainties=[
        "Exact debt terms and mineral access conditions (not publicly disclosed)",
        "Whether Zambia negotiated better terms than previous BRI recipients",
        "US/EU willingness to actually fund counter-offers vs just announce them",
        "Chinese domestic fiscal capacity to sustain BRI spending at current levels",
    ],
)


class StrategicAnalyzer:
    """
    Deep strategic analysis engine.

    Uses LLM for real-time analysis, but also maintains pre-computed
    generational plans and pattern recognition across events.
    """

    def __init__(self, llm_provider=None):
        self._llm = llm_provider
        self._plans = {p.country_iso3: p for p in GENERATIONAL_PLANS}
        self._analysis_history: list[StrategicAnalysis] = []

    async def analyze_event(
        self,
        headline: str,
        context: str = "",
        source: str = "",
    ) -> StrategicAnalysis:
        """
        Perform deep strategic analysis on a single event.

        This is the expensive call — it asks the LLM to think through
        5 orders of effects, hidden intent, and adversarial responses.
        Expect this to use 2000-4000 tokens per call.
        """
        if self._llm is None:
            logger.warning("No LLM provider — returning example analysis")
            return EXAMPLE_DEEP_ANALYSIS

        # Build rich context for the LLM
        actor_context = "\n".join(
            f"  {a.actor_id}: {a.name} — {a.role} ({a.country})"
            for a in ACTOR_REGISTRY[:20]
        )
        plan_context = "\n".join(
            f"  {p.country}: {p.plan_name} — {p.core_thesis[:100]}..."
            for p in GENERATIONAL_PLANS
        )

        from processing.scenario_engine import SCENARIO_REGISTRY
        scenario_context = "\n".join(
            f"  #{s.scenario_id}: {s.title} (P={s.probability_12m:.0%})"
            for s in SCENARIO_REGISTRY[:20]
        )

        system = STRATEGIC_ANALYSIS_SYSTEM.format(
            actor_context=actor_context,
            plan_context=plan_context,
            scenario_context=scenario_context,
        )

        user_prompt = f"EVENT: {headline}"
        if context:
            user_prompt += f"\nCONTEXT: {context}"
        if source:
            user_prompt += f"\nSOURCE: {source}"

        result = await self._llm.complete_json(system, user_prompt, temperature=0.4, max_tokens=4000)

        if "error" in result:
            logger.error("Strategic analysis failed: %s", result["error"])
            return EXAMPLE_DEEP_ANALYSIS

        # Parse into StrategicAnalysis (simplified — production would validate each field)
        analysis = StrategicAnalysis(
            event_summary=headline,
            timestamp="",
            effects=[OrderEffect(**e) for e in result.get("effects", [])],
            intent=StrategicIntent(**result.get("intent", {})) if "intent" in result else StrategicIntent("", "", [], [], [], ""),
            adversarial_responses=[AdversarialResponse(**r) for r in result.get("adversarial_responses", [])],
            resource_implications=result.get("resource_implications", []),
            influence_shift=result.get("influence_shift", {}),
            scenario_probability_shifts=result.get("scenario_probability_shifts", {}),
            analysis_confidence=result.get("analysis_confidence", 0.5),
            key_uncertainties=result.get("key_uncertainties", []),
        )

        self._analysis_history.append(analysis)
        return analysis

    def get_country_plan(self, country_iso3: str) -> Optional[GenerationalPlan]:
        """Retrieve a country's generational strategic plan."""
        return self._plans.get(country_iso3)

    def get_all_plans(self) -> list[GenerationalPlan]:
        """Get all generational plans."""
        return GENERATIONAL_PLANS

    def print_plan_summary(self, plan: GenerationalPlan) -> None:
        """Print a formatted generational plan summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("%s — %s", plan.country, plan.plan_name)
        logger.info("=" * 70)
        logger.info("  Thesis: %s", plan.core_thesis[:120])
        logger.info("  Key actors: %s", ", ".join(plan.key_actors))
        logger.info("  Competitors: %s", ", ".join(plan.primary_competitors))
        logger.info("  Biggest risk: %s", plan.biggest_risk)
        logger.info("")
        for obj in plan.objectives:
            status_icon = {
                "on_track": "🟢", "behind": "🟡", "ahead": "🔵",
                "stalled": "🔴", "pivoting": "🟠",
            }.get(obj.current_status, "⚪")
            logger.info(
                "  %s [%s] %s (%s)",
                status_icon, obj.timeframe.upper(), obj.title, obj.current_status,
            )
            logger.info("    %s", obj.description[:100])
            if obj.vulnerabilities:
                logger.info("    Vulnerabilities: %s", ", ".join(obj.vulnerabilities[:3]))

    def print_analysis(self, analysis: StrategicAnalysis) -> None:
        """Print a formatted strategic analysis."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("STRATEGIC ANALYSIS: %s", analysis.event_summary[:70])
        logger.info("=" * 70)

        logger.info("\n  MULTI-ORDER EFFECTS:")
        for e in analysis.effects:
            prefix = "  " * e.order
            logger.info(
                "  %s%s→ [P=%.0f%%, %s] %s",
                prefix, "│" * (e.order - 1),
                e.probability * 100, e.timeframe, e.effect[:80],
            )

        logger.info("\n  INTENT ANALYSIS:")
        logger.info("    Stated: %s", analysis.intent.stated_intent)
        logger.info("    Probable: %s", analysis.intent.probable_intent)
        for ha in analysis.intent.hidden_agendas:
            logger.info("    Hidden: %s", ha)

        logger.info("\n  ADVERSARIAL RESPONSES:")
        for r in analysis.adversarial_responses:
            concern_icon = {
                "unconcerned": "○", "watching": "◐",
                "concerned": "●", "alarmed": "◉", "threatened": "⊗",
            }.get(r.concern_level, "?")
            logger.info(
                "    %s %s [%s] — %s",
                concern_icon, r.responder, r.concern_level, r.likely_response[:60],
            )
            for cm in r.counter_moves:
                logger.info("      → %s", cm)

        if analysis.influence_shift:
            logger.info("\n  INFLUENCE SHIFT:")
            for country, direction in analysis.influence_shift.items():
                arrow = "↑" if direction == "gaining" else "↓"
                logger.info("    %s %s (%s)", arrow, country, direction)

        if analysis.key_uncertainties:
            logger.info("\n  KEY UNCERTAINTIES:")
            for u in analysis.key_uncertainties:
                logger.info("    ? %s", u)
