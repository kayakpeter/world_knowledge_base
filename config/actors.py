"""
Actor Registry — Key decision-makers and their influence on economic outcomes.

This layer maps the PEOPLE who move the numbers. Every stat in the KB is ultimately
the result of decisions made by specific humans — central bankers set rates,
trade negotiators set tariffs, defense ministers set spending, energy ministers
set policy. Tracking who these people are, what they're saying, and who they're
talking to is the qualitative intelligence that gives the quantitative model
its predictive edge.

Node types added to the graph:
  - Actor: A named individual with role, country affiliation, and influence domain
  - Communication: A news event, speech, meeting, or policy signal
  - Relationship: A directional link between actors (ally, adversary, negotiating)

The LLM processing phase parses raw news headlines into structured graph updates:
  headline → {actors, intent, affected_stats, sentiment, urgency}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Actor:
    """A key decision-maker in the global financial system."""
    actor_id: str               # e.g., "USA_POTUS", "CAN_PM"
    name: str                   # current holder of the role
    role: str                   # e.g., "President", "Prime Minister", "Fed Chair"
    role_type: str              # "head_of_state", "central_bank", "finance", "trade", "defense", "foreign_affairs", "institutional"
    country: str                # affiliated country or "International"
    country_iso3: str
    influence_domains: list[str]  # which stat categories this actor can move
    institution: str = ""       # e.g., "Federal Reserve", "ECB", "JPMorgan Chase"
    term_end: str = ""          # when their term/mandate expires (YYYY-MM or "indefinite")
    notes: str = ""


@dataclass
class ActorRelationship:
    """A directional relationship between two actors."""
    source_actor_id: str
    target_actor_id: str
    relationship_type: str      # "ally", "adversary", "negotiating", "subordinate", "institutional"
    strength: float             # 0.0 to 1.0
    context: str = ""           # brief description
    last_interaction: str = ""  # ISO date of most recent known interaction


# ─── Actor Registry (Key Roles per Country) ──────────────────────────────────
# This is seeded with known roles. The LLM processing phase keeps it current
# by parsing news for leadership changes, appointments, and departures.

ACTOR_REGISTRY: list[Actor] = [

    # ── United States ────────────────────────────────────────────────────
    Actor(
        actor_id="USA_POTUS", name="Donald Trump", role="President",
        role_type="head_of_state", country="United States", country_iso3="USA",
        influence_domains=["Macro/Solvency", "Trade/External", "Energy/Commodity", "Inst./Risk"],
        term_end="2029-01",
        notes="Second term. Aggressive tariff policy, Venezuela intervention, USMCA renegotiation.",
    ),
    Actor(
        actor_id="USA_TREASURY", name="Scott Bessent", role="Secretary of the Treasury",
        role_type="finance", country="United States", country_iso3="USA",
        influence_domains=["Macro/Solvency", "Monetary/Price", "Corporate/Business"],
        institution="US Treasury",
        notes="Former hedge fund manager. Managing OBBBA fiscal expansion and debt issuance.",
    ),
    Actor(
        actor_id="USA_FED", name="Jerome Powell", role="Chair of the Federal Reserve",
        role_type="central_bank", country="United States", country_iso3="USA",
        influence_domains=["Monetary/Price", "Corporate/Business", "Real Estate", "Consumer/Household"],
        institution="Federal Reserve",
        term_end="2026-05",
        notes="Term expires May 2026. Maintaining strategic plateau at 3.50-3.75%.",
    ),
    Actor(
        actor_id="USA_USTR", name="Jamieson Greer", role="US Trade Representative",
        role_type="trade", country="United States", country_iso3="USA",
        influence_domains=["Trade/External"],
        institution="USTR",
        notes="Leading USMCA renegotiation. Hardline on Chinese transshipment.",
    ),
    Actor(
        actor_id="USA_SEC_STATE", name="Marco Rubio", role="Secretary of State",
        role_type="foreign_affairs", country="United States", country_iso3="USA",
        influence_domains=["Trade/External", "Energy/Commodity", "Inst./Risk"],
        institution="State Department",
        notes="Slovakia/Hungary energy diplomacy. Iran pressure campaign with Netanyahu.",
    ),
    Actor(
        actor_id="USA_SEC_DEF", name="Pete Hegseth", role="Secretary of Defense",
        role_type="defense", country="United States", country_iso3="USA",
        influence_domains=["Macro/Solvency", "Infrastructure/Energy"],
        institution="Department of Defense",
        notes="Venezuela military operation. Defense budget implications.",
    ),
    Actor(
        actor_id="USA_JPM", name="Jamie Dimon", role="CEO, JPMorgan Chase",
        role_type="institutional", country="United States", country_iso3="USA",
        influence_domains=["Corporate/Business", "Monetary/Price", "Inst./Risk"],
        institution="JPMorgan Chase",
        notes="Fortress balance sheet strategy. Cockroach theory on private credit.",
    ),

    # ── Canada ───────────────────────────────────────────────────────────
    Actor(
        actor_id="CAN_PM", name="Mark Carney", role="Prime Minister",
        role_type="head_of_state", country="Canada", country_iso3="CAN",
        influence_domains=["Macro/Solvency", "Trade/External", "Monetary/Price", "Energy/Commodity"],
        notes="Former BoC and BoE governor. Appointed new chief trade negotiator to US.",
    ),
    Actor(
        actor_id="CAN_FM", name="Mélanie Joly", role="Minister of Foreign Affairs",
        role_type="foreign_affairs", country="Canada", country_iso3="CAN",
        influence_domains=["Trade/External", "Inst./Risk"],
        notes="Arctic security focus. NATO coordination.",
    ),
    Actor(
        actor_id="CAN_TRADE_NEG", name="TBD", role="Chief Trade Negotiator to US",
        role_type="trade", country="Canada", country_iso3="CAN",
        influence_domains=["Trade/External"],
        notes="Newly appointed by Carney. Key role in USMCA renegotiation.",
    ),
    Actor(
        actor_id="CAN_ANAND", name="Anita Anand", role="Minister of Foreign Affairs",
        role_type="foreign_affairs", country="Canada", country_iso3="CAN",
        influence_domains=["Trade/External", "Inst./Risk"],
        notes="Leading Canada at Munich Security Conference. Arctic security focus.",
    ),
    Actor(
        actor_id="CAN_BOC", name="Tiff Macklem", role="Governor, Bank of Canada",
        role_type="central_bank", country="Canada", country_iso3="CAN",
        influence_domains=["Monetary/Price", "Real Estate", "Consumer/Household"],
        institution="Bank of Canada",
        term_end="2027-06",
    ),

    # ── China ────────────────────────────────────────────────────────────
    Actor(
        actor_id="CHN_PRES", name="Xi Jinping", role="President",
        role_type="head_of_state", country="China", country_iso3="CHN",
        influence_domains=["Macro/Solvency", "Trade/External", "Energy/Commodity", "Inst./Risk"],
        term_end="indefinite",
        notes="Centralized authority. Tech decoupling response. Property sector intervention.",
    ),
    Actor(
        actor_id="CHN_PBOC", name="Pan Gongsheng", role="Governor, PBOC",
        role_type="central_bank", country="China", country_iso3="CHN",
        influence_domains=["Monetary/Price", "Real Estate", "Corporate/Business"],
        institution="People's Bank of China",
    ),
    Actor(
        actor_id="CHN_FM", name="Lan Fo'an", role="Minister of Finance",
        role_type="finance", country="China", country_iso3="CHN",
        influence_domains=["Macro/Solvency", "Infrastructure/Energy"],
        notes="Fiscal stimulus coordination. Local government debt management.",
    ),

    # ── Germany ──────────────────────────────────────────────────────────
    Actor(
        actor_id="DEU_CHAN", name="Friedrich Merz", role="Chancellor",
        role_type="head_of_state", country="Germany", country_iso3="DEU",
        influence_domains=["Macro/Solvency", "Trade/External", "Infrastructure/Energy"],
        notes="€500B infrastructure program. Fiscal reawakening from debt brake.",
    ),
    Actor(
        actor_id="DEU_FIN", name="TBD", role="Finance Minister",
        role_type="finance", country="Germany", country_iso3="DEU",
        influence_domains=["Macro/Solvency", "Monetary/Price"],
        notes="New government formation. Key role in fiscal expansion management.",
    ),

    # ── United Kingdom ───────────────────────────────────────────────────
    Actor(
        actor_id="GBR_PM", name="Keir Starmer", role="Prime Minister",
        role_type="head_of_state", country="United Kingdom", country_iso3="GBR",
        influence_domains=["Macro/Solvency", "Trade/External", "Infrastructure/Energy"],
    ),
    Actor(
        actor_id="GBR_BOE", name="Andrew Bailey", role="Governor, Bank of England",
        role_type="central_bank", country="United Kingdom", country_iso3="GBR",
        influence_domains=["Monetary/Price", "Real Estate", "Corporate/Business"],
        institution="Bank of England",
        term_end="2028-03",
    ),

    # ── Japan ────────────────────────────────────────────────────────────
    Actor(
        actor_id="JPN_PM", name="Shigeru Ishiba", role="Prime Minister",
        role_type="head_of_state", country="Japan", country_iso3="JPN",
        influence_domains=["Macro/Solvency", "Trade/External"],
    ),
    Actor(
        actor_id="JPN_BOJ", name="Kazuo Ueda", role="Governor, Bank of Japan",
        role_type="central_bank", country="Japan", country_iso3="JPN",
        influence_domains=["Monetary/Price", "Corporate/Business"],
        institution="Bank of Japan",
        term_end="2028-04",
        notes="Rate normalization to 0.75%. Yen carry trade implications.",
    ),

    # ── ECB / EU ─────────────────────────────────────────────────────────
    Actor(
        actor_id="EU_ECB", name="Christine Lagarde", role="President, ECB",
        role_type="central_bank", country="France", country_iso3="FRA",
        influence_domains=["Monetary/Price", "Corporate/Business", "Real Estate"],
        institution="European Central Bank",
        term_end="2027-10",
        notes="Potential cuts below 2%. Eurozone stagnation response.",
    ),
    Actor(
        actor_id="EU_DIPLOMAT", name="Kaja Kallas", role="EU High Representative",
        role_type="foreign_affairs", country="Netherlands", country_iso3="NLD",
        influence_domains=["Trade/External", "Inst./Risk"],
        institution="European Union",
        notes="Rejected US 'civilisational erasure' rhetoric. Defending EU trade sovereignty.",
    ),

    # ── India ────────────────────────────────────────────────────────────
    Actor(
        actor_id="IND_PM", name="Narendra Modi", role="Prime Minister",
        role_type="head_of_state", country="India", country_iso3="IND",
        influence_domains=["Macro/Solvency", "Trade/External", "Infrastructure/Energy"],
        notes="GDP overtaking Germany. All-of-the-above energy strategy.",
    ),
    Actor(
        actor_id="IND_RBI", name="Sanjay Malhotra", role="Governor, RBI",
        role_type="central_bank", country="India", country_iso3="IND",
        influence_domains=["Monetary/Price", "Consumer/Household"],
        institution="Reserve Bank of India",
    ),

    # ── Russia ───────────────────────────────────────────────────────────
    Actor(
        actor_id="RUS_PRES", name="Vladimir Putin", role="President",
        role_type="head_of_state", country="Russia", country_iso3="RUS",
        influence_domains=["Energy/Commodity", "Trade/External", "Inst./Risk"],
        term_end="2030",
        notes="War economy pivot. Energy exports to non-Western partners.",
    ),

    # ── Saudi Arabia ─────────────────────────────────────────────────────
    Actor(
        actor_id="SAU_MBS", name="Mohammed bin Salman", role="Crown Prince / PM",
        role_type="head_of_state", country="Saudi Arabia", country_iso3="SAU",
        influence_domains=["Energy/Commodity", "Macro/Solvency", "Infrastructure/Energy"],
        notes="Vision 2030. Fiscal strain from oil revenue shortfall.",
    ),

    # ── Israel ───────────────────────────────────────────────────────────
    Actor(
        actor_id="ISR_PM", name="Benjamin Netanyahu", role="Prime Minister",
        role_type="head_of_state", country="Israel", country_iso3="ISR",
        influence_domains=["Inst./Risk", "Energy/Commodity"],
        notes="Iran pressure alignment with Trump. Divergence on endgame.",
    ),

    # ── Iran ─────────────────────────────────────────────────────────────
    Actor(
        actor_id="IRN_FM", name="Abbas Araghchi", role="Foreign Minister",
        role_type="foreign_affairs", country="Iran", country_iso3="IRN",
        influence_domains=["Energy/Commodity", "Inst./Risk"],
        notes="Signaling nuclear deal compromise readiness. Critical for oil supply outlook.",
    ),

    # ── Turkey ───────────────────────────────────────────────────────────
    Actor(
        actor_id="TUR_PRES", name="Recep Tayyip Erdoğan", role="President",
        role_type="head_of_state", country="Turkey", country_iso3="TUR",
        influence_domains=["Macro/Solvency", "Monetary/Price", "Infrastructure/Energy"],
        notes="Earthquake reconstruction spending. Inflation management.",
    ),
    Actor(
        actor_id="TUR_CBRT", name="Fatih Karahan", role="Governor, CBRT",
        role_type="central_bank", country="Turkey", country_iso3="TUR",
        influence_domains=["Monetary/Price", "Consumer/Household"],
        institution="Central Bank of Turkey",
        notes="Tight monetary policy to control inflation. CB independence under pressure.",
    ),

    # ── Brazil ───────────────────────────────────────────────────────────
    Actor(
        actor_id="BRA_PRES", name="Luiz Inácio Lula da Silva", role="President",
        role_type="head_of_state", country="Brazil", country_iso3="BRA",
        influence_domains=["Macro/Solvency", "Trade/External", "Demographics/Social"],
        notes="Fiscal reform challenges. Women's labor force participation push.",
    ),

    # ── Mexico ───────────────────────────────────────────────────────────
    Actor(
        actor_id="MEX_PRES", name="Claudia Sheinbaum", role="President",
        role_type="head_of_state", country="Mexico", country_iso3="MEX",
        influence_domains=["Macro/Solvency", "Trade/External", "Energy/Commodity"],
        notes="USMCA renegotiation. Water treaty compliance. Energy reform.",
    ),

    # ── Poland ───────────────────────────────────────────────────────────
    Actor(
        actor_id="POL_PM", name="Donald Tusk", role="Prime Minister",
        role_type="head_of_state", country="Poland", country_iso3="POL",
        influence_domains=["Macro/Solvency", "Infrastructure/Energy"],
        notes="4.1% GDP on defense — highest in NATO. Security super-cycle.",
    ),

    # ── Indonesia ────────────────────────────────────────────────────────
    Actor(
        actor_id="IDN_PRES", name="Prabowo Subianto", role="President",
        role_type="head_of_state", country="Indonesia", country_iso3="IDN",
        influence_domains=["Macro/Solvency", "Trade/External", "Energy/Commodity"],
        notes="Mineral downstreaming. Business licensing reform.",
    ),

    # ── South Korea ──────────────────────────────────────────────────────
    Actor(
        actor_id="KOR_PRES", name="Lee Jae Myung", role="President",
        role_type="head_of_state", country="South Korea", country_iso3="KOR",
        influence_domains=["Macro/Solvency", "Trade/External"],
        term_end="2030-06",
        notes="21st President. Sworn in 2025-06-04 after Yoon Suk-yeol impeachment. "
              "Democratic Party. Pro-China engagement, cautious on USFK. "
              "Semiconductor/battery industrial policy focus.",
    ),
    Actor(
        actor_id="KOR_BOK", name="Rhee Chang-yong", role="Governor, Bank of Korea",
        role_type="central_bank", country="South Korea", country_iso3="KOR",
        influence_domains=["Monetary/Price"],
        institution="Bank of Korea",
    ),
]


# ─── Key Bilateral Relationships ────────────────────────────────────────────

ACTOR_RELATIONSHIPS: list[ActorRelationship] = [
    # US-Canada trade axis
    ActorRelationship("USA_POTUS", "CAN_PM", "negotiating", 0.6,
                      "USMCA renegotiation. Tariff tension but structural alliance."),
    ActorRelationship("USA_USTR", "CAN_TRADE_NEG", "negotiating", 0.7,
                      "Direct counterparts in USMCA talks."),

    # US-China strategic competition
    ActorRelationship("USA_POTUS", "CHN_PRES", "adversary", 0.8,
                      "Tech decoupling escalation. Tariff confrontation."),
    ActorRelationship("USA_SEC_STATE", "CHN_PRES", "adversary", 0.7,
                      "Diplomatic channel but adversarial posture."),

    # US-Mexico trade axis
    ActorRelationship("USA_POTUS", "MEX_PRES", "negotiating", 0.5,
                      "USMCA renegotiation. Water treaty tensions."),

    # US-Israel-Iran triangle
    ActorRelationship("USA_POTUS", "ISR_PM", "ally", 0.85,
                      "Aligned on Iran pressure. Split on endgame."),
    ActorRelationship("USA_POTUS", "IRN_FM", "adversary", 0.9,
                      "Maximum pressure. But Iran signaling compromise."),
    ActorRelationship("ISR_PM", "IRN_FM", "adversary", 0.95,
                      "Existential conflict. Nuclear program at center."),

    # US-Europe energy diplomacy
    ActorRelationship("USA_SEC_STATE", "EU_DIPLOMAT", "negotiating", 0.4,
                      "Rubio courting CEE on energy. Kallas pushing back on rhetoric."),

    # Russia-Europe tension
    ActorRelationship("RUS_PRES", "DEU_CHAN", "adversary", 0.8,
                      "Energy decoupling. Defense spending response."),
    ActorRelationship("RUS_PRES", "POL_PM", "adversary", 0.9,
                      "NATO frontline. Poland's defense super-cycle driven by Russia threat."),

    # Eurozone monetary coordination
    ActorRelationship("EU_ECB", "DEU_CHAN", "institutional", 0.6,
                      "Fiscal expansion vs monetary policy tension."),
    ActorRelationship("EU_ECB", "GBR_BOE", "institutional", 0.5,
                      "Parallel policy paths post-Brexit."),

    # Fed transition
    ActorRelationship("USA_POTUS", "USA_FED", "negotiating", 0.3,
                      "Political pressure for rate cuts. Powell term ending May 2026."),

    # Saudi-US energy axis
    ActorRelationship("SAU_MBS", "USA_POTUS", "ally", 0.7,
                      "Oil market coordination. Venezuela changes dynamic."),

    # BRICS axis
    ActorRelationship("CHN_PRES", "RUS_PRES", "ally", 0.7,
                      "Strategic partnership. Energy trade. Dollar alternative."),
    ActorRelationship("CHN_PRES", "BRA_PRES", "ally", 0.5,
                      "BRICS coordination. Commodity trade."),
    ActorRelationship("IND_PM", "USA_POTUS", "ally", 0.6,
                      "Strategic partnership on tech and defense. Trade growing."),

    # Japan monetary normalization
    ActorRelationship("JPN_BOJ", "USA_FED", "institutional", 0.6,
                      "Rate divergence driving carry trade dynamics."),
]


def get_actors_by_country(country_iso3: str) -> list[Actor]:
    """Get all actors affiliated with a country."""
    return [a for a in ACTOR_REGISTRY if a.country_iso3 == country_iso3]


def get_actors_by_role_type(role_type: str) -> list[Actor]:
    """Get all actors of a specific role type (e.g., 'central_bank')."""
    return [a for a in ACTOR_REGISTRY if a.role_type == role_type]


def get_actor_relationships(actor_id: str) -> list[ActorRelationship]:
    """Get all relationships involving an actor (as source or target)."""
    return [r for r in ACTOR_RELATIONSHIPS
            if r.source_actor_id == actor_id or r.target_actor_id == actor_id]
