"""
Agent country assignments for news interpretation.

Assignment rationale:
  Apollo     — geopolitical/security focus → USA/RUS/CHN/SAU/IRN/ISR/UAE/QAT (highest-tension actors + active Gulf theater)
  Minerva    — governance/policy focus     → GBR/DEU/FRA/ITA (largest European democracies)
  Prometheus — finance/risk focus          → JPN/KOR/AUS/NLD (Asia-Pacific + financial hubs)
  Daedalus   — technical/industrial        → TUR/ESP/POL/CAN (manufacturing, defence, energy)
  Hephaestus — reliability/infrastructure  → IND/BRA/IDN/MEX/ARG/ZAF (emerging markets, infra, commodities)

Dual-dispatch (SECONDARY_ASSIGNMENTS):
  ISR → apollo (primary) + minerva (secondary)
  Reason: IDF ops belong with Apollo (conflict), but Israeli governance/judiciary/
  settlement policy is Minerva-relevant (EU relations, ICJ, international law).

NOTE: Hephaestus was temporarily excluded (2026-02-20 to 2026-02-26) while migrating
from Gemini to Claude-based infrastructure. IND/BRA/IDN/MEX were covered as overflow
by the other agents during that window. Hephaestus rejoined 2026-02-27 — restored here.

NOTE: IRN/UAE/QAT/ARG/ZAF added 2026-03-11 — active conflict theater coverage + missing G20 members.
NOTE: ISR added 2026-03-13 — primary combatant, zero direct coverage gap closed.

NOTE: Hermes dropped from interpreter pool 2026-05-10 (Peter directive) — Hermes role
per /media/peter/fast-storage/agents/hermes/CLAUDE.md is coordinator/messenger, not
news-interpreter. Reassigned: UAE/QAT → apollo (Gulf theater extension of IRN/ISR/SAU);
ARG/ZAF → hephaestus (EM/commodities extension of IND/BRA/IDN/MEX).
"""
from __future__ import annotations

# ISO-3 → primary agent name
AGENT_ASSIGNMENTS: dict[str, str] = {
    # Apollo — geopolitical/security
    "USA": "apollo",
    "RUS": "apollo",
    "CHN": "apollo",
    "SAU": "apollo",
    # Minerva — governance/policy
    "GBR": "minerva",
    "DEU": "minerva",
    "FRA": "minerva",
    "ITA": "minerva",
    # Prometheus — finance/risk
    "JPN": "prometheus",
    "KOR": "prometheus",
    "AUS": "prometheus",
    "NLD": "prometheus",
    # Daedalus — technical/industrial
    "TUR": "daedalus",
    "ESP": "daedalus",
    "POL": "daedalus",
    "CAN": "daedalus",
    # Hephaestus — emerging markets / infrastructure / commodities
    "IND": "hephaestus",  # South Asia: trade policy, infrastructure, tech sector
    "BRA": "hephaestus",  # Latin America: commodities, governance, fiscal policy
    "IDN": "hephaestus",  # Southeast Asia: ASEAN, energy transition, Freeport mining
    "MEX": "hephaestus",  # North America EM: USMCA, nearshoring, cartel/security
    "ARG": "hephaestus",  # Argentina: Milei reforms, IMF, peso, lithium  (reassigned from hermes 2026-05-10)
    "ZAF": "hephaestus",  # South Africa: BRICS, platinum/gold, Eskom, rand  (reassigned from hermes 2026-05-10)
    # Apollo — active conflict theater (added 2026-03-11/13)
    "IRN": "apollo",      # Iran: IRGC ops, nuclear program, oil exports, Hormuz
    "ISR": "apollo",      # Israel: IDF ops, Gaza/Lebanon/Syria, coalition support
    "UAE": "apollo",      # UAE: Gulf conflict, oil, Dubai finance, state of defence  (reassigned from hermes 2026-05-10)
    "QAT": "apollo",      # Qatar: LNG, Al Udeid, ceasefire channels, Ras Laffan  (reassigned from hermes 2026-05-10)
}

# ISO-3 → additional agents (beyond the primary in AGENT_ASSIGNMENTS).
# Dispatcher sends to ALL listed agents for these countries.
SECONDARY_ASSIGNMENTS: dict[str, list[str]] = {
    "ISR": ["minerva"],  # ICJ, international law, EU-Israel relations, settlement policy
}

# Agent → list of ISO-3 codes (primary assignments)
COUNTRIES_BY_AGENT: dict[str, list[str]] = {}
for _iso3, _agent in AGENT_ASSIGNMENTS.items():
    COUNTRIES_BY_AGENT.setdefault(_agent, []).append(_iso3)

# Extend with secondary assignments
for _iso3, _secondary_agents in SECONDARY_ASSIGNMENTS.items():
    for _agent in _secondary_agents:
        if _iso3 not in COUNTRIES_BY_AGENT.get(_agent, []):
            COUNTRIES_BY_AGENT.setdefault(_agent, []).append(_iso3)

# Tier-1 subnational units — always monitor, regardless of story volume.
# These are the provinces/states whose leaders routinely move national-level news.
TIER1_SUBNATIONAL: dict[str, list[str]] = {
    "USA": ["California", "Texas", "New York", "Florida"],
    "CHN": ["Guangdong", "Shanghai", "Beijing", "Zhejiang"],
    "IND": ["Maharashtra", "Uttar Pradesh", "Tamil Nadu", "Gujarat"],
    "BRA": ["São Paulo", "Rio de Janeiro", "Minas Gerais"],
    "RUS": ["Moscow", "Saint Petersburg", "Tatarstan"],
    "DEU": ["Bavaria", "North Rhine-Westphalia", "Baden-Württemberg"],
    "CAN": ["Ontario", "Alberta", "British Columbia", "Quebec"],
    "AUS": ["New South Wales", "Victoria", "Queensland", "Western Australia"],
    "MEX": ["Nuevo León", "Jalisco", "Estado de México"],
    "ITA": ["Lombardy", "Lazio", "Veneto"],
    "ESP": ["Catalonia", "Madrid", "Andalusia", "Basque Country"],
}

# Tier-2: only monitor when generating national-level headlines
TIER2_SUBNATIONAL: dict[str, list[str]] = {
    "USA": ["Illinois", "Pennsylvania", "Georgia", "Michigan", "Ohio"],
    "CHN": ["Jiangsu", "Shandong", "Sichuan", "Hubei"],
    "IND": ["West Bengal", "Rajasthan", "Karnataka", "Telangana"],
    "BRA": ["Bahia", "Rio Grande do Sul", "Paraná"],
    "RUS": ["Siberia Federal District", "Ural Federal District"],
    "DEU": ["Saxony", "Hamburg", "Hesse"],
    "CAN": ["Saskatchewan", "Manitoba", "Nova Scotia"],
}
