"""
Agent country assignments for news interpretation.

Assignment rationale:
  Apollo     — geopolitical/security focus → USA/RUS/CHN/SAU/IRN (highest-tension actors + active conflict)
  Minerva    — governance/policy focus     → GBR/DEU/FRA/ITA (largest European democracies)
  Prometheus — finance/risk focus          → JPN/KOR/AUS/NLD (Asia-Pacific + financial hubs)
  Daedalus   — technical/industrial        → TUR/ESP/POL/CAN (manufacturing, defence, energy)
  Hephaestus — reliability/infrastructure  → IND/BRA/IDN/MEX (emerging markets, infra, commodities)
  Hermes     — overflow/critical coverage  → ARG/ZAF/UAE/QAT (G20 gaps + active Gulf theater)

NOTE: Hephaestus was temporarily excluded (2026-02-20 to 2026-02-26) while migrating
from Gemini to Claude-based infrastructure. IND/BRA/IDN/MEX were covered as overflow
by the other agents during that window. Hephaestus rejoined 2026-02-27 — restored here.

NOTE: IRN/UAE/QAT/ARG/ZAF added 2026-03-11 — active conflict theater coverage + missing G20 members.
"""
from __future__ import annotations

# ISO-3 → agent name
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
    # Apollo — active conflict theater (added 2026-03-11)
    "IRN": "apollo",      # Iran: IRGC ops, nuclear program, oil exports, Hormuz
    # Hermes — G20 gaps + Gulf theater (added 2026-03-11)
    "ARG": "hermes",      # Argentina: Milei reforms, IMF, peso, lithium
    "ZAF": "hermes",      # South Africa: BRICS, platinum/gold, Eskom, rand
    "UAE": "hermes",      # UAE: Gulf conflict, oil, Dubai finance, state of defence
    "QAT": "hermes",      # Qatar: LNG, Al Udeid, ceasefire channels, Ras Laffan
}

# Agent → list of ISO-3 codes
COUNTRIES_BY_AGENT: dict[str, list[str]] = {}
for _iso3, _agent in AGENT_ASSIGNMENTS.items():
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
