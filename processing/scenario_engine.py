"""
Scenario Engine — 50 scenarios from baseline to black swan.

Each scenario has:
- Probability estimate (refined by LLM)
- Affected nodes (countries + stats)
- Shock magnitude and propagation rules
- Causal chain (DAG of trigger events)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Scenario:
    """An economic scenario with probability, impact, and causal chain."""
    scenario_id: int
    title: str
    category: str  # "baseline", "likely", "moderate", "unlikely", "black_swan"
    probability_12m: float  # 0.0 to 1.0
    description: str
    primary_channel: str  # "trade", "financial", "commodity", "political", "technology"
    affected_countries: list[str]
    affected_stats: list[str]
    shock_magnitude: float  # estimated % impact on primary affected stat
    causal_chain: list[str]  # ordered sequence of trigger events
    severity: str = "medium"  # "low", "medium", "high", "critical"


# ─── The 50 Scenarios ────────────────────────────────────────────────────────
# Ordered by probability (most likely → black swan)

SCENARIO_REGISTRY: list[Scenario] = [
    # ── BASELINE (Prob > 0.60) ────────────────────────────────────────────
    Scenario(
        scenario_id=1, title="AI Capex Boom Continues",
        category="baseline", probability_12m=0.75,
        description="US cloud hyperscalers maintain 40%+ YoY AI infrastructure spending through 2026",
        primary_channel="technology",
        affected_countries=["United States", "South Korea", "Netherlands"],
        affected_stats=["real_gdp_growth", "power_consumption", "rd_to_gdp"],
        shock_magnitude=0.5, severity="low",
        causal_chain=["AI_demand_sustained", "semiconductor_orders_rise", "power_grid_strain"],
    ),
    Scenario(
        scenario_id=2, title="Fed Holds Rates Through H1",
        category="baseline", probability_12m=0.75,  # updated 2026-03-05: Iran oil shock extends hold
        description="FOMC maintains 3.50-3.75% through June 2026 on sticky inflation",
        primary_channel="financial",
        affected_countries=["United States"],
        affected_stats=["policy_rate", "yield_spread_10y3m", "lending_standards"],
        shock_magnitude=0.0, severity="low",
        causal_chain=["core_pce_sticky", "fomc_holds", "market_adjusts"],
    ),
    Scenario(
        scenario_id=3, title="China Stimulus Package",
        category="baseline", probability_12m=0.65,
        description="Beijing launches targeted fiscal stimulus to support property and consumption",
        primary_channel="financial",
        affected_countries=["China", "Australia", "Brazil", "Indonesia"],
        affected_stats=["real_gdp_growth", "private_credit_npls", "export_velocity"],
        shock_magnitude=1.0, severity="low",
        causal_chain=["property_weakness_persists", "pboc_eases", "commodity_demand_rises"],
    ),
    Scenario(
        scenario_id=4, title="USMCA Renegotiation Completes",
        category="baseline", probability_12m=0.50,
        description="US-Mexico-Canada reach updated trade agreement with stricter rules of origin",
        primary_channel="trade",
        affected_countries=["United States", "Mexico", "Canada"],
        affected_stats=["rule_of_origin_pct", "effective_tariff", "fdi_inflow"],
        shock_magnitude=-0.3, severity="low",
        causal_chain=["review_triggered", "negotiations_proceed", "compromise_reached"],
    ),
    Scenario(
        scenario_id=5, title="German Infrastructure Spend Ramps",
        category="baseline", probability_12m=0.60,
        description="€500B infrastructure and defense program begins material deployment",
        primary_channel="financial",
        affected_countries=["Germany", "Poland", "France", "Netherlands"],
        affected_stats=["public_investment_ratio", "real_gdp_growth", "debt_to_gdp"],
        shock_magnitude=0.5, severity="low",
        causal_chain=["fiscal_expansion_approved", "contracts_awarded", "gdp_lift"],
    ),

    # ── LIKELY (Prob 0.30 - 0.60) ────────────────────────────────────────
    Scenario(
        scenario_id=6, title="Private Credit NPL Spike",
        category="likely", probability_12m=0.45,
        description="Non-performing loans in private credit market rise 3-5x from 2025 levels",
        primary_channel="financial",
        affected_countries=["United States", "United Kingdom"],
        affected_stats=["private_credit_npls", "lending_standards", "bank_capital_adequacy"],
        shock_magnitude=3.0, severity="high",
        causal_chain=["maturity_wall_hits", "pik_toggles_exhaust", "npls_spike", "credit_tightens"],
    ),
    Scenario(
        scenario_id=7, title="India Overtakes Germany GDP",
        category="likely", probability_12m=0.55,
        description="India passes Germany in nominal GDP, accelerating emerging market weight shift",
        primary_channel="trade",
        affected_countries=["India", "Germany"],
        affected_stats=["real_gdp_growth", "fdi_inflow"],
        shock_magnitude=0.0, severity="low",
        causal_chain=["india_growth_sustains", "rupee_stable", "gdp_crossover"],
    ),
    Scenario(
        scenario_id=8, title="ECB Cuts Below 2%",
        category="likely", probability_12m=0.40,
        description="European Central Bank cuts deposit rate below 2% on weak growth",
        primary_channel="financial",
        affected_countries=["Germany", "France", "Italy", "Spain", "Netherlands"],
        affected_stats=["policy_rate", "real_rates", "currency_volatility"],
        shock_magnitude=-0.5, severity="medium",
        causal_chain=["eurozone_stagnation", "inflation_below_target", "ecb_cuts_aggressively"],
    ),
    Scenario(
        scenario_id=9, title="Oil Settles Below $55",
        category="likely", probability_12m=0.20,  # updated 2026-03-05: Iran war materially reduces probability
        description="WTI/Brent settle mid-$50s on Venezuela supply and weak China demand",
        primary_channel="commodity",
        affected_countries=["Saudi Arabia", "Russia", "Canada", "Brazil", "United States"],
        affected_stats=["crude_output", "wti_brent_spread", "current_account_gdp"],
        shock_magnitude=-15.0, severity="medium",
        causal_chain=["venezuela_supply_online", "opec_cuts_insufficient", "price_drops"],
    ),
    Scenario(
        scenario_id=10, title="BoJ Rate to 0.75%",
        category="likely", probability_12m=0.45,
        description="Bank of Japan continues normalization to 0.75% policy rate",
        primary_channel="financial",
        affected_countries=["Japan", "United States", "South Korea"],
        affected_stats=["policy_rate", "currency_volatility", "yield_spread_10y3m"],
        shock_magnitude=0.25, severity="medium",
        causal_chain=["japan_inflation_stable", "boj_hikes", "carry_trade_unwinds"],
    ),

    # ── MODERATE (Prob 0.15 - 0.30) ──────────────────────────────────────
    Scenario(
        scenario_id=11, title="US-China Tech Decoupling Escalation",
        category="moderate", probability_12m=0.30,
        description="US expands semiconductor export controls; China retaliates with rare earth restrictions",
        primary_channel="trade",
        affected_countries=["United States", "China", "South Korea", "Netherlands", "Japan"],
        affected_stats=["effective_tariff", "export_velocity", "rd_to_gdp"],
        shock_magnitude=-2.0, severity="high",
        causal_chain=["us_expands_controls", "china_retaliates", "supply_chains_fragment"],
    ),
    Scenario(
        scenario_id=12, title="EU CBAM Full Implementation Shock",
        category="moderate", probability_12m=0.25,
        description="EU Carbon Border Adjustment hits emerging market exports harder than expected",
        primary_channel="trade",
        affected_countries=["Turkey", "India", "Russia", "China", "Brazil"],
        affected_stats=["carbon_price", "export_velocity", "effective_tariff"],
        shock_magnitude=-1.5, severity="medium",
        causal_chain=["cbam_enforced", "compliance_costs_rise", "trade_diverted"],
    ),
    Scenario(
        scenario_id=13, title="Turkey Inflation Relapse",
        category="moderate", probability_12m=0.30,
        description="Turkish inflation reverses progress, returning above 40% on fiscal expansion",
        primary_channel="financial",
        affected_countries=["Turkey"],
        affected_stats=["core_cpi", "policy_rate", "currency_volatility", "real_rates"],
        shock_magnitude=15.0, severity="high",
        causal_chain=["earthquake_spending_continues", "lira_weakens", "inflation_surges"],
    ),
    Scenario(
        scenario_id=14, title="Shadow Banking Contagion",
        category="moderate", probability_12m=0.20,
        description="Non-bank financial intermediary failure triggers liquidity crisis in credit markets",
        primary_channel="financial",
        affected_countries=["United States", "United Kingdom", "China"],
        affected_stats=["shadow_bank_size", "lending_standards", "bank_capital_adequacy"],
        shock_magnitude=5.0, severity="critical",
        causal_chain=["nbfi_fails", "counterparty_risk_spikes", "credit_markets_freeze"],
    ),
    Scenario(
        scenario_id=15, title="Indonesian Currency Crisis",
        category="moderate", probability_12m=0.15,
        description="Rupiah depreciates sharply on current account widening and capital flight",
        primary_channel="financial",
        affected_countries=["Indonesia", "India", "Brazil", "Turkey"],
        affected_stats=["currency_volatility", "reserve_adequacy", "current_account_gdp"],
        shock_magnitude=-10.0, severity="high",
        causal_chain=["current_account_widens", "fed_holds_high", "em_outflows_accelerate"],
    ),
    Scenario(
        scenario_id=16, title="Poland Defense Spending Overheats Economy",
        category="moderate", probability_12m=0.20,
        description="4.1% GDP military spending creates labor shortages and inflation pressure",
        primary_channel="financial",
        affected_countries=["Poland"],
        affected_stats=["core_cpi", "labor_participation", "output_gap", "real_gdp_growth"],
        shock_magnitude=1.5, severity="medium",
        causal_chain=["defense_contracts_flood", "labor_market_tightens", "wages_spike"],
    ),
    Scenario(
        scenario_id=17, title="AI Productivity Gap Triggers Correction",
        category="moderate", probability_12m=0.25,
        description="Lack of measurable productivity gains from AI capex leads to tech stock selloff",
        primary_channel="financial",
        affected_countries=["United States", "South Korea", "Netherlands"],
        affected_stats=["real_gdp_growth", "housing_affordability"],
        shock_magnitude=-5.0, severity="high",
        causal_chain=["earnings_disappoint", "capex_questioned", "tech_selloff", "wealth_effect_reverses"],
    ),
    Scenario(
        scenario_id=18, title="Saudi Vision 2030 Fiscal Strain",
        category="moderate", probability_12m=0.25,
        description="Oil revenue shortfall forces Saudi to issue significant sovereign debt",
        primary_channel="commodity",
        affected_countries=["Saudi Arabia"],
        affected_stats=["debt_to_gdp", "primary_deficit", "crude_output"],
        shock_magnitude=5.0, severity="medium",
        causal_chain=["oil_revenue_drops", "vision_2030_spend_continues", "deficit_widens"],
    ),
    Scenario(
        scenario_id=19, title="Brazil Real Depreciation Spiral",
        category="moderate", probability_12m=0.20,
        description="Fiscal concerns and political uncertainty trigger real depreciation cycle",
        primary_channel="financial",
        affected_countries=["Brazil", "Mexico"],
        affected_stats=["currency_volatility", "policy_rate", "core_cpi", "net_external_debt"],
        shock_magnitude=-8.0, severity="high",
        causal_chain=["fiscal_reform_stalls", "rating_downgrade_risk", "capital_outflows"],
    ),
    Scenario(
        scenario_id=20, title="USMCA Withdrawal Threat",
        category="moderate", probability_12m=0.15,
        description="US signals USMCA termination, creating supply chain uncertainty",
        primary_channel="trade",
        affected_countries=["United States", "Mexico", "Canada"],
        affected_stats=["fdi_inflow", "export_velocity", "effective_tariff"],
        shock_magnitude=-3.0, severity="high",
        causal_chain=["negotiations_fail", "us_threatens_withdrawal", "investment_freezes"],
    ),

    # ── UNLIKELY (Prob 0.05 - 0.15) ─────────────────────────────────────
    Scenario(
        scenario_id=21, title="Italian Sovereign Debt Crisis",
        category="unlikely", probability_12m=0.10,
        description="Italian spread widens to 400bp+, triggering ECB Transmission Protection Instrument",
        primary_channel="financial",
        affected_countries=["Italy", "France", "Spain", "Germany"],
        affected_stats=["yield_spread_10y3m", "debt_to_gdp", "bank_capital_adequacy"],
        shock_magnitude=200.0, severity="critical",
        causal_chain=["growth_disappoints", "deficit_widens", "spreads_blow_out", "ecb_intervenes"],
    ),
    Scenario(
        scenario_id=22, title="Global Shipping Disruption",
        category="unlikely", probability_12m=0.12,
        description="Major shipping chokepoint (Suez/Malacca) blockage lasting 2+ months",
        primary_channel="trade",
        affected_countries=["China", "Germany", "Japan", "South Korea", "Netherlands"],
        affected_stats=["shipping_cost_index", "terms_of_trade", "inventory_levels", "core_cpi"],
        shock_magnitude=50.0, severity="high",
        causal_chain=["chokepoint_blocked", "shipping_reroutes", "costs_spike", "inflation_imports"],
    ),
    Scenario(
        scenario_id=23, title="US Debt Ceiling Default Scare",
        category="unlikely", probability_12m=0.08,
        description="Debt ceiling brinksmanship causes technical default or downgrade",
        primary_channel="financial",
        affected_countries=["United States"],
        affected_stats=["yield_spread_10y3m", "currency_volatility", "policy_rate"],
        shock_magnitude=50.0, severity="critical",
        causal_chain=["ceiling_not_raised", "treasury_misses_payment", "markets_panic"],
    ),
    Scenario(
        scenario_id=24, title="China Property Sector Collapse",
        category="unlikely", probability_12m=0.10,
        description="Major property developer default triggers systemic banking stress in China",
        primary_channel="financial",
        affected_countries=["China", "Australia", "Brazil", "Indonesia"],
        affected_stats=["private_credit_npls", "real_gdp_growth", "bank_capital_adequacy"],
        shock_magnitude=-3.0, severity="critical",
        causal_chain=["developer_defaults", "bank_exposure_revealed", "credit_freeze", "commodity_crash"],
    ),
    Scenario(
        scenario_id=25, title="European Energy Shock (Gas Supply)",
        category="unlikely", probability_12m=0.15,  # updated 2026-03-05: Iran war raises Gulf/LNG disruption risk
        description="LNG supply disruption forces European gas prices back above €100/MWh",
        primary_channel="commodity",
        affected_countries=["Germany", "France", "Italy", "Netherlands", "Poland"],
        affected_stats=["power_consumption", "core_cpi", "real_gdp_growth"],
        shock_magnitude=50.0, severity="high",
        causal_chain=["lng_supply_cut", "storage_drains", "prices_spike", "industry_curtails"],
    ),
    Scenario(
        scenario_id=26, title="Major Cyberattack on Financial Infrastructure",
        category="unlikely", probability_12m=0.08,
        description="State-sponsored cyberattack disrupts SWIFT, major exchange, or clearing house",
        primary_channel="financial",
        affected_countries=["United States", "United Kingdom"],
        affected_stats=["cyber_resilience", "bank_capital_adequacy", "currency_volatility"],
        shock_magnitude=10.0, severity="critical",
        causal_chain=["infrastructure_compromised", "settlements_freeze", "confidence_collapses"],
    ),
    Scenario(
        scenario_id=27, title="Mexico Water/Energy Treaty Violation Escalation",
        category="unlikely", probability_12m=0.10,
        description="US-Mexico water treaty dispute escalates to trade sanctions",
        primary_channel="trade",
        affected_countries=["Mexico", "United States"],
        affected_stats=["effective_tariff", "fdi_inflow", "current_account_gdp"],
        shock_magnitude=-2.0, severity="medium",
        causal_chain=["treaty_violation_declared", "us_imposes_penalties", "investment_chills"],
    ),
    Scenario(
        scenario_id=28, title="Indian Banking Stress Event",
        category="unlikely", probability_12m=0.08,
        description="Major Indian bank requires government intervention on hidden NPLs",
        primary_channel="financial",
        affected_countries=["India"],
        affected_stats=["private_credit_npls", "bank_capital_adequacy", "fdi_inflow"],
        shock_magnitude=5.0, severity="high",
        causal_chain=["npls_exposed", "bank_requires_bailout", "confidence_dips"],
    ),
    Scenario(
        scenario_id=29, title="Australian Housing Correction",
        category="unlikely", probability_12m=0.12,
        description="Australian housing prices correct 15%+ on rate sensitivity and China slowdown",
        primary_channel="financial",
        affected_countries=["Australia"],
        affected_stats=["housing_affordability", "private_credit_npls", "real_gdp_growth"],
        shock_magnitude=-15.0, severity="high",
        causal_chain=["rates_stay_high", "china_demand_falls", "housing_corrects", "wealth_effect_negative"],
    ),
    Scenario(
        scenario_id=30, title="South Korean Semiconductor Disruption",
        category="unlikely", probability_12m=0.07,
        description="Geopolitical event or natural disaster disrupts Korean semiconductor production",
        primary_channel="trade",
        affected_countries=["South Korea", "United States", "China", "Japan"],
        affected_stats=["export_velocity", "real_gdp_growth"],
        shock_magnitude=-5.0, severity="critical",
        causal_chain=["production_halted", "global_chip_shortage", "tech_sector_disrupted"],
    ),

    # ── REMOTE/TAIL (Prob 0.02 - 0.05) ──────────────────────────────────
    Scenario(
        scenario_id=31, title="Fed Leadership Shock",
        category="likely", probability_12m=0.45,  # updated 2026-03-05: Warsh nomination live; recategorized unlikely→likely
        description="Warsh confirmed as Fed chair; implements rate cuts + growth-oriented policy shift",
        primary_channel="financial",
        affected_countries=["United States"],
        affected_stats=["policy_rate", "yield_spread_10y3m", "real_rates"],
        shock_magnitude=1.5, severity="high",
        causal_chain=["powell_exits", "new_chair_surprises", "market_reprices"],
    ),
    Scenario(
        scenario_id=32, title="Russia-NATO Direct Confrontation",
        category="unlikely", probability_12m=0.03,
        description="Military incident between Russia and NATO member escalates",
        primary_channel="political",
        affected_countries=["Russia", "Poland", "Germany", "United Kingdom", "France", "United States"],
        affected_stats=["political_instability", "currency_volatility", "crude_output"],
        shock_magnitude=20.0, severity="critical",
        causal_chain=["incident_occurs", "article_5_invoked", "sanctions_maximum", "markets_crash"],
    ),
    Scenario(
        scenario_id=33, title="Dollar Hegemony Challenge",
        category="moderate", probability_12m=0.20,  # updated 2026-03-05: 49 CBDC pilots, China yuan stablecoin push, EU MiCA barriers
        description="BRICS payment system gains critical mass, reducing dollar reserve demand",
        primary_channel="financial",
        affected_countries=["United States", "China", "Russia", "India", "Brazil", "Saudi Arabia"],
        affected_stats=["reserve_adequacy", "currency_volatility", "current_account_gdp"],
        shock_magnitude=-5.0, severity="high",
        causal_chain=["brics_system_launches", "oil_priced_in_alt", "dollar_demand_drops"],
    ),
    Scenario(
        scenario_id=34, title="Pandemic 2.0 (Novel Pathogen)",
        category="unlikely", probability_12m=0.03,
        description="Novel pathogen triggers global containment measures and supply disruption",
        primary_channel="trade",
        affected_countries=list(("United States", "China", "Germany", "India", "Japan",
            "United Kingdom", "France", "Italy", "Russia", "Canada",
            "Brazil", "Spain", "Mexico", "Australia", "South Korea",
            "Turkey", "Indonesia", "Netherlands", "Saudi Arabia", "Poland")),
        affected_stats=["real_gdp_growth", "labor_participation", "shipping_cost_index"],
        shock_magnitude=-5.0, severity="critical",
        causal_chain=["pathogen_emerges", "borders_close", "supply_chains_break", "recession"],
    ),
    Scenario(
        scenario_id=35, title="Taiwan Strait Crisis",
        category="unlikely", probability_12m=0.04,
        description="Military escalation in Taiwan Strait disrupts global semiconductor supply",
        primary_channel="political",
        affected_countries=["China", "United States", "Japan", "South Korea", "Netherlands"],
        affected_stats=["political_instability", "export_velocity", "currency_volatility"],
        shock_magnitude=20.0, severity="critical",
        causal_chain=["military_escalation", "shipping_disrupted", "chip_supply_cut", "global_recession"],
    ),

    # ── BLACK SWAN (Prob < 0.02) ─────────────────────────────────────────
    Scenario(
        scenario_id=36, title="Cascading Sovereign Default Chain",
        category="black_swan", probability_12m=0.015,
        description="One major EM default triggers contagion across multiple sovereigns",
        primary_channel="financial",
        affected_countries=["Turkey", "Brazil", "Indonesia", "Mexico", "India"],
        affected_stats=["debt_to_gdp", "reserve_adequacy", "currency_volatility", "policy_rate"],
        shock_magnitude=30.0, severity="critical",
        causal_chain=["first_default", "contagion_spreads", "em_crisis", "capital_flight_global"],
    ),
    Scenario(
        scenario_id=37, title="US Treasury Market Liquidity Crisis",
        category="black_swan", probability_12m=0.02,
        description="Treasury market experiences flash crash or sustained illiquidity event",
        primary_channel="financial",
        affected_countries=["United States"],
        affected_stats=["yield_spread_10y3m", "policy_rate", "bank_capital_adequacy"],
        shock_magnitude=100.0, severity="critical",
        causal_chain=["auction_fails", "liquidity_evaporates", "fed_emergency_action"],
    ),
    Scenario(
        scenario_id=38, title="Global SWIFT System Compromise",
        category="black_swan", probability_12m=0.01,
        description="Cyberattack renders SWIFT inoperable for days, freezing international payments",
        primary_channel="financial",
        affected_countries=list(("United States", "China", "Germany", "India", "Japan",
            "United Kingdom", "France", "Italy", "Russia", "Canada",
            "Brazil", "Spain", "Mexico", "Australia", "South Korea",
            "Turkey", "Indonesia", "Netherlands", "Saudi Arabia", "Poland")),
        affected_stats=["cyber_resilience", "bank_capital_adequacy"],
        shock_magnitude=50.0, severity="critical",
        causal_chain=["swift_compromised", "payments_freeze", "trade_halts", "emergency_protocols"],
    ),
    Scenario(
        scenario_id=39, title="Eurozone Breakup Scenario",
        category="black_swan", probability_12m=0.01,
        description="Political crisis leads to major member state signaling euro exit",
        primary_channel="political",
        affected_countries=["Italy", "France", "Germany", "Spain", "Netherlands", "Poland"],
        affected_stats=["currency_volatility", "debt_to_gdp", "political_instability"],
        shock_magnitude=50.0, severity="critical",
        causal_chain=["political_crisis", "exit_rhetoric", "markets_price_breakup", "ecb_crisis"],
    ),
    Scenario(
        scenario_id=40, title="Correlated Natural Disaster Season",
        category="black_swan", probability_12m=0.015,
        description="Multiple simultaneous major natural disasters overwhelm global response capacity",
        primary_channel="commodity",
        affected_countries=["Japan", "Indonesia", "United States", "Mexico"],
        affected_stats=["real_gdp_growth", "debt_to_gdp", "shipping_cost_index"],
        shock_magnitude=-3.0, severity="critical",
        causal_chain=["disasters_cluster", "supply_chains_multiple_breaks", "insurance_crisis"],
    ),
    Scenario(
        scenario_id=41, title="AI-Driven Flash Crash Cascade",
        category="black_swan", probability_12m=0.02,
        description="Algorithmic trading systems trigger correlated selloff across major exchanges",
        primary_channel="financial",
        affected_countries=["United States", "Japan", "United Kingdom"],
        affected_stats=["bank_capital_adequacy", "lending_standards"],
        shock_magnitude=-15.0, severity="critical",
        causal_chain=["algo_feedback_loop", "circuit_breakers_fail", "liquidity_vanishes"],
    ),
    Scenario(
        scenario_id=42, title="Major Central Bank Digital Currency Disruption",
        category="black_swan", probability_12m=0.01,
        description="China's digital yuan or other CBDC causes unexpected capital flow disruption",
        primary_channel="financial",
        affected_countries=["China", "United States"],
        affected_stats=["m2_supply", "velocity_of_money", "currency_volatility"],
        shock_magnitude=5.0, severity="high",
        causal_chain=["cbdc_adoption_surges", "capital_controls_tighten", "fx_markets_disrupted"],
    ),
    Scenario(
        scenario_id=43, title="Global Rare Earth Supply Crisis",
        category="black_swan", probability_12m=0.015,
        description="China fully restricts rare earth exports, crippling global electronics/defense",
        primary_channel="commodity",
        affected_countries=["China", "United States", "Japan", "South Korea", "Germany"],
        affected_stats=["mineral_reserves", "export_velocity", "terms_of_trade"],
        shock_magnitude=30.0, severity="critical",
        causal_chain=["china_restricts_exports", "no_alternative_supply", "production_halts"],
    ),
    Scenario(
        scenario_id=44, title="Simultaneous EM Currency Crisis",
        category="black_swan", probability_12m=0.015,
        description="Dollar strength triggers synchronized currency crises across multiple EMs",
        primary_channel="financial",
        affected_countries=["Turkey", "Brazil", "Indonesia", "India", "Mexico"],
        affected_stats=["currency_volatility", "reserve_adequacy", "policy_rate", "net_external_debt"],
        shock_magnitude=-20.0, severity="critical",
        causal_chain=["dollar_surges", "em_reserves_deplete", "forced_rate_hikes", "recession_wave"],
    ),
    Scenario(
        scenario_id=45, title="Nuclear Power Plant Incident",
        category="black_swan", probability_12m=0.005,
        description="Major nuclear incident forces energy policy reversal globally",
        primary_channel="commodity",
        affected_countries=["France", "Japan", "South Korea", "United States"],
        affected_stats=["power_consumption", "green_capex", "carbon_price"],
        shock_magnitude=20.0, severity="critical",
        causal_chain=["incident_occurs", "nuclear_shutdown_wave", "energy_prices_spike"],
    ),
    Scenario(
        scenario_id=46, title="Global Internet Infrastructure Attack",
        category="black_swan", probability_12m=0.008,
        description="Undersea cable sabotage or DNS attack disrupts global internet for days",
        primary_channel="technology",
        affected_countries=list(("United States", "China", "Germany", "India", "Japan",
            "United Kingdom", "France", "Italy", "Russia", "Canada",
            "Brazil", "Spain", "Mexico", "Australia", "South Korea",
            "Turkey", "Indonesia", "Netherlands", "Saudi Arabia", "Poland")),
        affected_stats=["cyber_resilience", "real_gdp_growth"],
        shock_magnitude=-2.0, severity="critical",
        causal_chain=["infrastructure_attacked", "connectivity_lost", "commerce_halts"],
    ),
    Scenario(
        scenario_id=47, title="Runaway Sovereign Debt Spiral (US)",
        category="black_swan", probability_12m=0.02,
        description="US interest costs exceed 30% of revenue, triggering buyer's strike",
        primary_channel="financial",
        affected_countries=["United States"],
        affected_stats=["net_interest_to_revenue", "debt_to_gdp", "yield_spread_10y3m"],
        shock_magnitude=50.0, severity="critical",
        causal_chain=["interest_costs_compound", "auction_demand_drops", "yields_spike", "fiscal_crisis"],
    ),
    Scenario(
        scenario_id=48, title="Petrodollar System Collapse",
        category="black_swan", probability_12m=0.008,
        description="Saudi Arabia permanently shifts oil pricing away from USD",
        primary_channel="commodity",
        affected_countries=["United States", "Saudi Arabia", "China", "Russia"],
        affected_stats=["currency_volatility", "reserve_adequacy", "crude_output"],
        shock_magnitude=-10.0, severity="critical",
        causal_chain=["saudi_announces_shift", "dollar_reserve_demand_drops", "fx_reprices"],
    ),
    Scenario(
        scenario_id=49, title="Coordinated Global Cyber-Financial Attack",
        category="black_swan", probability_12m=0.005,
        description="State actor simultaneously attacks multiple financial systems globally",
        primary_channel="financial",
        affected_countries=list(("United States", "China", "Germany", "India", "Japan",
            "United Kingdom", "France", "Italy", "Russia", "Canada",
            "Brazil", "Spain", "Mexico", "Australia", "South Korea",
            "Turkey", "Indonesia", "Netherlands", "Saudi Arabia", "Poland")),
        affected_stats=["cyber_resilience", "bank_capital_adequacy", "political_instability"],
        shock_magnitude=50.0, severity="critical",
        causal_chain=["attack_launches", "systems_fail", "markets_close", "emergency_declared"],
    ),
    Scenario(
        scenario_id=50, title="Yellowstone/Supervolcano Event",
        category="black_swan", probability_12m=0.001,
        description="Major volcanic event disrupts global agriculture and climate for years",
        primary_channel="commodity",
        affected_countries=list(("United States", "China", "Germany", "India", "Japan",
            "United Kingdom", "France", "Italy", "Russia", "Canada",
            "Brazil", "Spain", "Mexico", "Australia", "South Korea",
            "Turkey", "Indonesia", "Netherlands", "Saudi Arabia", "Poland")),
        affected_stats=["real_gdp_growth", "core_cpi", "terms_of_trade"],
        shock_magnitude=-10.0, severity="critical",
        causal_chain=["eruption", "ash_cloud", "agriculture_collapses", "global_famine_risk"],
    ),

    # ── IRAN WAR + STABLECOIN LEGISLATIVE BATTLE (added 2026-03-05) ──────────
    Scenario(
        scenario_id=51, title="Iran War: Hormuz Closure / Sustained Oil Disruption",
        category="moderate", probability_12m=0.25,
        description=(
            "Iran-Israel-US conflict escalates to sustained disruption of Strait of Hormuz, "
            "cutting ~20% of global oil supply. Brent surges above $100, European gas nearly doubles. "
            "War is live as of 2026-02-28 — this scenario tracks escalation beyond current state."
        ),
        primary_channel="commodity",
        affected_countries=["Iran", "United States", "Saudi Arabia", "UAE", "China", "Japan", "South Korea", "India", "Germany"],
        affected_stats=["oil_price", "inflation_cpi", "real_gdp_growth", "current_account_balance", "policy_rate"],
        shock_magnitude=2.5, severity="critical",
        causal_chain=[
            "iran_war_escalates",
            "hormuz_threatened_or_closed",
            "oil_tankers_reroute_cape_of_good_hope",
            "brent_spikes_above_100",
            "global_inflation_surge",
            "rate_cut_path_blocked",
            "recession_risk_rises",
        ],
    ),
    Scenario(
        scenario_id=52, title="Warsh Confirmed: Fed Rate Cuts Begin H2 2026",
        category="likely", probability_12m=0.45,
        description=(
            "Kevin Warsh confirmed as Fed Chair after Powell exits May 2026. "
            "Delivers 2-3 rate cuts by year-end, reorienting monetary policy toward growth. "
            "Dollar weakens moderately; Treasury yields compress; crypto risk-on."
        ),
        primary_channel="financial",
        affected_countries=["United States", "Brazil", "India", "South Korea", "Germany"],
        affected_stats=["policy_rate", "yield_spread_10y3m", "real_gdp_growth", "exchange_rate_vs_usd", "equity_risk_premium"],
        shock_magnitude=0.75, severity="medium",
        causal_chain=[
            "powell_exits_may_2026",
            "warsh_senate_confirmed",
            "fomc_composition_shifts",
            "rate_cuts_signal_july",
            "dollar_weakens",
            "em_capital_inflows",
        ],
    ),
    Scenario(
        scenario_id=53, title="Clarity Act Passes Pre-July Recess",
        category="moderate", probability_12m=0.28,
        description=(
            "US Clarity Act (crypto market structure) passes Senate before July 2026 recess. "
            "Unlocks tokenization of securities, establishes dollar stablecoin framework globally. "
            "Opposed by banks, Warren Democrats, law enforcement, and foreign governments — "
            "passage requires White House political muscle holding through Iran war distraction."
        ),
        primary_channel="political",
        affected_countries=["United States", "China", "EU", "Singapore", "UAE"],
        affected_stats=["equity_risk_premium", "usd_dominance_index", "crypto_market_cap", "treasury_demand"],
        shock_magnitude=1.2, severity="high",
        causal_chain=[
            "genius_act_foundation_holds",
            "trump_political_capital_survives",
            "banking_lobby_loses_clarity_fight",
            "clarity_act_passes_july",
            "dollar_stablecoin_rails_global",
            "treasury_organic_demand_rises",
        ],
    ),
    Scenario(
        scenario_id=54, title="Clarity Act Stalls to 2027",
        category="likely", probability_12m=0.52,
        description=(
            "Clarity Act fails to pass before July 2026 recess due to Iran war consuming "
            "political oxygen, banking lobby entrenchment, unresolved Trump ethics provisions, "
            "and Democratic clock-running strategy ahead of midterms. "
            "Stablecoin strategy advances incrementally rather than transformatively. "
            "Note: Scenarios 53+54 sum to 0.80; residual 0.20 = partial/amended passage."
        ),
        primary_channel="political",
        affected_countries=["United States", "EU", "China", "Singapore"],
        affected_stats=["crypto_market_cap", "usd_dominance_index", "treasury_demand"],
        shock_magnitude=0.5, severity="low",
        causal_chain=[
            "iran_war_consumes_senate_bandwidth",
            "banking_lobby_delays_committee_vote",
            "ethics_provisions_unresolved",
            "july_recess_hits",
            "bill_punted_to_2027",
            "cbdc_alternatives_gain_ground",
        ],
    ),
    Scenario(
        scenario_id=55, title="Trump Political Crisis / Removal",
        category="unlikely", probability_12m=0.08,
        description=(
            "Trump removed, resigns, or politically incapacitated due to convergence of: "
            "Iran war crimes accusations from Europe, DOJ-Powell debacle fallout, "
            "family crypto conflict-of-interest exposure, and midterm political math. "
            "Vance succession likely; crypto strategy survives but Clarity Act stalls. "
            "Global read: deep American institutional instability → accelerates de-dollarization."
        ),
        primary_channel="political",
        affected_countries=["United States", "EU", "China", "Russia", "Israel"],
        affected_stats=["usd_dominance_index", "policy_rate", "equity_risk_premium", "sovereign_cds_spread"],
        shock_magnitude=3.0, severity="critical",
        causal_chain=[
            "iran_war_crimes_accusations",
            "doj_powell_investigation_escalates",
            "crypto_conflict_exposure",
            "political_support_collapses",
            "trump_removed_or_resigns",
            "vance_succession",
            "clarity_act_stalls",
            "dollar_confidence_shock",
        ],
    ),
    Scenario(
        scenario_id=56, title="Global Defense Procurement Supercycle",
        category="likely", probability_12m=0.60,
        description=(
            "Iran war demonstration effect triggers 5-10 year precision munitions procurement supercycle. "
            "Israel's iron dome depletion, Gulf states rearmament, NATO Article 3 stockpile concerns, "
            "and Asia-Pacific (Japan, South Korea, Taiwan, Australia) accelerated spending. "
            "Defense contractors see multi-year backlog growth. Feeds into sovereign fiscal pressure."
        ),
        primary_channel="political",
        affected_countries=["United States", "Israel", "Germany", "Poland", "Japan", "South Korea", "Australia", "Saudi Arabia"],
        affected_stats=["defense_spending_pct_gdp", "government_debt_to_gdp", "real_gdp_growth", "fiscal_balance"],
        shock_magnitude=1.0, severity="medium",
        causal_chain=[
            "iran_war_demonstrates_munitions_consumption_rate",
            "nato_stockpile_gap_revealed",
            "asia_pacific_procurement_accelerates",
            "defense_contractor_backlogs_surge",
            "sovereign_fiscal_pressure_rises",
            "debt_issuance_expands",
        ],
    ),
    Scenario(
        scenario_id=57, title="China Yuan Stablecoin Counter-Offensive",
        category="moderate", probability_12m=0.35,
        description=(
            "China accelerates yuan-backed stablecoin issuance via state-sponsored entities "
            "in response to US dollar stablecoin strategy. Targets Belt and Road corridor countries, "
            "Iranian trade settlement, and ASEAN cross-border payments. "
            "e-CNY + CIPS positioned as dollar stablecoin alternative for sanctioned entities."
        ),
        primary_channel="financial",
        affected_countries=["China", "Iran", "Saudi Arabia", "UAE", "Pakistan", "Russia", "Brazil"],
        affected_stats=["usd_dominance_index", "yuan_cross_border_share", "current_account_balance"],
        shock_magnitude=0.8, severity="medium",
        causal_chain=[
            "us_dollar_stablecoin_strategy_signals",  # fires on GENIUS passage OR Clarity Act debate
            "china_state_media_calls_for_yuan_stablecoins",
            "pboc_accelerates_ecny_international",
            "cips_expands_stablecoin_rails",
            "belt_and_road_yuan_settlement_rises",
            "dollar_share_in_em_trade_drops",
        ],
    ),
    Scenario(
        scenario_id=58, title="EM Sovereign Dollar Stablecoin Migration",
        category="moderate", probability_12m=0.25,
        description=(
            "Citizens in high-inflation emerging markets (Argentina, Turkey, Nigeria, Venezuela) "
            "mass-migrate savings into USDC/USDT, effectively dollarizing economies without "
            "government consent. Central banks lose monetary transmission. "
            "IMF warns of privatization of seigniorage — wealth concentrates in Coinbase/Circle/Tether."
        ),
        primary_channel="financial",
        affected_countries=["Argentina", "Turkey", "Nigeria", "Venezuela", "Egypt", "Pakistan"],
        affected_stats=["inflation_cpi", "exchange_rate_vs_usd", "current_account_balance", "fx_reserves"],
        shock_magnitude=2.0, severity="high",
        causal_chain=[
            "dollar_stablecoin_rails_accessible",
            "local_currency_inflation_persists",
            "citizens_adopt_usdc_usdt",
            "bank_deposit_flight",
            "monetary_policy_transmission_breaks",
            "central_bank_loses_control",
        ],
    ),
    Scenario(
        scenario_id=59, title="CBDC Coalition Acceleration",
        category="likely", probability_12m=0.60,
        description=(
            "Record 49 governments accelerate CBDC pilots in direct response to US stablecoin strategy. "
            "ECB digital euro wholesale live H2 2026, Japan yen stablecoin legal, South Korea won stablecoins. "
            "Creates parallel non-dollar settlement infrastructure across G20 + EM bloc. "
            "Dollar stablecoin dominance window narrows faster than Washington expects."
        ),
        primary_channel="financial",
        affected_countries=["EU", "Japan", "South Korea", "India", "Singapore", "Brazil", "South Africa"],
        affected_stats=["usd_dominance_index", "cross_border_payment_dollar_share", "fx_reserves"],
        shock_magnitude=0.6, severity="medium",
        causal_chain=[
            "genius_act_signals_us_dollar_stablecoin_push",
            "foreign_governments_read_as_dollar_weaponization",
            "49_cbdc_pilots_accelerate",
            "ecb_digital_euro_wholesale_live",
            "asia_instant_settlement_non_dollar",
            "dollar_optional_in_some_corridors",
        ],
    ),
]


def get_scenarios_by_category(category: str) -> list[Scenario]:
    """Get all scenarios in a given probability category."""
    return [s for s in SCENARIO_REGISTRY if s.category == category]


def get_scenarios_above_probability(threshold: float) -> list[Scenario]:
    """Get all scenarios with probability above a threshold."""
    return [s for s in SCENARIO_REGISTRY if s.probability_12m >= threshold]


def get_scenarios_affecting_country(country: str) -> list[Scenario]:
    """Get all scenarios that affect a specific country."""
    return [s for s in SCENARIO_REGISTRY if country in s.affected_countries]
