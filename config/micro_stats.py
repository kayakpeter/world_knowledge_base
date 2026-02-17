"""
Micro-Economic Stat Registry — Ground-level economic health indicators.

These 60 statistics complement the 50 macro-level stats by capturing the
real economy: households, corporations, real estate, labor, infrastructure,
and demographics. Many of these are LEADING indicators that show stress
6-18 months before it appears in macro aggregates.

Signal taxonomy:
  - LEADING: Changes direction before the overall economy
  - COINCIDENT: Moves in step with the economy
  - LAGGING: Changes direction after the economy has already shifted

Crack detection framework:
  A "thriving" economy has green across consumer, corporate, and labor.
  "Cracks appearing" shows in: rising delinquencies, falling business formation,
    CRE vacancy climbing, inventory-to-sales rising, consumer confidence diverging.
  "Crisis imminent" shows in: inverted yield curve + rising unemployment claims +
    corporate bond spreads widening + housing starts collapsing simultaneously.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config.settings import StatDefinition


# ─── Micro-Economic Stat Definitions (IDs 51-110) ───────────────────────────

MICRO_STAT_REGISTRY: list[StatDefinition] = [

    # ══════════════════════════════════════════════════════════════════════
    # CONSUMER / HOUSEHOLD (51-65)
    # These stats reflect the financial health of ordinary people.
    # Consumer stress is the earliest visible crack in any economy.
    # ══════════════════════════════════════════════════════════════════════

    StatDefinition(
        stat_id=51, name="median_household_income", category="Consumer/Household",
        source_type="api", primary_source="FRED / national stats",
        api_provider="fred",
        fred_series={"USA": "MEHOINUSA672N"},
        unit="lcu_annual",
        description="Median real household income (inflation-adjusted)",
    ),
    StatDefinition(
        stat_id=52, name="personal_savings_rate", category="Consumer/Household",
        source_type="api", primary_source="FRED / OECD",
        api_provider="fred",
        fred_series={"USA": "PSAVERT"},
        unit="percent",
        description="Personal savings as % of disposable income. LEADING — drops before recession.",
    ),
    StatDefinition(
        stat_id=53, name="household_debt_to_income", category="Consumer/Household",
        source_type="api", primary_source="BIS / FRED",
        api_provider="fred",
        fred_series={"USA": "HDTGPDUSQ163N"},
        unit="percent",
        description="Household debt as % of GDP. LEADING — rising ratio precedes credit stress.",
    ),
    StatDefinition(
        stat_id=54, name="consumer_confidence", category="Consumer/Household",
        source_type="api", primary_source="Conference Board / OECD CCI",
        api_provider="fred",
        fred_series={"USA": "CSCICP03USM665S"},
        unit="index",
        description="Consumer confidence index. LEADING — divergence from spending = warning.",
    ),
    StatDefinition(
        stat_id=55, name="credit_card_delinquency_rate", category="Consumer/Household",
        source_type="api", primary_source="FRED (Fed NY)",
        api_provider="fred",
        fred_series={"USA": "DRCCLACBS"},
        unit="percent",
        description="Credit card delinquency rate (90+ days). LEADING — spikes 6mo before recession.",
    ),
    StatDefinition(
        stat_id=56, name="auto_loan_delinquency_rate", category="Consumer/Household",
        source_type="api", primary_source="FRED (Fed NY)",
        api_provider="fred",
        fred_series={"USA": "DRSFRMACBS"},  # subprime auto proxy
        unit="percent",
        description="Auto loan delinquency rate. LEADING — subprime auto is early stress signal.",
    ),
    StatDefinition(
        stat_id=57, name="food_cpi", category="Consumer/Household",
        source_type="api", primary_source="FRED / national stats",
        api_provider="fred",
        fred_series={"USA": "CPIUFDSL"},
        unit="index",
        description="Food CPI index. COINCIDENT — food inflation erodes real income fastest for low earners.",
    ),
    StatDefinition(
        stat_id=58, name="energy_cpi", category="Consumer/Household",
        source_type="api", primary_source="FRED / national stats",
        api_provider="fred",
        fred_series={"USA": "CPIENGSL"},
        unit="index",
        description="Energy CPI (fuel + utilities). COINCIDENT — direct household cost pressure.",
    ),
    StatDefinition(
        stat_id=59, name="medical_expenditure_per_capita", category="Consumer/Household",
        source_type="api", primary_source="World Bank / CMS",
        api_provider="world_bank", wb_indicator="SH.XPD.CHEX.PC.CD",
        unit="usd_per_capita",
        description="Current health expenditure per capita. Structural burden on household budgets.",
    ),
    StatDefinition(
        stat_id=60, name="food_to_income_ratio", category="Consumer/Household",
        source_type="derived", primary_source="Derived: food CPI / median income",
        derive_from=["food_cpi", "median_household_income"],
        unit="ratio",
        description="Food spend as share of income. LEADING — above 30% = political instability risk.",
    ),
    StatDefinition(
        stat_id=61, name="consumer_credit_growth", category="Consumer/Household",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "TOTALSL"},
        unit="percent_yoy",
        description="Total consumer credit outstanding growth. LEADING — acceleration = late cycle.",
    ),
    StatDefinition(
        stat_id=62, name="retail_sales_growth", category="Consumer/Household",
        source_type="api", primary_source="FRED / national stats",
        api_provider="fred",
        fred_series={"USA": "RSXFS"},
        unit="percent_yoy",
        description="Retail sales ex food services. COINCIDENT — real-time consumer spending health.",
    ),
    StatDefinition(
        stat_id=63, name="income_inequality_gini", category="Consumer/Household",
        source_type="api", primary_source="World Bank",
        api_provider="world_bank", wb_indicator="SI.POV.GINI",
        unit="index_0_100",
        description="Gini coefficient. STRUCTURAL — rising inequality = slower growth + instability.",
    ),
    StatDefinition(
        stat_id=64, name="poverty_rate", category="Consumer/Household",
        source_type="api", primary_source="World Bank",
        api_provider="world_bank", wb_indicator="SI.POV.NAHC",
        unit="percent",
        description="National poverty headcount ratio. LAGGING — rises after recession hits.",
    ),
    StatDefinition(
        stat_id=65, name="cost_of_living_index", category="Consumer/Household",
        source_type="llm", primary_source="Numbeo / Expatistan → LLM",
        unit="index",
        description="Composite cost of living index. Cross-country comparability for household burden.",
    ),

    # ══════════════════════════════════════════════════════════════════════
    # REAL ESTATE (66-75)
    # Real estate is THE most interconnected sector — it touches banking,
    # construction employment, consumer wealth, and local government revenue.
    # CRE stress is currently the #1 crack in the US regional banking system.
    # ══════════════════════════════════════════════════════════════════════

    StatDefinition(
        stat_id=66, name="residential_price_growth", category="Real Estate",
        source_type="api", primary_source="FRED / BIS residential property",
        api_provider="fred",
        fred_series={"USA": "CSUSHPISA"},
        unit="percent_yoy",
        description="Residential home price YoY growth. COINCIDENT — wealth effect driver.",
    ),
    StatDefinition(
        stat_id=67, name="housing_starts", category="Real Estate",
        source_type="api", primary_source="FRED / national stats",
        api_provider="fred",
        fred_series={"USA": "HOUST"},
        unit="thousands_saar",
        description="New residential construction starts. LEADING — collapses before recession.",
    ),
    StatDefinition(
        stat_id=68, name="building_permits", category="Real Estate",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "PERMIT"},
        unit="thousands_saar",
        description="New building permits issued. LEADING — permits lead starts by 1-3 months.",
    ),
    StatDefinition(
        stat_id=69, name="mortgage_rate_30y", category="Real Estate",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "MORTGAGE30US"},
        unit="percent",
        description="30-year fixed mortgage rate. LEADING — rate shock kills demand with 3-6mo lag.",
    ),
    StatDefinition(
        stat_id=70, name="price_to_income_housing", category="Real Estate",
        source_type="derived", primary_source="Derived: home prices / median income",
        derive_from=["residential_price_growth", "median_household_income"],
        unit="ratio",
        description="House price to income ratio. STRUCTURAL — above 5x = affordability crisis.",
    ),
    StatDefinition(
        stat_id=71, name="cre_vacancy_rate", category="Real Estate",
        source_type="llm", primary_source="CBRE / JLL / CoStar → LLM",
        unit="percent",
        description="Commercial real estate vacancy rate (office). LEADING — drives regional bank stress.",
    ),
    StatDefinition(
        stat_id=72, name="cre_loan_delinquency", category="Real Estate",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "DRCLACBS"},
        unit="percent",
        description="CRE loan delinquency rate. LEADING — rising = regional bank capital erosion.",
    ),
    StatDefinition(
        stat_id=73, name="rental_vacancy_rate", category="Real Estate",
        source_type="api", primary_source="FRED / Census",
        api_provider="fred",
        fred_series={"USA": "RRVRUSQ156N"},
        unit="percent",
        description="Rental housing vacancy rate. COINCIDENT — tight = rent inflation, loose = overbuilt.",
    ),
    StatDefinition(
        stat_id=74, name="rent_to_income_ratio", category="Real Estate",
        source_type="llm", primary_source="Zillow / national indices → LLM",
        unit="percent",
        description="Median rent as % of median income. Above 30% = housing cost burden.",
    ),
    StatDefinition(
        stat_id=75, name="mortgage_delinquency_rate", category="Real Estate",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "DRSFRMACBS"},
        unit="percent",
        description="Residential mortgage delinquency rate. LAGGING — confirms household distress.",
    ),

    # ══════════════════════════════════════════════════════════════════════
    # CORPORATE / BUSINESS HEALTH (76-87)
    # Corporate stress propagates through layoffs → consumer stress → recession.
    # Bond spreads and business formation are the key early warning signals.
    # ══════════════════════════════════════════════════════════════════════

    StatDefinition(
        stat_id=76, name="ig_corporate_spread", category="Corporate/Business",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "BAMLC0A4CBBB"},
        unit="bps",
        description="Investment-grade corporate bond spread (BBB OAS). LEADING — widens before downturn.",
    ),
    StatDefinition(
        stat_id=77, name="hy_corporate_spread", category="Corporate/Business",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "BAMLH0A0HYM2"},
        unit="bps",
        description="High-yield corporate bond spread. LEADING — most sensitive credit risk barometer.",
    ),
    StatDefinition(
        stat_id=78, name="corporate_profit_margin", category="Corporate/Business",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "A446RC1Q027SBEA"},
        unit="percent",
        description="Corporate profits as % of GDP. COINCIDENT — margin compression = hiring slowdown.",
    ),
    StatDefinition(
        stat_id=79, name="business_inventory_to_sales", category="Corporate/Business",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "ISRATIO"},
        unit="ratio",
        description="Total business inventory-to-sales ratio. LEADING — rising = demand weakness.",
    ),
    StatDefinition(
        stat_id=80, name="new_business_formation", category="Corporate/Business",
        source_type="api", primary_source="FRED / Census",
        api_provider="fred",
        fred_series={"USA": "BABATOTALSAUS"},
        unit="thousands",
        description="New business applications. LEADING — falling = confidence/opportunity decline.",
    ),
    StatDefinition(
        stat_id=81, name="bankruptcy_filings", category="Corporate/Business",
        source_type="llm", primary_source="ABI / Epiq → LLM",
        unit="count",
        description="Commercial bankruptcy filings. COINCIDENT — rising = stress materializing.",
    ),
    StatDefinition(
        stat_id=82, name="small_business_optimism", category="Corporate/Business",
        source_type="api", primary_source="NFIB / FRED",
        api_provider="fred",
        fred_series={"USA": "STLFSI4"},  # Financial stress as proxy; NFIB not on FRED
        unit="index",
        description="Small business optimism index. LEADING — small firms feel tightening first.",
    ),
    StatDefinition(
        stat_id=83, name="ceo_to_worker_pay_ratio", category="Corporate/Business",
        source_type="llm", primary_source="EPI / SEC filings → LLM",
        unit="ratio",
        description="CEO-to-median-worker compensation ratio. STRUCTURAL — inequality + governance signal.",
    ),
    StatDefinition(
        stat_id=84, name="capex_growth", category="Corporate/Business",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "PNFI"},
        unit="percent_yoy",
        description="Private nonresidential fixed investment growth. LEADING — capex cuts precede layoffs.",
    ),
    StatDefinition(
        stat_id=85, name="pmi_manufacturing", category="Corporate/Business",
        source_type="api", primary_source="ISM / S&P Global",
        api_provider="fred",
        fred_series={"USA": "MANEMP"},  # manufacturing employment as proxy
        unit="index",
        description="Manufacturing PMI. LEADING — below 50 for 3+ months = contraction signal.",
    ),
    StatDefinition(
        stat_id=86, name="pmi_services", category="Corporate/Business",
        source_type="llm", primary_source="ISM / S&P Global → LLM",
        unit="index",
        description="Services PMI. LEADING — services is 70%+ of developed economies.",
    ),
    StatDefinition(
        stat_id=87, name="industry_concentration_hhi", category="Corporate/Business",
        source_type="llm", primary_source="OECD / national competition authorities → LLM",
        unit="hhi_index",
        description="Average industry concentration (HHI). STRUCTURAL — high = fragile, low = resilient.",
    ),

    # ══════════════════════════════════════════════════════════════════════
    # LABOR MARKET (88-95)
    # The labor market is where macro meets micro. Unemployment is lagging,
    # but initial claims, JOLTS, and quit rate are powerful leading indicators.
    # ══════════════════════════════════════════════════════════════════════

    StatDefinition(
        stat_id=88, name="unemployment_rate", category="Labor Market",
        source_type="api", primary_source="FRED / ILO",
        api_provider="fred",
        fred_series={"USA": "UNRATE"},
        unit="percent",
        description="Headline unemployment rate (U-3). LAGGING — rises after recession starts.",
    ),
    StatDefinition(
        stat_id=89, name="underemployment_rate", category="Labor Market",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "U6RATE"},
        unit="percent",
        description="U-6 underemployment rate. LAGGING — captures hidden labor market slack.",
    ),
    StatDefinition(
        stat_id=90, name="initial_jobless_claims", category="Labor Market",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "ICSA"},
        unit="thousands",
        description="Initial unemployment claims (weekly). LEADING — fastest labor market signal.",
    ),
    StatDefinition(
        stat_id=91, name="job_openings_rate", category="Labor Market",
        source_type="api", primary_source="FRED / JOLTS",
        api_provider="fred",
        fred_series={"USA": "JTSJOR"},
        unit="percent",
        description="Job openings rate (JOLTS). LEADING — falling openings = hiring freeze incoming.",
    ),
    StatDefinition(
        stat_id=92, name="quit_rate", category="Labor Market",
        source_type="api", primary_source="FRED / JOLTS",
        api_provider="fred",
        fred_series={"USA": "JTSQUR"},
        unit="percent",
        description="Voluntary quit rate. LEADING — falling quits = workers losing confidence.",
    ),
    StatDefinition(
        stat_id=93, name="wage_growth_real", category="Labor Market",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "CES0500000003"},
        unit="percent_yoy",
        description="Average hourly earnings growth (nominal; deflate for real). COINCIDENT.",
    ),
    StatDefinition(
        stat_id=94, name="youth_unemployment", category="Labor Market",
        source_type="api", primary_source="World Bank / ILO",
        api_provider="world_bank", wb_indicator="SL.UEM.1524.ZS",
        unit="percent",
        description="Youth unemployment (15-24). STRUCTURAL — high = social instability + lost generation.",
    ),
    StatDefinition(
        stat_id=95, name="labor_force_growth", category="Labor Market",
        source_type="api", primary_source="FRED / ILO",
        api_provider="fred",
        fred_series={"USA": "CLF16OV"},
        unit="percent_yoy",
        description="Civilian labor force growth. STRUCTURAL — shrinking = demographic drag on GDP.",
    ),

    # ══════════════════════════════════════════════════════════════════════
    # INFRASTRUCTURE / ENERGY GRID (96-102)
    # Infrastructure quality determines long-run competitiveness.
    # Energy grid adequacy is becoming the binding constraint on AI/growth.
    # ══════════════════════════════════════════════════════════════════════

    StatDefinition(
        stat_id=96, name="infrastructure_spend_pct_gdp", category="Infrastructure/Energy",
        source_type="api", primary_source="OECD / World Bank",
        api_provider="world_bank", wb_indicator="GC.XPN.COMP.ZS",
        unit="percent_gdp",
        description="Government infrastructure investment as % of GDP. STRUCTURAL.",
    ),
    StatDefinition(
        stat_id=97, name="electricity_generation_capacity", category="Infrastructure/Energy",
        source_type="api", primary_source="EIA / World Bank",
        api_provider="world_bank", wb_indicator="EG.ELC.PROD.KH",
        unit="gwh",
        description="Total electricity production. STRUCTURAL — grid capacity vs demand gap.",
    ),
    StatDefinition(
        stat_id=98, name="energy_grid_reserve_margin", category="Infrastructure/Energy",
        source_type="llm", primary_source="EIA / national grid operators → LLM",
        unit="percent",
        description="Generation reserve margin above peak demand. Below 15% = blackout risk.",
    ),
    StatDefinition(
        stat_id=99, name="broadband_penetration", category="Infrastructure/Energy",
        source_type="api", primary_source="World Bank / ITU",
        api_provider="world_bank", wb_indicator="IT.NET.BBND.P2",
        unit="per_100_people",
        description="Fixed broadband subscriptions per 100 people. STRUCTURAL — digital readiness.",
    ),
    StatDefinition(
        stat_id=100, name="freight_volume_index", category="Infrastructure/Energy",
        source_type="api", primary_source="FRED / BTS",
        api_provider="fred",
        fred_series={"USA": "TSIFRGHT"},
        unit="index",
        description="Transportation services freight index. LEADING — freight = real economy pulse.",
    ),
    StatDefinition(
        stat_id=101, name="airline_passenger_volume", category="Infrastructure/Energy",
        source_type="api", primary_source="FRED / IATA",
        api_provider="fred",
        fred_series={"USA": "LOADFACTOR"},
        unit="index",
        description="Airline passenger traffic. COINCIDENT — proxy for business activity + confidence.",
    ),
    StatDefinition(
        stat_id=102, name="renewable_share_generation", category="Infrastructure/Energy",
        source_type="api", primary_source="World Bank / IEA",
        api_provider="world_bank", wb_indicator="EG.ELC.RNEW.ZS",
        unit="percent",
        description="Renewable energy share of electricity. STRUCTURAL — energy transition progress.",
    ),

    # ══════════════════════════════════════════════════════════════════════
    # DEMOGRAPHICS / EDUCATION / SOCIAL (103-110)
    # Demographics are destiny over 10-20 year horizons.
    # Education quality determines innovation capacity and productivity.
    # These are the slowest-moving but most consequential indicators.
    # ══════════════════════════════════════════════════════════════════════

    StatDefinition(
        stat_id=103, name="dependency_ratio", category="Demographics/Social",
        source_type="api", primary_source="World Bank / UN",
        api_provider="world_bank", wb_indicator="SP.POP.DPND",
        unit="percent",
        description="Age dependency ratio (dependents / working-age). STRUCTURAL — fiscal pressure.",
    ),
    StatDefinition(
        stat_id=104, name="median_age", category="Demographics/Social",
        source_type="llm", primary_source="UN Population Division → LLM",
        unit="years",
        description="Median age of population. STRUCTURAL — aging = lower growth potential.",
    ),
    StatDefinition(
        stat_id=105, name="net_migration_rate", category="Demographics/Social",
        source_type="api", primary_source="World Bank / UN",
        api_provider="world_bank", wb_indicator="SM.POP.NETM",
        unit="per_1000",
        description="Net migration rate. STRUCTURAL — brain drain vs talent attraction.",
    ),
    StatDefinition(
        stat_id=106, name="education_expenditure_gdp", category="Demographics/Social",
        source_type="api", primary_source="World Bank / UNESCO",
        api_provider="world_bank", wb_indicator="SE.XPD.TOTL.GD.ZS",
        unit="percent_gdp",
        description="Government education expenditure as % of GDP. STRUCTURAL — human capital investment.",
    ),
    StatDefinition(
        stat_id=107, name="tertiary_enrollment_rate", category="Demographics/Social",
        source_type="api", primary_source="World Bank / UNESCO",
        api_provider="world_bank", wb_indicator="SE.TER.ENRR",
        unit="percent_gross",
        description="Tertiary education enrollment rate. STRUCTURAL — innovation pipeline.",
    ),
    StatDefinition(
        stat_id=108, name="life_expectancy", category="Demographics/Social",
        source_type="api", primary_source="World Bank",
        api_provider="world_bank", wb_indicator="SP.DYN.LE00.IN",
        unit="years",
        description="Life expectancy at birth. STRUCTURAL — composite health/development indicator.",
    ),
    StatDefinition(
        stat_id=109, name="urbanization_rate", category="Demographics/Social",
        source_type="api", primary_source="World Bank",
        api_provider="world_bank", wb_indicator="SP.URB.TOTL.IN.ZS",
        unit="percent",
        description="Urban population as % of total. STRUCTURAL — urbanization drives productivity.",
    ),
    StatDefinition(
        stat_id=110, name="homicide_rate", category="Demographics/Social",
        source_type="api", primary_source="World Bank / UNODC",
        api_provider="world_bank", wb_indicator="VC.IHR.PSRC.P5",
        unit="per_100k",
        description="Intentional homicides per 100k. STRUCTURAL — social stability proxy.",
    ),
]


# ─── Crack Detection Framework ──────────────────────────────────────────────
# These are the indicator combinations that signal regime transitions.
# Each "crack pattern" defines which micro stats to monitor together
# and what threshold combination signals trouble.

@dataclass
class CrackPattern:
    """
    A combination of indicators that collectively signal economic stress.

    When trigger_count or more indicators breach their thresholds simultaneously,
    the pattern is "active" and signals the specified regime transition.
    """
    pattern_id: int
    name: str
    description: str
    signal_type: str  # "leading", "confirming", "crisis"
    lead_time_months: int  # how far ahead this pattern typically signals
    indicators: list[dict]  # [{stat_name, direction, threshold, description}]
    trigger_count: int  # how many indicators must breach to activate
    target_regime: str  # "cracks_appearing" or "crisis_imminent"


CRACK_PATTERNS: list[CrackPattern] = [

    CrackPattern(
        pattern_id=1,
        name="Consumer Credit Stress",
        description="Household balance sheet deterioration preceding consumption collapse",
        signal_type="leading", lead_time_months=9,
        target_regime="cracks_appearing",
        trigger_count=3,
        indicators=[
            {"stat_name": "credit_card_delinquency_rate", "direction": "above", "threshold": 3.5,
             "description": "CC delinquencies above 3.5% signal consumer stress"},
            {"stat_name": "auto_loan_delinquency_rate", "direction": "above", "threshold": 4.0,
             "description": "Subprime auto is the canary in the coal mine"},
            {"stat_name": "personal_savings_rate", "direction": "below", "threshold": 3.0,
             "description": "Savings rate below 3% = no buffer left"},
            {"stat_name": "household_debt_to_income", "direction": "above", "threshold": 80.0,
             "description": "Debt-to-income above 80% = unsustainable"},
            {"stat_name": "consumer_confidence", "direction": "below_delta", "threshold": -15.0,
             "description": "Confidence dropping 15+ points in 3 months"},
        ],
    ),

    CrackPattern(
        pattern_id=2,
        name="Real Estate Distress",
        description="Property market stress radiating into banking sector",
        signal_type="leading", lead_time_months=12,
        target_regime="cracks_appearing",
        trigger_count=3,
        indicators=[
            {"stat_name": "cre_vacancy_rate", "direction": "above", "threshold": 20.0,
             "description": "Office vacancy above 20% = structural oversupply"},
            {"stat_name": "cre_loan_delinquency", "direction": "above", "threshold": 3.0,
             "description": "CRE delinquencies above 3% = bank capital stress"},
            {"stat_name": "housing_starts", "direction": "below_delta", "threshold": -25.0,
             "description": "Housing starts dropping 25%+ YoY"},
            {"stat_name": "mortgage_delinquency_rate", "direction": "above", "threshold": 4.0,
             "description": "Mortgage delinquencies above 4% = systemic"},
            {"stat_name": "residential_price_growth", "direction": "below", "threshold": -5.0,
             "description": "Home prices declining 5%+ YoY"},
        ],
    ),

    CrackPattern(
        pattern_id=3,
        name="Corporate Credit Deterioration",
        description="Business sector leveraging up while margins compress",
        signal_type="leading", lead_time_months=6,
        target_regime="cracks_appearing",
        trigger_count=3,
        indicators=[
            {"stat_name": "hy_corporate_spread", "direction": "above", "threshold": 500,
             "description": "HY spreads above 500bp = risk aversion"},
            {"stat_name": "ig_corporate_spread", "direction": "above", "threshold": 200,
             "description": "IG spreads above 200bp = broad credit stress"},
            {"stat_name": "corporate_profit_margin", "direction": "below_delta", "threshold": -2.0,
             "description": "Profit margins compressing 2pp+ = layoffs incoming"},
            {"stat_name": "business_inventory_to_sales", "direction": "above", "threshold": 1.45,
             "description": "Inventory-to-sales above 1.45 = demand weakness"},
            {"stat_name": "new_business_formation", "direction": "below_delta", "threshold": -20.0,
             "description": "Business formation dropping 20%+ = confidence collapse"},
        ],
    ),

    CrackPattern(
        pattern_id=4,
        name="Labor Market Cooling",
        description="Jobs market transitioning from tight to loose",
        signal_type="leading", lead_time_months=6,
        target_regime="cracks_appearing",
        trigger_count=3,
        indicators=[
            {"stat_name": "initial_jobless_claims", "direction": "above", "threshold": 300,
             "description": "Weekly claims above 300K = labor market weakening"},
            {"stat_name": "job_openings_rate", "direction": "below_delta", "threshold": -1.5,
             "description": "JOLTS openings rate dropping 1.5pp = hiring freeze"},
            {"stat_name": "quit_rate", "direction": "below", "threshold": 2.0,
             "description": "Quit rate below 2% = workers scared to leave"},
            {"stat_name": "wage_growth_real", "direction": "below", "threshold": 0.0,
             "description": "Negative real wage growth = purchasing power erosion"},
            {"stat_name": "capex_growth", "direction": "below", "threshold": -5.0,
             "description": "Capex declining 5%+ = companies pulling back"},
        ],
    ),

    CrackPattern(
        pattern_id=5,
        name="Full Recession Signal",
        description="Multiple systems failing simultaneously — crisis imminent",
        signal_type="confirming", lead_time_months=3,
        target_regime="crisis_imminent",
        trigger_count=4,
        indicators=[
            {"stat_name": "yield_spread_10y3m", "direction": "below", "threshold": -0.5,
             "description": "Deeply inverted yield curve"},
            {"stat_name": "pmi_manufacturing", "direction": "below", "threshold": 45.0,
             "description": "Manufacturing in deep contraction"},
            {"stat_name": "initial_jobless_claims", "direction": "above", "threshold": 350,
             "description": "Claims spiking above 350K"},
            {"stat_name": "hy_corporate_spread", "direction": "above", "threshold": 700,
             "description": "HY spreads blowing out above 700bp"},
            {"stat_name": "consumer_confidence", "direction": "below", "threshold": 80,
             "description": "Consumer confidence collapsing"},
            {"stat_name": "freight_volume_index", "direction": "below_delta", "threshold": -10.0,
             "description": "Freight volumes dropping 10%+ = trade contracting"},
        ],
    ),

    CrackPattern(
        pattern_id=6,
        name="Emerging Market Vulnerability",
        description="Combination signaling EM fragility — applicable to non-G7 countries",
        signal_type="leading", lead_time_months=6,
        target_regime="cracks_appearing",
        trigger_count=3,
        indicators=[
            {"stat_name": "food_to_income_ratio", "direction": "above", "threshold": 0.30,
             "description": "Food > 30% of income = social unrest risk"},
            {"stat_name": "youth_unemployment", "direction": "above", "threshold": 25.0,
             "description": "Youth unemployment above 25% = instability"},
            {"stat_name": "currency_volatility", "direction": "above", "threshold": 15.0,
             "description": "FX vol above 15% = capital flight risk"},
            {"stat_name": "reserve_adequacy", "direction": "below", "threshold": 3.0,
             "description": "Reserves below 3 months imports = crisis vulnerable"},
            {"stat_name": "net_external_debt", "direction": "above", "threshold": 60.0,
             "description": "External debt above 60% of GNI = default risk"},
        ],
    ),

    CrackPattern(
        pattern_id=7,
        name="Infrastructure Constraint",
        description="Growth hitting physical infrastructure limits",
        signal_type="confirming", lead_time_months=12,
        target_regime="cracks_appearing",
        trigger_count=2,
        indicators=[
            {"stat_name": "energy_grid_reserve_margin", "direction": "below", "threshold": 15.0,
             "description": "Grid reserve below 15% = blackout risk, growth ceiling"},
            {"stat_name": "freight_volume_index", "direction": "above_delta", "threshold": 10.0,
             "description": "Freight surging while infrastructure static = bottleneck"},
            {"stat_name": "broadband_penetration", "direction": "below", "threshold": 30.0,
             "description": "Low broadband = digital economy excluded"},
        ],
    ),
]
