"""
Global Financial KB — Configuration

Maps the 50 statistics to real API endpoints for 20 countries.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

# ─── Storage Paths ───────────────────────────────────────────────────────────

DATA_ROOT = Path("/fast-storage/global_financial_kb")
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
GRAPH_DIR = DATA_ROOT / "graph"

# For dev/testing when /fast-storage isn't available
DEV_DATA_ROOT = Path(__file__).parent.parent / "data"
DEV_RAW_DIR = DEV_DATA_ROOT / "raw"
DEV_PROCESSED_DIR = DEV_DATA_ROOT / "processed"
DEV_GRAPH_DIR = DEV_DATA_ROOT / "graph"


# ─── Countries ───────────────────────────────────────────────────────────────

COUNTRIES = [
    "United States", "China", "Germany", "India", "Japan",
    "United Kingdom", "France", "Italy", "Russia", "Canada",
    "Brazil", "Spain", "Mexico", "Australia", "South Korea",
    "Turkey", "Indonesia", "Netherlands", "Saudi Arabia", "Poland",
]

# ISO-3166 alpha-3 codes for API lookups
COUNTRY_CODES = {
    "United States": {"iso3": "USA", "iso2": "US", "imf": "111", "fred_prefix": ""},
    "China": {"iso3": "CHN", "iso2": "CN", "imf": "924"},
    "Germany": {"iso3": "DEU", "iso2": "DE", "imf": "134"},
    "India": {"iso3": "IND", "iso2": "IN", "imf": "534"},
    "Japan": {"iso3": "JPN", "iso2": "JP", "imf": "158"},
    "United Kingdom": {"iso3": "GBR", "iso2": "GB", "imf": "112"},
    "France": {"iso3": "FRA", "iso2": "FR", "imf": "132"},
    "Italy": {"iso3": "ITA", "iso2": "IT", "imf": "136"},
    "Russia": {"iso3": "RUS", "iso2": "RU", "imf": "922"},
    "Canada": {"iso3": "CAN", "iso2": "CA", "imf": "156"},
    "Brazil": {"iso3": "BRA", "iso2": "BR", "imf": "223"},
    "Spain": {"iso3": "ESP", "iso2": "ES", "imf": "184"},
    "Mexico": {"iso3": "MEX", "iso2": "MX", "imf": "273"},
    "Australia": {"iso3": "AUS", "iso2": "AU", "imf": "193"},
    "South Korea": {"iso3": "KOR", "iso2": "KR", "imf": "542"},
    "Turkey": {"iso3": "TUR", "iso2": "TR", "imf": "186"},
    "Indonesia": {"iso3": "IDN", "iso2": "ID", "imf": "536"},
    "Netherlands": {"iso3": "NLD", "iso2": "NL", "imf": "138"},
    "Saudi Arabia": {"iso3": "SAU", "iso2": "SA", "imf": "456"},
    "Poland": {"iso3": "POL", "iso2": "PL", "imf": "964"},
}


# ─── Stat Definitions ────────────────────────────────────────────────────────

@dataclass
class StatDefinition:
    """One of the 50 statistics tracked per country."""
    stat_id: int
    name: str
    category: str
    source_type: str  # "api", "derived", "llm"
    primary_source: str
    api_provider: Optional[str] = None  # "fred", "world_bank", "imf", "bis", "eia", "oecd"
    fred_series: Optional[dict] = None  # country_iso3 -> FRED series ID
    wb_indicator: Optional[str] = None  # World Bank indicator code
    imf_indicator: Optional[str] = None
    oecd_indicator: Optional[str] = None  # stat_name passed to OecdFetcher.fetch_indicator()
    bis_indicator: Optional[str] = None   # stat_name passed to BisFetcher as fallback
    eia_indicator: Optional[str] = None   # "productId:activityId" for EIA API
    unit: str = "percent"
    description: str = ""
    derive_from: list[str] = field(default_factory=list)  # stat names needed for derivation


# The full registry of 50 statistics with real API mappings
STAT_REGISTRY: list[StatDefinition] = [
    # ── Macro/Solvency (1-10) ────────────────────────────────────────────
    StatDefinition(
        stat_id=1, name="real_gdp_growth", category="Macro/Solvency",
        source_type="api", primary_source="World Bank / IMF",
        api_provider="world_bank", wb_indicator="NY.GDP.MKTP.KD.ZG",
        unit="percent_yoy", description="Annual real GDP growth rate",
    ),
    StatDefinition(
        stat_id=2, name="debt_to_gdp", category="Macro/Solvency",
        source_type="api", primary_source="IMF Fiscal Monitor",
        api_provider="imf", imf_indicator="GGXWDG_NGDP",
        unit="percent", description="General government gross debt as % of GDP",
    ),
    StatDefinition(
        stat_id=3, name="primary_deficit", category="Macro/Solvency",
        source_type="api", primary_source="IMF",
        api_provider="imf", imf_indicator="GGXONLB_G01_GDP_PT",
        unit="percent_gdp", description="Primary net lending/borrowing as % of GDP",
    ),
    StatDefinition(
        stat_id=4, name="net_interest_to_revenue", category="Macro/Solvency",
        source_type="api", primary_source="FRED / IMF",
        api_provider="fred",
        fred_series={"USA": "A091RC1Q027SBEA"},
        unit="percent", description="Net interest payments as % of government revenue",
    ),
    StatDefinition(
        stat_id=5, name="net_savings", category="Macro/Solvency",
        source_type="api", primary_source="World Bank",
        api_provider="world_bank", wb_indicator="NY.ADJ.NNAT.GN.ZS",
        unit="percent_gni", description="Adjusted net national savings as % of GNI",
    ),
    StatDefinition(
        stat_id=6, name="fiscal_multiplier", category="Macro/Solvency",
        source_type="llm", primary_source="IMF Research → LLM inference",
        unit="ratio", description="Estimated fiscal multiplier from literature + context",
    ),
    StatDefinition(
        stat_id=7, name="entitlement_spend", category="Macro/Solvency",
        source_type="api", primary_source="OECD SOCX",
        api_provider="oecd", oecd_indicator="entitlement_spend",
        unit="percent_gdp", description="Public social expenditure as % of GDP",
    ),
    StatDefinition(
        stat_id=8, name="tax_to_gdp", category="Macro/Solvency",
        source_type="api", primary_source="World Bank / OECD Revenue Statistics",
        api_provider="world_bank", wb_indicator="GC.TAX.TOTL.GD.ZS",
        unit="percent_gdp", description="Tax revenue as % of GDP",
    ),
    StatDefinition(
        stat_id=9, name="public_investment_ratio", category="Macro/Solvency",
        source_type="api", primary_source="World Bank / IMF",
        api_provider="world_bank", wb_indicator="NE.GDI.TOTL.ZS",
        unit="percent_gdp", description="Gross capital formation as % of GDP (proxy for investment capacity)",
    ),
    StatDefinition(
        stat_id=10, name="output_gap", category="Macro/Solvency",
        source_type="api", primary_source="IMF WEO",
        api_provider="imf", imf_indicator="GGXCNL_NGDP",
        unit="percent", description="Overall fiscal balance as % of GDP (WEO; output gap via LLM where DataMapper unavailable)",
    ),

    # ── Monetary/Price (11-20) ───────────────────────────────────────────
    StatDefinition(
        stat_id=11, name="core_cpi", category="Monetary/Price",
        source_type="api", primary_source="FRED / national stats",
        api_provider="fred",
        fred_series={
            "USA": "CPILFESL",
            "GBR": "GBRCPIALLMINMEI",
            "JPN": "JPNCPIALLMINMEI",
        },
        unit="percent_yoy", description="Core CPI (ex food & energy) YoY",
    ),
    StatDefinition(
        stat_id=12, name="policy_rate", category="Monetary/Price",
        source_type="api", primary_source="BIS / FRED / OECD",
        api_provider="fred",
        fred_series={
            # G7 + primary central banks — dedicated FRED series
            "USA": "DFEDTARU",         # Fed funds target rate upper bound
            "GBR": "BOERUKM",          # Bank of England base rate
            "JPN": "IRSTCB01JPM156N",  # Bank of Japan
            "CAN": "IRSTCB01CAM156N",  # Bank of Canada
            # ECB member states — all use the ECB main refinancing rate
            "DEU": "ECBMLFR",          # ECB Main Lending Facility Rate (covers all €-zone)
            "FRA": "ECBMLFR",
            "ITA": "ECBMLFR",
            "ESP": "ECBMLFR",
            "NLD": "ECBMLFR",
            # Other G20 — OECD/FRED interbank overnight rates (track policy closely)
            "AUS": "IRSTCB01AUM156N",  # Reserve Bank of Australia
            "KOR": "IRSTCB01KRM156N",  # Bank of Korea
            "MEX": "IRSTCB01MXM156N",  # Banco de México
            "TUR": "IRSTCB01TRM156N",  # Central Bank of Turkey
            "POL": "IRSTCB01PLM156N",  # National Bank of Poland
            "CHN": "IRSTCB01CNM156N",  # PBoC (overnight interbank proxy)
            "IND": "IRSTCB01INM156N",  # Reserve Bank of India
            "BRA": "IRSTCB01BRM156N",  # Banco Central do Brasil (SELIC proxy)
            # IDN, RUS, SAU: no reliable FRED series; BIS fallback when API stabilises
        },
        bis_indicator="policy_rate",  # BIS fallback for IDN, RUS, SAU when API available
        unit="percent", description="Central bank policy rate",
    ),
    StatDefinition(
        stat_id=13, name="yield_spread_10y3m", category="Monetary/Price",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "T10Y3M"},
        unit="percent", description="10Y-3M Treasury yield spread",
    ),
    StatDefinition(
        stat_id=14, name="m2_supply", category="Monetary/Price",
        source_type="api", primary_source="FRED / central banks",
        api_provider="fred",
        fred_series={
            "USA": "M2SL",
            "CHN": "MABMM201CNM189S",
            "GBR": "MABMM201GBM189S",
            "JPN": "MABMM201JPM189S",
        },
        unit="billion_lcu", description="M2 money supply",
    ),
    StatDefinition(
        stat_id=15, name="real_rates", category="Monetary/Price",
        source_type="derived", primary_source="Derived: policy_rate - core_cpi",
        derive_from=["policy_rate", "core_cpi"],
        unit="percent", description="Real interest rate (policy rate minus core inflation)",
    ),
    StatDefinition(
        stat_id=16, name="tips_breakeven", category="Monetary/Price",
        source_type="api", primary_source="FRED",
        api_provider="fred",
        fred_series={"USA": "T10YIE"},
        unit="percent", description="10Y TIPS breakeven inflation rate",
    ),
    StatDefinition(
        stat_id=17, name="lending_standards", category="Monetary/Price",
        source_type="api", primary_source="FRED SLOOS",
        api_provider="fred",
        fred_series={"USA": "DRTSCILM"},
        unit="net_percent", description="Net % of banks tightening C&I loan standards",
    ),
    StatDefinition(
        stat_id=18, name="velocity_of_money", category="Monetary/Price",
        source_type="derived", primary_source="Derived: GDP / M2",
        derive_from=["real_gdp_growth", "m2_supply"],
        unit="ratio", description="Velocity of M2 money stock",
    ),
    StatDefinition(
        stat_id=19, name="currency_volatility", category="Monetary/Price",
        source_type="derived", primary_source="Derived from FX data",
        unit="percent", description="30-day realized volatility of currency vs USD",
    ),
    StatDefinition(
        stat_id=20, name="cb_independence_score", category="Monetary/Price",
        source_type="llm", primary_source="Academic indices → LLM assessment",
        unit="score_0_100", description="Central bank independence score (0-100)",
    ),

    # ── Trade/External (21-30) ───────────────────────────────────────────
    StatDefinition(
        stat_id=21, name="rule_of_origin_pct", category="Trade/External",
        source_type="llm", primary_source="WTO / trade agreements → LLM",
        unit="percent", description="Avg rule-of-origin content requirement in major FTAs",
    ),
    StatDefinition(
        stat_id=22, name="current_account_gdp", category="Trade/External",
        source_type="api", primary_source="IMF BOP",
        api_provider="world_bank", wb_indicator="BN.CAB.XOKA.GD.ZS",
        unit="percent_gdp", description="Current account balance as % of GDP",
    ),
    StatDefinition(
        stat_id=23, name="effective_tariff", category="Trade/External",
        source_type="api", primary_source="World Bank / WTO",
        api_provider="world_bank", wb_indicator="TM.TAX.MRCH.WM.AR.ZS",
        unit="percent", description="Weighted mean applied tariff rate, all products",
    ),
    StatDefinition(
        stat_id=24, name="reserve_adequacy", category="Trade/External",
        source_type="api", primary_source="World Bank",
        api_provider="world_bank", wb_indicator="FI.RES.TOTL.MO",
        unit="months_imports", description="Total reserves in months of imports",
    ),
    StatDefinition(
        stat_id=25, name="net_external_debt", category="Trade/External",
        source_type="api", primary_source="World Bank",
        api_provider="world_bank", wb_indicator="DT.DOD.DECT.GN.ZS",
        unit="percent_gni", description="External debt stocks as % of GNI",
    ),
    StatDefinition(
        stat_id=26, name="export_velocity", category="Trade/External",
        source_type="api", primary_source="World Bank",
        api_provider="world_bank", wb_indicator="NE.EXP.GNFS.ZS",
        unit="percent_gdp", description="Exports of goods and services as % of GDP",
    ),
    StatDefinition(
        stat_id=27, name="fdi_inflow", category="Trade/External",
        source_type="api", primary_source="World Bank / UNCTAD",
        api_provider="world_bank", wb_indicator="BX.KLT.DINV.WD.GD.ZS",
        unit="percent_gdp", description="FDI net inflows as % of GDP",
    ),
    StatDefinition(
        stat_id=28, name="terms_of_trade", category="Trade/External",
        source_type="api", primary_source="World Bank",
        api_provider="world_bank", wb_indicator="TT.PRI.MRCH.XD.WD",
        unit="index", description="Net barter terms of trade index (2015=100)",
    ),
    StatDefinition(
        stat_id=29, name="shipping_cost_index", category="Trade/External",
        source_type="api", primary_source="Freightos Baltic Index",
        api_provider="fred",
        fred_series={"USA": "FBXGLOBAL"},  # proxy — Freightos on FRED
        unit="usd_per_feu", description="Global container freight rate index",
    ),
    StatDefinition(
        stat_id=30, name="trade_concentration", category="Trade/External",
        source_type="derived", primary_source="Derived from export data",
        derive_from=["export_velocity"],
        unit="hhi_index", description="Herfindahl index of export partner concentration",
    ),

    # ── Energy/Commodity (31-40) ─────────────────────────────────────────
    StatDefinition(
        stat_id=31, name="crude_output", category="Energy/Commodity",
        source_type="api", primary_source="EIA",
        api_provider="fred",
        fred_series={"USA": "MCRFPUS2"},
        unit="mbpd", description="Crude oil field production (million barrels/day)",
    ),
    StatDefinition(
        stat_id=32, name="wti_brent_spread", category="Energy/Commodity",
        source_type="api", primary_source="EIA / FRED",
        api_provider="fred",
        fred_series={"USA": "DCOILWTICO"},  # need both WTI and Brent
        unit="usd_per_barrel", description="WTI-Brent crude oil price spread",
    ),
    StatDefinition(
        stat_id=33, name="mineral_reserves", category="Energy/Commodity",
        source_type="llm", primary_source="USGS Mineral Commodity Summaries → LLM",
        unit="index", description="Composite mineral reserve adequacy score",
    ),
    StatDefinition(
        stat_id=34, name="esg_disclosure_score", category="Energy/Commodity",
        source_type="llm", primary_source="MSCI → LLM assessment",
        unit="score_0_100", description="Sovereign ESG disclosure quality score",
    ),
    StatDefinition(
        stat_id=35, name="power_consumption", category="Energy/Commodity",
        source_type="api", primary_source="World Bank / EIA",
        api_provider="world_bank", wb_indicator="EG.USE.ELEC.KH.PC",
        unit="kwh_per_capita", description="Electric power consumption per capita",
    ),
    StatDefinition(
        stat_id=36, name="green_capex", category="Energy/Commodity",
        source_type="llm", primary_source="IEA / BloombergNEF → LLM",
        unit="percent_gdp", description="Green/clean energy capital expenditure as % of GDP",
    ),
    StatDefinition(
        stat_id=37, name="carbon_price", category="Energy/Commodity",
        source_type="api", primary_source="ICAP / EU ETS",
        api_provider="fred",
        fred_series={"USA": "EUETSACALEGP"},  # EU ETS on FRED as proxy
        unit="eur_per_ton", description="Carbon credit price (EU ETS benchmark)",
    ),
    StatDefinition(
        stat_id=38, name="refining_margin", category="Energy/Commodity",
        source_type="derived", primary_source="Derived from crude + product prices",
        derive_from=["crude_output", "wti_brent_spread"],
        unit="usd_per_barrel", description="Crack spread / refining margin estimate",
    ),
    StatDefinition(
        stat_id=39, name="inventory_levels", category="Energy/Commodity",
        source_type="api", primary_source="EIA Weekly Petroleum Status",
        api_provider="fred",
        fred_series={"USA": "WCESTUS1"},
        unit="million_barrels", description="US crude oil commercial inventory",
    ),
    StatDefinition(
        stat_id=40, name="subsidies_gdp", category="Energy/Commodity",
        source_type="api", primary_source="IMF / OECD",
        api_provider="world_bank", wb_indicator="GC.XPN.TRFT.ZS",
        unit="percent_gdp", description="Government subsidies and transfers as % of GDP",
    ),

    # ── Institutional/Risk (41-50) ───────────────────────────────────────
    StatDefinition(
        stat_id=41, name="private_credit_npls", category="Inst./Risk",
        source_type="api", primary_source="World Bank",
        api_provider="world_bank", wb_indicator="FB.AST.NPER.ZS",
        unit="percent", description="Bank non-performing loans as % of total gross loans",
    ),
    StatDefinition(
        stat_id=42, name="bank_capital_adequacy", category="Inst./Risk",
        source_type="api", primary_source="World Bank / BIS",
        api_provider="world_bank", wb_indicator="FB.BNK.CAPA.ZS",
        unit="percent", description="Bank capital to assets ratio",
    ),
    StatDefinition(
        stat_id=43, name="corruption_index", category="Inst./Risk",
        source_type="api", primary_source="Transparency International CPI",
        api_provider="world_bank", wb_indicator="CC.PER.RNK",
        unit="percentile_rank", description="Control of corruption percentile rank",
    ),
    StatDefinition(
        stat_id=44, name="political_instability", category="Inst./Risk",
        source_type="api", primary_source="World Bank WGI",
        api_provider="world_bank", wb_indicator="PV.PER.RNK",
        unit="percentile_rank", description="Political stability percentile rank",
    ),
    StatDefinition(
        stat_id=45, name="shadow_bank_size", category="Inst./Risk",
        source_type="llm", primary_source="FSB Global Monitoring → LLM",
        unit="percent_gdp", description="Non-bank financial intermediation as % of GDP",
    ),
    StatDefinition(
        stat_id=46, name="cyber_resilience", category="Inst./Risk",
        source_type="llm", primary_source="ITU GCI → LLM",
        unit="score_0_100", description="National cybersecurity resilience score",
    ),
    StatDefinition(
        stat_id=47, name="demographic_drag", category="Inst./Risk",
        source_type="api", primary_source="World Bank / UN Population",
        api_provider="world_bank", wb_indicator="SP.POP.GROW",
        unit="percent_yoy", description="Population growth rate",
    ),
    StatDefinition(
        stat_id=48, name="labor_participation", category="Inst./Risk",
        source_type="api", primary_source="World Bank / ILO",
        api_provider="world_bank", wb_indicator="SL.TLF.CACT.ZS",
        unit="percent", description="Labor force participation rate (% of total pop 15+)",
    ),
    StatDefinition(
        stat_id=49, name="rd_to_gdp", category="Inst./Risk",
        source_type="api", primary_source="World Bank / OECD",
        api_provider="world_bank", wb_indicator="GB.XPD.RSDV.GD.ZS",
        unit="percent_gdp", description="Research and development expenditure as % of GDP",
    ),
    StatDefinition(
        stat_id=50, name="housing_affordability", category="Inst./Risk",
        source_type="api", primary_source="OECD / BIS property prices",
        api_provider="fred",
        fred_series={"USA": "CSUSHPINSA"},
        unit="index", description="House price index (S&P Case-Shiller for US)",
    ),
]


def _build_full_registry() -> list[StatDefinition]:
    """Combine macro and micro stat registries into a single unified registry."""
    try:
        from config.micro_stats import MICRO_STAT_REGISTRY
        return STAT_REGISTRY + MICRO_STAT_REGISTRY
    except ImportError:
        return STAT_REGISTRY


# The unified registry: 50 macro + 60 micro = 110 stats per country
FULL_STAT_REGISTRY: list[StatDefinition] = _build_full_registry()


def get_stat_by_name(name: str) -> Optional[StatDefinition]:
    """Look up a stat definition by its name."""
    for stat in FULL_STAT_REGISTRY:
        if stat.name == name:
            return stat
    return None


def get_stats_by_category(category: str) -> list[StatDefinition]:
    """Get all stats for a given category."""
    return [s for s in FULL_STAT_REGISTRY if s.category == category]


def get_api_stats() -> list[StatDefinition]:
    """Get all stats that come from direct API calls."""
    return [s for s in FULL_STAT_REGISTRY if s.source_type == "api"]


def get_llm_stats() -> list[StatDefinition]:
    """Get all stats that require LLM inference."""
    return [s for s in FULL_STAT_REGISTRY if s.source_type == "llm"]


def get_derived_stats() -> list[StatDefinition]:
    """Get all stats that are calculated from other stats."""
    return [s for s in FULL_STAT_REGISTRY if s.source_type == "derived"]


def get_macro_stats() -> list[StatDefinition]:
    """Get the original 50 macro-level stats."""
    return STAT_REGISTRY


def get_micro_stats() -> list[StatDefinition]:
    """Get the 60 micro-economic stats."""
    try:
        from config.micro_stats import MICRO_STAT_REGISTRY
        return MICRO_STAT_REGISTRY
    except ImportError:
        return []


def get_all_categories() -> list[str]:
    """Get all unique stat categories."""
    return sorted(set(s.category for s in FULL_STAT_REGISTRY))
