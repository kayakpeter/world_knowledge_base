# Global Financial Knowledge Base — Architecture

## Overview
A sovereign economic knowledge graph system that ingests real macroeconomic
AND micro-economic data for the top 20 economies, uses LLM inference to derive
relationship weights and scenario probabilities, and populates a queryable
knowledge base with Markov state-transition modeling, shock propagation,
and early-warning crack detection.

## Scale
- 110 statistics per country (50 macro + 60 micro)
- 20 countries → 2,200 stat nodes + 20 sovereign nodes = 2,220 total nodes
- ~2,400 edges (monitors + contagion + trade dependencies)
- 50 scenarios (baseline → black swan)
- 7 crack detection patterns with multi-indicator trigger logic

## Compute Phases

### Phase 1: Ingestion (I/O-bound)
- Pull 1,000 data points (50 stats × 20 countries) from real APIs
- Sources: FRED, World Bank, IMF, BIS, EIA, OECD
- Store raw observations as Parquet via Polars
- Runs on any machine — bottleneck is API rate limits

### Phase 2: LLM Processing (GPU-bound)
- Infer edge weights between nodes from correlation + contextual reasoning
- Generate probability distributions for 50 scenarios
- Extract structured data from unstructured sources (news, reports)
- **Dev mode**: Claude API (no GPU needed)
- **Lambda mode**: Local model (Mixtral/Llama-70B on A6000 48GB)

### Phase 3: KB Population + Inference (Memory-bound)
- Build NetworkX DiGraph with typed nodes and weighted edges
- Run proper HMM with Baum-Welch parameter estimation
- Shock propagation via breadth-first contagion with decay
- Scenario simulation with Monte Carlo over transition matrix

## Data Source Registry

### Macro/Solvency (10 stats)
| # | Statistic | Primary Source | API | Free |
|---|-----------|---------------|-----|------|
| 1 | Real GDP Growth | World Bank / IMF WEO | REST | Yes |
| 2 | Debt-to-GDP | IMF Fiscal Monitor | REST | Yes |
| 3 | Primary Deficit | IMF / FRED (US) | REST | Yes |
| 4 | Net Interest/Revenue | FRED / national treasuries | REST | Yes |
| 5 | Net Savings | World Bank | REST | Yes |
| 6 | Fiscal Multiplier | IMF Research → LLM-derived | LLM | N/A |
| 7 | Entitlement Spend | OECD Social Expenditure | REST | Yes |
| 8 | Tax-to-GDP | OECD Revenue Statistics | REST | Yes |
| 9 | Public Investment Ratio | IMF Investment & Capital Stock | REST | Yes |
| 10 | Output Gap | IMF WEO / OECD | REST | Yes |

### Monetary/Price (10 stats)
| # | Statistic | Primary Source | API | Free |
|---|-----------|---------------|-----|------|
| 11 | Core CPI/PCE | FRED / national stats offices | REST | Yes |
| 12 | Policy Rate | BIS / FRED | REST | Yes |
| 13 | Yield Spread (10Y-3M) | FRED / national bond data | REST | Yes |
| 14 | M2 Supply | FRED / national central banks | REST | Yes |
| 15 | Real Rates | Derived (policy rate - inflation) | Calc | N/A |
| 16 | TIPS Breakeven | FRED (US) / derived elsewhere | REST | Yes |
| 17 | Lending Standards | FRED SLOOS / ECB BLS | REST | Yes |
| 18 | Velocity of Money | Derived (GDP / M2) | Calc | N/A |
| 19 | Currency Volatility | Derived from FX data | Calc | N/A |
| 20 | CB Independence Score | Academic indices → LLM | LLM | N/A |

### Trade/External (10 stats)
| # | Statistic | Primary Source | API | Free |
|---|-----------|---------------|-----|------|
| 21 | Rule of Origin % | WTO / trade agreements → LLM | LLM | N/A |
| 22 | Current Account/GDP | IMF BOP | REST | Yes |
| 23 | Effective Tariff | WTO / UNCTAD TRAINS | REST | Yes |
| 24 | Reserve Adequacy | IMF COFER | REST | Yes |
| 25 | Net External Debt | World Bank / IMF | REST | Yes |
| 26 | Export Velocity | UN Comtrade / WTO | REST | Yes |
| 27 | FDI Inflow | UNCTAD / World Bank | REST | Yes |
| 28 | Terms of Trade | World Bank / national stats | REST | Yes |
| 29 | Shipping Cost Index | Freightos Baltic Index | REST | Freemium |
| 30 | Trade Concentration | UN Comtrade → derived | Calc | N/A |

### Energy/Commodity (10 stats)
| # | Statistic | Primary Source | API | Free |
|---|-----------|---------------|-----|------|
| 31 | Crude Output | EIA / OPEC MOMR | REST | Yes |
| 32 | WTI/Brent Spread | EIA / FRED | REST | Yes |
| 33 | Mineral Reserves | USGS Mineral Commodity | REST | Yes |
| 34 | ESG Disclosure Score | MSCI → LLM assessment | LLM | N/A |
| 35 | Power Consumption | EIA / IEA | REST | Yes |
| 36 | Green Capex | IEA / BloombergNEF → LLM | LLM | N/A |
| 37 | Carbon Price | ICAP / EU ETS | REST | Freemium |
| 38 | Refining Margin | EIA → derived | Calc | N/A |
| 39 | Inventory Levels | EIA / API Weekly | REST | Yes |
| 40 | Subsidies/GDP | IMF / OECD | REST | Yes |

### Institutional/Risk (10 stats)
| # | Statistic | Primary Source | API | Free |
|---|-----------|---------------|-----|------|
| 41 | Private Credit NPLs | BIS / national regulators | REST | Yes |
| 42 | Bank Capital Adequacy | BIS Basel III monitoring | REST | Yes |
| 43 | Corruption Index | Transparency Intl CPI | REST | Yes |
| 44 | Political Instability | World Bank WGI | REST | Yes |
| 45 | Shadow Bank Size | FSB Global Monitoring | PDF→LLM | N/A |
| 46 | Cyber Resilience | ITU GCI → LLM | LLM | N/A |
| 47 | Demographic Drag | UN Population Division | REST | Yes |
| 48 | Labor Participation | ILO / FRED / national | REST | Yes |
| 49 | R&D/GDP | OECD / UNESCO | REST | Yes |
| 50 | Housing Affordability | OECD / national indices | REST | Yes |

**Summary**: ~35 stats from direct APIs, ~8 derived/calculated, ~7 requiring LLM inference

## Micro-Economic Layer (Stats 51-110)

### Consumer/Household (15 stats)
Median household income, savings rate, household debt-to-income, consumer confidence,
credit card delinquencies, auto loan delinquencies, food CPI, energy CPI, medical
expenditure, food-to-income ratio, consumer credit growth, retail sales, Gini
coefficient, poverty rate, cost of living index.

### Real Estate (10 stats)
Residential price growth, housing starts, building permits, mortgage rates,
price-to-income ratio, CRE vacancy, CRE loan delinquency, rental vacancy,
rent-to-income ratio, mortgage delinquency.

### Corporate/Business (12 stats)
IG corporate spreads, HY corporate spreads, corporate profit margins,
inventory-to-sales ratio, new business formation, bankruptcy filings,
small business optimism, CEO-to-worker pay ratio, capex growth,
manufacturing PMI, services PMI, industry concentration.

### Labor Market (8 stats)
Unemployment (U-3), underemployment (U-6), initial jobless claims,
job openings rate (JOLTS), quit rate, real wage growth, youth unemployment,
labor force growth.

### Infrastructure/Energy (7 stats)
Infrastructure spend/GDP, electricity generation, grid reserve margin,
broadband penetration, freight volume, airline passenger volume,
renewable share of generation.

### Demographics/Social (8 stats)
Dependency ratio, median age, net migration rate, education spend/GDP,
tertiary enrollment, life expectancy, urbanization rate, homicide rate.

**Micro summary**: 86 from APIs, 17 LLM-inferred, 7 derived

## Crack Detection Framework

The crack detector monitors 7 multi-indicator patterns that signal regime transitions:

| # | Pattern | Indicators | Trigger | Lead Time | Target Regime |
|---|---------|-----------|---------|-----------|---------------|
| 1 | Consumer Credit Stress | 5 | 3 | 9 months | Cracks Appearing |
| 2 | Real Estate Distress | 5 | 3 | 12 months | Cracks Appearing |
| 3 | Corporate Credit Deterioration | 5 | 3 | 6 months | Cracks Appearing |
| 4 | Labor Market Cooling | 5 | 3 | 6 months | Cracks Appearing |
| 5 | Full Recession Signal | 6 | 4 | 3 months | Crisis Imminent |
| 6 | Emerging Market Vulnerability | 5 | 3 | 6 months | Cracks Appearing |
| 7 | Infrastructure Constraint | 3 | 2 | 12 months | Cracks Appearing |

Each indicator has a threshold and direction. When trigger_count indicators
breach simultaneously, the pattern activates and signals the regime transition.

## Graph Schema

### Node Types
- `Sovereign` — country node with GDP rank, region, currency
- `Statistic` — individual metric with category, value, delta, source
- `Scenario` — event node with probability, severity, affected nodes
- `MarkovState` — one of {Tranquil, Turbulent, Crisis} per country

### Edge Types
- `MONITORS` — Sovereign → Statistic (weight=1.0, always present)
- `CONTAGION` — Statistic → Statistic cross-border (weight=correlation)
- `TRIGGERS` — Statistic → Scenario (weight=probability)
- `TRANSITIONS` — MarkovState → MarkovState (weight=transition prob)
- `TRADE_DEPENDS` — Sovereign → Sovereign (weight=trade volume share)

## HMM Design
Not a static 3×3 matrix. Proper implementation:
- **Hidden states**: {Tranquil, Turbulent, Crisis} per country (60 total)
- **Observations**: The 50 statistics per country, discretized into {Low, Normal, High}
- **Emission matrix**: P(observation | state), learned from historical data
- **Transition matrix**: Starts from prior, updated via Baum-Welch
- **Inference**: Viterbi for most-likely state sequence, Forward-Backward for posteriors

## Lambda Deployment Notes
- A6000 48GB VRAM → can run Mixtral-8x7B or Llama-3-70B (GPTQ 4-bit)
- Graph with 2,220 nodes + 2,400 edges fits easily in CPU RAM (~50MB)
- Full ingestion: 110 stats × 20 countries = 2,200 API calls → ~30 min with rate limiting
- LLM inference: ~17 LLM-inferred stats × 20 countries = 340 calls for stat values
- Edge weight inference: ~7,000 LLM calls for cross-border contagion weights
- Total LLM processing: ~7,340 calls → ~3-4 hours on A6000 with 70B model
- Crack detection: CPU-only, runs in milliseconds per country
- Monte Carlo scenario sim (10,000 runs × 20 countries) → minutes on CPU
- **Threadripper case**: Full daily refresh with all 20 countries = viable on A6000,
  but expanding to 50+ countries or sub-national data will need the Threadripper's
  memory bandwidth for graph operations + multiple GPU inference
