"""
Global Financial Knowledge Base — Main Entry Point

Orchestrates the full pipeline:
1. Ingest data from real APIs (World Bank, FRED, IMF)
2. Build the knowledge graph with typed nodes and weighted edges
3. Run HMM state estimation per country
4. Execute scenario analysis with shock propagation
5. (Optional) LLM processing for edge weights and stat inference

Usage:
    # Full pipeline (requires FRED_API_KEY env var for US data)
    python main.py --mode full

    # Graph-only (use cached/mock data, skip API calls)
    python main.py --mode graph-only

    # Scenario analysis only
    python main.py --mode scenarios

    # Test HMM on synthetic data
    python main.py --mode hmm-test
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    COUNTRIES,
    COUNTRY_CODES,
    STAT_REGISTRY,
    FULL_STAT_REGISTRY,
    DEV_DATA_ROOT,
    DEV_RAW_DIR,
    DEV_PROCESSED_DIR,
    DEV_GRAPH_DIR,
)
from ingestion.pipeline import IngestionPipeline
from knowledge_base.graph_builder import KnowledgeGraphBuilder
from models.hmm import SovereignHMM, HMMParams, discretize_observations, STATE_NAMES
from processing.scenario_engine import (
    SCENARIO_REGISTRY,
    get_scenarios_by_category,
    get_scenarios_affecting_country,
)
from processing.crack_detector import CrackDetector
from processing.news_intelligence import (
    NewsIntelligenceProcessor,
    DEMO_EVENTS,
    SAMPLE_HEADLINES,
)
from config.actors import ACTOR_REGISTRY
from processing.strategic_analysis import (
    StrategicAnalyzer,
    GENERATIONAL_PLANS,
    EXAMPLE_DEEP_ANALYSIS,
)
from production.briefing_generator import WeeklyBriefingGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("main")


async def run_ingestion(fred_api_key: str | None = None) -> pl.DataFrame:
    """Phase 1: Ingest data from external APIs."""
    pipeline = IngestionPipeline(
        fred_api_key=fred_api_key,
        output_dir=DEV_RAW_DIR,
    )
    observations_df = await pipeline.run()
    return observations_df


def build_knowledge_graph(observations_df: pl.DataFrame) -> KnowledgeGraphBuilder:
    """Phase 2: Build the knowledge graph from observations."""
    builder = KnowledgeGraphBuilder()
    builder.build_from_observations(observations_df)

    # Export for inspection
    DEV_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    builder.export_graph(DEV_GRAPH_DIR / "financial_kg.json")

    return builder


def run_hmm_analysis(observations_df: pl.DataFrame) -> dict[str, dict]:
    """
    Phase 3: Run HMM state estimation for each country.

    Uses the ingested observations to estimate current economic states.
    """
    logger.info("Running HMM state estimation for %d countries...", len(COUNTRIES))

    results: dict[str, dict] = {}

    for country in COUNTRIES:
        codes = COUNTRY_CODES.get(country, {})
        iso3 = codes.get("iso3", "")

        # Get all observations for this country
        if not observations_df.is_empty():
            country_obs = observations_df.filter(pl.col("country_iso3") == iso3)
        else:
            country_obs = pl.DataFrame()

        # Create country-specific HMM
        params = HMMParams(country=country, country_iso3=iso3)
        hmm = SovereignHMM(params)

        if not country_obs.is_empty() and "value" in country_obs.columns:
            # Discretize the observations into {Low=0, Normal=1, High=2}
            values = country_obs["value"].to_numpy()
            observations_discrete = discretize_observations(values)

            if len(observations_discrete) >= 3:
                # Estimate current state
                state_estimate = hmm.current_state_estimate(observations_discrete)

                # Run Monte Carlo forecast
                ml_state_idx = np.argmax(
                    [state_estimate["probabilities"][s] for s in STATE_NAMES]
                )
                forecast = hmm.forecast_summary(
                    current_state_idx=ml_state_idx,
                    n_steps=12,
                    n_simulations=5000,
                )

                results[country] = {
                    "state_estimate": state_estimate,
                    "forecast": forecast,
                    "n_observations": len(observations_discrete),
                }
                continue

        # Fallback: use prior probabilities
        results[country] = {
            "state_estimate": {
                "country": country,
                "state": "Tranquil",
                "probabilities": {"Tranquil": 0.60, "Turbulent": 0.30, "Crisis": 0.10},
                "n_observations": 0,
            },
            "forecast": hmm.forecast_summary(0, 12, 5000),
            "n_observations": 0,
        }

    return results


def run_scenario_analysis(kb: KnowledgeGraphBuilder) -> None:
    """Phase 4: Scenario analysis with shock propagation."""
    logger.info("=" * 70)
    logger.info("SCENARIO ANALYSIS")
    logger.info("=" * 70)

    categories = ["baseline", "likely", "moderate", "unlikely", "black_swan"]

    for cat in categories:
        scenarios = get_scenarios_by_category(cat)
        logger.info(
            "\n%s (%d scenarios):", cat.upper().replace("_", " "), len(scenarios)
        )
        for s in scenarios:
            logger.info(
                "  [%02d] %s — P=%.1f%% — Severity=%s — Channel=%s",
                s.scenario_id, s.title,
                s.probability_12m * 100, s.severity, s.primary_channel,
            )

    # Demonstrate shock propagation for a high-impact scenario
    logger.info("\n" + "=" * 70)
    logger.info("SHOCK PROPAGATION DEMO: Private Credit NPL Spike (Scenario #6)")
    logger.info("=" * 70)

    impacts = kb.propagate_shock(
        source_node="USA_private_credit_npls",
        shock_magnitude=3.0,  # 3x increase in NPLs
        decay_rate=0.6,
    )

    if impacts:
        # Sort by absolute impact
        sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        logger.info("Top affected nodes:")
        for node, impact in sorted_impacts[:15]:
            logger.info("  %s → impact=%.4f", node, impact)

    # Show country-specific scenario exposure
    logger.info("\n" + "=" * 70)
    logger.info("COUNTRY SCENARIO EXPOSURE")
    logger.info("=" * 70)

    for country in COUNTRIES[:5]:  # Top 5 for brevity
        country_scenarios = get_scenarios_affecting_country(country)
        critical = [s for s in country_scenarios if s.severity == "critical"]
        high = [s for s in country_scenarios if s.severity == "high"]
        logger.info(
            "  %s: %d total scenarios (%d critical, %d high)",
            country, len(country_scenarios), len(critical), len(high),
        )


def run_hmm_test() -> None:
    """Test the HMM implementation with synthetic data."""
    logger.info("=" * 70)
    logger.info("HMM TEST — Synthetic Data")
    logger.info("=" * 70)

    # Create a country HMM with known parameters
    params = HMMParams(country="Test Country", country_iso3="TST")
    hmm = SovereignHMM(params)

    # Generate synthetic observations: start Tranquil, transition to Turbulent
    np.random.seed(42)
    true_states = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 1, 1, 0, 0])
    observations = np.array([
        np.random.choice(3, p=params.emission_matrix[s]) for s in true_states
    ])

    logger.info("True states:     %s", [STATE_NAMES[s] for s in true_states])
    logger.info("Observations:    %s", observations.tolist())

    # Viterbi decoding
    decoded_states, log_prob = hmm.viterbi(observations)
    logger.info("Decoded states:  %s", [STATE_NAMES[s] for s in decoded_states])
    logger.info("Log probability: %.4f", log_prob)

    # State posteriors
    gamma = hmm.state_posteriors(observations)
    logger.info("\nState posteriors (last 5 time steps):")
    for t in range(len(observations) - 5, len(observations)):
        probs = {STATE_NAMES[i]: f"{gamma[t, i]:.3f}" for i in range(3)}
        logger.info("  t=%d: %s", t, probs)

    # Current state estimate
    estimate = hmm.current_state_estimate(observations)
    logger.info("\nCurrent state estimate: %s", estimate)

    # Monte Carlo forecast
    forecast = hmm.forecast_summary(
        current_state_idx=np.argmax(gamma[-1]),
        n_steps=12,
        n_simulations=10000,
    )
    logger.info("\n12-month crisis probability trajectory:")
    for t, p in enumerate(forecast["crisis_trajectory"]):
        bar = "█" * int(p * 50)
        logger.info("  Month %2d: %.1f%% %s", t + 1, p * 100, bar)

    # Baum-Welch parameter learning
    logger.info("\nRunning Baum-Welch parameter estimation...")
    final_ll = hmm.baum_welch(observations, max_iterations=50)
    logger.info("Final log-likelihood: %.4f", final_ll)
    logger.info("Learned transition matrix:\n%s", np.array2string(
        params.transition_matrix, precision=3, suppress_small=True
    ))


def run_crack_detection_demo() -> None:
    """
    Demonstrate the crack detection engine with realistic scenario data.
    """
    logger.info("\n" + "=" * 70)
    logger.info("CRACK DETECTION ENGINE — Demo")
    logger.info("=" * 70)

    detector = CrackDetector()

    # Scenario 1: US economy showing cracks (late 2025 conditions)
    us_current = {
        "credit_card_delinquency_rate": 3.8,   # above 3.5 threshold — BREACH
        "auto_loan_delinquency_rate": 4.2,      # above 4.0 — BREACH
        "personal_savings_rate": 3.5,            # above 3.0 — OK
        "household_debt_to_income": 75.0,        # below 80 — OK
        "consumer_confidence": 95.0,             # above 80 — OK
        "cre_vacancy_rate": 22.0,                # above 20 — BREACH
        "cre_loan_delinquency": 3.5,             # above 3.0 — BREACH
        "housing_starts": 1400.0,                # not a delta check directly
        "mortgage_delinquency_rate": 2.5,        # below 4 — OK
        "residential_price_growth": 3.0,         # above -5 — OK
        "hy_corporate_spread": 420.0,            # below 500 — OK but rising
        "ig_corporate_spread": 160.0,            # below 200 — OK
        "corporate_profit_margin": 10.0,         # stable
        "business_inventory_to_sales": 1.42,     # approaching 1.45
        "new_business_formation": 400.0,         # stable
        "initial_jobless_claims": 245.0,         # below 300 — OK
        "job_openings_rate": 4.5,                # declining
        "quit_rate": 2.1,                        # above 2.0 — OK barely
        "wage_growth_real": 1.0,                 # positive — OK
        "capex_growth": 2.0,                     # positive — OK
        "yield_spread_10y3m": 0.5,               # positive — OK
        "pmi_manufacturing": 48.0,               # below 50 — soft
        "freight_volume_index": 110.0,           # stable
        "food_to_income_ratio": 0.12,            # low for US
        "youth_unemployment": 8.5,               # OK for US
        "currency_volatility": 7.0,              # low
        "reserve_adequacy": 12.0,                # ample
        "net_external_debt": 40.0,               # manageable
        "energy_grid_reserve_margin": 18.0,      # adequate
        "broadband_penetration": 42.0,           # good
    }

    report_us = detector.evaluate_country("United States", "USA", us_current)
    detector.print_report(report_us)

    # Scenario 2: Turkey in deeper trouble
    turkey_current = {
        "credit_card_delinquency_rate": 5.5,     # BREACH
        "auto_loan_delinquency_rate": 6.0,       # BREACH
        "personal_savings_rate": 1.5,            # BREACH
        "household_debt_to_income": 30.0,        # OK (Turkish households don't leverage like US)
        "consumer_confidence": 70.0,             # Low but above 80 threshold — depends on index
        "hy_corporate_spread": 650.0,            # BREACH
        "ig_corporate_spread": 280.0,            # BREACH
        "corporate_profit_margin": 6.0,          # compressed
        "business_inventory_to_sales": 1.5,      # BREACH
        "new_business_formation": 200.0,         # low
        "initial_jobless_claims": 350.0,         # BREACH
        "job_openings_rate": 2.0,                # very low
        "quit_rate": 1.5,                        # BREACH
        "wage_growth_real": -8.0,                # deeply negative — BREACH
        "capex_growth": -12.0,                   # BREACH
        "yield_spread_10y3m": -0.8,              # inverted — BREACH
        "pmi_manufacturing": 44.0,               # deep contraction — BREACH
        "freight_volume_index": 85.0,            # declining
        "food_to_income_ratio": 0.35,            # BREACH — above 30%
        "youth_unemployment": 28.0,              # BREACH
        "currency_volatility": 22.0,             # BREACH
        "reserve_adequacy": 2.5,                 # BREACH
        "net_external_debt": 55.0,               # approaching danger
        "energy_grid_reserve_margin": 12.0,      # BREACH
        "broadband_penetration": 25.0,           # BREACH
    }

    report_tr = detector.evaluate_country("Turkey", "TUR", turkey_current)
    detector.print_report(report_tr)

    # Scenario 3: India thriving
    india_current = {
        "credit_card_delinquency_rate": 1.8,
        "auto_loan_delinquency_rate": 2.0,
        "personal_savings_rate": 18.0,
        "household_debt_to_income": 35.0,
        "consumer_confidence": 120.0,
        "hy_corporate_spread": 280.0,
        "ig_corporate_spread": 100.0,
        "corporate_profit_margin": 14.0,
        "business_inventory_to_sales": 1.2,
        "new_business_formation": 800.0,
        "initial_jobless_claims": 150.0,
        "job_openings_rate": 5.5,
        "quit_rate": 3.0,
        "wage_growth_real": 4.0,
        "capex_growth": 12.0,
        "yield_spread_10y3m": 1.5,
        "pmi_manufacturing": 56.0,
        "freight_volume_index": 125.0,
        "food_to_income_ratio": 0.28,
        "youth_unemployment": 22.0,
        "currency_volatility": 8.0,
        "reserve_adequacy": 8.0,
        "net_external_debt": 20.0,
        "energy_grid_reserve_margin": 20.0,
        "broadband_penetration": 15.0,  # low but improving
    }

    report_in = detector.evaluate_country("India", "IND", india_current)
    detector.print_report(report_in)

    # Summary
    logger.info("\n" + "-" * 70)
    logger.info("CRACK DETECTION SUMMARY:")
    for report in [report_us, report_tr, report_in]:
        regime_map = {
            "thriving": "🟢",
            "cracks_appearing": "🟡",
            "crisis_imminent": "🔴",
        }
        logger.info(
            "  %s %-20s  %s  confidence=%.0f%%  active_patterns=%d  breached=%d/%d",
            regime_map.get(report.overall_regime, "⚪"),
            report.country,
            report.overall_regime.upper().replace("_", " "),
            report.regime_confidence * 100,
            len(report.active_patterns),
            report.leading_indicators_breached,
            report.total_indicators_monitored,
        )


def run_news_intelligence_demo() -> None:
    """
    Demonstrate the news intelligence layer using pre-analyzed headlines.

    Shows how raw news gets structured into graph updates:
    actors, intent, affected stats, scenario shifts, relationship changes.
    """
    logger.info("\n" + "=" * 70)
    logger.info("NEWS INTELLIGENCE — Daily Briefing (2026-02-15)")
    logger.info("=" * 70)

    processor = NewsIntelligenceProcessor()

    # Display each event
    for event in DEMO_EVENTS:
        processor.print_event_summary(event)

    # Generate aggregate graph updates
    updates = processor.generate_graph_updates(DEMO_EVENTS)

    logger.info("\n" + "-" * 70)
    logger.info("GRAPH UPDATE SUMMARY:")
    logger.info("  Events processed: %d", updates["total_events_processed"])
    logger.info("  Stat node signals: %d", len(updates["node_updates"]))
    logger.info("  Edge updates: %d", len(updates["edge_updates"]))
    logger.info("  New actors discovered: %d", len(updates["actor_additions"]))

    if updates["scenario_shifts"]:
        logger.info("  Scenario probability shifts:")
        from processing.scenario_engine import SCENARIO_REGISTRY
        scenario_lookup = {s.scenario_id: s for s in SCENARIO_REGISTRY}
        for sid, delta in sorted(updates["scenario_shifts"].items(), key=lambda x: abs(x[1]), reverse=True):
            sc = scenario_lookup.get(sid)
            name = sc.title if sc else f"Scenario #{sid}"
            base_prob = sc.probability_12m if sc else 0.0
            new_prob = max(0.0, min(1.0, base_prob + delta))
            sign = "+" if delta > 0 else ""
            logger.info(
                "    %s: %.1f%% → %.1f%% (%s%.1f%%)",
                name, base_prob * 100, new_prob * 100, sign, delta * 100,
            )

    if updates["actor_additions"]:
        logger.info("  New actors to register:")
        for actor in updates["actor_additions"]:
            logger.info("    + %s — %s (%s)", actor["name"], actor["role"], actor["country_iso3"])

    # Show communication network
    logger.info("\n" + "-" * 70)
    logger.info("COMMUNICATION NETWORK (today's signals):")
    actor_lookup = {a.actor_id: a for a in ACTOR_REGISTRY}
    for edge in updates["edge_updates"]:
        if edge["edge_type"] == "COMMUNICATES":
            src = actor_lookup.get(edge["source"])
            tgt = actor_lookup.get(edge["target"])
            src_name = f"{src.name}" if src else edge["source"]
            tgt_name = f"{tgt.name}" if tgt else edge["target"]
            logger.info("    %s ↔ %s [%s]", src_name, tgt_name, edge["sentiment"])
        elif edge["edge_type"] == "RELATIONSHIP":
            src = actor_lookup.get(edge["source"])
            tgt = actor_lookup.get(edge["target"])
            src_name = f"{src.name}" if src else edge["source"]
            tgt_name = f"{tgt.name}" if tgt else edge["target"]
            logger.info(
                "    %s → %s [%s, strength=%.1f] %s",
                src_name, tgt_name,
                edge.get("relationship_type", ""),
                edge.get("strength", 0),
                edge.get("context", "")[:60],
            )

    # Actor-to-country influence chain
    logger.info("\n" + "-" * 70)
    logger.info("ACTOR → COUNTRY → STAT INFLUENCE CHAINS:")

    from config.actors import ACTOR_REGISTRY as actors
    chains_shown = 0
    for event in DEMO_EVENTS:
        if event.affected_stats and event.actors_mentioned:
            actor = actor_lookup.get(event.actors_mentioned[0])
            if actor:
                for stat in event.affected_stats[:2]:
                    direction = event.stat_direction.get(stat, "?")
                    for country in event.affected_countries[:1]:
                        logger.info(
                            "    %s (%s) → %s → %s_%s [%s]",
                            actor.name, actor.role[:20], actor.country,
                            country, stat, direction,
                        )
                        chains_shown += 1
        if chains_shown >= 8:
            break


def run_strategic_analysis_demo() -> None:
    """
    Demonstrate the strategic analysis engine:
    - Generational plans for major powers
    - Deep multi-order analysis of a sample event
    - Adversarial response modeling
    - Cross-event pattern recognition
    """
    analyzer = StrategicAnalyzer()

    logger.info("\n" + "=" * 70)
    logger.info("STRATEGIC ANALYSIS ENGINE — Generational Plans & Deep Intelligence")
    logger.info("=" * 70)

    # ── Part 1: Generational Plans ────────────────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info("PART 1: GENERATIONAL STRATEGIC PLANS")
    logger.info("─" * 70)

    for plan in GENERATIONAL_PLANS:
        analyzer.print_plan_summary(plan)

    # ── Part 2: Cross-Plan Competition Matrix ────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info("PART 2: STRATEGIC COMPETITION MATRIX")
    logger.info("─" * 70)
    logger.info("\n  Who competes with whom, and over what:\n")

    competition_pairs = [
        ("USA", "CHN", "Semiconductors, AI, reserve currency, Pacific dominance, rare earths"),
        ("USA", "RUS", "Arctic, European security architecture, energy markets, SWIFT hegemony"),
        ("USA", "CAN", "Trade terms (USMCA), Arctic sovereignty, critical minerals processing"),
        ("CHN", "IND", "Manufacturing hub status, demographic advantage, Indian Ocean influence"),
        ("CHN", "RUS", "Central Asian influence, Arctic resources (cooperative-competitive)"),
        ("DEU", "USA", "Industrial energy costs, defense burden-sharing, trade rules"),
        ("SAU", "USA", "Oil pricing power, petrodollar terms, Middle East security architecture"),
        ("CAN", "AUS", "Critical mineral exports, China+1 positioning, immigration talent"),
        ("IND", "IDN", "Manufacturing FDI, demographic dividend, ASEAN influence"),
    ]
    for c1, c2, over_what in competition_pairs:
        logger.info("    %s ↔ %s : %s", c1, c2, over_what)

    # ── Part 3: Resource Control Mapping ─────────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info("PART 3: CRITICAL RESOURCE CONTROL MAP")
    logger.info("─" * 70)

    resource_map = [
        ("Semiconductors (advanced <7nm)", "USA/NLD/JPN/KOR control", "CHN excluded", "Taiwan is the single point of failure"),
        ("Rare Earths (processing)", "CHN controls 87% of processing", "USA/CAN/AUS racing", "5-10yr to build alternatives"),
        ("Oil (swing production)", "USA/SAU/RUS control marginal barrel", "Everyone else dependent", "Venezuela blockade shifts balance"),
        ("LNG (export capacity)", "USA/QAT/AUS dominate", "EU/JPN/KOR importing", "New capacity takes 4-5yr to build"),
        ("Copper (EV critical)", "CHL/PER/ZMB/CHN mines", "Everyone needs it", "China's Africa strategy is copper strategy"),
        ("Lithium (battery grade)", "AUS/CHL/ARG/CHN", "Everyone needs it", "Processing bottleneck not mining"),
        ("Food (caloric security)", "USA/BRA/ARG export", "CHN/EGY/SAU import dependent", "Climate volatility rising"),
        ("Water (freshwater)", "CAN/BRA/RUS abundant", "SAU/IND/MEX stressed", "The hidden constraint on everything"),
        ("AI Compute (GPU access)", "USA controls via NVIDIA/AMD", "CHN developing alternatives", "Energy is the binding constraint"),
        ("Submarine cables", "USA/GBR/FRA own most", "CHN building alternatives", "95% of global data flows through them"),
    ]
    for resource, who_controls, who_needs, note in resource_map:
        logger.info("\n    %s", resource)
        logger.info("      Controls: %s", who_controls)
        logger.info("      Needs: %s", who_needs)
        logger.info("      Note: %s", note)

    # ── Part 4: Deep Analysis Example ────────────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info("PART 4: DEEP ANALYSIS — 'China Builds Hospital in Zambia'")
    logger.info("─" * 70)
    logger.info("  (Demonstrating multi-order effects, hidden intent, adversarial responses)")

    analyzer.print_analysis(EXAMPLE_DEEP_ANALYSIS)

    # ── Part 5: Today's Headlines Through Strategic Lens ─────────────────
    logger.info("\n" + "─" * 70)
    logger.info("PART 5: TODAY'S HEADLINES — STRATEGIC CONTEXT")
    logger.info("─" * 70)

    # Map each headline to which generational plans it affects
    headline_plan_links = [
        (
            "Carney announces new Chief Trade Negotiator to US",
            [
                ("CAN", "CAN_OBJ_1", "advances", "Dedicated negotiator signals Canada taking USMCA seriously"),
                ("USA", "USA_OBJ_4", "advances", "USMCA renewal supports reshoring via North American supply chains"),
            ],
        ),
        (
            "Rubio courts Slovakia/Hungary on energy switch from Russia",
            [
                ("USA", "USA_OBJ_1", "advances", "Expanding LNG export markets in CEE"),
                ("RUS", "RUS_OBJ_1", "threatens", "Direct attack on Russian energy revenue in Europe"),
                ("DEU", "DEU_OBJ_1", "advances", "CEE energy diversification reduces European Russia-dependence"),
            ],
        ),
        (
            "Iran signals nuclear deal compromise",
            [
                ("SAU", "SAU_OBJ_2", "threatens", "Iran supply return could push oil below Saudi fiscal breakeven"),
                ("USA", "USA_OBJ_1", "advances", "More global supply = lower prices = competitive US energy"),
                ("IND", "IND_OBJ_2", "advances", "More supply options for India's diversification strategy"),
            ],
        ),
        (
            "US forces board tanker fleeing Venezuela blockade",
            [
                ("USA", "USA_OBJ_1", "advances", "Controlling Venezuela oil = controlling marginal barrel"),
                ("RUS", "RUS_OBJ_1", "threatens", "Shadow fleet interdiction precedent applies to Russian tankers too"),
                ("CHN", "CHN_OBJ_3", "threatens", "Venezuela was a Chinese resource security partner"),
            ],
        ),
        (
            "EU diplomat rejects US 'civilisational erasure' rhetoric",
            [
                ("USA", "USA_OBJ_3", "threatens", "Transatlantic friction weakens dollar coalition"),
                ("DEU", "DEU_OBJ_2", "advances", "Rhetoric gap strengthens case for European defense autonomy"),
                ("CHN", "CHN_OBJ_4", "advances", "Western division creates openings for RMB bilateral deals with EU"),
            ],
        ),
        (
            "Canada at Munich: Arctic security focus",
            [
                ("CAN", "CAN_OBJ_2", "advances", "Arctic sovereignty signaling to NATO allies"),
                ("RUS", "RUS_OBJ_2", "neutral", "Russia notes but doesn't feel directly threatened yet"),
            ],
        ),
        (
            "Constant rain makes farming more difficult (UK)",
            [
                ("GBR", None, "threatens", "Food CPI pressure on already strained households"),
                ("IND", "IND_OBJ_2", "neutral", "Climate disruption elsewhere validates India's multi-source strategy"),
            ],
        ),
    ]

    for headline, plan_impacts in headline_plan_links:
        logger.info("\n  📰 %s", headline)
        for country, obj_id, impact, explanation in plan_impacts:
            plan = analyzer.get_country_plan(country)
            if plan and obj_id:
                obj = next((o for o in plan.objectives if o.objective_id == obj_id), None)
                obj_name = obj.title if obj else obj_id
            else:
                obj_name = "General"

            impact_icon = {"advances": "✅", "threatens": "⚠️", "neutral": "➖"}.get(impact, "?")
            logger.info(
                "    %s %s → %s: %s",
                impact_icon, country, obj_name, explanation,
            )

    # ── Part 6: "What's Next" Assessment ─────────────────────────────────
    logger.info("\n" + "─" * 70)
    logger.info("PART 6: WHAT'S NEXT — NEAR-TERM PREDICTIONS BY COUNTRY")
    logger.info("─" * 70)

    predictions = [
        ("USA", [
            "USMCA negotiations formalize Q2 2026 — Greer vs Canada's new negotiator",
            "Fed leadership transition May 2026 — Trump's pick reshapes monetary policy",
            "Venezuela operation either succeeds (US controls 500K bbl/d) or becomes quagmire",
            "CHIPS Act fabs come online 2026-2027 — first test of reshoring strategy",
        ]),
        ("CHN", [
            "Property sector: managed decline or uncontrolled collapse by mid-2026",
            "Huawei next-gen chip: success proves self-sufficiency, failure means decade delay",
            "BRI debt restructuring wave — Zambia/Sri Lanka/Pakistan all renegotiating",
            "Taiwan Strait: military exercises calibrated to US election cycle (2028)",
        ]),
        ("CAN", [
            "Trade negotiator's first offer to USTR by Q3 2026 — defines Carney's legacy",
            "Korean submarine deal finalizes — manufacturing plant locations announced",
            "Critical minerals processing: first non-Chinese rare earth refinery in Saskatchewan?",
            "Arctic: NORAD modernization funding allocation reveals real commitment level",
        ]),
        ("RUS", [
            "Energy revenue: Rubio's CEE tour + Iran deal could cut gas income 15-20%",
            "Shadow fleet: US tanker boarding sets precedent for broader interdiction",
            "War economy: can't sustain current burn rate beyond 18 months at current oil prices",
            "Arctic: accelerating Northern Sea Route development as climate opens access",
        ]),
        ("DEU", [
            "€500B infrastructure program: first tranche allocation by H2 2026",
            "Energy prices: industrial electricity must drop 30% or automotive exodus continues",
            "Defense: Rheinmetall and KMW capacity expansion — can German industry actually deliver?",
            "Merz's fiscal gamble: bond market reaction determines if paradigm shift sticks",
        ]),
        ("IND", [
            "Semiconductor fab (Tata + TSMC): ground broken, 3yr timeline to first chips",
            "Manufacturing: Apple/Samsung supply chain shift accelerating — can India absorb?",
            "Demographics: 2026 census will reveal true labor participation challenge",
            "Energy: largest solar tender in history (50GW) — execution is the test",
        ]),
        ("SAU", [
            "Oil below $55 scenario: forces Vision 2030 cuts to NEOM/entertainment",
            "Iran deal: existential threat to Saudi fiscal model if oil supply expands",
            "RMB oil sales: how far does MBS push this before US reacts?",
            "OPEC+ discipline: if Russia cheats, entire cartel framework at risk",
        ]),
    ]

    for country, preds in predictions:
        plan = analyzer.get_country_plan(country)
        plan_name = plan.plan_name if plan else "N/A"
        logger.info("\n  🔮 %s — %s:", country, plan_name)
        for p in preds:
            logger.info("    → %s", p)

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("STRATEGIC ASSESSMENT COMPLETE")
    logger.info("=" * 70)
    logger.info("  Countries with generational plans: %d", len(GENERATIONAL_PLANS))
    logger.info("  Total strategic objectives tracked: %d",
                sum(len(p.objectives) for p in GENERATIONAL_PLANS))
    logger.info("  Resource control vectors: %d", len(resource_map))
    logger.info("  Competition pairs: %d", len(competition_pairs))
    logger.info("  Headline → plan linkages today: %d",
                sum(len(impacts) for _, impacts in headline_plan_links))


def print_dashboard(kb: KnowledgeGraphBuilder, country: str) -> None:
    """Print the full stat dashboard for a country."""
    dashboard = kb.get_country_dashboard(country)

    logger.info("\n" + "=" * 70)
    logger.info("DASHBOARD: %s", country)
    logger.info("=" * 70)

    for category, stats in dashboard.items():
        logger.info("\n  %s:", category)
        for stat in stats:
            value_str = f"{stat['value']:.2f}" if stat["value"] is not None else "PENDING"
            source_tag = f"[{stat['source_type']}]"
            logger.info(
                "    %2d. %-30s %12s %s %s",
                stat["stat_id"], stat["name"], value_str, stat["unit"], source_tag,
            )


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Global Financial Knowledge Base")
    parser.add_argument(
        "--mode",
        choices=["full", "graph-only", "scenarios", "hmm-test", "dashboard", "crack-detect", "news", "strategic", "briefing"],
        default="full",
        help="Execution mode",
    )
    parser.add_argument("--fred-key", type=str, default=None, help="FRED API key")
    parser.add_argument("--country", type=str, default="United States", help="Country for dashboard/deep dive mode")
    parser.add_argument("--episode", type=int, default=1, help="Episode number for briefing mode")
    args = parser.parse_args()

    # Ensure data directories exist
    for d in [DEV_DATA_ROOT, DEV_RAW_DIR, DEV_PROCESSED_DIR, DEV_GRAPH_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if args.mode == "hmm-test":
        run_hmm_test()
        return

    if args.mode == "crack-detect":
        run_crack_detection_demo()
        return

    if args.mode == "news":
        run_news_intelligence_demo()
        return

    if args.mode == "strategic":
        run_strategic_analysis_demo()
        return

    if args.mode == "briefing":
        briefing_dir = DEV_DATA_ROOT / "briefings" / f"episode_{args.episode:03d}"
        briefing_dir.mkdir(parents=True, exist_ok=True)
        generator = WeeklyBriefingGenerator(briefing_dir, episode_number=args.episode)
        package = generator.generate_episode(focus_country=args.country)
        logger.info("\n" + "=" * 70)
        logger.info("EPISODE PACKAGE COMPLETE")
        logger.info("=" * 70)
        logger.info("  YouTube Title: %s", package.youtube_title)
        logger.info("  Duration: ~%.0f minutes", package.duration_estimate)
        logger.info("  Script: %s", package.script_file)
        logger.info("  Charts: %d files in %s", len(package.chart_files), briefing_dir / "charts")
        logger.info("  Thumbnail concept saved in script file")
        logger.info("\n  Social Clips (Short-Form):")
        all_segs = [package.cold_open] + package.segments + [package.closing]
        for seg in all_segs:
            if seg.social_clip_hook:
                logger.info("    [%s] %s", seg.title, seg.social_clip_hook)
        return

    # Phase 1: Ingestion
    if args.mode == "full":
        logger.info("Starting full pipeline with live API ingestion...")
        observations_df = await run_ingestion(fred_api_key=args.fred_key)
    else:
        logger.info("Skipping ingestion — using empty DataFrame for graph structure...")
        observations_df = pl.DataFrame()

    # Phase 2: Knowledge Graph
    kb = build_knowledge_graph(observations_df)

    if args.mode == "dashboard":
        print_dashboard(kb, args.country)
        return

    # Phase 3: HMM Analysis
    hmm_results = run_hmm_analysis(observations_df)

    logger.info("\n" + "=" * 70)
    logger.info("HMM STATE ESTIMATES")
    logger.info("=" * 70)
    for country, result in hmm_results.items():
        est = result["state_estimate"]
        probs = est["probabilities"]
        logger.info(
            "  %-20s → %s (T=%.0f%% Tu=%.0f%% C=%.0f%%) [%d obs]",
            country, est["state"],
            probs["Tranquil"] * 100,
            probs["Turbulent"] * 100,
            probs["Crisis"] * 100,
            result["n_observations"],
        )

    # Phase 4: Scenario Analysis
    run_scenario_analysis(kb)

    # Phase 5: Dashboard for top economy
    print_dashboard(kb, "United States")

    # Summary
    metrics = kb.get_metrics()
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("  Observations ingested: %d", len(observations_df) if not observations_df.is_empty() else 0)
    logger.info("  Graph nodes: %d", metrics.total_nodes)
    logger.info("  Graph edges: %d", metrics.total_edges)
    logger.info("  Countries analyzed: %d", len(hmm_results))
    logger.info("  Scenarios modeled: %d", len(SCENARIO_REGISTRY))
    logger.info("  LLM stats pending: %d", len([s for s in FULL_STAT_REGISTRY if s.source_type == "llm"]))
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Set FRED_API_KEY and run --mode full for live data")
    logger.info("  2. Set ANTHROPIC_API_KEY for LLM-inferred stats and edge weights")
    logger.info("  3. Deploy to Lambda A6000 with --llm-mode local for GPU inference")


if __name__ == "__main__":
    asyncio.run(main())
