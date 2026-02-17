"""
build_initial_kb.py — One-time initial Knowledge Base build script.

Run this on a Lambda Labs A6000/A100 instance to perform all GPU-intensive
work and produce a downloadable snapshot for daily home-system use.

Usage:
    # Lambda: local Llama-70B via vLLM (no API cost, full GPU inference)
    LLM_MODE=local FRED_API_KEY=xxx python build_initial_kb.py

    # Dev/test: Claude API (no GPU required, costs ~$5-10 in API calls)
    LLM_MODE=claude ANTHROPIC_API_KEY=xxx FRED_API_KEY=xxx python build_initial_kb.py

    # Resume from a checkpoint (skip completed phases)
    LLM_MODE=local FRED_API_KEY=xxx python build_initial_kb.py --resume

Output:
    kb_snapshot_YYYYMMDD.tar.gz  (~50-200MB, contains everything needed
                                   for daily home-system incremental updates)

Phases:
    1. Full parallel ingestion — all 20 countries × 50 stats from real APIs
    2. LLM stat inference    — 17 LLM-required stats × 20 countries (batched)
    3. Edge weight inference  — cross-border contagion weights for trade corridors
    4. Graph construction     — build full typed NetworkX graph with all data
    5. HMM training          — Baum-Welch per country, save learned parameters
    6. Snapshot export        — pack all outputs into a single tar.gz
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pickle
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    COUNTRIES,
    COUNTRY_CODES,
    STAT_REGISTRY,
    FULL_STAT_REGISTRY,
    get_llm_stats,
    DEV_RAW_DIR,
    DEV_GRAPH_DIR,
    DEV_DATA_ROOT,
)
from ingestion.pipeline import IngestionPipeline
from knowledge_base.graph_builder import KnowledgeGraphBuilder, TRADE_CORRIDORS
from models.hmm import SovereignHMM, HMMParams, discretize_observations, STATE_NAMES
from processing.llm_interface import LLMProcessor, get_llm_provider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_kb")

# ─── Paths ────────────────────────────────────────────────────────────────────

BUILD_DIR = DEV_DATA_ROOT / "build"
CHECKPOINT_DIR = BUILD_DIR / "checkpoints"
SNAPSHOT_DIR = BUILD_DIR / "snapshots"

for _d in (BUILD_DIR, CHECKPOINT_DIR, SNAPSHOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─── LLM Stat Batching ────────────────────────────────────────────────────────

# Batch multiple LLM-inferred stats per API call to reduce cost and latency.
# Each batch covers all LLM stats for one country in 1-2 calls.
STAT_INFERENCE_BATCH_SYSTEM = """You are a sovereign economic data analyst with expertise in
global macroeconomics. Given a country and a list of statistics that require expert estimation,
provide your best estimates based on publicly available reports, academic research, and
institutional publications (IMF, World Bank, BIS, OECD, national statistics offices).

All estimates should reflect conditions as of late 2025 / early 2026.

Respond ONLY with a JSON object where each key is the stat_name and value is:
{
  "value": <float>,
  "confidence": <float 0.0-1.0>,
  "unit": "<string>",
  "reasoning": "<one sentence>"
}"""

# ─── Edge Weight Batching ─────────────────────────────────────────────────────

# For each trade corridor, infer weights for the 6 most economically
# significant stat pairs. 31 corridors × 6 pairs = 186 targeted calls.
# Batched at 6 pairs per call = ~31 API calls total.
TOP_CONTAGION_PAIRS = [
    ("real_gdp_growth", "real_gdp_growth"),
    ("policy_rate", "policy_rate"),
    ("yield_spread_10y3m", "yield_spread_10y3m"),
    ("private_credit_npls", "bank_capital_adequacy"),
    ("crude_output", "wti_brent_spread"),
    ("currency_volatility", "reserve_adequacy"),
]

EDGE_WEIGHT_BATCH_SYSTEM = """You are a quantitative macroeconomist specialising in cross-border
economic transmission. For each stat pair listed, estimate the causal transmission weight and
mechanism between the two countries.

Respond ONLY with a JSON array, one object per pair:
[
  {
    "source_stat": "<stat_name>",
    "target_stat": "<stat_name>",
    "weight": <float 0.0-1.0>,
    "lag_months": <int>,
    "mechanism": "<one sentence transmission channel>",
    "confidence": <float 0.0-1.0>
  }
]"""


# ─── Checkpoint Helpers ───────────────────────────────────────────────────────

def checkpoint_path(phase: str) -> Path:
    return CHECKPOINT_DIR / f"phase_{phase}.done"


def is_done(phase: str) -> bool:
    return checkpoint_path(phase).exists()


def mark_done(phase: str, metadata: dict | None = None) -> None:
    data = {"completed_at": datetime.now(timezone.utc).isoformat()}
    if metadata:
        data.update(metadata)
    checkpoint_path(phase).write_text(json.dumps(data, indent=2))
    logger.info("  ✓ Phase %s checkpointed", phase)


def phase_header(num: str, title: str) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PHASE %s: %s", num, title)
    logger.info("=" * 70)


# ─── Phase 1: Full Ingestion ─────────────────────────────────────────────────

async def phase1_ingestion(fred_api_key: str, resume: bool) -> pl.DataFrame:
    phase_header("1", "Full Parallel Ingestion")

    # Check for existing observations to resume from
    obs_files = sorted(DEV_RAW_DIR.glob("observations_*.parquet"), reverse=True)
    if resume and obs_files:
        logger.info("RESUME: Loading existing observations from %s", obs_files[0])
        df = pl.read_parquet(obs_files[0])
        logger.info("  Loaded %d observations", len(df))
        return df

    pipeline = IngestionPipeline(
        fred_api_key=fred_api_key,
        output_dir=DEV_RAW_DIR,
        start_year=2018,   # More history for HMM training
        end_year=2026,
    )

    t0 = time.time()
    # skip_cache=True: Lambda build always fetches fresh — no stale data
    df = await pipeline.run(skip_cache=True)
    elapsed = time.time() - t0

    logger.info(
        "Ingestion complete: %d observations in %.1f seconds",
        len(df), elapsed,
    )
    return df


# ─── Phase 2: LLM Stat Inference ─────────────────────────────────────────────

async def phase2_llm_stats(
    processor: LLMProcessor,
    observations_df: pl.DataFrame,
    resume: bool,
) -> pl.DataFrame:
    phase_header("2", "LLM Stat Inference (17 stats × 20 countries)")

    cache_path = CHECKPOINT_DIR / "llm_stats_cache.parquet"

    # Load existing cache if resuming
    existing_cache: dict[str, dict] = {}
    if resume and cache_path.exists():
        cached_df = pl.read_parquet(cache_path)
        for row in cached_df.iter_rows(named=True):
            existing_cache[row["cache_key"]] = row
        logger.info("RESUME: Loaded %d cached LLM stat inferences", len(existing_cache))

    llm_stats = get_llm_stats()
    if not llm_stats:
        logger.info("No LLM stats to infer — skipping")
        return pl.DataFrame()

    # Build context from what we already have from APIs
    api_context: dict[str, dict] = {}
    if not observations_df.is_empty():
        latest = (
            observations_df
            .sort("period", descending=True)
            .group_by(["country_iso3", "stat_name"])
            .first()
        )
        for row in latest.iter_rows(named=True):
            key = row["country_iso3"]
            if key not in api_context:
                api_context[key] = {}
            api_context[key][row["stat_name"]] = row["value"]

    new_rows: list[dict] = []
    total = len(COUNTRIES)
    now_iso = datetime.now(timezone.utc).isoformat()

    for i, country in enumerate(COUNTRIES, 1):
        codes = COUNTRY_CODES.get(country, {})
        iso3 = codes.get("iso3", "")

        # Check if all stats for this country are already cached
        all_cached = all(
            f"{iso3}_{s.name}" in existing_cache for s in llm_stats
        )
        if resume and all_cached:
            logger.info("  SKIP (%d/%d): %s — fully cached", i, total, country)
            continue

        logger.info("  Inferring (%d/%d): %s — %d stats", i, total, country, len(llm_stats))

        # Build the batch prompt — all LLM stats for this country in one call
        stats_list = "\n".join(
            f'  - "{s.name}": {s.description or s.name} (unit: {s.unit})'
            for s in llm_stats
            if f"{iso3}_{s.name}" not in existing_cache
        )

        context = api_context.get(iso3, {})
        context_text = ""
        if context:
            # Include the 5 most relevant API-sourced stats as context
            sample = dict(list(context.items())[:5])
            context_text = f"\nKnown data for {country}: {json.dumps(sample, default=str)}"

        user_prompt = (
            f"Country: {country} (ISO: {iso3}){context_text}\n\n"
            f"Estimate the following statistics:\n{stats_list}\n\n"
            f"Return a JSON object with each stat_name as key."
        )

        try:
            result = await processor.provider.complete_json(
                STAT_INFERENCE_BATCH_SYSTEM,
                user_prompt,
                temperature=0.1,
                max_tokens=3000,
            )

            for s in llm_stats:
                cache_key = f"{iso3}_{s.name}"
                if cache_key in existing_cache:
                    continue

                stat_result = result.get(s.name, {})
                if not stat_result or "value" not in stat_result:
                    logger.warning(
                        "    No result for %s/%s", country, s.name
                    )
                    continue

                row = {
                    "cache_key": cache_key,
                    "country": country,
                    "country_iso3": iso3,
                    "stat_id": s.stat_id,
                    "stat_name": s.name,
                    "category": s.category,
                    "value": float(stat_result["value"]),
                    "period": "llm_inferred_2026",
                    "unit": stat_result.get("unit", s.unit),
                    "source": "LLM inference",
                    "source_url": "",
                    "retrieved_at": now_iso,
                    "node_id": f"{iso3}_{s.name}",
                    "confidence": float(stat_result.get("confidence", 0.5)),
                    "reasoning": str(stat_result.get("reasoning", "")),
                }
                new_rows.append(row)

        except Exception as exc:
            logger.error("  LLM inference failed for %s: %s", country, exc)

        # Save cache after each country — if Lambda dies mid-run, we resume
        if new_rows:
            checkpoint_df = pl.DataFrame(new_rows)
            if cache_path.exists():
                existing_df = pl.read_parquet(cache_path)
                checkpoint_df = pl.concat([existing_df, checkpoint_df])
            checkpoint_df.write_parquet(cache_path, compression="zstd")

        # Small pause to respect API rate limits (Claude: ~50 req/min)
        await asyncio.sleep(1.2)

    # Load full cache (existing + new)
    if cache_path.exists():
        return pl.read_parquet(cache_path)
    if new_rows:
        return pl.DataFrame(new_rows)
    return pl.DataFrame()


# ─── Phase 3: Edge Weight Inference ──────────────────────────────────────────

async def phase3_edge_weights(
    processor: LLMProcessor,
    resume: bool,
) -> list[dict]:
    phase_header("3", "Cross-Border Edge Weight Inference")

    cache_path = CHECKPOINT_DIR / "edge_weights_cache.parquet"
    existing_keys: set[str] = set()
    existing_rows: list[dict] = []

    if resume and cache_path.exists():
        cached_df = pl.read_parquet(cache_path)
        existing_rows = cached_df.to_dicts()
        existing_keys = {r["edge_key"] for r in existing_rows}
        logger.info("RESUME: Loaded %d cached edge weights", len(existing_rows))

    new_rows: list[dict] = []
    total_corridors = len(TRADE_CORRIDORS)
    now_iso = datetime.now(timezone.utc).isoformat()

    for i, (src_country, tgt_country, trade_weight) in enumerate(TRADE_CORRIDORS, 1):
        # Check if all pairs for this corridor are already cached
        all_pairs_cached = all(
            f"{src_country}→{tgt_country}:{src_s}→{tgt_s}" in existing_keys
            for src_s, tgt_s in TOP_CONTAGION_PAIRS
        )
        if resume and all_pairs_cached:
            logger.info("  SKIP (%d/%d): %s → %s — cached", i, total_corridors, src_country, tgt_country)
            continue

        logger.info(
            "  Weighting (%d/%d): %s → %s (trade=%.2f)",
            i, total_corridors, src_country, tgt_country, trade_weight,
        )

        # Build batch prompt for all 6 stat pairs in this corridor
        pairs_to_infer = [
            (src_s, tgt_s)
            for src_s, tgt_s in TOP_CONTAGION_PAIRS
            if f"{src_country}→{tgt_country}:{src_s}→{tgt_s}" not in existing_keys
        ]

        if not pairs_to_infer:
            continue

        pairs_text = "\n".join(
            f"  {j+1}. {src_country}.{src_s} → {tgt_country}.{tgt_s}"
            for j, (src_s, tgt_s) in enumerate(pairs_to_infer)
        )

        user_prompt = (
            f"Trade corridor: {src_country} → {tgt_country} "
            f"(bilateral trade intensity: {trade_weight:.2f})\n\n"
            f"Estimate the economic contagion weight for each pair:\n{pairs_text}\n\n"
            f"For each pair, how strongly does movement in the source stat "
            f"cause movement in the target stat, given this trade relationship?"
        )

        try:
            result = await processor.provider.complete_json(
                EDGE_WEIGHT_BATCH_SYSTEM,
                user_prompt,
                temperature=0.1,
                max_tokens=2000,
            )

            if isinstance(result, list):
                for pair_data, (src_s, tgt_s) in zip(result, pairs_to_infer):
                    edge_key = f"{src_country}→{tgt_country}:{src_s}→{tgt_s}"
                    new_rows.append({
                        "edge_key": edge_key,
                        "source_country": src_country,
                        "target_country": tgt_country,
                        "source_stat": src_s,
                        "target_stat": tgt_s,
                        "weight": float(pair_data.get("weight", 0.0)),
                        "lag_months": int(pair_data.get("lag_months", 3)),
                        "mechanism": str(pair_data.get("mechanism", "")),
                        "confidence": float(pair_data.get("confidence", 0.5)),
                        "trade_weight": trade_weight,
                        "inferred_at": now_iso,
                    })
            else:
                logger.warning("Unexpected result format for %s→%s", src_country, tgt_country)

        except Exception as exc:
            logger.error("  Edge weight failed for %s→%s: %s", src_country, tgt_country, exc)

        # Save after each corridor
        if new_rows:
            all_rows = existing_rows + new_rows
            pl.DataFrame(all_rows).write_parquet(cache_path, compression="zstd")

        await asyncio.sleep(1.2)

    all_rows = existing_rows + new_rows
    logger.info("Edge weights complete: %d edges inferred", len(all_rows))
    return all_rows


# ─── Phase 4: Graph Construction ─────────────────────────────────────────────

def phase4_build_graph(
    observations_df: pl.DataFrame,
    llm_stats_df: pl.DataFrame,
    edge_weights: list[dict],
) -> KnowledgeGraphBuilder:
    phase_header("4", "Full Graph Construction")

    # Merge API observations + LLM-inferred stats
    all_observations = observations_df

    if not llm_stats_df.is_empty():
        # Select only the columns that match the observations schema
        llm_cols = [
            "country", "country_iso3", "stat_id", "stat_name",
            "category", "value", "period", "unit", "source",
            "source_url", "retrieved_at", "node_id",
        ]
        llm_for_merge = llm_stats_df.select(
            [c for c in llm_cols if c in llm_stats_df.columns]
        )
        if not all_observations.is_empty():
            all_observations = pl.concat([all_observations, llm_for_merge])
        else:
            all_observations = llm_for_merge

    logger.info("Total observations (API + LLM): %d", len(all_observations))

    builder = KnowledgeGraphBuilder()
    builder.build_from_observations(all_observations)

    # Apply LLM-inferred edge weights on top of the seed contagion channels
    if edge_weights:
        logger.info("Applying %d LLM-inferred edge weights...", len(edge_weights))
        applied = 0
        for ew in edge_weights:
            if ew["weight"] < 0.05:
                continue  # Skip negligible weights

            src_iso3 = COUNTRY_CODES.get(ew["source_country"], {}).get("iso3", "")
            tgt_iso3 = COUNTRY_CODES.get(ew["target_country"], {}).get("iso3", "")
            src_node = f"{src_iso3}_{ew['source_stat']}"
            tgt_node = f"{tgt_iso3}_{ew['target_stat']}"

            if src_node in builder.graph and tgt_node in builder.graph:
                # Blend LLM weight with existing seed weight if present
                existing = builder.graph.edges.get((src_node, tgt_node), {})
                if existing:
                    blended = (existing.get("weight", 0) * 0.3) + (ew["weight"] * 0.7)
                    builder.graph[src_node][tgt_node]["weight"] = blended
                    builder.graph[src_node][tgt_node]["mechanism"] = ew["mechanism"]
                else:
                    builder.graph.add_edge(
                        src_node, tgt_node,
                        edge_type="CONTAGION",
                        weight=ew["weight"],
                        lag_months=ew.get("lag_months", 3),
                        mechanism=ew.get("mechanism", ""),
                        confidence=ew.get("confidence", 0.5),
                        source_country=ew["source_country"],
                        target_country=ew["target_country"],
                    )
                applied += 1

        logger.info("Applied %d edge weights (skipped %d < 0.05)", applied, len(edge_weights) - applied)

    metrics = builder.get_metrics()
    logger.info(
        "Graph: %d nodes | %d edges | density=%.4f",
        metrics.total_nodes, metrics.total_edges, metrics.density,
    )
    logger.info(
        "  Edges: %d MONITORS | %d CONTAGION | %d TRADE",
        metrics.monitors_edges, metrics.contagion_edges, metrics.trade_edges,
    )

    return builder


# ─── Phase 5: HMM Training ───────────────────────────────────────────────────

def phase5_hmm_training(
    builder: KnowledgeGraphBuilder,
    observations_df: pl.DataFrame,
) -> dict[str, SovereignHMM]:
    phase_header("5", "HMM Baum-Welch Training (20 countries)")

    hmm_models: dict[str, SovereignHMM] = {}

    # Key macro stats to use as HMM observations
    HMM_STATS = [
        "real_gdp_growth", "core_cpi", "policy_rate",
        "yield_spread_10y3m", "private_credit_npls",
    ]

    for country in COUNTRIES:
        codes = COUNTRY_CODES.get(country, {})
        iso3 = codes.get("iso3", "")

        # Extract time series for this country's HMM stats
        obs_array = np.array([], dtype=int)

        if not observations_df.is_empty():
            country_df = (
                observations_df
                .filter(
                    (pl.col("country_iso3") == iso3) &
                    (pl.col("stat_name").is_in(HMM_STATS))
                )
                .sort("period")
            )

            if not country_df.is_empty():
                period_groups = country_df.group_by("period").agg(
                    pl.col("value").mean().alias("avg_value")
                ).sort("period")

                values = period_groups["avg_value"].to_numpy()
                if len(values) >= 5:
                    obs_array = discretize_observations(values)

        params = HMMParams(country=country, country_iso3=iso3)
        hmm = SovereignHMM(params=params)

        if len(obs_array) >= 5:
            try:
                log_likelihood = hmm.baum_welch(obs_array)
                state_result = hmm.current_state_estimate(obs_array)
                logger.info(
                    "  %-20s → %d obs | LL=%.2f | state=%s",
                    country,
                    len(obs_array),
                    log_likelihood,
                    state_result.get("state", "?"),
                )
            except Exception as exc:
                logger.warning("  HMM training failed for %s: %s", country, exc)
                state_result = {}
        else:
            logger.info(
                "  %-20s → insufficient obs (%d) — using priors",
                country, len(obs_array),
            )
            state_result = {}

        hmm_models[country] = hmm

        # Update the graph node with the estimated Markov state
        if state_result and country in builder.graph:
            state_name = state_result.get("state", "Tranquil")
            state_idx = STATE_NAMES.index(state_name) if state_name in STATE_NAMES else 0
            builder.graph.nodes[country]["markov_state"] = f"S{state_idx}_{state_name}"
            builder.graph.nodes[country]["state_probs"] = state_result.get("probabilities", {})

    logger.info("HMM training complete for %d countries", len(hmm_models))
    return hmm_models


# ─── Phase 6: Snapshot Export ─────────────────────────────────────────────────

def phase6_export_snapshot(
    builder: KnowledgeGraphBuilder,
    hmm_models: dict[str, SovereignHMM],
    observations_df: pl.DataFrame,
    llm_stats_df: pl.DataFrame,
    edge_weights: list[dict],
) -> Path:
    phase_header("6", "Snapshot Export")

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    snapshot_dir = SNAPSHOT_DIR / f"kb_{date_str}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 6a: Export knowledge graph
    graph_path = snapshot_dir / "financial_kg.json"
    builder.export_graph(graph_path)
    logger.info("  ✓ Graph exported (%s)", graph_path)

    # 6b: Export HMM parameters
    hmm_params = {}
    for country, hmm in hmm_models.items():
        try:
            p = hmm.params
            hmm_params[country] = {
                "country": p.country,
                "country_iso3": p.country_iso3,
                "trans_matrix": p.transition_matrix.tolist(),
                "emit_matrix": p.emission_matrix.tolist(),
                "start_prob": p.initial_probs.tolist(),
                "n_states": p.n_states,
                "n_obs_levels": p.n_obs_levels,
            }
        except Exception:
            hmm_params[country] = {"error": "export_failed"}

    hmm_path = snapshot_dir / "hmm_params.json"
    hmm_path.write_text(json.dumps(hmm_params, indent=2, default=str))
    logger.info("  ✓ HMM parameters exported (%s)", hmm_path)

    # 6c: Export observations (API + LLM merged)
    if not observations_df.is_empty():
        obs_path = snapshot_dir / "observations.parquet"
        observations_df.write_parquet(obs_path, compression="zstd")
        logger.info("  ✓ Observations exported (%d rows)", len(observations_df))

    if not llm_stats_df.is_empty():
        llm_path = snapshot_dir / "llm_stats.parquet"
        llm_stats_df.write_parquet(llm_path, compression="zstd")
        logger.info("  ✓ LLM stats exported (%d rows)", len(llm_stats_df))

    # 6d: Export edge weights
    if edge_weights:
        ew_path = snapshot_dir / "edge_weights.parquet"
        pl.DataFrame(edge_weights).write_parquet(ew_path, compression="zstd")
        logger.info("  ✓ Edge weights exported (%d)", len(edge_weights))

    # 6e: Write build manifest
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "graph_nodes": builder.graph.number_of_nodes(),
        "graph_edges": builder.graph.number_of_edges(),
        "countries": len(COUNTRIES),
        "api_observations": len(observations_df),
        "llm_observations": len(llm_stats_df),
        "edge_weights": len(edge_weights),
        "hmm_countries": len(hmm_models),
        "python_version": sys.version,
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 6f: Pack everything into a single tar.gz
    tarball_path = SNAPSHOT_DIR / f"kb_snapshot_{date_str}.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(snapshot_dir, arcname=f"kb_{date_str}")

    size_mb = tarball_path.stat().st_size / 1_048_576
    logger.info("")
    logger.info("=" * 70)
    logger.info("  SNAPSHOT READY: %s (%.1f MB)", tarball_path, size_mb)
    logger.info("  Download with:")
    logger.info("    rsync -avz --progress \\")
    logger.info("      lambda:~/global_financial_kb/data/build/snapshots/%s \\", tarball_path.name)
    logger.info("      /media/peter/fast-storage/projects/world_knowledge_base/")
    logger.info("=" * 70)

    return tarball_path


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(description="Build the global financial knowledge base")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoints — skip already-completed phases",
    )
    parser.add_argument(
        "--skip-ingestion", action="store_true",
        help="Skip Phase 1 (use existing observations parquet)",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip Phases 2-3 (build graph with API data only, no LLM inference)",
    )
    args = parser.parse_args()

    # ── Environment check ──────────────────────────────────────────────────────
    fred_key = os.environ.get("FRED_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    llm_mode = os.environ.get("LLM_MODE", "claude").lower()

    logger.info("")
    logger.info("=" * 70)
    logger.info("  GLOBAL FINANCIAL KB — INITIAL BUILD")
    logger.info("  %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    logger.info("=" * 70)
    logger.info("  LLM mode:     %s", llm_mode)
    logger.info("  FRED key:     %s", "✓ set" if fred_key else "✗ MISSING — US/EU data will be limited")
    logger.info("  Anthropic:    %s", "✓ set" if anthropic_key else ("✗ not needed (local mode)" if llm_mode == "local" else "✗ MISSING"))
    logger.info("  Resume:       %s", "yes" if args.resume else "no (full rebuild)")
    logger.info("  Output dir:   %s", SNAPSHOT_DIR)
    logger.info("")

    if llm_mode == "claude" and not anthropic_key and not args.skip_llm:
        logger.error("ANTHROPIC_API_KEY required for claude mode. Set it or use --skip-llm.")
        sys.exit(1)

    build_start = time.time()

    # ── Phase 1: Ingestion ─────────────────────────────────────────────────────
    if args.skip_ingestion and args.resume:
        obs_files = sorted(DEV_RAW_DIR.glob("observations_*.parquet"), reverse=True)
        if obs_files:
            observations_df = pl.read_parquet(obs_files[0])
            logger.info("Using existing observations: %s (%d rows)", obs_files[0].name, len(observations_df))
        else:
            logger.error("--skip-ingestion specified but no observations parquet found in %s", DEV_RAW_DIR)
            sys.exit(1)
    else:
        observations_df = await phase1_ingestion(fred_key, resume=args.resume)

    # ── Phases 2-3: LLM Inference ─────────────────────────────────────────────
    llm_stats_df = pl.DataFrame()
    edge_weights: list[dict] = []

    if not args.skip_llm:
        provider = get_llm_provider(mode=llm_mode)
        processor = LLMProcessor(provider)

        llm_stats_df = await phase2_llm_stats(processor, observations_df, resume=args.resume)
        edge_weights = await phase3_edge_weights(processor, resume=args.resume)
    else:
        logger.info("Skipping LLM phases (--skip-llm)")

    # ── Phase 4: Graph ─────────────────────────────────────────────────────────
    builder = phase4_build_graph(observations_df, llm_stats_df, edge_weights)

    # ── Phase 5: HMM Training ─────────────────────────────────────────────────
    hmm_models = phase5_hmm_training(builder, observations_df)

    # ── Phase 6: Export ────────────────────────────────────────────────────────
    snapshot_path = phase6_export_snapshot(
        builder, hmm_models, observations_df, llm_stats_df, edge_weights
    )

    total_elapsed = time.time() - build_start
    logger.info("")
    logger.info("TOTAL BUILD TIME: %.1f minutes", total_elapsed / 60)
    logger.info("SNAPSHOT: %s", snapshot_path)


if __name__ == "__main__":
    asyncio.run(main())
