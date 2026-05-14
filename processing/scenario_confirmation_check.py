#!/usr/bin/env python3
"""
scenario_confirmation_check.py

For each unconfirmed scenario assigned to this agent, searches GDELT
for recent news (last 48h) and asks Ollama to assess whether the event
has been confirmed. Updates the scenario JSON confirmed flag if evidence found.

Usage:
    python -m processing.scenario_confirmation_check --agent apollo
    python -m processing.scenario_confirmation_check --agent hermes
    python -m processing.scenario_confirmation_check --agent hephaestus
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from processing.scenario_archiver import archive_expired_scenarios

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scenario_confirmation")

# ── Scenario assignments per agent ────────────────────────────────────────────

AGENT_SCENARIOS: dict[str, list[dict]] = {
    "apollo": [
        {
            "trigger": "B61-12 nuclear gravity bomb deployed to Middle East theater",
            "search_terms": "B61 nuclear bomb Middle East deployed theater",
            "key_entities": ["B61-12", "nuclear bomb", "Middle East deployment"],
        },
        {
            "trigger": "Bushehr nuclear power plant struck — radioactive release risk",
            "search_terms": "Bushehr nuclear power plant struck attack radiation",
            "key_entities": ["Bushehr", "nuclear plant", "attack", "radiation"],
        },
        {
            "trigger": "China confirms military materiel supply to Iran — US imposes secondary sanctions",
            "search_terms": "China military weapons supply Iran secondary sanctions",
            "key_entities": ["China", "Iran", "military supply", "secondary sanctions"],
        },
        {
            "trigger": "China declares Taiwan Air Defense Identification Zone closure — warplanes breach median line",
            "search_terms": "China Taiwan ADIZ closure warplanes median line",
            "key_entities": ["China", "Taiwan", "ADIZ", "median line"],
        },
        {
            "trigger": "Fordow underground nuclear facility struck — enrichment status unknown",
            "search_terms": "Fordow nuclear facility struck attack enrichment",
            "key_entities": ["Fordow", "nuclear facility", "struck", "enrichment"],
        },
        {
            "trigger": "Iron Dome interceptor stockpile depleted — mass Israeli civilian casualties",
            "search_terms": "Iron Dome depleted stockpile interceptors Israel casualties",
            "key_entities": ["Iron Dome", "depleted", "Israeli casualties"],
        },
        {
            "trigger": "Kharg Island second strike — Iran crude export terminal destroyed",
            "search_terms": "Kharg Island strike attack Iran oil terminal",
            "key_entities": ["Kharg Island", "strike", "Iran oil terminal"],
        },
        {
            "trigger": "Mojtaba Khamenei confirmed killed — IRGC succession crisis",
            "search_terms": "Mojtaba Khamenei killed dead IRGC succession",
            "key_entities": ["Mojtaba Khamenei", "killed", "IRGC succession"],
        },
    ],
    "hermes": [
        {
            "trigger": "IDF launches Lebanon ground invasion — 2006-scale operation",
            "search_terms": "IDF Israel Lebanon ground invasion troops",
            "key_entities": ["IDF", "Lebanon", "ground invasion"],
        },
        {
            "trigger": "Iran mines Strait of Hormuz — commercial shipping blocked",
            "search_terms": "Iran mines Strait Hormuz shipping blocked",
            "key_entities": ["Iran", "mines", "Strait of Hormuz", "shipping blocked"],
        },
    ],
    "hephaestus": [
        {
            "trigger": "India formally exits Iran war neutrality — aligns with US coalition",
            "search_terms": "India exits neutrality Iran war US coalition alignment",
            "key_entities": ["India", "neutrality", "Iran war", "US coalition"],
        },
        {
            "trigger": "India-Pakistan nuclear signalling — Pakistan places strategic forces on alert",
            "search_terms": "Pakistan nuclear forces alert India-Pakistan strategic",
            "key_entities": ["Pakistan", "nuclear forces", "alert", "India-Pakistan"],
        },
    ],
}

# ── GDELT search ──────────────────────────────────────────────────────────────

GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc"

async def gdelt_search(terms: str, timespan: str = "48h", max_records: int = 15) -> list[dict]:
    """Search GDELT Doc API for recent articles matching terms. Retries on 429."""
    params = {
        "query":      terms,
        "mode":       "artlist",
        "format":     "json",
        "maxrecords": max_records,
        "sort":       "DateDesc",
        "timespan":   timespan,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(3):
            try:
                await asyncio.sleep(2.0 * (attempt + 1))  # 2s, 4s, 6s
                resp = await client.get(GDELT_BASE, params=params)
                if resp.status_code == 429:
                    logger.warning("GDELT 429 on attempt %d for '%s' — retrying", attempt + 1, terms[:40])
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data.get("articles", [])
            except httpx.HTTPStatusError:
                if attempt == 2:
                    logger.warning("GDELT search failed after retries for '%s'", terms[:40])
                    return []
            except Exception as e:
                logger.warning("GDELT search error for '%s': %s", terms[:40], e)
                return []
    return []


# ── Ollama confirmation assessment ───────────────────────────────────────────

OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:70b")

CONFIRMATION_PROMPT = """\
You are a geopolitical intelligence analyst. Your task is to assess whether a specific event has been CONFIRMED based on recent news headlines.

TARGET EVENT: {trigger}

RECENT NEWS HEADLINES (last 48 hours):
{headlines}

Assess whether the news headlines provide evidence that the target event has occurred or been confirmed.

Respond in JSON only:
{{
  "confirmed": true or false,
  "confidence": 0.0-1.0,
  "evidence": "brief summary of supporting headlines, or 'none' if not confirmed",
  "key_headline": "the single most relevant headline, or null"
}}
"""

async def assess_confirmation(trigger: str, articles: list[dict]) -> dict:
    """Ask Ollama whether articles confirm the trigger event."""
    if not articles:
        return {"confirmed": False, "confidence": 0.0, "evidence": "no articles found", "key_headline": None}

    headlines = "\n".join(
        f"- {a.get('title', '')} ({a.get('seendate', '')[:10]})"
        for a in articles[:15]
    )

    prompt = CONFIRMATION_PROMPT.format(trigger=trigger, headlines=headlines)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(OLLAMA_URL, json={
                "model":       OLLAMA_MODEL,
                "messages":    [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens":  300,
            })
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Extract JSON from response
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.warning("Ollama assessment failed: %s", e)
            return {"confirmed": False, "confidence": 0.0, "evidence": f"assessment error: {e}", "key_headline": None}


# ── Scenario file update ──────────────────────────────────────────────────────

SCENARIOS_DIR = Path("/media/peter/fast-storage/projects/world_knowledge_base/global_financial_kb/data/scenarios/active")

def find_scenario_files(trigger: str) -> list[Path]:
    """Find active scenario JSON files matching this trigger event."""
    matches = []
    for f in SCENARIOS_DIR.glob("*.json"):
        if "test-" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            if data.get("trigger_event", "").strip() == trigger.strip():
                matches.append(f)
        except Exception:
            continue
    return matches


def update_scenario_confirmed(path: Path, assessment: dict) -> None:
    """Update the scenario JSON confirmed flag and add confirmation metadata."""
    data = json.loads(path.read_text())
    data["confirmed"] = True
    data["confirmation"] = {
        "confirmed_at":  datetime.now(timezone.utc).isoformat(),
        "confidence":    assessment.get("confidence", 0.0),
        "evidence":      assessment.get("evidence", ""),
        "key_headline":  assessment.get("key_headline"),
        "source":        "scenario_confirmation_check",
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_agent_checks(agent: str) -> None:
    scenarios = AGENT_SCENARIOS.get(agent)
    if not scenarios:
        logger.error("Unknown agent '%s'. Valid: %s", agent, list(AGENT_SCENARIOS))
        sys.exit(1)

    logger.info("=== Scenario confirmation check — %s (%d scenarios) ===", agent, len(scenarios))

    archived = archive_expired_scenarios()
    if archived:
        logger.info("Pre-check archival: moved %d expired scenario(s)", len(archived))

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    results = []
    for sc in scenarios:
        trigger = sc["trigger"]
        terms   = sc["search_terms"]
        logger.info("[%s] Searching GDELT: '%s'", agent, terms)

        articles = await gdelt_search(terms)
        logger.info("[%s] Found %d articles", agent, len(articles))

        assessment = await assess_confirmation(trigger, articles)
        confirmed  = assessment.get("confirmed", False)
        confidence = assessment.get("confidence", 0.0)
        evidence   = assessment.get("evidence", "")
        headline   = assessment.get("key_headline")

        status = "CONFIRMED" if confirmed else "unconfirmed"
        logger.info("[%s] %s | conf=%.2f | %s", agent, status, confidence, evidence[:80])

        if headline:
            logger.info("[%s]   headline: %s", agent, headline)

        if confirmed and confidence >= 0.6:
            files = find_scenario_files(trigger)
            if files:
                for f in files:
                    update_scenario_confirmed(f, assessment)
                logger.info("[%s] Updated %d scenario file(s) → confirmed=True", agent, len(files))
            else:
                logger.warning("[%s] No scenario files found for trigger: %s", agent, trigger[:60])

        results.append({
            "trigger":    trigger,
            "confirmed":  confirmed,
            "confidence": confidence,
            "evidence":   evidence,
            "headline":   headline,
            "articles_found": len(articles),
        })

    # Write results summary
    log_dir = Path(f"/media/peter/fast-storage/agents/{agent}/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / f"scenario_confirmation_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"agent": agent, "run_at": now, "results": results}, indent=2))
    logger.info("Results written to %s", out)

    # Print summary
    confirmed_count = sum(1 for r in results if r["confirmed"])
    logger.info("=== Summary: %d/%d confirmed ===", confirmed_count, len(results))
    for r in results:
        tag = "✓ CONFIRMED" if r["confirmed"] else "  watch    "
        logger.info("  %s  %.2f  %s", tag, r["confidence"], r["trigger"][:65])


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario confirmation checker")
    parser.add_argument("--agent", required=True, choices=list(AGENT_SCENARIOS.keys()),
                        help="Agent to run checks for")
    args = parser.parse_args()
    asyncio.run(run_agent_checks(args.agent))


if __name__ == "__main__":
    main()
