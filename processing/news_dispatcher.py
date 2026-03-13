"""
news_dispatcher.py — Route news parquets to agent inboxes by country assignment.

Flow:
  1. Find all news_*.parquet files in data/raw/ not yet in the dispatch log.
  2. For each file, split rows by country_iso3 → agent via AGENT_ASSIGNMENTS.
  3. Write a per-agent JSON batch to data/dispatched/.
  4. Send an inbox message to each agent (via agent-tools/send.py) with the
     batch file attached and a structured headline briefing in the body.
  5. Record the dispatch in data/raw/news_dispatch_log.json.

Dispatch log format (JSON):
  [{"parquet": "news_20260220_184049.parquet",
    "dispatched_at": "2026-02-20T18:41:00+00:00",
    "agents": {"apollo": 12, "minerva": 8, ...}}, ...]

Batch file format (JSON):
  data/dispatched/news_<agent>_<timestamp>.json
  → list of row dicts matching the NewsItem field names
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import re

import polars as pl

try:
    from langdetect import detect as _langdetect_detect
    from langdetect import LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


def _detected_language(text: str) -> str | None:
    """Return ISO-639-1 language code for text, or None if detection fails/unavailable."""
    if not _LANGDETECT_AVAILABLE or not text or len(text) < 10:
        return None
    try:
        return _langdetect_detect(text)
    except Exception:
        return None

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.agent_assignments import AGENT_ASSIGNMENTS, COUNTRIES_BY_AGENT

logger = logging.getLogger(__name__)

PROJ               = Path(__file__).parent.parent
RAW_DIR            = PROJ / "data" / "raw"
DISPATCHED_DIR     = PROJ / "data" / "dispatched"
DISPATCH_LOG_PATH  = RAW_DIR / "news_dispatch_log.json"
SEND_PY            = Path.home() / ".claude" / "agent-tools" / "send.py"
PYTHON             = sys.executable

# ── Content-level exclusion filters ────────────────────────────────────────────
# These catch items that are geo-mis-tagged at the source (e.g. a German news
# aggregator syndicating Brazilian football assigns country_iso3=DEU; a French
# financial wire republishes US class action boilerplate as FRA content).
# Applied per-agent in _apply_content_filters() after the country_iso3 split.

# US investor/securities wire boilerplate — very formulaic press releases from
# BusinessWire / GlobeNewswire / PR Newswire that contaminate EU/APAC feeds.
_SECURITIES_WIRE_RE = re.compile(
    r"(?i)\b("
    r"INVESTOR (DEADLINE|NOTICE|ALERT)"
    r"|SHAREHOLDER (ALERT|NOTICE|DEADLINE)"
    r"|DEADLINE ALERT"
    r"|class action lawsuit"
    r"|lead plaintiff deadline"
    r"|securities fraud class action"
    r"|FINAL DEADLINE\b.*\binvestor"
    r")\b"
)

# South American football clubs / competitions that appear mis-tagged in EU feeds.
_SA_FOOTBALL_RE = re.compile(
    r"(?i)\b("
    r"Flamengo|Corinthians|Palmeiras|Grêmio|Gremio|Fluminense"
    r"|Vasco da Gama|Botafogo|Cruzeiro|Atletico Mineiro|Internacional"
    r"|São Paulo FC|Santos FC|Fortaleza EC"
    r"|Copa Libertadores|Copa Sudamericana|Copa do Brasil"
    r")\b"
)

# Low-relevance lifestyle / entertainment patterns for macro-focused EU agents.
_EU_LIFESTYLE_RE = re.compile(
    r"(?i)("
    r"cheapest (places|areas|towns|villages|cities) to (live|buy|rent)"
    r"|passport (advice|tips|renewal checklist|check list)"
    r"|\bhoroscope\b"
    r"|\bcelebrity (couple|split|feud|baby|wedding)\b"
    r"|record(s)? (album|track|single) (by|from|with)\b"
    r")"
)

# Per-agent pattern list: (compiled_regex, description_for_logging)
_AGENT_EXCLUSIONS: dict[str, list[tuple[re.Pattern, str]]] = {
    "minerva": [
        (_SECURITIES_WIRE_RE, "US-securities-wire"),
        (_SA_FOOTBALL_RE,     "SA-football-mis-tag"),
        (_EU_LIFESTYLE_RE,    "EU-lifestyle"),
    ],
    "prometheus": [
        (_SECURITIES_WIRE_RE, "US-securities-wire"),
        (_SA_FOOTBALL_RE,     "SA-football-mis-tag"),
    ],
    "daedalus": [
        (_SECURITIES_WIRE_RE, "US-securities-wire"),
        (_SA_FOOTBALL_RE,     "SA-football-mis-tag"),
    ],
    "apollo": [
        (_SECURITIES_WIRE_RE, "US-securities-wire"),
    ],
    "hermes": [
        (_SECURITIES_WIRE_RE, "US-securities-wire"),
        (_SA_FOOTBALL_RE,     "SA-football-mis-tag"),
    ],
    # hephaestus handles BRA/IND/IDN — SA football belongs in BRA feed; keep.
    # Only strip wire boilerplate.
    "hephaestus": [
        (_SECURITIES_WIRE_RE, "US-securities-wire"),
    ],
}

# Max headlines shown per country in the message body
MAX_HEADLINES_PER_COUNTRY = 5


class NewsDispatcher:
    """
    Routes news parquets to agent inboxes.

    Usage:
        dispatcher = NewsDispatcher()
        n = dispatcher.dispatch_pending()       # all undispatched parquets
        n = dispatcher.dispatch_parquet(path)   # one specific file
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        DISPATCHED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def dispatch_pending(self) -> int:
        """Dispatch all news_*.parquet and equity_news_*.parquet files not yet in the dispatch log.

        Returns:
            Total number of NewsItems dispatched across all agents.
        """
        log = self._load_log()
        dispatched_names = {entry["parquet"] for entry in log}

        # Sovereign news + equity/company news (same schema, country_iso3=USA routes to Apollo)
        parquets = sorted(
            list(RAW_DIR.glob("news_2*.parquet")) +
            list(RAW_DIR.glob("equity_news_2*.parquet"))
        )
        pending = [p for p in parquets if p.name not in dispatched_names]

        if not pending:
            logger.info("NewsDispatcher: no pending parquets — nothing to dispatch")
            return 0

        total = 0
        for path in pending:
            total += self.dispatch_parquet(path, _log=log)
        return total

    def dispatch_parquet(
        self,
        path: Path,
        *,
        _log: Optional[list] = None,   # pass pre-loaded log to avoid re-reads
    ) -> int:
        """Dispatch a single news parquet to all relevant agent inboxes.

        Returns:
            Number of NewsItems dispatched (sum across all agents).
        """
        path = Path(path)
        if not path.exists():
            logger.error("NewsDispatcher: parquet not found: %s", path)
            return 0

        df = pl.read_parquet(path)
        if df.is_empty():
            logger.info("NewsDispatcher: %s is empty — skipping", path.name)
            return 0

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        agent_counts: dict[str, int] = {}
        total = 0

        # Group rows by assigned agent
        batches = self._split_by_agent(df)

        for agent, agent_df in batches.items():
            if agent_df.is_empty():
                continue
            n = len(agent_df)
            agent_counts[agent] = n
            total += n

            if not self.dry_run:
                batch_path = self._write_batch(agent, agent_df, timestamp)
                self._send_to_agent(agent, batch_path, agent_df, path.name)
            else:
                batch_name = f"news_{agent}_{timestamp}.json"
                logger.info(
                    "DRY RUN — would send %d items to %s (batch: %s)",
                    n, agent, batch_name,
                )

        if not self.dry_run and total > 0:
            log = _log if _log is not None else self._load_log()
            log.append({
                "parquet":       path.name,
                "dispatched_at": datetime.now(timezone.utc).isoformat(),
                "agents":        agent_counts,
            })
            self._save_log(log)

        logger.info(
            "NewsDispatcher: %s → %d items across %d agents%s",
            path.name, total, len(agent_counts),
            " (dry run)" if self.dry_run else "",
        )
        return total

    # ── Internal ───────────────────────────────────────────────────────────────

    def _split_by_agent(self, df: pl.DataFrame) -> dict[str, pl.DataFrame]:
        """Group DataFrame rows by assigned agent, based on country_iso3.

        Items where language != 'english' are silently dropped — the news API
        returns only a paywalled summary placeholder for non-English sources,
        leaving agents nothing to interpret beyond an untranslated headline.

        Content-level exclusion filters (_AGENT_EXCLUSIONS) are applied after
        the country_iso3 split to remove mis-tagged and low-relevance items.
        """
        # Drop non-English items before dispatch
        # The news API uses both "en" and "english" as language codes.
        if "language" in df.columns:
            n_before = len(df)
            lang = pl.col("language").str.to_lowercase()
            df = df.filter(lang.is_in(["english", "en"]))
            n_dropped = n_before - len(df)
            if n_dropped:
                logger.info(
                    "NewsDispatcher: dropped %d non-English items before dispatch",
                    n_dropped,
                )

        # Secondary language check: catch sources that mislabel non-English
        # articles as language='en' (e.g. Politico Europe publishing German
        # content with an 'en' tag).  Uses langdetect on the headline when
        # available; silently skips if library is absent or detection fails.
        if _LANGDETECT_AVAILABLE and "headline" in df.columns:
            headlines = df["headline"].to_list()
            keep_mask = []
            n_lang_dropped = 0
            for h in headlines:
                detected = _detected_language(h)
                if detected is not None and detected != "en":
                    keep_mask.append(False)
                    n_lang_dropped += 1
                    logger.info(
                        "NewsDispatcher: lang-mismatch drop (tag=en, detected=%s): '%s'",
                        detected, (h or "")[:80],
                    )
                else:
                    keep_mask.append(True)
            if n_lang_dropped:
                df = df.filter(pl.Series(keep_mask))

        result: dict[str, pl.DataFrame] = {}
        for agent, iso3_list in COUNTRIES_BY_AGENT.items():
            agent_df = df.filter(pl.col("country_iso3").is_in(iso3_list))
            if not agent_df.is_empty():
                agent_df = self._apply_content_filters(agent, agent_df)
            if not agent_df.is_empty():
                result[agent] = agent_df
        return result

    def _apply_content_filters(
        self, agent: str, df: pl.DataFrame
    ) -> pl.DataFrame:
        """Apply per-agent content exclusion patterns to remove mis-tagged or
        low-relevance items before dispatch.

        Filters are applied to the 'headline' column only.  Each dropped item
        is logged at DEBUG level with the matched pattern name.
        """
        exclusions = _AGENT_EXCLUSIONS.get(agent)
        if not exclusions:
            return df

        headlines = df["headline"].to_list()
        keep_mask = [True] * len(headlines)

        for pattern, label in exclusions:
            for i, h in enumerate(headlines):
                if keep_mask[i] and h and pattern.search(h):
                    keep_mask[i] = False
                    logger.debug(
                        "NewsDispatcher [%s]: dropped '%s' (filter: %s)",
                        agent, h[:80], label,
                    )

        n_dropped = keep_mask.count(False)
        if n_dropped:
            logger.info(
                "NewsDispatcher [%s]: content filters dropped %d item(s)",
                agent, n_dropped,
            )
            df = df.filter(pl.Series(keep_mask))

        return df

    def _write_batch(
        self, agent: str, df: pl.DataFrame, timestamp: str
    ) -> Path:
        """Serialise agent rows to a JSON batch file.  Returns the file path."""
        batch_path = DISPATCHED_DIR / f"news_{agent}_{timestamp}.json"
        rows = df.to_dicts()
        # raw_actors is stored as pipe-separated string in parquet; restore to list
        for row in rows:
            actors_raw = row.get("raw_actors", "") or ""
            row["raw_actors"] = [a for a in actors_raw.split("|") if a]
        batch_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
        return batch_path

    def _send_to_agent(
        self,
        agent: str,
        batch_path: Path,
        df: pl.DataFrame,
        source_parquet: str,
    ) -> None:
        """Send an inbox message to an agent with the batch file attached."""
        subject = self._build_subject(agent, df)
        body    = self._build_body(agent, df, batch_path, source_parquet)

        cmd = [
            PYTHON, str(SEND_PY),
            "--from",    "hermes",
            "--to",      agent,
            "--type",    "task",
            "--subject", subject,
            "--body",    body,
            "--attach",  str(batch_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode != 0:
                logger.error(
                    "send.py error for %s: %s", agent, result.stderr.strip()
                )
            else:
                logger.info("Dispatched to %s: %s", agent, subject)
        except subprocess.TimeoutExpired:
            logger.error("send.py timed out for %s", agent)
        except Exception as exc:
            logger.error("send.py exception for %s: %s", agent, exc)

    def _build_subject(self, agent: str, df: pl.DataFrame) -> str:
        countries = sorted(df["country_iso3"].unique().to_list())
        n = len(df)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return f"News briefing: {', '.join(countries)} — {n} items ({date_str})"

    def _build_body(
        self,
        agent: str,
        df: pl.DataFrame,
        batch_path: Path,
        source_parquet: str,
    ) -> str:
        n = len(df)
        countries = sorted(df["country_iso3"].unique().to_list())
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"NEWS BRIEFING — {agent.upper()} — {date_str}",
            f"Source: {source_parquet}",
            f"Total items: {n} across {len(countries)} countries",
            "",
        ]

        # Per-country breakdown with sample headlines
        by_country = df.group_by("country_iso3").agg(
            pl.col("headline").alias("headlines"),
            pl.col("country").first().alias("country_name"),
            pl.len().alias("n"),
        ).sort("n", descending=True)

        for row in by_country.iter_rows(named=True):
            iso3   = row["country_iso3"]
            name   = row["country_name"]
            count  = row["n"]
            heads  = row["headlines"][:MAX_HEADLINES_PER_COUNTRY]
            lines.append(f"── {name} ({iso3}) — {count} items ──")
            for h in heads:
                lines.append(f"  • {h}")
            if count > MAX_HEADLINES_PER_COUNTRY:
                lines.append(f"  … and {count - MAX_HEADLINES_PER_COUNTRY} more")
            lines.append("")

        lines += [
            "── INTERPRETATION INSTRUCTIONS ──",
            "Full items are in the attached JSON file. For each item, produce a",
            "NewsInterpretation with: actor_ids, intent, affected_stats, sentiment,",
            "urgency, cross_country flags.",
            "",
            f"Write results to: data/interpretations/interpretations_{agent}_<timestamp>.parquet",
            "",
            f"Attached: {batch_path}",
        ]

        return "\n".join(lines)

    # ── Dispatch log ───────────────────────────────────────────────────────────

    def _load_log(self) -> list:
        if not DISPATCH_LOG_PATH.exists():
            return []
        try:
            return json.loads(DISPATCH_LOG_PATH.read_text())
        except Exception as exc:
            logger.warning("Could not read dispatch log: %s — starting fresh", exc)
            return []

    def _save_log(self, log: list) -> None:
        try:
            DISPATCH_LOG_PATH.write_text(json.dumps(log, indent=2))
        except Exception as exc:
            logger.error("Could not save dispatch log: %s", exc)
