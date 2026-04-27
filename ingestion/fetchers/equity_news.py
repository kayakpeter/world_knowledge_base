"""
equity_news.py — Equity & sector news fetcher for Phase 5.

Fetches financial market news for:
  1. 15 major sector / macro ETFs (XLE, XLF, XLK, SPY, etc.) via RSS
  2. Top ~91 S&P 500 + sector ETF component companies via GNews
  3. Broad financial market news via curated RSS feeds

Outputs NewsItem objects in the same schema as the sovereign news fetcher,
with:
  - country = "United States", country_iso3 = "USA"
  - fetch_source = "equity_rss" | "equity_gnews"
  - raw_actors = [ticker, ...] for companies explicitly mentioned

Usage:
    fetcher = EquityNewsFetcher()
    items = await fetcher.fetch_all(since_hours=6)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

from processing.news_queue import NewsItem

logger = logging.getLogger(__name__)


# ─── ETF Definitions ─────────────────────────────────────────────────────────
# (ticker, kb_sector, description)
SECTOR_ETFS: list[tuple[str, str, str]] = [
    ("SPY",  "Other",                "S&P 500 broad market"),
    ("QQQ",  "Tech",                 "NASDAQ-100 technology"),
    ("XLE",  "Energy",               "Energy Select Sector"),
    ("XLF",  "Financial",            "Financial Select Sector"),
    ("XLK",  "Tech",                 "Technology Select Sector"),
    ("XLV",  "Pharma",               "Health Care Select Sector"),
    ("XLI",  "Other",                "Industrial Select Sector"),
    ("XLP",  "Retail",               "Consumer Staples Select Sector"),
    ("XLY",  "Retail",               "Consumer Discretionary Select Sector"),
    ("XLB",  "Mining",               "Materials Select Sector"),
    ("XLRE", "Other",                "Real Estate Select Sector"),
    ("XLU",  "Other",                "Utilities Select Sector"),
    ("XLC",  "Tech",                 "Communication Services Select Sector"),
    ("GLD",  "Mining",               "Gold ETF"),
    ("TLT",  "Financial",            "20+ Year Treasury Bond ETF"),
    ("IWM",  "Other",                "Russell 2000 small-cap"),
    ("USO",  "Energy",               "US Oil Fund"),
]

# ─── Bellwether ticker registry (single source of truth) ─────────────────────
# Used for: (1) Yahoo Finance per-ticker RSS fetch, (2) post-fetch ticker
# scanner via _ALL_TICKERS, (3) coverage_monitor.py per-bellwether check.
# Each entry: (ticker, company_name, sector_label).
#
# Curated 2026-04-27 to repair the equity-news pipeline. INTC was the
# headline missed-catalyst case (Q1 earnings 2026-04-24 +22-23% blowout
# missed by the prior 34-name list — INTC absent, MAX_COMPANY_GNEWS=20
# truncating, and equity_gnews dead from GNews 12h delay vs 6h cron window).
#
# 65 names, sector-balanced. NOT exhaustive — bellwether-grade only.
TOP_COMPANY_TARGETS: list[tuple[str, str, str]] = [
    # ── AI / semiconductors (12) ──────────────────────────────────────────
    ("NVDA",  "NVIDIA",                "AI/semis"),
    ("AMD",   "AMD",                   "AI/semis"),
    ("INTC",  "Intel",                 "AI/semis"),
    ("AVGO",  "Broadcom",              "AI/semis"),
    ("MRVL",  "Marvell",               "AI/semis"),
    ("MU",    "Micron",                "AI/semis"),
    ("QCOM",  "Qualcomm",              "AI/semis"),
    ("TXN",   "Texas Instruments",     "AI/semis"),
    ("TSM",   "TSMC ADR",              "AI/semis"),
    ("ARM",   "Arm Holdings",          "AI/semis"),
    ("AMAT",  "Applied Materials",     "AI/semis"),
    ("KLAC",  "KLA Corp",              "AI/semis"),
    # ── Mag 7 (7) ─────────────────────────────────────────────────────────
    ("AAPL",  "Apple",                 "Mag7"),
    ("MSFT",  "Microsoft",             "Mag7"),
    ("GOOGL", "Alphabet",              "Mag7"),
    ("META",  "Meta Platforms",        "Mag7"),
    ("AMZN",  "Amazon",                "Mag7"),
    ("NFLX",  "Netflix",               "Mag7"),
    ("TSLA",  "Tesla",                 "Mag7"),
    # ── Other megacap tech (5) ────────────────────────────────────────────
    ("ORCL",  "Oracle",                "BigTech"),
    ("CRM",   "Salesforce",            "BigTech"),
    ("ADBE",  "Adobe",                 "BigTech"),
    ("NOW",   "ServiceNow",            "BigTech"),
    ("IBM",   "IBM",                   "BigTech"),
    # ── Banks / large finance (8) ────────────────────────────────────────
    ("JPM",   "JPMorgan Chase",        "Banks"),
    ("BAC",   "Bank of America",       "Banks"),
    ("GS",    "Goldman Sachs",         "Banks"),
    ("MS",    "Morgan Stanley",        "Banks"),
    ("WFC",   "Wells Fargo",           "Banks"),
    ("C",     "Citigroup",             "Banks"),
    ("BLK",   "BlackRock",             "Banks"),
    ("BX",    "Blackstone",            "Banks"),
    # ── Berkshire / private credit risk (3) ──────────────────────────────
    ("BRK-B", "Berkshire Hathaway",    "Holdcos"),
    ("KKR",   "KKR",                   "PrivateCredit"),
    ("APO",   "Apollo Global",         "PrivateCredit"),
    # ── Defense (5) ──────────────────────────────────────────────────────
    ("LMT",   "Lockheed Martin",       "Defense"),
    ("RTX",   "RTX",                   "Defense"),
    ("NOC",   "Northrop Grumman",      "Defense"),
    ("GD",    "General Dynamics",      "Defense"),
    ("BA",    "Boeing",                "Defense"),
    # ── Energy (5) ───────────────────────────────────────────────────────
    ("XOM",   "ExxonMobil",            "Energy"),
    ("CVX",   "Chevron",               "Energy"),
    ("COP",   "ConocoPhillips",        "Energy"),
    ("OXY",   "Occidental",            "Energy"),
    ("SLB",   "SLB",                   "Energy"),
    # ── Healthcare / pharma (8) ──────────────────────────────────────────
    ("UNH",   "UnitedHealth",          "Healthcare"),
    ("LLY",   "Eli Lilly",             "Healthcare"),
    ("JNJ",   "Johnson & Johnson",     "Healthcare"),
    ("PFE",   "Pfizer",                "Healthcare"),
    ("MRK",   "Merck",                 "Healthcare"),
    ("ABBV",  "AbbVie",                "Healthcare"),
    ("BMY",   "Bristol-Myers Squibb",  "Healthcare"),
    ("ABT",   "Abbott",                "Healthcare"),
    # ── Industrials / cyclicals (4) ──────────────────────────────────────
    ("CAT",   "Caterpillar",           "Industrials"),
    ("DE",    "John Deere",            "Industrials"),
    ("GE",    "GE",                    "Industrials"),
    ("EMR",   "Emerson",               "Industrials"),
    # ── Retail / consumer staples + discretionary (5) ────────────────────
    ("WMT",   "Walmart",               "Retail"),
    ("COST",  "Costco",                "Retail"),
    ("HD",    "Home Depot",            "Retail"),
    ("TGT",   "Target",                "Retail"),
    ("PG",    "Procter & Gamble",      "Retail"),
    # ── Auto / transport (3) ─────────────────────────────────────────────
    ("F",     "Ford",                  "Auto"),
    ("GM",    "General Motors",        "Auto"),
    ("FDX",   "FedEx",                 "Transport"),
]

# ─── RSS Feeds ────────────────────────────────────────────────────────────────
# Curated free financial RSS feeds. Each entry: (label, url, [tickers_to_tag])
# tickers_to_tag: pre-tagged tickers (e.g. broad market = SPY); else scanner runs post-fetch

EQUITY_RSS_FEEDS: list[tuple[str, str, list[str]]] = [
    # ── Broad market ─────────────────────────────────────────────────────────
    # Note: CNBC disabled public RSS (403 as of 2026). Replaced with working alternatives.
    ("MarketWatch",        "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines", ["SPY"]),
    ("WSJ US Business",    "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",                   ["SPY", "TLT"]),
    ("Seeking Alpha",      "https://seekingalpha.com/market_currents.xml",                       ["SPY"]),
    # ── Press releases ───────────────────────────────────────────────────────
    ("PR Newswire",        "https://www.prnewswire.com/rss/news-releases-list.rss",             []),
    # ── Benzinga (Phase 2 of C, added 2026-04-27) ────────────────────────────
    # Per-symbol Benzinga feeds (e.g. /quote/<TICKER>/feed) are 404 — gated
    # behind paid Benzinga Pro. The free general-news feeds work and tend to
    # carry breaking earnings/movers items minutes ahead of aggregator RSS.
    # No pre-tagged tickers — rely on the post-fetch _scan_tickers() scanner.
    ("Benzinga News",      "https://www.benzinga.com/news/feed",                                 []),
    ("Benzinga Earnings",  "https://www.benzinga.com/news/earnings/feed",                        []),
    ("Benzinga Markets",   "https://www.benzinga.com/markets/feed",                              []),
]

# ─── Ticker scanner ───────────────────────────────────────────────────────────
# Regex to find ticker symbols in headline/summary text.
# Matches $AAPL or standalone uppercase symbols 1-5 chars (avoids common words).
_COMMON_WORDS = {
    "A", "I", "IT", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "MY",
    "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE", "AI", "GDP", "CEO",
    "IPO", "ETF", "SEC", "FED", "IMF", "GDP", "CPI", "EPS", "FDA", "IRS",
    "THE", "AND", "FOR", "ARE", "NOT", "BUT", "CAN", "ALL", "HAS", "ITS",
    "NEW", "NOW", "ONE", "OUT", "TOP", "TWO", "WAY", "WHO", "WHY", "YET",
    "MAY", "NET", "BAD", "BIG", "BUY", "CUT", "DUE", "END", "EPS", "GET",
    "HOW", "KEY", "LOW", "OIL", "PAY", "PUT", "RUN", "SAY", "SEE", "SET",
    "TAX", "USE", "WIN", "YET", "Q1", "Q2", "Q3", "Q4",
}

# Ticker symbols we actually care about. Single source of truth — the
# coverage_monitor's bellwether floor reads the same TOP_COMPANY_TARGETS.
_ALL_TICKERS: set[str] = (
    {t for t, _, _ in SECTOR_ETFS} |
    {t for t, _, _ in TOP_COMPANY_TARGETS}
)

_DOLLAR_TICKER_RE = re.compile(r'\$([A-Z]{1,5})\b')
_BARE_TICKER_RE   = re.compile(r'\b([A-Z]{2,5})\b')


def _scan_tickers(text: str) -> list[str]:
    """Return known ticker symbols found in text."""
    found: list[str] = []
    for m in _DOLLAR_TICKER_RE.finditer(text):
        t = m.group(1)
        if t in _ALL_TICKERS:
            found.append(t)
    for m in _BARE_TICKER_RE.finditer(text):
        t = m.group(1)
        if t in _ALL_TICKERS and t not in _COMMON_WORDS:
            found.append(t)
    return list(dict.fromkeys(found))  # deduplicate, preserve order


# ─── Helpers (shared with sovereign news.py) ─────────────────────────────────

def _item_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _strip_html(text: str) -> str:
    if not text or "<" not in text:
        return text
    try:
        parsed = ET.fromstring(f"<r>{text}</r>")
        return " ".join(parsed.itertext()).strip()
    except ET.ParseError:
        return re.sub(r"<[^>]+>", "", text).strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _since_dt(hours: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _parse_dt(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            continue
    try:
        return parsedate_to_datetime(raw)
    except Exception:
        return None


def _is_recent(published_at: Optional[str], since: datetime) -> bool:
    if not published_at:
        return True  # for equity news, include undated items (PR Newswire often omits)
    dt = _parse_dt(published_at)
    if dt is None:
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= since


def _make_item(
    url: str,
    headline: str,
    summary: str,
    source: str,
    published_at: str,
    fetch_source: str,
    tickers: list[str],
) -> NewsItem:
    combined = f"{headline} {summary}"
    scanned  = _scan_tickers(combined)
    all_tickers = list(dict.fromkeys(tickers + scanned))
    return NewsItem(
        item_id      = _item_id(url),
        country      = "United States",
        country_iso3 = "USA",
        headline     = headline[:500],
        summary      = summary[:1500],
        source       = source,
        source_url   = url,
        published_at = published_at,
        fetched_at   = _now_iso(),
        language     = "en",
        fetch_source = fetch_source,
        tone         = 0.0,
        raw_actors   = all_tickers,
    )


# ─── Main Fetcher ─────────────────────────────────────────────────────────────

class EquityNewsFetcher:
    """
    Async equity + sector news fetcher.

    Sources:
      - Curated financial RSS feeds (MarketWatch, WSJ, Seeking Alpha
        general, Benzinga news/earnings/markets, PR Newswire)
      - Yahoo Finance per-ticker RSS for the 65 bellwether tickers in
        TOP_COMPANY_TARGETS. ~20 items per ticker, properly dated.
      - Seeking Alpha per-ticker RSS for the same 65 bellwether tickers
        via their /api/sa/combined endpoint (~30 items per ticker,
        analyst-leaning). The seekingalpha.com/symbol/<TICKER>/news URL
        in Hermes's brief is 403-CDN-blocked; the api/sa/combined path
        is the working alternative.
      The combined per-ticker fan-out (130 fetches per cycle) is bounded
      by independent semaphores; URL-hash dedup in fetch_all() collapses
      cross-source overlap. Replaces the dead GNews path (12h free-tier
      delay > 6h cron window).
    """

    YAHOO_FINANCE_RSS  = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    SEEKING_ALPHA_RSS  = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"

    def __init__(
        self,
        timeout: float = 20.0,
    ):
        self._timeout = timeout
        self._seen: set[str] = set()
        # Polite concurrent-request cap per source. 8 is well within both
        # Yahoo's and SA's public-RSS tolerance and keeps a 65-ticker fan-out
        # under ~10s end-to-end per source.
        self._yh_sem = asyncio.Semaphore(8)
        self._sa_sem = asyncio.Semaphore(8)

    async def fetch_all(self, since_hours: int = 6) -> list[NewsItem]:
        """
        Fetch equity news from all sources.

        Returns:
            Deduplicated list of NewsItem, sorted newest-first.
        """
        since = _since_dt(since_hours)
        items: list[NewsItem] = []

        results = await asyncio.gather(
            self._fetch_equity_rss(since),
            self._fetch_yahoo_per_ticker(since),
            self._fetch_sa_per_ticker(since),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error("Equity news source error: %s", result)
                continue
            for item in result:
                if item.item_id not in self._seen:
                    self._seen.add(item.item_id)
                    items.append(item)

        logger.info(
            "EquityNewsFetcher: %d unique items from last %d hours",
            len(items), since_hours,
        )
        return items

    # ── RSS ───────────────────────────────────────────────────────────────────

    async def _fetch_equity_rss(self, since: datetime) -> list[NewsItem]:
        items: list[NewsItem] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [
                self._fetch_one_rss(client, label, url, tickers, since)
                for label, url, tickers in EQUITY_RSS_FEEDS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.debug("RSS error: %s", r)
                    continue
                items.extend(r)
        logger.info("  RSS: %d equity items", len(items))
        return items

    async def _fetch_one_rss(
        self,
        client: httpx.AsyncClient,
        label: str,
        url: str,
        pre_tickers: list[str],
        since: datetime,
    ) -> list[NewsItem]:
        try:
            resp = await client.get(url, follow_redirects=True)
            if resp.status_code != 200:
                logger.debug("RSS %s: HTTP %d", label, resp.status_code)
                return []

            # Some CDNs (e.g. Cloudflare on Benzinga) inject a trailing
            # <script> after </rss>, breaking strict XML. Truncate at the
            # last close-of-root we recognise so the inner document parses.
            body = resp.text
            for end_tag in ("</rss>", "</feed>"):
                idx = body.rfind(end_tag)
                if idx != -1:
                    body = body[: idx + len(end_tag)]
                    break

            root = ET.fromstring(body)
            ns   = {"atom": "http://www.w3.org/2005/Atom"}

            # Handle both RSS 2.0 (<item>) and Atom (<entry>) formats
            entries = root.findall(".//item") or root.findall(".//atom:entry", ns)
            items: list[NewsItem] = []

            for entry in entries:
                def _t(tag: str, default: str = "") -> str:
                    # ElementTree elements are falsy when they have no children,
                    # so we must use 'is not None' instead of 'or'.
                    el = entry.find(tag)
                    if el is None:
                        el = entry.find(f"atom:{tag}", ns)
                    if el is None:
                        return default
                    return (el.text or "").strip()

                headline = _strip_html(_t("title"))
                if not headline:
                    continue

                # URL: prefer <link> text, else href attribute
                link_el = entry.find("link")
                if link_el is None:
                    link_el = entry.find("atom:link", ns)
                link_url = ""
                if link_el is not None:
                    link_url = (link_el.text or "").strip() or link_el.get("href", "")

                if not link_url:
                    continue

                pub_raw = _t("pubDate") or _t("published") or _t("updated")
                if not _is_recent(pub_raw, since):
                    continue

                pub_iso = ""
                dt = _parse_dt(pub_raw)
                if dt:
                    pub_iso = dt.isoformat()

                summary = _strip_html(_t("description") or _t("summary") or _t("content"))

                items.append(_make_item(
                    url        = link_url,
                    headline   = headline,
                    summary    = summary,
                    source     = label,
                    published_at = pub_iso,
                    fetch_source = "equity_rss",
                    tickers    = pre_tickers,
                ))

            logger.debug("RSS %s: %d items", label, len(items))
            return items

        except ET.ParseError as exc:
            logger.debug("RSS %s: XML parse error — %s", label, exc)
            return []
        except Exception as exc:
            logger.debug("RSS %s: %s", label, exc)
            return []

    # ── Yahoo Finance per-ticker ─────────────────────────────────────────────

    async def _fetch_yahoo_per_ticker(self, since: datetime) -> list[NewsItem]:
        """Fetch the per-ticker RSS feed for every bellwether in
        TOP_COMPANY_TARGETS. Each feed returns ~20 items; cross-mentions are
        heavy (a Mag 7 earnings story appears in 5+ feeds), so URL-hash dedup
        in fetch_all() collapses the volume. Net result is typically
        200-300 unique items per cycle across the 65-ticker list.
        """
        items: list[NewsItem] = []
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; KB-fetcher/1.0)"},
        ) as client:
            tasks = [
                self._fetch_one_yahoo(client, ticker, name, since)
                for ticker, name, _sector in TOP_COMPANY_TARGETS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            n_failed = 0
            for r in results:
                if isinstance(r, Exception):
                    n_failed += 1
                    logger.debug("Yahoo Finance fetch error: %s", r)
                    continue
                items.extend(r)
            if n_failed:
                logger.warning(
                    "  Yahoo Finance: %d/%d ticker fetches failed",
                    n_failed, len(TOP_COMPANY_TARGETS),
                )

        logger.info(
            "  Yahoo Finance: %d equity items across %d tickers",
            len(items), len(TOP_COMPANY_TARGETS),
        )
        return items

    async def _fetch_one_yahoo(
        self,
        client: httpx.AsyncClient,
        ticker: str,
        name: str,
        since: datetime,
    ) -> list[NewsItem]:
        """Fetch one ticker's Yahoo Finance RSS feed."""
        url = self.YAHOO_FINANCE_RSS.format(ticker=ticker)
        async with self._yh_sem:
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.debug("Yahoo Finance %s: HTTP %d", ticker, resp.status_code)
                    return []
                root = ET.fromstring(resp.text)
            except (httpx.HTTPError, ET.ParseError) as exc:
                logger.debug("Yahoo Finance %s: %s", ticker, exc)
                return []

        out: list[NewsItem] = []
        for entry in root.findall(".//item"):
            title_el = entry.find("title")
            link_el  = entry.find("link")
            desc_el  = entry.find("description")
            pub_el   = entry.find("pubDate")

            link_url = (link_el.text or "").strip() if link_el is not None else ""
            headline = _strip_html((title_el.text or "")) if title_el is not None else ""
            if not link_url or not headline:
                continue

            pub_raw = (pub_el.text or "").strip() if pub_el is not None else ""
            if not _is_recent(pub_raw, since):
                continue
            pub_iso = pub_raw
            dt = _parse_dt(pub_raw)
            if dt:
                pub_iso = dt.isoformat()

            summary = _strip_html((desc_el.text or "")) if desc_el is not None else ""

            out.append(_make_item(
                url          = link_url,
                headline     = headline,
                summary      = summary,
                source       = f"Yahoo Finance ({ticker})",
                published_at = pub_iso,
                fetch_source = "equity_yahoo",
                tickers      = [ticker],
            ))
        return out

    # ── Seeking Alpha per-ticker ─────────────────────────────────────────────

    async def _fetch_sa_per_ticker(self, since: datetime) -> list[NewsItem]:
        """Fetch the per-ticker SA RSS for every bellwether in TOP_COMPANY_TARGETS.
        SA's per-ticker feed is 30 items deep and analyst-leaning; complements
        Yahoo's broader market mix. Cross-mentions and Yahoo-overlap collapse
        via URL-hash dedup in fetch_all().
        """
        items: list[NewsItem] = []
        async with httpx.AsyncClient(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; KB-fetcher/1.0)"},
        ) as client:
            tasks = [
                self._fetch_one_sa(client, ticker, since)
                for ticker, _name, _sector in TOP_COMPANY_TARGETS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            n_failed = 0
            for r in results:
                if isinstance(r, Exception):
                    n_failed += 1
                    logger.debug("Seeking Alpha fetch error: %s", r)
                    continue
                items.extend(r)
            if n_failed:
                logger.warning(
                    "  Seeking Alpha: %d/%d ticker fetches failed",
                    n_failed, len(TOP_COMPANY_TARGETS),
                )
        logger.info(
            "  Seeking Alpha: %d equity items across %d tickers",
            len(items), len(TOP_COMPANY_TARGETS),
        )
        return items

    async def _fetch_one_sa(
        self,
        client: httpx.AsyncClient,
        ticker: str,
        since: datetime,
    ) -> list[NewsItem]:
        """Fetch one ticker's Seeking Alpha api/sa/combined RSS feed."""
        url = self.SEEKING_ALPHA_RSS.format(ticker=ticker)
        async with self._sa_sem:
            try:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.debug("Seeking Alpha %s: HTTP %d", ticker, resp.status_code)
                    return []
                # Same Cloudflare-trailing-junk safeguard as _fetch_one_rss.
                body = resp.text
                for end_tag in ("</rss>", "</feed>"):
                    idx = body.rfind(end_tag)
                    if idx != -1:
                        body = body[: idx + len(end_tag)]
                        break
                root = ET.fromstring(body)
            except (httpx.HTTPError, ET.ParseError) as exc:
                logger.debug("Seeking Alpha %s: %s", ticker, exc)
                return []

        out: list[NewsItem] = []
        for entry in root.findall(".//item"):
            title_el = entry.find("title")
            link_el  = entry.find("link")
            desc_el  = entry.find("description")
            pub_el   = entry.find("pubDate")

            link_url = (link_el.text or "").strip() if link_el is not None else ""
            headline = _strip_html((title_el.text or "")) if title_el is not None else ""
            if not link_url or not headline:
                continue

            pub_raw = (pub_el.text or "").strip() if pub_el is not None else ""
            if not _is_recent(pub_raw, since):
                continue
            pub_iso = pub_raw
            dt = _parse_dt(pub_raw)
            if dt:
                pub_iso = dt.isoformat()

            summary = _strip_html((desc_el.text or "")) if desc_el is not None else ""

            out.append(_make_item(
                url          = link_url,
                headline     = headline,
                summary      = summary,
                source       = f"Seeking Alpha ({ticker})",
                published_at = pub_iso,
                fetch_source = "equity_sa",
                tickers      = [ticker],
            ))
        return out

    async def health_check(self) -> dict[str, bool]:
        """Quick health check for each source."""
        results: dict[str, bool] = {}

        async with httpx.AsyncClient(timeout=10.0) as client:
            # RSS: try first feed
            try:
                r = await client.get(EQUITY_RSS_FEEDS[0][1], follow_redirects=True)
                results["equity_rss"] = r.status_code == 200
            except Exception:
                results["equity_rss"] = False

            # Yahoo Finance: try one ticker (NVDA — high traffic, almost never down)
            try:
                r = await client.get(
                    self.YAHOO_FINANCE_RSS.format(ticker="NVDA"),
                    follow_redirects=True,
                )
                results["equity_yahoo"] = r.status_code == 200
            except Exception:
                results["equity_yahoo"] = False

            # Seeking Alpha per-ticker
            try:
                r = await client.get(
                    self.SEEKING_ALPHA_RSS.format(ticker="NVDA"),
                    follow_redirects=True,
                )
                results["equity_sa"] = r.status_code == 200
            except Exception:
                results["equity_sa"] = False

        return results
