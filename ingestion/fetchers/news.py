"""
Multi-source news fetcher.

Sources (in priority order):
  1. NewsData.io  — primary structured API, 76 countries, 16 languages, full text
  2. GNews        — Google News quality for Western countries, named-entity search
  3. GDELT DOC    — free backbone for Russia, China, Arabic-language markets
  4. RSS feeds    — curated state/official media for countries with weak API coverage

Each source contributes NewsItem objects. Deduplication is by item_id
(sha256 of canonical URL, first 16 hex chars).

Usage:
    fetcher = NewsFetcher()
    items = await fetcher.fetch_all(since_hours=6)
    # → list[NewsItem], deduplicated, all 20 countries
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

import httpx

from processing.news_queue import NewsItem

logger = logging.getLogger(__name__)

# ─── Country Code Maps ────────────────────────────────────────────────────────

# ISO-3 → ISO-2 lowercase (NewsData.io format)
ISO3_TO_NEWSDATA: dict[str, str] = {
    "USA": "us", "CHN": "cn", "DEU": "de", "IND": "in", "JPN": "jp",
    "GBR": "gb", "FRA": "fr", "ITA": "it", "RUS": "ru", "CAN": "ca",
    "BRA": "br", "ESP": "es", "MEX": "mx", "AUS": "au", "KOR": "kr",
    "TUR": "tr", "IDN": "id", "NLD": "nl", "SAU": "sa", "POL": "pl",
    "ARG": "ar", "ZAF": "za", "IRN": "ir", "UAE": "ae", "QAT": "qa",
    "ISR": "il",
}

# ISO-3 → ISO-2 uppercase (GNews / GDELT format)
ISO3_TO_ISO2: dict[str, str] = {k: v.upper() for k, v in ISO3_TO_NEWSDATA.items()}

# ISO-3 → full country name (for NewsItem.country)
ISO3_TO_NAME: dict[str, str] = {
    # G20 members
    "USA": "United States", "CHN": "China", "DEU": "Germany",
    "IND": "India", "JPN": "Japan", "GBR": "United Kingdom",
    "FRA": "France", "ITA": "Italy", "RUS": "Russia", "CAN": "Canada",
    "BRA": "Brazil", "ESP": "Spain", "MEX": "Mexico", "AUS": "Australia",
    "KOR": "South Korea", "TUR": "Turkey", "IDN": "Indonesia",
    "NLD": "Netherlands", "SAU": "Saudi Arabia", "POL": "Poland",
    "ARG": "Argentina", "ZAF": "South Africa",
    # Critical non-G20 (active conflict theater / Gulf)
    "IRN": "Iran", "UAE": "United Arab Emirates", "QAT": "Qatar",
    "ISR": "Israel",
}

# ─── RSS Feeds ────────────────────────────────────────────────────────────────
# Curated for countries with weak commercial API coverage.
# Each entry: (label, url, iso3)

# Feeds whose RSS items lack <pubDate> (or any equivalent timestamp).
# Without a pubDate, _is_recent() returned False and dropped every item.
# For these feeds we accept items unconditionally and use the URL-hash
# seen-IDs cache (data/raw/news_seen_ids.parquet, 7d retention) to prevent
# reprocessing on subsequent runs. First-fix surfaced for IRN: PressTV +
# Iran International each ship ~100 items per fetch with no pubDate, so
# the entire IRN feed had been silently dry since 2026-03-11.
DATELESS_FEED_FALLBACK = {
    "PressTV", "Iran International",
}

RSS_FEEDS: list[tuple[str, str, str]] = [
    # China — xinhuanet.com worldrss.xml serves stale 2017-2018 content; replaced.
    ("SCMP China",      "https://www.scmp.com/rss/2/feed",                        "CHN"),
    ("Global Times",    "https://www.globaltimes.cn/rss/outbrain.xml",            "CHN"),
    ("CGTN World",      "https://www.cgtn.com/subscribe/rss/section/world.xml",   "CHN"),
    # Russia
    ("TASS English",    "https://tass.com/rss/v2.xml",                            "RUS"),
    ("RT News",         "https://www.rt.com/rss/news/",                           "RUS"),
    # Saudi Arabia / Gulf — arabnews and alarabiya block scrapers; use Gulf News
    ("Gulf News",       "https://gulfnews.com/rss/world",                         "SAU"),
    ("Saudi Gazette",   "https://saudigazette.com.sa/feed",                       "SAU"),
    # Indonesia
    ("Antara News",     "https://en.antaranews.com/rss/news.xml",                 "IDN"),
    ("Jakarta Post",    "https://www.thejakartapost.com/rss/category/indonesia",  "IDN"),
    # Turkey — dailysabah has malformed XML (BOM); use Anadolu Agency
    ("Anadolu Agency",  "https://www.aa.com.tr/en/rss/default?cat=world",        "TUR"),
    ("Hurriyet Daily",  "https://www.hurriyetdailynews.com/rss/news",             "TUR"),
    # Japan
    ("NHK World",       "https://www3.nhk.or.jp/rss/news/cat0.xml",              "JPN"),
    ("Japan Times",     "https://www.japantimes.co.jp/feed/",                     "JPN"),
    ("Yonhap News",     "https://en.yna.co.kr/RSS/news.xml",                      "KOR"),
    ("Korea JoongAng",  "https://koreajoongangdaily.joins.com/rss/feeds/latest.xml", "KOR"),
    # India
    ("Times of India",  "https://timesofindia.indiatimes.com/rssfeedstopstories.cms", "IND"),
    ("Hindu Business",  "https://www.thehindubusinessline.com/feeder/default.rss", "IND"),
    # Brazil
    ("Agencia Brasil",  "https://agenciabrasil.ebc.com.br/rss/ultimasnoticias/feed.xml", "BRA"),
    # Poland — pap.pl has malformed XML; use Notes From Poland
    ("Notes From PL",   "https://notesfrompoland.com/feed/",                      "POL"),
    # Argentina
    ("Buenos Aires Herald", "https://buenosairesherald.com/feed",                 "ARG"),
    ("MercoPress",          "https://en.mercopress.com/rss",                      "ARG"),
    # South Africa
    ("Daily Maverick",      "https://www.dailymaverick.co.za/feed/",              "ZAF"),
    ("Business Day ZA",     "https://www.businesslive.co.za/rss/bd/",            "ZAF"),
    # Iran — PressTV is state media but only English-language source available
    ("PressTV",             "https://www.presstv.ir/rss.xml",                     "IRN"),
    ("Iran International",  "https://www.iranintl.com/en/rss",                   "IRN"),
    # Israel — Jerusalem Post + Times of Israel cover IDF ops, government, conflict
    ("Jerusalem Post",      "https://www.jpost.com/rss/rssfeedsfrontpage.aspx",   "ISR"),
    ("Times of Israel",     "https://www.timesofisrael.com/feed/",                "ISR"),
    ("Haaretz English",     "https://www.haaretz.com/srv/haaretz-rss.xml",        "ISR"),
    # UAE
    ("The National UAE",    "https://www.thenationalnews.com/rss.xml",            "UAE"),
    ("Gulf News UAE",       "https://gulfnews.com/rss/uae",                       "UAE"),
    # Qatar
    ("Al Jazeera English",  "https://www.aljazeera.com/xml/rss/all.xml",          "QAT"),
    ("Qatar Tribune",       "https://www.qatar-tribune.com/rss.xml",              "QAT"),
    # United Kingdom — country-specific feeds only (world feeds bleed in lifestyle/sports)
    ("BBC UK News",         "https://feeds.bbci.co.uk/news/uk/rss.xml",           "GBR"),
    ("BBC UK Politics",     "https://feeds.bbci.co.uk/news/politics/rss.xml",     "GBR"),
    ("The Guardian UK",     "https://www.theguardian.com/uk/rss",                 "GBR"),
    # Germany — DW Germany feed (not "all" which includes world sports/entertainment)
    ("DW Germany",          "https://rss.dw.com/xml/rss-en-ger",                  "DEU"),
    ("DW Europe",           "https://rss.dw.com/xml/rss-en-eu",                   "DEU"),
    # France — /en/france/rss is stale (last updated 2026-03-02); Europe feed is current
    ("France 24 Europe",    "https://www.france24.com/en/europe/rss",             "FRA"),
    # Italy — slim English coverage; Italian Insider covers politics/business
    ("Italian Insider",     "https://www.italianinsider.it/?q=rss.xml",           "ITA"),
    # Politico Europe — EU-wide governance/policy; assign to GBR (Minerva) for broad European coverage
    ("Politico Europe",     "https://www.politico.eu/feed/",                      "GBR"),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _item_id(url: str) -> str:
    """Stable dedup key: first 16 hex chars of sha256 of the URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string using the stdlib XML parser (no deps)."""
    if not text or "<" not in text:
        return text
    try:
        # Wrap in a root element so the parser handles fragments cleanly
        parsed = ET.fromstring(f"<r>{text}</r>")
        return " ".join((parsed.itertext())).strip()
    except ET.ParseError:
        # Fallback: naive tag strip
        import re
        return re.sub(r"<[^>]+>", "", text).strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _since_dt(hours: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def _parse_dt(raw: Optional[str]) -> Optional[datetime]:
    """Try to parse a date string into a UTC-aware datetime."""
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
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    try:
        return parsedate_to_datetime(raw)
    except Exception:
        return None


def _is_recent(published_at: Optional[str], since: datetime) -> bool:
    if not published_at:
        return False  # exclude if no date — avoids pulling stale undated content
    dt = _parse_dt(published_at)
    if dt is None:
        return False  # exclude if date unparseable — avoids old Xinhua-style articles
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= since


# ─── Main Fetcher ─────────────────────────────────────────────────────────────

class NewsFetcher:
    """
    Async multi-source news fetcher for all 20 tracked countries.

    Pulls from NewsData.io, GNews, GDELT, and curated RSS feeds.
    Returns deduplicated NewsItem objects from the last `since_hours` hours.
    """

    NEWSDATA_BASE = "https://newsdata.io/api/1/news"
    GNEWS_BASE    = "https://gnews.io/api/v4/search"
    GDELT_BASE    = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Default GNews query — broad enough to catch sovereign intelligence stories.
    # Override via gnews_query param when more precision is needed.
    DEFAULT_GNEWS_QUERY = (
        "government OR minister OR trade OR sanctions OR economy "
        "OR military OR election OR policy OR central bank OR GDP"
    )

    def __init__(
        self,
        newsdata_api_key: Optional[str] = None,
        gnews_api_key: Optional[str] = None,
        timeout: float = 20.0,
        newsdata_batch_size: int = 1,
        gnews_query: Optional[str] = None,
    ):
        """
        Args:
            newsdata_batch_size: Countries per NewsData.io request.
                Free tier = 1 (only 1 country code allowed).
                Paid tier = 5 (API limit). Pass 5 on paid keys to run 4× faster.
            gnews_query: Override the default GNews query string.
                Default covers sovereign-intelligence topics broadly.
        """
        self._nd_key     = newsdata_api_key or os.environ.get("NEWSDATA_API_KEY", "")
        self._gn_key     = gnews_api_key    or os.environ.get("GNEWS_API_KEY", "")
        self._timeout    = timeout
        self._nd_batch   = newsdata_batch_size
        self._gn_query   = gnews_query or self.DEFAULT_GNEWS_QUERY
        # _seen deduplicates within a single fetch_all() call (and across repeated
        # calls on the same instance, e.g. in streaming loops). Intentional.
        self._seen: set[str] = set()
        # Rate-limit concurrent requests per API to avoid 429s.
        # NewsData free tier: ~10 req/s burst tolerated; use 3 concurrent + 0.2s gap.
        # GNews free tier: ~2 req/s; use 2 concurrent + 0.5s gap.
        # GDELT public API: ~1 req/s; use 1 (sequential) + 0.5s gap.
        self._nd_sem    = asyncio.Semaphore(3)
        self._gn_sem    = asyncio.Semaphore(2)
        self._gdelt_sem = asyncio.Semaphore(1)

        if not self._nd_key:
            logger.warning("No NEWSDATA_API_KEY — NewsData.io source disabled")
        if not self._gn_key:
            logger.warning("No GNEWS_API_KEY — GNews source disabled")

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_all(
        self,
        iso3_list: Optional[list[str]] = None,
        since_hours: int = 6,
    ) -> list[NewsItem]:
        """
        Fetch news for all (or a subset of) countries from all sources.

        Args:
            iso3_list: ISO-3 codes to fetch. Defaults to all 20 tracked countries.
            since_hours: Only return articles published in the last N hours.

        Returns:
            Deduplicated list of NewsItem, sorted newest-first.
        """
        if iso3_list is None:
            iso3_list = list(ISO3_TO_NAME.keys())

        since = _since_dt(since_hours)
        items: list[NewsItem] = []

        # Run all sources concurrently
        results = await asyncio.gather(
            self._fetch_newsdata(iso3_list, since),
            self._fetch_gnews(iso3_list, since),
            self._fetch_gdelt(iso3_list, since),
            self._fetch_rss(iso3_list, since),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error("News source error: %s", result)
                continue
            for item in result:
                if item.item_id not in self._seen:
                    self._seen.add(item.item_id)
                    items.append(item)

        logger.info(
            "NewsFetcher: %d unique items from last %d hours across %d countries",
            len(items), since_hours, len(iso3_list),
        )
        return items

    async def health_check(self) -> dict[str, bool]:
        """Verify each source is reachable."""
        results: dict[str, bool] = {}

        async with httpx.AsyncClient(timeout=10.0) as client:
            # NewsData.io
            try:
                r = await client.get(
                    self.NEWSDATA_BASE,
                    params={"apikey": self._nd_key, "country": "us", "size": 1},
                )
                results["newsdata"] = r.status_code == 200
            except Exception:
                results["newsdata"] = False

            # GNews
            try:
                r = await client.get(
                    self.GNEWS_BASE,
                    params={"token": self._gn_key, "q": "economy", "country": "us", "max": 1},
                )
                results["gnews"] = r.status_code == 200
            except Exception:
                results["gnews"] = False

            # GDELT
            try:
                r = await client.get(
                    self.GDELT_BASE,
                    params={"query": "economy", "mode": "artlist", "format": "json",
                            "maxrecords": 1},
                )
                results["gdelt"] = r.status_code == 200
            except Exception:
                results["gdelt"] = False

        results["rss"] = True  # RSS has no central health endpoint
        return results

    # ── NewsData.io ───────────────────────────────────────────────────────────

    async def _fetch_newsdata(
        self,
        iso3_list: list[str],
        since: datetime,
    ) -> list[NewsItem]:
        """
        NewsData.io: fetch up to 10 countries per request (API limit: 5 codes).
        Batches countries into groups of 5, runs concurrently.
        """
        if not self._nd_key:
            return []

        # Map to NewsData country codes, drop any without a mapping
        nd_codes = [
            (iso3, ISO3_TO_NEWSDATA[iso3])
            for iso3 in iso3_list
            if iso3 in ISO3_TO_NEWSDATA
        ]

        batch_size = self._nd_batch  # 1 for free tier, up to 5 for paid
        batches: list[list[tuple[str, str]]] = []
        for i in range(0, len(nd_codes), batch_size):
            batches.append(nd_codes[i : i + batch_size])

        tasks = [self._newsdata_batch(batch, since) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[NewsItem] = []
        for r in results:
            if isinstance(r, list):
                items.extend(r)
        return items

    async def _newsdata_batch(
        self,
        batch: list[tuple[str, str]],  # [(iso3, nd_code), ...]
        since: datetime,
    ) -> list[NewsItem]:
        iso3_by_nd = {nd: iso3 for iso3, nd in batch}
        country_param = ",".join(nd for _, nd in batch)
        fetched = _now_iso()
        items: list[NewsItem] = []

        params = {
            "apikey":   self._nd_key,
            "country":  country_param,
            # NOTE: from_date is paid-only (/1/archive endpoint).
            # Free /1/news returns the ~200 most recent articles regardless of age.
            # Temporal coverage is filtered client-side by _is_recent().
            # Agent interpretations should note that NewsData items may span >6 h.
            "size":     10,
        }

        async with self._nd_sem:
            await asyncio.sleep(0.2)
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(self.NEWSDATA_BASE, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                    for art in data.get("results", []):
                        url = art.get("link", "")
                        if not url:
                            continue
                        pub = art.get("pubDate", "")
                        if not _is_recent(pub, since):
                            continue

                        # Resolve country: NewsData returns a list of country codes
                        art_countries = art.get("country", []) or []
                        iso3 = "UNK"
                        for nd_code in art_countries:
                            if nd_code in iso3_by_nd:
                                iso3 = iso3_by_nd[nd_code]
                                break
                        if iso3 == "UNK" and batch:
                            iso3 = batch[0][0]  # fallback to first in batch

                        summary = (art.get("content") or art.get("description") or "")[:1500]

                        items.append(NewsItem(
                            item_id=_item_id(url),
                            country=ISO3_TO_NAME.get(iso3, iso3),
                            country_iso3=iso3,
                            headline=art.get("title", "").strip(),
                            summary=summary.strip(),
                            source=art.get("source_id", ""),
                            source_url=url,
                            published_at=pub,
                            fetched_at=fetched,
                            language=art.get("language", "en"),
                            fetch_source="newsdata",
                            tone=0.0,
                            raw_actors=art.get("creator") or [],
                        ))

                    logger.debug(
                        "NewsData.io: %d items for %s",
                        len(items), country_param,
                    )

            except httpx.HTTPStatusError as exc:
                logger.error("NewsData.io HTTP %s for %s", exc.response.status_code, country_param)
            except Exception as exc:
                logger.error("NewsData.io error for %s: %s", country_param, exc)

        return items

    # ── GNews ─────────────────────────────────────────────────────────────────

    async def _fetch_gnews(
        self,
        iso3_list: list[str],
        since: datetime,
    ) -> list[NewsItem]:
        """
        GNews: one request per country (country filter is a single value).
        Focuses on English-language Western countries where GNews quality excels.
        """
        if not self._gn_key:
            return []

        # GNews is strongest for English-language markets
        priority = {"USA", "GBR", "DEU", "FRA", "ITA", "CAN", "AUS", "NLD",
                    "JPN", "KOR", "ESP", "POL"}
        targets = [iso3 for iso3 in iso3_list if iso3 in priority]

        tasks = [self._gnews_country(iso3, since) for iso3 in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[NewsItem] = []
        for r in results:
            if isinstance(r, list):
                items.extend(r)
        return items

    async def _gnews_country(self, iso3: str, since: datetime) -> list[NewsItem]:
        iso2 = ISO3_TO_ISO2.get(iso3, "").lower()
        if not iso2:
            return []

        fetched = _now_iso()
        items: list[NewsItem] = []
        from_str = since.strftime("%Y-%m-%dT%H:%M:%SZ")

        params = {
            "token":   self._gn_key,
            "q":       self._gn_query,
            "country": iso2,
            "lang":    "en",
            "from":    from_str,
            "max":     10,
        }

        async with self._gn_sem:
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(self.GNEWS_BASE, params=params)
                    resp.raise_for_status()
                    data = resp.json()

                    for art in data.get("articles", []):
                        url = art.get("url", "")
                        if not url:
                            continue
                        pub = art.get("publishedAt", "")
                        if not _is_recent(pub, since):
                            continue

                        source = art.get("source", {}).get("name", "")
                        summary = (art.get("content") or art.get("description") or "")[:1500]

                        items.append(NewsItem(
                            item_id=_item_id(url),
                            country=ISO3_TO_NAME.get(iso3, iso3),
                            country_iso3=iso3,
                            headline=art.get("title", "").strip(),
                            summary=summary.strip(),
                            source=source,
                            source_url=url,
                            published_at=pub,
                            fetched_at=fetched,
                            language="en",
                            fetch_source="gnews",
                            tone=0.0,
                            raw_actors=[],
                        ))

                    logger.debug("GNews: %d items for %s", len(items), iso3)

            except httpx.HTTPStatusError as exc:
                logger.error("GNews HTTP %s for %s", exc.response.status_code, iso3)
            except Exception as exc:
                logger.error("GNews error for %s: %s", iso3, exc)

        return items

    # ── GDELT ─────────────────────────────────────────────────────────────────

    async def _fetch_gdelt(
        self,
        iso3_list: list[str],
        since: datetime,
    ) -> list[NewsItem]:
        """
        GDELT DOC API: no key required.
        Strongest for Russia, China, Arabic-language, and Asian markets.
        One request per country to avoid over-querying.
        """
        # GDELT adds most value for non-Western markets
        priority = {"RUS", "CHN", "SAU", "IND", "IDN", "TUR", "BRA", "MEX",
                    "JPN", "KOR", "POL"}
        targets = [iso3 for iso3 in iso3_list if iso3 in priority]

        tasks = [self._gdelt_country(iso3, since) for iso3 in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[NewsItem] = []
        for r in results:
            if isinstance(r, list):
                items.extend(r)
        return items

    async def _gdelt_country(self, iso3: str, since: datetime) -> list[NewsItem]:
        iso2 = ISO3_TO_ISO2.get(iso3, "")
        if not iso2:
            return []

        fetched = _now_iso()
        items: list[NewsItem] = []

        start = since.strftime("%Y%m%d%H%M%S")
        end   = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

        params = {
            "query":         f"sourcecountry:{iso2}",
            "mode":          "artlist",
            "format":        "json",
            "maxrecords":    25,
            "startdatetime": start,
            "enddatetime":   end,
            "sort":          "DateDesc",
        }

        # Semaphore limits concurrent GDELT requests to 2 to avoid 429s.
        # Sleep inside the lock staggers requests by at least 0.5 s.
        async with self._gdelt_sem:
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(self.GDELT_BASE, params=params)
                    if resp.status_code != 200:
                        logger.debug("GDELT non-200 for %s: %s", iso3, resp.status_code)
                        return []

                    body = resp.text.strip()
                    if not body:
                        logger.debug("GDELT: empty response for %s (no results)", iso3)
                        return []
                    try:
                        data = resp.json()
                    except ValueError:
                        # GDELT occasionally returns HTML or plain-text on overload
                        logger.debug(
                            "GDELT: non-JSON 200 for %s (len=%d) — skipping",
                            iso3, len(body),
                        )
                        return []
                    for art in data.get("articles", []):
                        url = art.get("url", "")
                        if not url:
                            continue

                        pub_raw = art.get("seendate", "")
                        # GDELT seendate format: "20260220T123456Z"
                        pub_iso = ""
                        if pub_raw:
                            try:
                                dt = datetime.strptime(pub_raw, "%Y%m%dT%H%M%SZ")
                                dt = dt.replace(tzinfo=timezone.utc)
                                pub_iso = dt.isoformat()
                                if dt < since:
                                    continue
                            except ValueError:
                                pub_iso = pub_raw

                        tone_normalised = 0.0  # artlist mode doesn't return tone

                        items.append(NewsItem(
                            item_id=_item_id(url),
                            country=ISO3_TO_NAME.get(iso3, iso3),
                            country_iso3=iso3,
                            headline=art.get("title", "").strip(),
                            summary="",  # GDELT artlist returns no body text
                            source=art.get("domain", ""),
                            source_url=url,
                            published_at=pub_iso,
                            fetched_at=fetched,
                            language=art.get("language", "en").lower()[:2],
                            fetch_source="gdelt",
                            tone=tone_normalised,
                            raw_actors=[],
                        ))

                    logger.debug("GDELT: %d items for %s", len(items), iso3)

            except Exception as exc:
                logger.error("GDELT error for %s: %s", iso3, exc)

        return items

    # ── RSS ───────────────────────────────────────────────────────────────────

    async def _fetch_rss(
        self,
        iso3_list: list[str],
        since: datetime,
    ) -> list[NewsItem]:
        """
        Curated RSS feeds for state/official media in hard-to-cover countries.
        Each feed is fetched independently; failures are logged and skipped.
        """
        iso3_set = set(iso3_list)
        relevant_feeds = [(label, url, iso3) for label, url, iso3 in RSS_FEEDS
                          if iso3 in iso3_set]

        tasks = [self._rss_feed(label, url, iso3, since)
                 for label, url, iso3 in relevant_feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        items: list[NewsItem] = []
        for r in results:
            if isinstance(r, list):
                items.extend(r)
        return items

    async def _rss_feed(
        self,
        label: str,
        url: str,
        iso3: str,
        since: datetime,
    ) -> list[NewsItem]:
        fetched = _now_iso()
        items: list[NewsItem] = []

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; KB-fetcher/1.0)"},
            ) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.debug("RSS %s returned %s", label, resp.status_code)
                    return []

                root = ET.fromstring(resp.content)
                # Handle both RSS 2.0 (<item>) and Atom (<entry>)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall(".//item") or root.findall(".//atom:entry", ns)

                for entry in entries:
                    def _text(tag: str) -> str:
                        # NOTE: must use `is not None` — bool(Element) is False for
                        # leaf elements (no children) even when they have text content.
                        el = entry.find(tag)
                        if el is None:
                            el = entry.find(f"atom:{tag}", ns)
                        return (el.text or "").strip() if el is not None else ""

                    link_el = entry.find("link")
                    if link_el is None:
                        link_el = entry.find("atom:link", ns)
                    art_url = (
                        (link_el.text or link_el.get("href", "")).strip()
                        if link_el is not None else ""
                    )
                    if not art_url:
                        continue

                    pub_raw = _text("pubDate") or _text("published") or _text("updated")
                    if not pub_raw and label in DATELESS_FEED_FALLBACK:
                        # Feed publishes no pubDate; accept and stamp with
                        # fetch time. URL-hash dedup prevents reprocessing.
                        pub_iso = fetched
                    elif _is_recent(pub_raw, since):
                        pub_iso = pub_raw
                        dt = _parse_dt(pub_raw)
                        if dt:
                            pub_iso = dt.isoformat()
                    else:
                        continue

                    headline = _text("title")
                    summary  = _strip_html(
                        _text("description") or _text("summary") or _text("content")
                    )[:1500]

                    items.append(NewsItem(
                        item_id=_item_id(art_url),
                        country=ISO3_TO_NAME.get(iso3, iso3),
                        country_iso3=iso3,
                        headline=headline,
                        summary=summary,
                        source=label,
                        source_url=art_url,
                        published_at=pub_iso,
                        fetched_at=fetched,
                        language="en",
                        fetch_source="rss",
                        tone=0.0,
                        raw_actors=[],
                    ))

                logger.debug("RSS %s: %d items", label, len(items))

        except ET.ParseError as exc:
            logger.warning("RSS parse error for %s: %s", label, exc)
        except Exception as exc:
            logger.error("RSS fetch error for %s: %s", label, exc)

        return items
