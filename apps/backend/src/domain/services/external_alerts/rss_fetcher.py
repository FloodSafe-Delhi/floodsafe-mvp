"""
RSS Fetcher - Fetches flood-related news from RSS feeds.

Supported feeds:
Delhi (7 sources):
- Hindustan Times, Times of India, Indian Express, NDTV
- The Hindu Delhi, Economic Times, News18 India

Bangalore (4 sources):
- Hindustan Times, Times of India
- Deccan Herald, The Hindu Bangalore

No API keys required.
"""

import asyncio
import aiohttp
import feedparser
from datetime import datetime, timezone
from typing import Optional
from email.utils import parsedate_to_datetime
import logging

from .base_fetcher import BaseFetcher, ExternalAlertCreate
from src.core.config import settings

logger = logging.getLogger(__name__)


# RSS feeds configuration per city
RSS_FEEDS = {
    "delhi": [
        {
            "url": "https://www.hindustantimes.com/feeds/rss/delhi-news/rss.xml",
            "name": "Hindustan Times",
            "short_name": "HT"
        },
        {
            "url": "https://timesofindia.indiatimes.com/rssfeeds/-2128838597.cms",
            "name": "Times of India",
            "short_name": "TOI"
        },
        {
            "url": "https://indianexpress.com/section/cities/delhi/feed/",
            "name": "Indian Express",
            "short_name": "IE"
        },
        {
            "url": "https://feeds.feedburner.com/ndtvnews-delhi-news",
            "name": "NDTV Delhi",
            "short_name": "NDTV"
        },
        {
            "url": "https://www.thehindu.com/news/cities/Delhi/feeder/default.rss",
            "name": "The Hindu Delhi",
            "short_name": "TH"
        },
        {
            "url": "https://economictimes.indiatimes.com/news/politics-and-nation/rssfeeds/1052732854.cms",
            "name": "Economic Times",
            "short_name": "ET"
        },
        {
            "url": "https://www.news18.com/rss/india.xml",
            "name": "News18 India",
            "short_name": "N18"
        },
    ],
    "bangalore": [
        {
            "url": "https://www.hindustantimes.com/feeds/rss/cities/bengaluru-news/rss.xml",
            "name": "Hindustan Times Bangalore",
            "short_name": "HT"
        },
        {
            "url": "https://timesofindia.indiatimes.com/rssfeeds/4084750.cms",
            "name": "Times of India Bangalore",
            "short_name": "TOI"
        },
        {
            "url": "https://www.deccanherald.com/feeds/feed-1-22.rss",
            "name": "Deccan Herald",
            "short_name": "DH"
        },
        {
            "url": "https://www.thehindu.com/news/cities/bangalore/feeder/default.rss",
            "name": "The Hindu Bangalore",
            "short_name": "TH"
        },
    ],
}

# Maximum age of articles to fetch (in days)
MAX_ARTICLE_AGE_DAYS = 7


class RSSFetcher(BaseFetcher):
    """Fetches flood-related news from RSS feeds."""

    def __init__(self, timeout: int = 30, max_retries: int = 3):
        """
        Initialize RSS fetcher.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts per feed
        """
        self.timeout = timeout
        self.max_retries = max_retries

    def get_source_name(self) -> str:
        return "rss"

    def is_enabled(self) -> bool:
        # RSS feeds are always enabled - no API keys required
        return getattr(settings, 'RSS_FEEDS_ENABLED', True)

    async def fetch(self, city: str) -> list[ExternalAlertCreate]:
        """
        Fetch flood-related news from RSS feeds for a city.

        Args:
            city: City identifier ('delhi', 'bangalore')

        Returns:
            List of ExternalAlertCreate objects
        """
        self.log_fetch_start(city)

        feeds = RSS_FEEDS.get(city.lower(), [])
        if not feeds:
            logger.warning(f"[RSS] No feeds configured for city: {city}")
            return []

        alerts = []

        # Fetch all feeds concurrently
        tasks = [self._fetch_feed(feed, city) for feed in feeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"[RSS] Feed fetch error: {result}")
                continue
            alerts.extend(result)

        self.log_fetch_complete(city, len(alerts))
        return alerts

    async def _fetch_feed(self, feed_config: dict, city: str) -> list[ExternalAlertCreate]:
        """
        Fetch and parse a single RSS feed with retry logic.

        Args:
            feed_config: Feed configuration dict with url, name, short_name
            city: City identifier

        Returns:
            List of ExternalAlertCreate objects
        """
        url = feed_config["url"]
        feed_name = feed_config["name"]

        for attempt in range(1, self.max_retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                        headers={"User-Agent": "FloodSafe/1.0 (+https://floodsafe.app)"}
                    ) as response:
                        if response.status != 200:
                            logger.warning(f"[RSS] {feed_name}: HTTP {response.status}")
                            if attempt < self.max_retries:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            return []

                        content = await response.text()
                        return self._parse_feed(content, feed_config, city)

            except asyncio.TimeoutError:
                logger.warning(f"[RSS] {feed_name}: Timeout (attempt {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
            except aiohttp.ClientError as e:
                logger.warning(f"[RSS] {feed_name}: Client error: {e} (attempt {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"[RSS] {feed_name}: Unexpected error: {e}")
                break

        return []

    def _parse_feed(self, content: str, feed_config: dict, city: str) -> list[ExternalAlertCreate]:
        """
        Parse RSS feed content and extract flood-related articles.

        Args:
            content: Raw RSS XML content
            feed_config: Feed configuration
            city: City identifier

        Returns:
            List of ExternalAlertCreate objects
        """
        alerts = []
        feed_name = feed_config["name"]

        try:
            feed = feedparser.parse(content)

            if feed.bozo and feed.bozo_exception:
                logger.warning(f"[RSS] {feed_name}: Parse warning: {feed.bozo_exception}")

            entries = feed.entries[:50]  # Limit to 50 most recent
            now = datetime.now(timezone.utc)

            for entry in entries:
                # Get title and description
                title = entry.get("title", "")
                description = entry.get("summary", entry.get("description", ""))

                # Clean HTML from description
                description = self.clean_html(description)

                # Check if article is flood-related
                combined_text = f"{title} {description}"
                if not self.filter_by_keywords(combined_text):
                    continue

                # Parse publication date
                pub_date = self._parse_date(entry)
                if pub_date:
                    # Skip articles older than MAX_ARTICLE_AGE_DAYS
                    age = now - pub_date
                    if age.days > MAX_ARTICLE_AGE_DAYS:
                        continue
                else:
                    pub_date = now

                # Get article URL
                url = entry.get("link", "")

                # Generate unique source_id
                source_id = self.generate_source_id(url, title)

                # Determine severity from keywords
                severity = self._infer_severity(combined_text)

                # Create alert
                alert = ExternalAlertCreate(
                    source="rss",
                    source_id=source_id,
                    source_name=feed_name,
                    city=city,
                    title=self.truncate_text(title, 500),
                    message=self.truncate_text(description, 2000),
                    severity=severity,
                    url=url,
                    raw_data={
                        "feed_url": feed_config["url"],
                        "feed_name": feed_name,
                        "published": pub_date.isoformat() if pub_date else None,
                        "guid": entry.get("id", entry.get("guid", "")),
                    }
                )
                alerts.append(alert)

            logger.debug(f"[RSS] {feed_name}: Found {len(alerts)} flood-related articles")

        except Exception as e:
            logger.error(f"[RSS] {feed_name}: Parse error: {e}")

        return alerts

    def _parse_date(self, entry: dict) -> Optional[datetime]:
        """
        Parse publication date from RSS entry.

        Args:
            entry: feedparser entry dict

        Returns:
            datetime object or None
        """
        # Try different date fields
        date_str = entry.get("published", entry.get("updated", entry.get("created")))

        if not date_str:
            return None

        try:
            # Try standard RFC 822 format
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            pass

        try:
            # feedparser sometimes provides parsed time
            if "published_parsed" in entry and entry["published_parsed"]:
                from time import mktime
                return datetime.fromtimestamp(mktime(entry["published_parsed"]), tz=timezone.utc)
        except (ValueError, TypeError, OverflowError):
            pass

        return None

    def _infer_severity(self, text: str) -> Optional[str]:
        """
        Infer severity from text content.

        Args:
            text: Combined title and description

        Returns:
            Inferred severity level
        """
        text_lower = text.lower()

        # Severe indicators
        severe_keywords = ['death', 'dead', 'casualt', 'rescue', 'evacuate',
                         'emergency', 'disaster', 'crisis', 'severe', 'extreme']
        if any(kw in text_lower for kw in severe_keywords):
            return 'severe'

        # High indicators
        high_keywords = ['warning', 'alert', 'danger', 'rising', 'overflow',
                        'submerge', 'stranded', 'trapped']
        if any(kw in text_lower for kw in high_keywords):
            return 'high'

        # Moderate indicators
        moderate_keywords = ['waterlog', 'flood', 'inundat', 'heavy rain',
                           'traffic jam', 'disruption']
        if any(kw in text_lower for kw in moderate_keywords):
            return 'moderate'

        # Default to low for general flood news
        return 'low'


async def test_rss_fetcher():
    """Test function to verify RSS fetcher works."""
    fetcher = RSSFetcher()

    print(f"RSS Fetcher enabled: {fetcher.is_enabled()}")
    print(f"Source name: {fetcher.get_source_name()}")

    print("\nFetching Delhi RSS feeds...")
    alerts = await fetcher.fetch("delhi")

    print(f"\nFound {len(alerts)} flood-related articles:")
    for alert in alerts[:5]:  # Show first 5
        print(f"\n  [{alert.source_name}] {alert.severity or 'N/A'}")
        print(f"  Title: {alert.title[:80]}...")
        print(f"  URL: {alert.url}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rss_fetcher())
