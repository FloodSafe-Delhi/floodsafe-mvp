"""
Twitter Fetcher - Fetches flood-related tweets from Twitter/X API v2.

API Documentation: https://developer.twitter.com/en/docs/twitter-api

Requirements:
- TWITTER_BEARER_TOKEN environment variable
- Free tier: 1,500 tweets/month (track usage to avoid exceeding)

Monitored accounts:
- @IMDDelhi - IMD Delhi official
- @DelhiTraffic - Delhi Traffic Police
- @ndmaindia - National Disaster Management Authority
- @DDNewslive - DD News
"""

import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Optional
import logging
import os

from .base_fetcher import BaseFetcher, ExternalAlertCreate
from src.core.config import settings

logger = logging.getLogger(__name__)


# Twitter API v2 endpoints
TWITTER_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

# Search queries per city
CITY_QUERIES = {
    "delhi": {
        # Main flood search query
        "flood_query": "(delhi OR ncr OR \"new delhi\") (flood OR floods OR waterlog OR waterlogged OR inundation) -is:retweet lang:en",
        # Official accounts query
        "official_query": "(from:IMDDelhi OR from:DelhiTraffic OR from:ndmaindia) (flood OR rain OR warning OR alert) -is:retweet",
        # Yamuna specific
        "yamuna_query": "yamuna (level OR rising OR flood OR danger) -is:retweet lang:en",
    },
    "bangalore": {
        "flood_query": "(bangalore OR bengaluru) (flood OR floods OR waterlog OR waterlogged) -is:retweet lang:en",
        "official_query": "(from:ABORIG_TRAFFIC OR from:maborig) (flood OR rain OR waterlog) -is:retweet",
    },
}

# Rate limit tracking (free tier: 1,500 tweets/month)
MAX_TWEETS_PER_MONTH = 1500
TWEETS_PER_REQUEST = 10  # Conservative to stay within limits


class TwitterFetcher(BaseFetcher):
    """Fetches flood-related tweets from Twitter API v2."""

    def __init__(self, timeout: int = 30):
        """
        Initialize Twitter fetcher.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._bearer_token: Optional[str] = None
        self._monthly_usage = 0
        self._usage_reset_date: Optional[datetime] = None

    def get_source_name(self) -> str:
        return "twitter"

    def is_enabled(self) -> bool:
        """Check if Twitter fetcher is enabled (has bearer token)."""
        token = self._get_bearer_token()
        if not token:
            logger.debug("[Twitter] No bearer token configured")
            return False
        return True

    def _get_bearer_token(self) -> Optional[str]:
        """Get Twitter bearer token from config or environment."""
        if self._bearer_token:
            return self._bearer_token

        # Try config first, then environment
        self._bearer_token = getattr(settings, 'TWITTER_BEARER_TOKEN', None)
        if not self._bearer_token:
            self._bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')

        return self._bearer_token

    def _check_rate_limit(self) -> bool:
        """
        Check if we're within rate limits.

        Returns:
            True if we can make more requests this month
        """
        now = datetime.now(timezone.utc)

        # Reset monthly usage at start of month
        if self._usage_reset_date is None or now.month != self._usage_reset_date.month:
            self._monthly_usage = 0
            self._usage_reset_date = now

        remaining = MAX_TWEETS_PER_MONTH - self._monthly_usage
        if remaining <= 0:
            logger.warning(f"[Twitter] Monthly rate limit reached ({MAX_TWEETS_PER_MONTH} tweets)")
            return False

        return True

    def _track_usage(self, tweet_count: int):
        """Track API usage for rate limiting."""
        self._monthly_usage += tweet_count
        logger.debug(f"[Twitter] Monthly usage: {self._monthly_usage}/{MAX_TWEETS_PER_MONTH}")

    async def fetch(self, city: str) -> list[ExternalAlertCreate]:
        """
        Fetch flood-related tweets for a city.

        Args:
            city: City identifier ('delhi', 'bangalore')

        Returns:
            List of ExternalAlertCreate objects
        """
        if not self.is_enabled():
            logger.debug("[Twitter] Fetcher disabled - no bearer token")
            return []

        if not self._check_rate_limit():
            return []

        self.log_fetch_start(city)

        city_queries = CITY_QUERIES.get(city.lower(), {})
        if not city_queries:
            logger.warning(f"[Twitter] No queries configured for city: {city}")
            return []

        alerts = []
        bearer_token = self._get_bearer_token()

        # Execute each query
        for query_name, query in city_queries.items():
            try:
                query_alerts = await self._execute_search(query, city, bearer_token)
                alerts.extend(query_alerts)

                # Small delay between queries to be nice to API
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"[Twitter] Query '{query_name}' failed: {e}")

        # Deduplicate by tweet ID
        seen_ids = set()
        unique_alerts = []
        for alert in alerts:
            tweet_id = alert.raw_data.get("tweet_id") if alert.raw_data else None
            if tweet_id and tweet_id not in seen_ids:
                seen_ids.add(tweet_id)
                unique_alerts.append(alert)

        self.log_fetch_complete(city, len(unique_alerts))
        return unique_alerts

    async def _execute_search(self, query: str, city: str, bearer_token: str) -> list[ExternalAlertCreate]:
        """
        Execute a Twitter search query.

        Args:
            query: Twitter search query
            city: City identifier
            bearer_token: Twitter API bearer token

        Returns:
            List of ExternalAlertCreate objects
        """
        alerts = []

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {bearer_token}",
                    "User-Agent": "FloodSafe/1.0",
                }

                params = {
                    "query": query,
                    "max_results": TWEETS_PER_REQUEST,
                    "tweet.fields": "created_at,author_id,geo,public_metrics,source",
                    "expansions": "author_id,geo.place_id",
                    "user.fields": "username,name,verified",
                    "place.fields": "full_name,geo",
                }

                async with session.get(
                    TWITTER_SEARCH_URL,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 401:
                        logger.error("[Twitter] Invalid bearer token")
                        return []

                    if response.status == 429:
                        logger.warning("[Twitter] Rate limited by API")
                        return []

                    if response.status != 200:
                        error_text = await response.text()
                        logger.warning(f"[Twitter] API error {response.status}: {error_text[:200]}")
                        return []

                    data = await response.json()

                    # Track usage
                    result_count = data.get("meta", {}).get("result_count", 0)
                    self._track_usage(result_count)

                    # Parse tweets
                    tweets = data.get("data", [])
                    users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
                    places = {p["id"]: p for p in data.get("includes", {}).get("places", [])}

                    for tweet in tweets:
                        alert = self._parse_tweet(tweet, users, places, city)
                        if alert:
                            alerts.append(alert)

        except aiohttp.ClientError as e:
            logger.warning(f"[Twitter] Connection error: {e}")
        except Exception as e:
            logger.error(f"[Twitter] Search error: {e}")

        return alerts

    def _parse_tweet(self, tweet: dict, users: dict, places: dict, city: str) -> Optional[ExternalAlertCreate]:
        """
        Parse a tweet into an ExternalAlertCreate.

        Args:
            tweet: Tweet data from API
            users: User data lookup
            places: Place data lookup
            city: City identifier

        Returns:
            ExternalAlertCreate or None
        """
        try:
            tweet_id = tweet.get("id")
            text = tweet.get("text", "")

            # Skip if too short or doesn't contain flood keywords
            if len(text) < 20:
                return None

            if not self.filter_by_keywords(text):
                return None

            # Get author info
            author_id = tweet.get("author_id")
            author = users.get(author_id, {})
            author_name = author.get("name", "Unknown")
            author_username = author.get("username", "unknown")
            is_verified = author.get("verified", False)

            # Parse timestamp
            created_at_str = tweet.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.now(timezone.utc)

            # Skip tweets older than 7 days
            if (datetime.now(timezone.utc) - created_at).days > 7:
                return None

            # Get location if available
            latitude = None
            longitude = None
            geo = tweet.get("geo", {})
            if geo:
                place_id = geo.get("place_id")
                if place_id and place_id in places:
                    place = places[place_id]
                    bbox = place.get("geo", {}).get("bbox", [])
                    if len(bbox) >= 4:
                        # Use center of bounding box
                        longitude = (bbox[0] + bbox[2]) / 2
                        latitude = (bbox[1] + bbox[3]) / 2

            # Determine severity from content
            severity = self._infer_severity(text)

            # Build title
            if is_verified:
                title = f"@{author_username} (Verified)"
            else:
                title = f"@{author_username}"

            # Generate tweet URL
            tweet_url = f"https://twitter.com/{author_username}/status/{tweet_id}"

            return ExternalAlertCreate(
                source="twitter",
                source_id=f"twitter_{tweet_id}",
                source_name=f"Twitter - {author_name}",
                city=city,
                title=title,
                message=text,
                severity=severity,
                url=tweet_url,
                latitude=latitude,
                longitude=longitude,
                raw_data={
                    "tweet_id": tweet_id,
                    "author_id": author_id,
                    "author_username": author_username,
                    "author_name": author_name,
                    "verified": is_verified,
                    "metrics": tweet.get("public_metrics", {}),
                    "source": tweet.get("source", ""),
                    "created_at": created_at_str,
                }
            )

        except Exception as e:
            logger.error(f"[Twitter] Error parsing tweet: {e}")
            return None

    def _infer_severity(self, text: str) -> str:
        """
        Infer severity from tweet content.

        Official accounts and certain keywords get higher severity.
        """
        text_lower = text.lower()

        # Severe indicators
        severe_keywords = ['death', 'rescue', 'emergency', 'evacuate', 'severe', 'danger']
        if any(kw in text_lower for kw in severe_keywords):
            return 'severe'

        # High indicators
        high_keywords = ['warning', 'alert', 'rising', 'overflow', 'stranded']
        if any(kw in text_lower for kw in high_keywords):
            return 'high'

        # Moderate
        moderate_keywords = ['waterlog', 'flood', 'traffic', 'disruption']
        if any(kw in text_lower for kw in moderate_keywords):
            return 'moderate'

        return 'low'


async def test_twitter_fetcher():
    """Test function to verify Twitter fetcher works."""
    fetcher = TwitterFetcher()

    print(f"Twitter Fetcher enabled: {fetcher.is_enabled()}")
    print(f"Source name: {fetcher.get_source_name()}")

    if not fetcher.is_enabled():
        print("\n[SKIP] Twitter fetcher disabled - no TWITTER_BEARER_TOKEN set")
        print("Set TWITTER_BEARER_TOKEN environment variable to test")
        return

    print("\nFetching Delhi Twitter alerts...")
    alerts = await fetcher.fetch("delhi")

    print(f"\nFound {len(alerts)} tweets:")
    for alert in alerts[:5]:
        print(f"\n  [{alert.severity}] {alert.title}")
        print(f"  Message: {alert.message[:100]}...")
        print(f"  URL: {alert.url}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_twitter_fetcher())
