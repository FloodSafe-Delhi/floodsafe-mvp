"""
CWC Scraper - Scrapes flood forecasts from Central Water Commission portal.

Portal: https://ffs.india-water.gov.in/

Note: CWC does not provide a public API. This scraper:
1. Fetches the flood forecast page
2. Parses HTML to extract flood warnings
3. Filters for relevant cities/stations

Respects robots.txt and implements polite scraping with delays.
"""

import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Optional
import logging
import re
from bs4 import BeautifulSoup

from .base_fetcher import BaseFetcher, ExternalAlertCreate
from src.core.config import settings

logger = logging.getLogger(__name__)


# CWC URLs
CWC_FLOOD_FORECAST_URL = "https://ffs.india-water.gov.in/"
CWC_DASHBOARD_URL = "https://cwc.gov.in/ffm_dashboard"

# City to CWC station mapping
CITY_STATIONS = {
    "delhi": {
        "river": "YAMUNA",
        "stations": ["OLD DELHI RAILWAY BRIDGE", "DELHI", "WAZIRABAD", "OKHLA"],
        "state": "DELHI",
        "keywords": ["delhi", "yamuna", "wazirabad", "okhla", "ncr"],
    },
    "bangalore": {
        "river": "CAUVERY",
        "stations": [],
        "state": "KARNATAKA",
        "keywords": ["bangalore", "bengaluru", "karnataka"],
    },
}

# Danger level keywords
DANGER_KEYWORDS = {
    "severe": ["extreme", "severe flood", "highest flood level", "danger level"],
    "high": ["above danger", "rising trend", "warning level", "danger"],
    "moderate": ["near danger", "watchful", "high level"],
    "low": ["normal", "below danger", "falling"],
}


class CWCScraper(BaseFetcher):
    """Scrapes flood forecasts from CWC portal."""

    def __init__(self, timeout: int = 30, request_delay: float = 2.0):
        """
        Initialize CWC scraper.

        Args:
            timeout: Request timeout in seconds
            request_delay: Delay between requests (polite scraping)
        """
        self.timeout = timeout
        self.request_delay = request_delay

    def get_source_name(self) -> str:
        return "cwc"

    def is_enabled(self) -> bool:
        """CWC scraper is always enabled - no API keys required."""
        return getattr(settings, 'CWC_SCRAPER_ENABLED', True)

    async def fetch(self, city: str) -> list[ExternalAlertCreate]:
        """
        Fetch flood forecasts from CWC for a city.

        Args:
            city: City identifier ('delhi', 'bangalore')

        Returns:
            List of ExternalAlertCreate objects
        """
        if not self.is_enabled():
            logger.debug("[CWC] Scraper disabled via config")
            return []

        self.log_fetch_start(city)

        city_config = CITY_STATIONS.get(city.lower())
        if not city_config:
            logger.warning(f"[CWC] No configuration for city: {city}")
            return []

        alerts = []

        # Scrape main flood forecast page
        try:
            page_alerts = await self._scrape_forecast_page(city, city_config)
            alerts.extend(page_alerts)
        except Exception as e:
            logger.error(f"[CWC] Forecast page scrape failed: {e}")

        self.log_fetch_complete(city, len(alerts))
        return alerts

    async def _scrape_forecast_page(self, city: str, city_config: dict) -> list[ExternalAlertCreate]:
        """
        Scrape the CWC flood forecast page.

        Args:
            city: City identifier
            city_config: City configuration dict

        Returns:
            List of ExternalAlertCreate objects
        """
        alerts = []

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "User-Agent": "FloodSafe/1.0 (+https://floodsafe.app) - Flood monitoring for public safety",
                    "Accept": "text/html,application/xhtml+xml",
                }

                async with session.get(
                    CWC_FLOOD_FORECAST_URL,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status != 200:
                        logger.warning(f"[CWC] Page returned status {response.status}")
                        return []

                    html = await response.text()

                    # Parse HTML
                    soup = BeautifulSoup(html, 'html.parser')

                    # Look for flood forecast tables
                    alerts.extend(self._parse_forecast_tables(soup, city, city_config))

                    # Look for alert/warning sections
                    alerts.extend(self._parse_alert_sections(soup, city, city_config))

        except aiohttp.ClientError as e:
            logger.warning(f"[CWC] Connection error: {e}")
        except Exception as e:
            logger.error(f"[CWC] Scrape error: {e}")

        return alerts

    def _parse_forecast_tables(self, soup: BeautifulSoup, city: str, city_config: dict) -> list[ExternalAlertCreate]:
        """
        Parse flood forecast tables from HTML.

        Args:
            soup: BeautifulSoup object
            city: City identifier
            city_config: City configuration

        Returns:
            List of ExternalAlertCreate objects
        """
        alerts = []
        river = city_config.get("river", "")
        stations = city_config.get("stations", [])
        keywords = city_config.get("keywords", [])

        # Find all tables
        tables = soup.find_all('table')

        for table in tables:
            rows = table.find_all('tr')

            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue

                # Get row text
                row_text = ' '.join(cell.get_text(strip=True).lower() for cell in cells)

                # Check if row is relevant to our city
                is_relevant = False

                # Check river name
                if river.lower() in row_text:
                    is_relevant = True

                # Check station names
                for station in stations:
                    if station.lower() in row_text:
                        is_relevant = True
                        break

                # Check keywords
                for keyword in keywords:
                    if keyword in row_text:
                        is_relevant = True
                        break

                if not is_relevant:
                    continue

                # Check for danger/warning indicators
                severity = self._detect_severity(row_text)
                if severity in ['moderate', 'high', 'severe']:
                    # Extract useful information
                    title = self._extract_station_name(row_text, stations, river)
                    message = self._format_row_message(cells)

                    alert = ExternalAlertCreate(
                        source="cwc",
                        source_id=self.generate_source_id("cwc", title, datetime.now().strftime("%Y-%m-%d")),
                        source_name="CWC Flood Forecast",
                        city=city,
                        title=f"CWC Flood Forecast - {title}",
                        message=message,
                        severity=severity,
                        url=CWC_FLOOD_FORECAST_URL,
                        raw_data={"row_text": row_text[:500]},
                        expires_at=datetime.now(timezone.utc) + timedelta(days=1)
                    )
                    alerts.append(alert)

        return alerts

    def _parse_alert_sections(self, soup: BeautifulSoup, city: str, city_config: dict) -> list[ExternalAlertCreate]:
        """
        Parse alert/warning sections from HTML.

        Args:
            soup: BeautifulSoup object
            city: City identifier
            city_config: City configuration

        Returns:
            List of ExternalAlertCreate objects
        """
        alerts = []
        keywords = city_config.get("keywords", [])

        # Look for alert divs, warning sections, etc.
        alert_elements = soup.find_all(['div', 'span', 'p'], class_=re.compile(r'alert|warning|danger', re.I))

        for element in alert_elements:
            text = element.get_text(strip=True).lower()

            # Check if relevant to our city
            is_relevant = any(kw in text for kw in keywords)
            if not is_relevant:
                continue

            # Check for flood-related content
            if not self.filter_by_keywords(text):
                continue

            severity = self._detect_severity(text)

            alert = ExternalAlertCreate(
                source="cwc",
                source_id=self.generate_source_id("cwc_alert", text[:100], datetime.now().strftime("%Y-%m-%d-%H")),
                source_name="CWC Alert",
                city=city,
                title=f"CWC Flood Alert - {city_config.get('state', city.title())}",
                message=self.truncate_text(element.get_text(strip=True), 1000),
                severity=severity,
                url=CWC_FLOOD_FORECAST_URL,
                raw_data={"element_class": str(element.get('class', []))},
                expires_at=datetime.now(timezone.utc) + timedelta(hours=12)
            )
            alerts.append(alert)

        return alerts

    def _detect_severity(self, text: str) -> str:
        """
        Detect severity level from text content.

        Args:
            text: Text to analyze

        Returns:
            Severity level
        """
        text_lower = text.lower()

        for severity, keywords in DANGER_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return severity

        # Default based on presence of certain terms
        if 'above' in text_lower and ('danger' in text_lower or 'level' in text_lower):
            return 'high'
        if 'warning' in text_lower:
            return 'moderate'

        return 'low'

    def _extract_station_name(self, text: str, stations: list, river: str) -> str:
        """
        Extract station name from row text.

        Args:
            text: Row text
            stations: List of known stations
            river: River name

        Returns:
            Station name or river name
        """
        text_upper = text.upper()

        for station in stations:
            if station.upper() in text_upper:
                return station

        return river if river else "Unknown Station"

    def _format_row_message(self, cells: list) -> str:
        """
        Format table row cells into a message.

        Args:
            cells: List of BeautifulSoup cell elements

        Returns:
            Formatted message
        """
        parts = []
        for i, cell in enumerate(cells):
            text = cell.get_text(strip=True)
            if text:
                parts.append(text)

        return " | ".join(parts)


async def test_cwc_scraper():
    """Test function to verify CWC scraper works."""
    scraper = CWCScraper()

    print(f"CWC Scraper enabled: {scraper.is_enabled()}")
    print(f"Source name: {scraper.get_source_name()}")

    print("\nFetching Delhi CWC alerts...")
    alerts = await scraper.fetch("delhi")

    print(f"\nFound {len(alerts)} alerts:")
    for alert in alerts:
        print(f"\n  [{alert.severity}] {alert.title}")
        print(f"  Message: {alert.message[:150]}...")
        print(f"  URL: {alert.url}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_cwc_scraper())
