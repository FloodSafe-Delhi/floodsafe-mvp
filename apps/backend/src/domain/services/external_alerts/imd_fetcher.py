"""
IMD Fetcher - Fetches weather warnings from India Meteorological Department.

API Documentation: https://mausam.imd.gov.in/imd_latest/contents/api.pdf

Note: IMD API requires IP whitelist. This fetcher will:
1. Test API accessibility on first call
2. Disable itself if API is not accessible
3. Fall back to scraping Flash Flood Bulletin if API fails

Warning Codes:
- 1: Green (Low)
- 2: Yellow (Moderate)
- 3: Orange (High)
- 4: Red (Severe)
"""

import asyncio
import aiohttp
from datetime import datetime, timezone, timedelta
from typing import Optional
import logging
import re

from .base_fetcher import BaseFetcher, ExternalAlertCreate
from src.core.config import settings

logger = logging.getLogger(__name__)


# IMD API endpoints
IMD_BASIN_QPF_API = "https://mausam.imd.gov.in/api/basin_qpf_api.php"
IMD_DISTRICT_WARNING_URL = "https://mausam.imd.gov.in/responsive/districtWiseWarning.php"
IMD_FLASH_FLOOD_URL = "https://mausam.imd.gov.in/responsive/flashFloodBulletin.php"

# City to basin/district mapping
CITY_BASIN_MAP = {
    "delhi": {
        "basin_ids": ["100", "101"],  # Upper Yamuna, Lower Yamuna
        "state": "DELHI",
        "districts": ["NEW DELHI", "NORTH DELHI", "SOUTH DELHI", "EAST DELHI", "WEST DELHI"],
    },
    "bangalore": {
        "basin_ids": ["200"],
        "state": "KARNATAKA",
        "districts": ["BENGALURU URBAN", "BENGALURU RURAL"],
    },
}

# Warning code to severity mapping
WARNING_SEVERITY_MAP = {
    "1": "low",      # Green
    "2": "moderate", # Yellow
    "3": "high",     # Orange
    "4": "severe",   # Red
}


class IMDFetcher(BaseFetcher):
    """Fetches weather warnings from IMD API."""

    def __init__(self, timeout: int = 30):
        """
        Initialize IMD fetcher.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._api_accessible: Optional[bool] = None  # None = not tested yet

    def get_source_name(self) -> str:
        return "imd"

    def is_enabled(self) -> bool:
        """Check if IMD fetcher is enabled via config."""
        return getattr(settings, 'IMD_API_ENABLED', True)

    async def fetch(self, city: str) -> list[ExternalAlertCreate]:
        """
        Fetch weather warnings from IMD for a city.

        Args:
            city: City identifier ('delhi', 'bangalore')

        Returns:
            List of ExternalAlertCreate objects
        """
        if not self.is_enabled():
            logger.debug("[IMD] Fetcher disabled via config")
            return []

        self.log_fetch_start(city)

        city_config = CITY_BASIN_MAP.get(city.lower())
        if not city_config:
            logger.warning(f"[IMD] No configuration for city: {city}")
            return []

        alerts = []

        # Try Basin QPF API first
        try:
            basin_alerts = await self._fetch_basin_qpf(city, city_config)
            alerts.extend(basin_alerts)
        except Exception as e:
            logger.warning(f"[IMD] Basin QPF API failed: {e}")

        # Try scraping Flash Flood Bulletin as fallback/supplement
        try:
            flash_alerts = await self._scrape_flash_flood_bulletin(city, city_config)
            alerts.extend(flash_alerts)
        except Exception as e:
            logger.warning(f"[IMD] Flash Flood Bulletin scrape failed: {e}")

        self.log_fetch_complete(city, len(alerts))
        return alerts

    async def _fetch_basin_qpf(self, city: str, city_config: dict) -> list[ExternalAlertCreate]:
        """
        Fetch from IMD Basin QPF API.

        Note: This API may require IP whitelist.
        """
        alerts = []

        try:
            async with aiohttp.ClientSession() as session:
                # Try fetching all basins
                async with session.get(
                    IMD_BASIN_QPF_API,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers={"User-Agent": "FloodSafe/1.0"}
                ) as response:
                    if response.status == 403:
                        logger.info("[IMD] Basin QPF API returned 403 - IP not whitelisted")
                        self._api_accessible = False
                        return []

                    if response.status != 200:
                        logger.warning(f"[IMD] Basin QPF API returned {response.status}")
                        return []

                    self._api_accessible = True
                    data = await response.json()

                    # Parse API response
                    # Expected format: list of basins with warning codes
                    if isinstance(data, list):
                        for basin in data:
                            basin_id = str(basin.get("id", ""))
                            if basin_id in city_config.get("basin_ids", []):
                                alert = self._parse_basin_warning(basin, city)
                                if alert:
                                    alerts.append(alert)

        except aiohttp.ClientError as e:
            logger.warning(f"[IMD] Basin QPF API connection error: {e}")
        except Exception as e:
            logger.error(f"[IMD] Basin QPF API error: {e}")

        return alerts

    def _parse_basin_warning(self, basin: dict, city: str) -> Optional[ExternalAlertCreate]:
        """Parse a basin warning from API response."""
        try:
            basin_name = basin.get("Name", basin.get("name", "Unknown Basin"))

            # Check warning codes for days 1-5
            has_warning = False
            max_severity = "low"
            warnings = []

            for day in range(1, 6):
                code_key = f"Day{day}" if f"Day{day}" in basin else f"day{day}"
                code = str(basin.get(code_key, "1"))

                if code in ["2", "3", "4"]:
                    has_warning = True
                    severity = WARNING_SEVERITY_MAP.get(code, "low")
                    warnings.append(f"Day {day}: {severity.title()}")

                    # Track max severity
                    if code == "4":
                        max_severity = "severe"
                    elif code == "3" and max_severity not in ["severe"]:
                        max_severity = "high"
                    elif code == "2" and max_severity not in ["severe", "high"]:
                        max_severity = "moderate"

            if not has_warning:
                return None

            # Create alert
            title = f"IMD Weather Warning - {basin_name}"
            message = f"Weather warnings issued for {basin_name} river basin.\n\n" + "\n".join(warnings)

            return ExternalAlertCreate(
                source="imd",
                source_id=self.generate_source_id("imd", basin.get("id"), datetime.now().strftime("%Y-%m-%d")),
                source_name="IMD Weather",
                city=city,
                title=title,
                message=message,
                severity=max_severity,
                url=IMD_DISTRICT_WARNING_URL,
                raw_data=basin,
                expires_at=datetime.now(timezone.utc) + timedelta(days=1)
            )

        except Exception as e:
            logger.error(f"[IMD] Error parsing basin warning: {e}")
            return None

    async def _scrape_flash_flood_bulletin(self, city: str, city_config: dict) -> list[ExternalAlertCreate]:
        """
        Scrape IMD Flash Flood Bulletin page.

        Fallback when API is not accessible.
        """
        alerts = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    IMD_FLASH_FLOOD_URL,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers={"User-Agent": "FloodSafe/1.0"}
                ) as response:
                    if response.status != 200:
                        return []

                    html = await response.text()

                    # Parse HTML for flood alerts
                    # Look for state/district mentions with warnings
                    state = city_config.get("state", "")
                    districts = city_config.get("districts", [])

                    # Simple regex-based extraction
                    # Look for warning indicators near city/state names
                    pattern = rf"({state}|{'|'.join(districts)}).*?(warning|alert|heavy rain|flood|rainfall)"

                    matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)

                    if matches:
                        # Create a general alert
                        alert = ExternalAlertCreate(
                            source="imd",
                            source_id=self.generate_source_id("imd_flash", city, datetime.now().strftime("%Y-%m-%d")),
                            source_name="IMD Flash Flood",
                            city=city,
                            title=f"IMD Flash Flood Bulletin - {state}",
                            message=f"Flash flood warnings have been issued for {state}. Check IMD website for details.",
                            severity="high",
                            url=IMD_FLASH_FLOOD_URL,
                            raw_data={"matches": len(matches)},
                            expires_at=datetime.now(timezone.utc) + timedelta(days=1)
                        )
                        alerts.append(alert)

        except Exception as e:
            logger.warning(f"[IMD] Flash Flood scrape error: {e}")

        return alerts

    async def test_api_access(self) -> bool:
        """
        Test if IMD API is accessible (IP whitelisted).

        Returns:
            True if API is accessible
        """
        if self._api_accessible is not None:
            return self._api_accessible

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    IMD_BASIN_QPF_API,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    self._api_accessible = response.status == 200
                    logger.info(f"[IMD] API access test: {'accessible' if self._api_accessible else 'blocked'}")
                    return self._api_accessible
        except Exception as e:
            logger.warning(f"[IMD] API access test failed: {e}")
            self._api_accessible = False
            return False


async def test_imd_fetcher():
    """Test function to verify IMD fetcher works."""
    fetcher = IMDFetcher()

    print(f"IMD Fetcher enabled: {fetcher.is_enabled()}")
    print(f"Source name: {fetcher.get_source_name()}")

    print("\nTesting API access...")
    accessible = await fetcher.test_api_access()
    print(f"API accessible: {accessible}")

    print("\nFetching Delhi IMD alerts...")
    alerts = await fetcher.fetch("delhi")

    print(f"\nFound {len(alerts)} alerts:")
    for alert in alerts:
        print(f"\n  [{alert.source_name}] {alert.severity}")
        print(f"  Title: {alert.title}")
        print(f"  Message: {alert.message[:100]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_imd_fetcher())
