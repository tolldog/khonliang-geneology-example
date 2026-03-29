"""
Geni API engine — OAuth2 access to the Geni world family tree.

Docs: https://www.geni.com/platform/developer/help
API Version: 1
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

GENI_API = "https://www.geni.com/api"
GENI_OAUTH = "https://www.geni.com/platform/oauth"


class GeniClient:
    """
    Client for the Geni.com API.

    Uses OAuth2 for authentication. For server-to-server access,
    use client credentials flow.

    Example:
        client = GeniClient(
            app_id="1999",
            api_key="...",
            api_secret="...",
        )
        client.authenticate()
        profile = client.get_profile("6000000004578010257")
    """

    def __init__(
        self,
        app_id: str = "",
        api_key: str = "",
        api_secret: str = "",
    ):
        self.app_id = app_id or os.environ.get("GENI_APP_ID", "")
        self.api_key = api_key or os.environ.get("GENI_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("GENI_API_SECRET", "")
        self.access_token = ""
        self.session = requests.Session()

    def authenticate(self) -> bool:
        """
        Authenticate with Geni using client credentials.

        Returns True if successful.
        """
        try:
            resp = self.session.post(
                f"{GENI_OAUTH}/request_token",
                data={
                    "client_id": self.api_key,
                    "client_secret": self.api_secret,
                    "grant_type": "client_credentials",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self.access_token = data.get("access_token", "")
            if self.access_token:
                logger.info("Geni authentication successful")
                return True
            logger.warning(f"Geni auth failed: {data}")
            return False
        except Exception as e:
            logger.warning(f"Geni authentication error: {e}")
            return False

    def _request(
        self, endpoint: str, method: str = "GET", **params
    ) -> Optional[Dict]:
        """Make an authenticated API request."""
        if not self.access_token:
            logger.warning("Not authenticated with Geni")
            return None

        params["access_token"] = self.access_token
        url = f"{GENI_API}/{endpoint}"

        try:
            if method == "GET":
                resp = self.session.get(url, params=params, timeout=15)
            else:
                resp = self.session.post(url, data=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Geni API error ({endpoint}): {e}")
            return None

    def get_profile(self, profile_id: str) -> Optional[Dict]:
        """Get a profile by Geni ID."""
        return self._request(f"profile-{profile_id}")

    def get_profile_by_url(self, url: str) -> Optional[Dict]:
        """Get a profile by Geni URL."""
        # Extract profile ID from URL
        if "/people/" in url:
            parts = url.split("/")
            for i, p in enumerate(parts):
                if p == "people":
                    profile_id = parts[i + 1] if i + 1 < len(parts) else ""
                    if profile_id:
                        return self.get_profile(profile_id)
        return None

    def get_parents(self, profile_id: str) -> Optional[List[Dict]]:
        """Get parents of a profile."""
        result = self._request(f"profile-{profile_id}/parents")
        if result and "results" in result:
            return result["results"]
        return None

    def get_children(self, profile_id: str) -> Optional[List[Dict]]:
        """Get children of a profile."""
        result = self._request(f"profile-{profile_id}/children")
        if result and "results" in result:
            return result["results"]
        return None

    def get_spouses(self, profile_id: str) -> Optional[List[Dict]]:
        """Get spouses/partners of a profile."""
        result = self._request(f"profile-{profile_id}/partners")
        if result and "results" in result:
            return result["results"]
        return None

    def search(
        self,
        names: str = "",
        first_name: str = "",
        last_name: str = "",
    ) -> Optional[List[Dict]]:
        """Search for profiles."""
        params = {}
        if names:
            params["names"] = names
        if first_name:
            params["first_name"] = first_name
        if last_name:
            params["last_name"] = last_name

        result = self._request("profile/search", **params)
        if result and "results" in result:
            return result["results"]
        return None

    def format_profile(self, profile: Dict) -> str:
        """Format a Geni profile as readable text."""
        name = profile.get("name", "Unknown")
        parts = [name]

        birth = profile.get("birth", {})
        death = profile.get("death", {})

        bd = birth.get("date", {}).get("formatted_date", "")
        dd = death.get("date", {}).get("formatted_date", "")
        if bd or dd:
            parts.append(f"({bd or '?'} - {dd or '?'})")

        bl = birth.get("location", {}).get("city", "")
        if bl:
            parts.append(f"b. {bl}")

        return " ".join(parts)
