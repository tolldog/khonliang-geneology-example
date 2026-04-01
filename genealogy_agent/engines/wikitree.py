"""
WikiTree API engine — free read-only access to WikiTree profiles.

Endpoint: https://api.wikitree.com/api.php
No auth required, just an appId for rate limiting.
Returns JSON.

Docs: https://github.com/wikitree/wikitree-api
"""

import logging
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

WIKITREE_API = "https://api.wikitree.com/api.php"


class WikiTreeClient:
    """
    Client for the WikiTree API.

    Example:
        client = WikiTreeClient()
        person = client.get_person("Tolle-1")
        ancestors = client.get_ancestors("Tolle-1", depth=5)
    """

    def __init__(self, app_id: str = "khonliang-genealogy"):
        self.app_id = app_id
        self.session = requests.Session()

    def _request(self, action: str, **params) -> Optional[Dict]:
        """Make an API request."""
        params["action"] = action
        params["appId"] = self.app_id
        params["format"] = "json"

        try:
            resp = self.session.post(
                WIKITREE_API, data=params, timeout=15
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"WikiTree API error: {e}")
            return None

    def get_person(
        self, key: str, fields: str = "*"
    ) -> Optional[Dict]:
        """
        Get a person's profile.

        Args:
            key: WikiTree ID (e.g. "Tolle-1") or person ID number
            fields: Comma-separated fields or "*" for all
        """
        result = self._request("getPerson", key=key, fields=fields)
        if result and isinstance(result, list) and len(result) > 0:
            return result[0].get("person")
        return None

    def get_profile(self, key: str) -> Optional[Dict]:
        """Get a profile (alias for getPerson with bioFormat)."""
        result = self._request(
            "getProfile", key=key, fields="*", bioFormat="text"
        )
        if result and isinstance(result, list) and len(result) > 0:
            return result[0].get("profile")
        return None

    def get_ancestors(
        self, key: str, depth: int = 5
    ) -> Optional[List[Dict]]:
        """Get ancestors of a person."""
        result = self._request(
            "getAncestors", key=key, depth=depth, fields="*"
        )
        if result and isinstance(result, list) and len(result) > 0:
            return result[0].get("ancestors", [])
        return None

    def get_descendants(
        self, key: str, depth: int = 3
    ) -> Optional[List[Dict]]:
        """Get descendants of a person."""
        result = self._request(
            "getDescendants", key=key, depth=depth, fields="*"
        )
        if result and isinstance(result, list) and len(result) > 0:
            return result[0].get("descendants", [])
        return None

    def get_relatives(
        self, keys: List[str], get_parents: bool = True,
        get_children: bool = True, get_spouses: bool = True,
    ) -> Optional[List[Dict]]:
        """Get relatives for one or more persons."""
        params = {
            "keys": ",".join(keys),
            "fields": "*",
            "getParents": "1" if get_parents else "0",
            "getChildren": "1" if get_children else "0",
            "getSpouses": "1" if get_spouses else "0",
        }
        result = self._request("getRelatives", **params)
        if result and isinstance(result, list):
            items = []
            for entry in result:
                items.extend(entry.get("items", []))
            return items
        return None

    def search_person(
        self, first_name: str = "", last_name: str = "",
        birth_date: str = "", death_date: str = "",
        birth_location: str = "",
    ) -> Optional[List[Dict]]:
        """Search for persons by name/dates/location."""
        params = {}
        if first_name:
            params["FirstName"] = first_name
        if last_name:
            params["LastName"] = last_name
        if birth_date:
            params["BirthDate"] = birth_date
        if death_date:
            params["DeathDate"] = death_date
        if birth_location:
            params["BirthLocation"] = birth_location

        result = self._request("searchPerson", **params)
        if result and isinstance(result, list):
            return result
        return None

    def format_person(self, person: Dict) -> str:
        """Format a WikiTree person dict as readable text."""
        name = (
            f"{person.get('FirstName', '')} "
            f"{person.get('LastNameAtBirth', person.get('LastNameCurrent', ''))}"
        ).strip()

        parts = [name]
        bd = person.get("BirthDate", "")
        dd = person.get("DeathDate", "")
        if bd or dd:
            parts.append(f"({bd or '?'} - {dd or '?'})")

        bl = person.get("BirthLocation", "")
        if bl:
            parts.append(f"b. {bl}")

        bio = person.get("bio", "")
        if bio:
            # First 200 chars of bio
            parts.append(f"\n  {bio[:200]}")

        return " ".join(parts)
