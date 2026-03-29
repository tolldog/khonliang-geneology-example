"""
Web search module — finds external information about family members.

Uses DuckDuckGo search (no API key required) to find:
- Historical context for places and time periods
- Obituaries, census records, immigration records
- Wikipedia articles about relevant historical events
- Find A Grave, Ancestry hints, FamilySearch references

Results are formatted for LLM context injection or direct display.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS

    HAS_DDG = True
except ImportError:
    try:
        from duckduckgo_search import DDGS

        HAS_DDG = True
    except ImportError:
        HAS_DDG = False

HAS_GOOGLE = False
try:
    import requests
    from bs4 import BeautifulSoup

    HAS_GOOGLE = True
except ImportError:
    pass


@dataclass
class SearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str
    source: str = ""  # domain name
    relevance: float = 0.0  # 0.0-1.0, set by filter

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
        }


class GenealogySearcher:
    """
    Web search tailored for genealogy research.

    Builds targeted search queries from person data and returns
    formatted results suitable for LLM context or display.

    Example:
        searcher = GenealogySearcher()
        results = searcher.search_person("Timothy Tolle", birth_year=1683, place="Maryland")
        context = searcher.build_context(results)
    """

    # Sites particularly useful for genealogy
    GENEALOGY_SITES = [
        "findagrave.com",
        "familysearch.org",
        "ancestry.com",
        "wikitree.com",
        "geni.com",
    ]

    def __init__(self, max_results: int = 5):
        if not HAS_DDG:
            logger.warning(
                "duckduckgo-search not installed. "
                "Install with: pip install duckduckgo-search"
            )
        self.max_results = max_results

    def search(self, query: str, max_results: Optional[int] = None) -> List[SearchResult]:
        """Run a raw web search."""
        if not HAS_DDG:
            return []

        limit = max_results or self.max_results
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=limit))
            results = []
            for r in raw:
                url = r.get("href", "")
                domain = url.split("/")[2] if url.count("/") >= 2 else ""
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=url,
                    snippet=r.get("body", ""),
                    source=domain,
                ))
            return results
        except Exception as e:
            logger.warning(f"DDG search failed: {e}")
            return []

    def google_search(
        self, query: str, max_results: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search via Google HTML scraping. Slower but different results than DDG.
        Respects rate limits — use sparingly.
        """
        if not HAS_GOOGLE:
            return []

        limit = max_results or self.max_results
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            params = {"q": query, "num": limit}
            resp = requests.get(
                "https://www.google.com/search",
                params=params,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            results = []

            for div in soup.select("div.g"):
                link = div.select_one("a")
                title_el = div.select_one("h3")
                snippet_el = div.select_one("div.VwiC3b")

                if link and title_el:
                    url = link.get("href", "")
                    if not url.startswith("http"):
                        continue
                    domain = url.split("/")[2] if url.count("/") >= 2 else ""
                    results.append(SearchResult(
                        title=title_el.get_text(strip=True),
                        url=url,
                        snippet=(
                            snippet_el.get_text(strip=True) if snippet_el else ""
                        ),
                        source=domain,
                    ))

                if len(results) >= limit:
                    break

            return results
        except Exception as e:
            logger.warning(f"Google search failed: {e}")
            return []

    def bing_search(
        self, query: str, max_results: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search via Bing HTML scraping.
        """
        if not HAS_GOOGLE:
            return []

        limit = max_results or self.max_results
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            params = {"q": query, "count": limit}
            resp = requests.get(
                "https://www.bing.com/search",
                params=params,
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")
            results = []

            for li in soup.select("li.b_algo"):
                link = li.select_one("a")
                snippet_el = li.select_one("p") or li.select_one("div.b_caption p")

                if link:
                    url = link.get("href", "")
                    if not url.startswith("http"):
                        continue
                    domain = url.split("/")[2] if url.count("/") >= 2 else ""
                    results.append(SearchResult(
                        title=link.get_text(strip=True),
                        url=url,
                        snippet=(
                            snippet_el.get_text(strip=True) if snippet_el else ""
                        ),
                        source=domain,
                    ))

                if len(results) >= limit:
                    break

            return results
        except Exception as e:
            logger.warning(f"Bing search failed: {e}")
            return []

    def multi_search(
        self,
        query: str,
        max_per_engine: int = 5,
    ) -> List[SearchResult]:
        """
        Search across DDG, Google, and Bing in parallel. Deduplicate by URL.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        engines = {
            "ddg": lambda: self.search(query, max_results=max_per_engine),
            "google": lambda: self.google_search(query, max_results=max_per_engine),
            "bing": lambda: self.bing_search(query, max_results=max_per_engine),
        }

        seen_urls: set = set()
        all_results: List[SearchResult] = []

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="search") as pool:
            futures = {
                pool.submit(fn): name for name, fn in engines.items()
            }
            for future in as_completed(futures, timeout=15):
                engine_name = futures[future]
                try:
                    for r in future.result():
                        if r.url not in seen_urls:
                            seen_urls.add(r.url)
                            all_results.append(r)
                except Exception as e:
                    logger.debug(f"{engine_name} search failed: {e}")

        return all_results

    def fetch_page(
        self,
        url: str,
        max_chars: int = 5000,
    ) -> Optional[str]:
        """
        Fetch a URL and extract readable text content.

        Returns cleaned text suitable for LLM context, or None on failure.
        """
        if not HAS_GOOGLE:
            logger.warning("requests/beautifulsoup not available for fetch")
            return None

        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove script, style, nav elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text = "\n".join(lines)

            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            return text
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def search_person(
        self,
        name: str,
        birth_year: Optional[int] = None,
        death_year: Optional[int] = None,
        place: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Search for a specific person with genealogy context.

        Builds a targeted query using name, dates, and places.
        """
        parts = [f'"{name}"', "genealogy"]
        if birth_year:
            parts.append(str(birth_year))
        if place:
            # Use just the first significant part of the place
            place_parts = [p.strip() for p in place.split(",") if p.strip()]
            if place_parts:
                parts.append(place_parts[0])
        if death_year:
            parts.append(str(death_year))

        query = " ".join(parts)
        return self.search(query, max_results)

    def search_genealogy_sites(
        self,
        name: str,
        place: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[SearchResult]:
        """Search specifically on genealogy websites."""
        site_filter = " OR ".join(f"site:{s}" for s in self.GENEALOGY_SITES)
        parts = [f'"{name}"']
        if place:
            place_parts = [p.strip() for p in place.split(",") if p.strip()]
            if place_parts:
                parts.append(place_parts[0])
        parts.append(f"({site_filter})")

        query = " ".join(parts)
        return self.search(query, max_results)

    def search_historical_context(
        self,
        place: str,
        year: int,
        topic: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for historical context about a place and time period."""
        parts = [f'"{place}"', str(year), "history"]
        if topic:
            parts.append(topic)
        query = " ".join(parts)
        return self.search(query)

    def search_migration(
        self,
        surname: str,
        from_place: Optional[str] = None,
        to_place: Optional[str] = None,
        era: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search for migration patterns for a surname."""
        parts = [f'"{surname}"', "family", "migration"]
        if from_place:
            parts.append(f'"{from_place}"')
        if to_place:
            parts.append(f'"{to_place}"')
        if era:
            parts.append(era)
        query = " ".join(parts)
        return self.search(query)

    def build_context(
        self,
        results: List[SearchResult],
        max_chars: int = 3000,
    ) -> str:
        """Format search results as context for LLM injection."""
        if not results:
            return "No web search results found."

        lines = [f"Web search results ({len(results)}):"]
        total = 0
        for r in results:
            entry = f"\n- [{r.source}] {r.title}\n  {r.snippet}"
            if total + len(entry) > max_chars:
                break
            lines.append(entry)
            total += len(entry)

        return "\n".join(lines)

    def filter_relevant(
        self,
        results: List[SearchResult],
        name: str,
        birth_year: Optional[int] = None,
        place: Optional[str] = None,
        family_names: Optional[List[str]] = None,
        min_relevance: float = 0.2,
    ) -> List[SearchResult]:
        """
        Score and filter results for relevance to a genealogy query.

        Scores based on:
        - Name appears in title/snippet
        - Surname appears in title/snippet
        - Year range matches (within 50 years)
        - Place appears in snippet
        - Family names appear in snippet
        - Source is a genealogy site (bonus)
        - Penalty for clearly unrelated content (serial killers, etc.)

        Returns only results above min_relevance, sorted by score.
        """
        # Extract name parts
        name_parts = name.lower().split()
        surname = name_parts[-1] if name_parts else ""
        given = name_parts[0] if name_parts else ""

        # Place keywords
        place_words = set()
        if place:
            for part in place.lower().replace(",", " ").split():
                part = part.strip()
                if len(part) > 2 and part not in ("usa", "united", "states"):
                    place_words.add(part)

        family_lower = {n.lower() for n in (family_names or [])}

        # Negative signals — clearly unrelated content
        negative_terms = {
            "serial killer", "murder", "convicted", "arrested",
            "crime", "robbery", "assault", "prison", "sentenced",
            "sports", "football", "basketball", "baseball",
            "movie", "film", "actor", "actress", "singer",
            "restaurant", "hotel", "real estate", "for sale",
        }

        scored = []
        for r in results:
            text = f"{r.title} {r.snippet}".lower()
            score = 0.0

            # Negative check first
            if any(neg in text for neg in negative_terms):
                continue

            # Surname in text (strong signal)
            if surname and surname in text:
                score += 0.3

            # Given name in text
            if given and given in text:
                score += 0.2

            # Full name in text
            if name.lower() in text:
                score += 0.2

            # Year match (within 50 years)
            if birth_year:
                import re
                years = [int(m) for m in re.findall(r"\b\d{4}\b", text)]
                if any(abs(y - birth_year) <= 50 for y in years):
                    score += 0.15

            # Place match
            if place_words:
                matches = sum(1 for w in place_words if w in text)
                score += min(0.2, matches * 0.1)

            # Family name match
            if family_lower:
                matches = sum(1 for n in family_lower if n in text)
                score += min(0.15, matches * 0.05)

            # Genealogy site bonus
            if r.source in self.GENEALOGY_SITES:
                score += 0.1

            # Genealogy keyword bonus
            if any(kw in text for kw in [
                "genealogy", "family tree", "ancestor", "born", "died",
                "census", "burial", "cemetery", "obituary", "marriage",
            ]):
                score += 0.1

            r.relevance = min(1.0, score)
            if score >= min_relevance:
                scored.append(r)

        scored.sort(key=lambda r: -r.relevance)
        return scored

    def quick_scan(
        self,
        name: str,
        birth_year: Optional[int] = None,
        place: Optional[str] = None,
        family_names: Optional[List[str]] = None,
    ) -> str:
        """
        Quick scan — search, filter for relevance, and format results.

        Automatically excludes unrelated content (serial killers, sports,
        entertainment, etc.) by scoring against known facts.
        """
        seen_urls = set()
        all_results: List[SearchResult] = []

        # General person search
        for r in self.search_person(name, birth_year=birth_year, place=place):
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)

        # Genealogy site search
        for r in self.search_genealogy_sites(name, place=place, max_results=3):
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)

        # Filter for relevance
        filtered = self.filter_relevant(
            all_results,
            name=name,
            birth_year=birth_year,
            place=place,
            family_names=family_names,
        )

        if filtered:
            return self.build_context(filtered)

        # If everything was filtered out, return top 2 unfiltered
        # with a caveat
        if all_results:
            return (
                "Note: results may not be directly relevant.\n"
                + self.build_context(all_results[:2])
            )

        return "No web search results found."
