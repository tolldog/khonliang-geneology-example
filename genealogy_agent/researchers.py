"""
Genealogy researchers — pluggable workers for the research pool.

Each researcher handles specific task types and returns structured results.
Register them with a ResearchPool to process queued research tasks.
"""

import logging
import re
from typing import Optional

from khonliang.research.base import BaseResearcher
from khonliang.research.models import ResearchResult, ResearchTask

from genealogy_agent.gedcom_parser import GedcomTree
from genealogy_agent.web_search import GenealogySearcher

logger = logging.getLogger(__name__)


class WebSearchResearcher(BaseResearcher):
    """
    Web search researcher — looks up people, places, and history online.

    Handles:
        - person_lookup: search for a specific person
        - web_search: general web search
        - historical_context: search for historical info about a place/time
        - migration: search for migration patterns
    """

    name = "web_search"
    capabilities = [
        "person_lookup",
        "web_search",
        "historical_context",
        "migration",
    ]
    max_concurrent = 3  # multiple web searches can run in parallel

    def __init__(self, tree: Optional[GedcomTree] = None, max_results: int = 5):
        self.searcher = GenealogySearcher(max_results=max_results)
        self.tree = tree

    async def research(self, task: ResearchTask) -> ResearchResult:
        if task.task_type == "person_lookup":
            return self._person_lookup(task)
        elif task.task_type == "historical_context":
            return self._historical_context(task)
        elif task.task_type == "migration":
            return self._migration_search(task)
        else:
            return self._general_search(task)

    def _person_lookup(self, task: ResearchTask) -> ResearchResult:
        """
        Look up a person using multiple search engines, tree enrichment,
        and relevance filtering.

        Strategy:
        1. Enrich query with tree data (dates, places, family names)
        2. Search DDG (general + genealogy sites)
        3. Search Google
        4. Deduplicate by URL
        5. Filter all results for relevance
        6. Return only relevant results
        """
        name = task.query
        birth_year = None
        place = None
        family_names = []

        # Enrich from tree data
        if self.tree:
            person = self.tree.find_person(name)
            if person:
                year_match = re.search(r"\d{4}", person.birth_date)
                birth_year = int(year_match.group()) if year_match else None
                place = person.birth_place

                for p in self.tree.get_parents(person.xref):
                    family_names.append(p.surname)
                    family_names.append(p.given_name)
                for s in self.tree.get_spouses(person.xref):
                    family_names.append(s.surname)
                for c in self.tree.get_children(person.xref):
                    family_names.append(c.given_name)
                family_names = [n for n in family_names if n]

        # Run all search strategies in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        seen_urls: set = set()
        all_raw = []
        lock = __import__("threading").Lock()

        def _collect(results):
            with lock:
                for r in results:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        all_raw.append(r)

        # Build queries
        multi_query_parts = [f'"{name}"']
        if place:
            place_parts = [p.strip() for p in place.split(",") if p.strip()]
            if place_parts:
                multi_query_parts.append(place_parts[0])
        if birth_year:
            multi_query_parts.append(str(birth_year))
        multi_query_parts.append("genealogy OR family")
        multi_query = " ".join(multi_query_parts)

        strategies = [
            lambda: self.searcher.search_person(
                name, birth_year=birth_year, place=place, max_results=8
            ),
            lambda: self.searcher.search_genealogy_sites(
                name, place=place, max_results=5
            ),
            lambda: self.searcher.multi_search(multi_query, max_per_engine=5),
        ]

        # Add raw query strategy if there's extra context
        if "," in task.query or len(task.query.split()) > 3:
            raw_query = task.query
            strategies.append(
                lambda: self.searcher.multi_search(raw_query, max_per_engine=5)
            )

        with ThreadPoolExecutor(
            max_workers=len(strategies), thread_name_prefix="lookup"
        ) as pool:
            futures = [pool.submit(fn) for fn in strategies]
            for future in as_completed(futures, timeout=20):
                try:
                    _collect(future.result())
                except Exception as e:
                    logger.debug(f"Search strategy failed: {e}")

        # Filter all results for relevance
        # Include extra context words from the query as family_names
        # so "Taylor University" boosts results mentioning Taylor
        extra_terms = [
            w.strip(".,")
            for w in task.query.split()
            if len(w.strip(".,")) > 2
        ]
        filter_names = family_names + extra_terms

        results = self.searcher.filter_relevant(
            all_raw,
            name=name,
            birth_year=birth_year,
            place=place,
            family_names=filter_names,
        )
        sources = [r.url for r in results if r.url]
        content = self.searcher.build_context(results)

        if not results and all_raw:
            content = (
                "Note: results may not be directly relevant.\n"
                + self.searcher.build_context(all_raw[:3])
            )

        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title=f"Research: {name}",
            content=content,
            confidence=0.7,
            sources=sources,
            scope=task.scope,
        )

    def _historical_context(self, task: ResearchTask) -> ResearchResult:
        """Look up historical context for a place and time."""
        query = task.query
        year = task.metadata.get("year", 1800)

        # Try to extract year from query
        year_match = re.search(r"\d{4}", query)
        if year_match:
            year = int(year_match.group())
            query = query.replace(year_match.group(), "").strip()

        results = self.searcher.search_historical_context(query, year)
        context = self.searcher.build_context(results)
        sources = [r.url for r in results if r.url]

        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title=f"History: {query} ({year})",
            content=context,
            confidence=0.6,
            sources=sources,
            scope=task.scope,
        )

    def _migration_search(self, task: ResearchTask) -> ResearchResult:
        """Search for migration patterns."""
        results = self.searcher.search_migration(
            surname=task.query,
            from_place=task.metadata.get("from_place"),
            to_place=task.metadata.get("to_place"),
            era=task.metadata.get("era"),
        )
        context = self.searcher.build_context(results)
        sources = [r.url for r in results if r.url]

        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title=f"Migration: {task.query}",
            content=context,
            confidence=0.6,
            sources=sources,
            scope=task.scope,
        )

    def _general_search(self, task: ResearchTask) -> ResearchResult:
        """General web search."""
        results = self.searcher.search(task.query)
        context = self.searcher.build_context(results)
        sources = [r.url for r in results if r.url]

        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title=f"Search: {task.query}",
            content=context,
            confidence=0.5,
            sources=sources,
            scope=task.scope,
        )


class TreeResearcher(BaseResearcher):
    """
    Tree data researcher — looks up structured data from the GEDCOM tree.

    Handles:
        - tree_lookup: find person details
        - tree_ancestors: get ancestor chain
        - tree_migration: trace migration from tree data
    """

    name = "tree_data"
    capabilities = ["tree_lookup", "tree_ancestors", "tree_migration"]
    max_concurrent = 5  # tree lookups are fast, allow many

    def __init__(self, tree: GedcomTree):
        self.tree = tree

    async def research(self, task: ResearchTask) -> ResearchResult:
        if task.task_type == "tree_lookup":
            return self._lookup(task)
        elif task.task_type == "tree_ancestors":
            return self._ancestors(task)
        elif task.task_type == "tree_migration":
            return self._migration(task)
        return self._lookup(task)

    def _lookup(self, task: ResearchTask) -> ResearchResult:
        person = self.tree.find_person(task.query)
        if not person:
            return ResearchResult(
                task_id=task.task_id,
                task_type=task.task_type,
                title=f"Not found: {task.query}",
                content=f"No person matching '{task.query}' in the tree.",
                confidence=1.0,
                scope=task.scope,
            )

        context = self.tree.build_context(person.xref, depth=2)
        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title=f"Tree: {person.full_name}",
            content=context,
            confidence=1.0,
            scope=task.scope,
        )

    def _ancestors(self, task: ResearchTask) -> ResearchResult:
        person = self.tree.find_person(task.query)
        if not person:
            return ResearchResult(
                task_id=task.task_id,
                task_type=task.task_type,
                title=f"Not found: {task.query}",
                content=f"No person matching '{task.query}' in the tree.",
                confidence=1.0,
                scope=task.scope,
            )

        gens = task.metadata.get("generations", 4)
        ancestors = self.tree.get_ancestors(person.xref, generations=gens)
        lines = [f"Ancestors of {person.full_name} ({len(ancestors)}):"]
        for a in ancestors:
            lines.append(f"  - {a.display}")

        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title=f"Ancestors: {person.full_name}",
            content="\n".join(lines),
            confidence=1.0,
            scope=task.scope,
        )

    def _migration(self, task: ResearchTask) -> ResearchResult:
        person = self.tree.find_person(task.query)
        if not person:
            return ResearchResult(
                task_id=task.task_id,
                task_type=task.task_type,
                title=f"Not found: {task.query}",
                content=f"No person matching '{task.query}' in the tree.",
                confidence=1.0,
                scope=task.scope,
            )

        ancestors = self.tree.get_ancestors(person.xref, generations=10)
        all_people = [person] + ancestors

        moves = []
        for p in all_people:
            year_match = re.search(r"\d{4}", p.birth_date or "")
            year = int(year_match.group()) if year_match else None
            place = p.birth_place or p.death_place
            if place:
                moves.append((year or 9999, p.full_name, place))

        moves.sort()
        lines = [f"Migration for {person.full_name}'s line:"]
        for year, name, place in moves:
            yr = str(year) if year != 9999 else "????"
            lines.append(f"  {yr}: {name} — {place}")

        return ResearchResult(
            task_id=task.task_id,
            task_type=task.task_type,
            title=f"Migration: {person.full_name}",
            content="\n".join(lines),
            confidence=1.0,
            scope=task.scope,
        )
