"""
Genealogy agent roles — LLM-backed family history research.

- ResearcherRole: answers questions about the family tree
- FactCheckerRole: validates relationships, dates, and flags contradictions
- NarratorRole: generates family stories from genealogy data
"""

from typing import Any, Dict, List, Optional

from khonliang.roles.base import BaseRole

from genealogy_agent.gedcom_parser import GedcomTree, Person


def _build_multi_context(
    tree: GedcomTree, message: str, max_persons: int = 10
) -> str:
    """
    Smart context builder — handles both single and multi-person queries.

    Tries multiple strategies:
    1. Search for all matching persons (handles "all the Tims", "Toll family")
    2. Find a specific person mentioned by name
    3. Fall back to tree summary
    """
    msg_lower = message.lower()

    # Strategy 1: broad search — look for name-like words in the query
    # Skip common non-name words
    skip = {
        "the", "a", "an", "all", "of", "in", "about", "tell", "me",
        "who", "were", "was", "are", "is", "what", "when", "where",
        "how", "did", "do", "does", "my", "our", "their", "his", "her",
        "family", "tree", "parents", "children", "ancestors", "descendants",
        "grandparents", "siblings", "married", "born", "died", "lived",
        "from", "to", "and", "with", "for", "this", "that", "these",
        "those", "have", "had", "been", "any", "some", "every", "each",
        "story", "history", "migration", "check", "verify", "validate",
        "narrative", "describe", "explain", "find", "search", "list",
        "show", "give", "get", "can", "you", "please", "could",
    }

    # Extract candidate search terms
    # Strip punctuation, possessives, and normalize quotes
    import re

    words = message.split()
    candidates = []
    for w in words:
        # Strip punctuation and possessive forms
        cleaned = re.sub(r"['\u2019]s$", "", w)  # Tim's -> Tim
        cleaned = re.sub(r"[.,!?'\"\u2018\u2019\u201c\u201d]", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned.lower() not in skip and len(cleaned) > 1:
            candidates.append(cleaned)

    # Try each candidate as a search term
    all_matches: List[Person] = []
    seen_xrefs = set()
    for term in candidates:
        for person in tree.search_persons(term):
            if person.xref not in seen_xrefs:
                seen_xrefs.add(person.xref)
                all_matches.append(person)

    # If we found multiple matches, return a list context
    if len(all_matches) > 1:
        lines = [f"Found {len(all_matches)} matching persons:"]
        for p in all_matches[:max_persons]:
            lines.append(f"\n- {p.display}")
            parents = tree.get_parents(p.xref)
            if parents:
                lines.append(
                    f"  Parents: {', '.join(par.full_name for par in parents)}"
                )
            spouses = tree.get_spouses(p.xref)
            if spouses:
                lines.append(
                    f"  Spouse: {', '.join(s.full_name for s in spouses)}"
                )
        if len(all_matches) > max_persons:
            lines.append(
                f"\n... and {len(all_matches) - max_persons} more"
            )
        return "\n".join(lines)

    # Strategy 2: single person — try multi-word name matching
    for i in range(len(words)):
        for j in range(i + 1, min(i + 4, len(words) + 1)):
            name_guess = " ".join(words[i:j])
            person = tree.find_person(name_guess)
            if person:
                return tree.build_context(person.xref, depth=2)

    # Strategy 3: if we found exactly one match from search
    if len(all_matches) == 1:
        return tree.build_context(all_matches[0].xref, depth=2)

    # Fallback: tree summary
    return tree.get_summary()


class ResearcherRole(BaseRole):
    """
    Answers questions about the family tree using parsed GEDCOM data.

    Injects relevant family context into the prompt so the LLM can
    answer questions like "who were John's grandparents?" or "where
    did the Smith family come from?"
    """

    def __init__(self, model_pool, tree: GedcomTree, **kwargs):
        super().__init__(role="researcher", model_pool=model_pool, **kwargs)
        self.tree = tree
        self._system_prompt = (
            "You are a genealogy research assistant. You have access to a "
            "parsed family tree database. Answer questions about family "
            "relationships, dates, and places based on the provided data. "
            "Be precise about what the records show vs what is uncertain. "
            "If the data doesn't contain the answer, say so."
        )

    def build_context(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        return _build_multi_context(self.tree, message)

    async def handle(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = self.build_context(message, context)
        prompt = (
            f"Family tree data:\n{ctx}\n\n"
            f"Question: {message}\n\nAnswer:"
        )

        response, elapsed_ms = await self._timed_generate(
            prompt=prompt, system=self.system_prompt
        )
        return {
            "response": response.strip(),
            "metadata": {
                "role": self.role,
                "generation_time_ms": elapsed_ms,
            },
        }


class FactCheckerRole(BaseRole):
    """
    Validates genealogy data for contradictions and anomalies.

    Checks things like:
    - Birth before parent's birth
    - Death before birth
    - Unreasonable age gaps
    - Missing data patterns
    """

    def __init__(self, model_pool, tree: GedcomTree, **kwargs):
        super().__init__(role="fact_checker", model_pool=model_pool, **kwargs)
        self.tree = tree
        self._system_prompt = (
            "You are a genealogy fact-checker. Analyze the provided family "
            "data for inconsistencies, contradictions, or suspicious patterns. "
            "Look for: impossible dates (child born before parent), unreasonable "
            "age gaps, missing data, and historical plausibility issues. "
            "Be specific about what looks wrong and suggest corrections."
        )

    def build_context(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        return _build_multi_context(self.tree, message, max_persons=15)

    async def handle(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = self.build_context(message, context)
        prompt = (
            f"Family tree data to validate:\n{ctx}\n\n"
            f"Request: {message}\n\nAnalysis:"
        )

        response, elapsed_ms = await self._timed_generate(
            prompt=prompt, system=self.system_prompt
        )
        return {
            "response": response.strip(),
            "metadata": {
                "role": self.role,
                "generation_time_ms": elapsed_ms,
            },
        }


class NarratorRole(BaseRole):
    """
    Generates readable family narratives from genealogy data.

    Turns dry dates and places into engaging family stories.
    """

    def __init__(self, model_pool, tree: GedcomTree, **kwargs):
        super().__init__(role="narrator", model_pool=model_pool, **kwargs)
        self.tree = tree
        self._system_prompt = (
            "You are a genealogy storyteller. Transform dry family tree data "
            "into engaging, readable narratives. Include historical context "
            "where relevant (wars, migrations, historical events that may have "
            "affected the family). Stick to facts from the data but weave them "
            "into a compelling story. Clearly mark any speculation."
        )

    def build_context(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        return _build_multi_context(self.tree, message)

    async def handle(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ctx = self.build_context(message, context)
        prompt = (
            f"Family tree data:\n{ctx}\n\n"
            f"Request: {message}\n\nNarrative:"
        )

        response, elapsed_ms = await self._timed_generate(
            prompt=prompt, system=self.system_prompt
        )
        return {
            "response": response.strip(),
            "metadata": {
                "role": self.role,
                "generation_time_ms": elapsed_ms,
            },
        }
