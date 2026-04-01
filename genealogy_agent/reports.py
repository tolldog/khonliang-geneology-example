"""
Research reports — present accumulated knowledge to the user.

Report types:
- Session report: what happened since last connection
- Person report: everything we know about a specific person
- Research report: what background research found
- Gap report: what's still missing and what we tried
- Knowledge report: state of the knowledge store
"""

import logging
import re
import time
from typing import Optional

from khonliang.knowledge.store import KnowledgeStore, Tier

from genealogy_agent.gedcom_parser import GedcomTree

logger = logging.getLogger(__name__)


class ReportBuilder:
    """
    Builds reports from tree data, knowledge store, and research results.

    Example:
        builder = ReportBuilder(tree, knowledge_store)
        report = builder.person_report("Timothy Toll")
        report = builder.knowledge_report()
        report = builder.gap_report("Timothy Toll")
    """

    def __init__(
        self,
        tree: GedcomTree,
        knowledge_store: Optional[KnowledgeStore] = None,
    ):
        self.tree = tree
        self.store = knowledge_store

    def person_report(self, name: str) -> str:
        """
        Everything we know about a person — tree data + research findings.

        Combines structured GEDCOM data with any knowledge accumulated
        through research, web searches, and API lookups.
        """
        person = self.tree.find_person(name)
        if not person:
            return f"Person '{name}' not found in the tree."

        sections = []

        # Header
        sections.append(f"# Report: {person.full_name}")
        sections.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
        sections.append("")

        # Tree data
        sections.append("## Tree Data")
        sections.append(self.tree.build_context(person.xref, depth=2))
        sections.append("")

        # Family summary
        parents = self.tree.get_parents(person.xref)
        children = self.tree.get_children(person.xref)
        spouses = self.tree.get_spouses(person.xref)
        siblings = self.tree.get_siblings(person.xref)

        sections.append("## Family Summary")
        sections.append(f"- Parents: {len(parents)}")
        sections.append(f"- Spouses: {len(spouses)}")
        sections.append(f"- Children: {len(children)}")
        sections.append(f"- Siblings: {len(siblings)}")
        sections.append("")

        # Knowledge findings
        if self.store:
            entries = self.store.search(
                person.full_name, limit=10
            )
            if entries:
                sections.append("## Research Findings")
                for entry in entries:
                    tier_label = {
                        Tier.AXIOM: "RULE",
                        Tier.IMPORTED: "VERIFIED",
                        Tier.DERIVED: "RESEARCHED",
                    }.get(entry.tier, "?")
                    sections.append(
                        f"### [{tier_label}] {entry.title} "
                        f"(confidence: {entry.confidence:.0%})"
                    )
                    sections.append(f"Source: {entry.source}")
                    sections.append(entry.content[:500])
                    sections.append("")
            else:
                sections.append("## Research Findings")
                sections.append("No research has been done on this person yet.")
                sections.append(
                    f"Try: `!lookup {person.full_name}` to search online."
                )
                sections.append("")

        # Gaps for this person
        from genealogy_agent.tree_analysis import TreeAnalyzer

        analyzer = TreeAnalyzer(self.tree)

        # Missing data
        missing = []
        if not person.birth_date:
            missing.append("birth date")
        if not person.birth_place:
            missing.append("birth place")
        if not person.death_date:
            year = self._extract_year(person.birth_date)
            if year and year < 1940:
                missing.append("death date")
        if not person.occupation:
            missing.append("occupation")

        if missing or not parents:
            sections.append("## Gaps & Research Opportunities")
            if missing:
                sections.append(f"- Missing: {', '.join(missing)}")
            if not parents:
                sections.append("- **Dead end**: no parents recorded")
                sections.append(
                    f"  Search: `!lookup {analyzer.search_name(person)} "
                    f"{person.birth_place or ''} genealogy parents`"
                )
            sections.append("")

        # Ancestor dead ends
        dead_ends = analyzer.find_dead_ends_for(person.full_name)
        if dead_ends:
            sections.append(f"## Ancestor Dead Ends ({len(dead_ends)})")
            for gap in dead_ends[:10]:
                sections.append(f"- {gap.description}")
            if len(dead_ends) > 10:
                sections.append(f"- ... and {len(dead_ends) - 10} more")
            sections.append("")

        return "\n".join(sections)

    def knowledge_report(self) -> str:
        """Report on the state of the knowledge store."""
        if not self.store:
            return "No knowledge store configured."

        sections = []
        sections.append("# Knowledge Report")
        sections.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
        sections.append("")

        stats = self.store.get_stats()
        sections.append("## Summary")
        sections.append(f"- Total entries: {stats.get('total_entries', 0)}")

        by_tier = stats.get("by_tier", {})
        sections.append(f"- Axioms (Tier 1): {by_tier.get('axiom', 0)}")
        sections.append(f"- Imported (Tier 2): {by_tier.get('imported', 0)}")
        sections.append(f"- Derived (Tier 3): {by_tier.get('derived', 0)}")
        sections.append("")

        by_scope = stats.get("by_scope", {})
        if by_scope:
            sections.append("## By Scope")
            for scope, count in sorted(by_scope.items()):
                sections.append(f"- {scope}: {count}")
            sections.append("")

        # Show recent derived entries (most recent research)
        derived = self.store.get_by_tier(Tier.DERIVED)
        if derived:
            # Sort by updated_at descending
            derived.sort(key=lambda e: e.updated_at, reverse=True)
            sections.append(f"## Recent Research ({len(derived)} entries)")
            for entry in derived[:10]:
                conf = f"{entry.confidence:.0%}"
                sections.append(
                    f"- [{conf}] {entry.title} (source: {entry.source})"
                )
            sections.append("")

        # Show promoted entries (high quality)
        imported = self.store.get_by_tier(Tier.IMPORTED)
        promoted = [e for e in imported if e.source != "system"]
        if promoted:
            sections.append(f"## Validated Knowledge ({len(promoted)} entries)")
            for entry in promoted[:10]:
                sections.append(
                    f"- {entry.title} ({entry.confidence:.0%})"
                )
            sections.append("")

        # Axioms
        axioms = self.store.get_axioms()
        if axioms:
            sections.append(f"## Active Rules ({len(axioms)})")
            for a in axioms:
                sections.append(f"- {a.title}: {a.content[:80]}")
            sections.append("")

        return "\n".join(sections)

    def gap_report(self, name: Optional[str] = None) -> str:
        """Report on tree gaps, anomalies, and research opportunities."""
        from genealogy_agent.tree_analysis import TreeAnalyzer

        analyzer = TreeAnalyzer(self.tree)

        sections = []
        sections.append("# Gap Analysis Report")
        sections.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}")
        sections.append("")

        if name:
            person = self.tree.find_person(name)
            if not person:
                return f"Person '{name}' not found."
            sections.append(f"## For: {person.full_name}")
            sections.append("")

            dead_ends = analyzer.find_dead_ends_for(name)
            sections.append(f"### Dead-End Lines ({len(dead_ends)})")
            for gap in dead_ends[:15]:
                sections.append(f"- {gap.description}")
                if gap.research_query:
                    sections.append(f"  Query: {gap.research_query}")
            if len(dead_ends) > 15:
                sections.append(f"- ... and {len(dead_ends) - 15} more")
            sections.append("")
        else:
            sections.append(f"## Tree: {len(self.tree.persons)} persons, "
                            f"{len(self.tree.families)} families")
            sections.append("")

        # Anomalies
        anomalies = analyzer.find_date_anomalies()
        if anomalies:
            high = [a for a in anomalies if a.severity == "high"]
            med = [a for a in anomalies if a.severity == "medium"]
            sections.append(
                f"### Date Anomalies ({len(anomalies)} total, "
                f"{len(high)} high, {len(med)} medium)"
            )
            for a in high[:10]:
                sections.append(f"- [HIGH] {a.person_name}: {a.description}")
            for a in med[:5]:
                sections.append(f"- [MED] {a.person_name}: {a.description}")
            sections.append("")

        # Missing data summary
        missing = analyzer.find_missing_data()
        if missing:
            sections.append(f"### Missing Data ({len(missing)} persons)")
            sections.append(
                "Persons with 2+ missing fields "
                "(birth date, place, death date, etc.)"
            )
            sections.append("")

        # Incomplete families
        incomplete = analyzer.find_incomplete_families()
        if incomplete:
            sections.append(f"### Incomplete Families ({len(incomplete)})")
            sections.append("Families with unknown spouse")
            sections.append("")

        # Research coverage (if knowledge store available)
        if self.store:
            total_persons = len(self.tree.persons)
            researched = set()
            for entry in self.store.get_by_tier(Tier.DERIVED):
                # Extract names from entry titles
                for person in self.tree.persons.values():
                    if person.full_name.lower() in entry.title.lower():
                        researched.add(person.xref)

            coverage = len(researched) / total_persons * 100 if total_persons else 0
            sections.append("### Research Coverage")
            sections.append(
                f"- {len(researched)} of {total_persons} persons "
                f"have been researched ({coverage:.1f}%)"
            )
            sections.append("")

        return "\n".join(sections)

    def session_report(self) -> str:
        """Brief summary suitable for showing on connect."""
        sections = []
        sections.append("# Session Summary")
        sections.append("")

        # Tree stats
        sections.append(
            f"Tree: {len(self.tree.persons)} persons, "
            f"{len(self.tree.families)} families"
        )

        # Knowledge stats
        if self.store:
            stats = self.store.get_stats()
            total = stats.get("total_entries", 0)
            derived = stats.get("by_tier", {}).get("derived", 0)
            imported = stats.get("by_tier", {}).get("imported", 0)
            sections.append(
                f"Knowledge: {total} entries "
                f"({imported} verified, {derived} researched)"
            )

            # Recent additions
            recent = self.store.get_by_tier(Tier.DERIVED)
            recent.sort(key=lambda e: e.updated_at, reverse=True)
            if recent:
                sections.append("")
                sections.append("Recent research:")
                for entry in recent[:5]:
                    sections.append(f"  - {entry.title}")

        sections.append("")
        sections.append(
            "Commands: !report <name>, !knowledge, !gaps, "
            "!dead-ends <name>, !anomalies"
        )

        return "\n".join(sections)

    @staticmethod
    def _extract_year(date_str: str) -> Optional[int]:
        if not date_str:
            return None
        match = re.search(r"\d{4}", date_str)
        return int(match.group()) if match else None
