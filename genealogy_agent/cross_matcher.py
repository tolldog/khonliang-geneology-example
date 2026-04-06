"""
CrossMatcher — heuristic-based person matching across trees.

Pure algorithmic scoring with no LLM calls. Designed for fast batch
scanning with surname pre-filtering. Use MatchAgent for LLM-backed
evaluation of top candidates.

Scoring weights:
  - Name similarity:    40%
  - Date proximity:     25%
  - Place overlap:      20%
  - Family structure:   15%
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from genealogy_agent.forest import QualifiedPerson, TreeForest
from genealogy_agent.gedcom_parser import GedcomTree, Person


@dataclass
class MatchCandidate:
    """A potential match between two persons in different trees."""

    person_a: QualifiedPerson
    person_b: QualifiedPerson
    score: float              # 0.0 - 1.0 composite
    name_score: float = 0.0
    date_score: float = 0.0
    place_score: float = 0.0
    family_score: float = 0.0
    conflicts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "person_a": self.person_a.to_dict(),
            "person_b": self.person_b.to_dict(),
            "score": round(self.score, 3),
            "name_score": round(self.name_score, 3),
            "date_score": round(self.date_score, 3),
            "place_score": round(self.place_score, 3),
            "family_score": round(self.family_score, 3),
            "conflicts": self.conflicts,
        }

    @property
    def display(self) -> str:
        return (
            f"{self.person_a.display} <-> {self.person_b.display} "
            f"(score={self.score:.0%})"
        )


# Scoring weights
W_NAME = 0.40
W_DATE = 0.25
W_PLACE = 0.20
W_FAMILY = 0.15


class CrossMatcher:
    """Heuristic-based cross-tree person matching."""

    def __init__(self, forest: TreeForest):
        self.forest = forest

    def scan(
        self,
        tree_a: str,
        tree_b: str,
        min_score: float = 0.6,
        max_results: int = 50,
    ) -> List[MatchCandidate]:
        """Scan two trees for matching persons.

        Pre-filters by surname for performance, then scores all pairs
        with matching surnames.
        """
        ta = self.forest.get_tree(tree_a)
        tb = self.forest.get_tree(tree_b)
        if not ta or not tb:
            return []

        # Build surname index for tree B
        surname_index_b: Dict[str, List[Person]] = {}
        for person in tb.persons.values():
            key = person.surname.lower()
            if key:
                surname_index_b.setdefault(key, []).append(person)

        candidates: List[MatchCandidate] = []
        for person_a in ta.persons.values():
            key = person_a.surname.lower()
            if not key or key not in surname_index_b:
                continue

            for person_b in surname_index_b[key]:
                match = self.score_pair(
                    person_a, person_b, ta, tb, tree_a, tree_b
                )
                if match.score >= min_score:
                    candidates.append(match)

        # Sort by score descending
        candidates.sort(key=lambda m: -m.score)
        return candidates[:max_results]

    def compare(
        self, qxref_a: str, qxref_b: str
    ) -> Optional[MatchCandidate]:
        """Compare two specific persons by qualified xref."""
        qp_a = self.forest.get_person(qxref_a)
        qp_b = self.forest.get_person(qxref_b)
        if not qp_a or not qp_b:
            return None

        ta = self.forest.get_tree(qp_a.tree_name)
        tb = self.forest.get_tree(qp_b.tree_name)
        if not ta or not tb:
            return None

        return self.score_pair(
            qp_a.person, qp_b.person, ta, tb,
            qp_a.tree_name, qp_b.tree_name,
        )

    def score_pair(
        self,
        a: Person, b: Person,
        tree_a: GedcomTree, tree_b: GedcomTree,
        name_a: str, name_b: str,
    ) -> MatchCandidate:
        """Score a pair of persons across all dimensions."""
        ns = self._name_score(a, b)
        ds = self._date_score(a, b)
        ps = self._place_score(a, b)
        fs = self._family_score(a, b, tree_a, tree_b)
        conflicts = self._detect_conflicts(a, b)

        # Conflict penalty
        penalty = min(len(conflicts) * 0.15, 0.5)
        composite = (
            W_NAME * ns + W_DATE * ds + W_PLACE * ps + W_FAMILY * fs
        ) * (1.0 - penalty)

        return MatchCandidate(
            person_a=QualifiedPerson(tree_name=name_a, person=a),
            person_b=QualifiedPerson(tree_name=name_b, person=b),
            score=max(0.0, min(1.0, composite)),
            name_score=ns,
            date_score=ds,
            place_score=ps,
            family_score=fs,
            conflicts=conflicts,
        )

    def _name_score(self, a: Person, b: Person) -> float:
        """Score name similarity. Surname match is 0.6 base."""
        if not a.surname or not b.surname:
            return 0.0

        score = 0.0
        # Surname comparison
        if a.surname.lower() == b.surname.lower():
            score = 0.6
        else:
            return 0.0  # pre-filter should prevent this

        # Given name comparison
        ga = a.given_name.lower()
        gb = b.given_name.lower()
        if not ga or not gb:
            return score

        if ga == gb:
            score += 0.4
        elif ga.startswith(gb) or gb.startswith(ga):
            # Nickname/abbreviation match (e.g., "Wm" matches "William")
            score += 0.25
        elif ga[0] == gb[0]:
            # Same initial
            score += 0.1

        return min(1.0, score)

    def _date_score(self, a: Person, b: Person) -> float:
        """Score date proximity. Same year = 1.0, -0.1 per year apart."""
        ya = self._extract_year(a.birth_date)
        yb = self._extract_year(b.birth_date)

        if ya is None and yb is None:
            # No dates to compare — neutral
            return 0.5

        if ya is None or yb is None:
            # One has a date, one doesn't — slight penalty
            return 0.3

        diff = abs(ya - yb)
        if diff == 0:
            return 1.0
        if diff <= 2:
            return 0.9
        if diff <= 5:
            return 0.7
        return max(0.0, 1.0 - diff * 0.1)

    def _place_score(self, a: Person, b: Person) -> float:
        """Score place overlap via token intersection."""
        places_a = self._place_tokens(a)
        places_b = self._place_tokens(b)

        if not places_a or not places_b:
            return 0.3  # neutral if no place data

        intersection = places_a & places_b
        union = places_a | places_b

        if not union:
            return 0.3

        return len(intersection) / len(union)

    def _family_score(
        self, a: Person, b: Person,
        tree_a: GedcomTree, tree_b: GedcomTree,
    ) -> float:
        """Score family structure similarity."""
        score = 0.0
        checks = 0

        # Compare spouse surnames
        spouses_a = tree_a.get_spouses(a.xref)
        spouses_b = tree_b.get_spouses(b.xref)
        if spouses_a and spouses_b:
            checks += 1
            a_surnames = {s.surname.lower() for s in spouses_a if s.surname}
            b_surnames = {s.surname.lower() for s in spouses_b if s.surname}
            if a_surnames & b_surnames:
                score += 1.0

        # Compare parent surnames
        parents_a = tree_a.get_parents(a.xref)
        parents_b = tree_b.get_parents(b.xref)
        if parents_a and parents_b:
            checks += 1
            a_names = {p.surname.lower() for p in parents_a if p.surname}
            b_names = {p.surname.lower() for p in parents_b if p.surname}
            if a_names & b_names:
                score += 1.0

        # Compare child count (rough proxy)
        children_a = tree_a.get_children(a.xref)
        children_b = tree_b.get_children(b.xref)
        if children_a and children_b:
            checks += 1
            diff = abs(len(children_a) - len(children_b))
            if diff == 0:
                score += 1.0
            elif diff <= 2:
                score += 0.5

        if checks == 0:
            return 0.3  # neutral if no family data
        return score / checks

    def _detect_conflicts(self, a: Person, b: Person) -> List[str]:
        """Detect hard conflicts that make a match unlikely."""
        conflicts = []

        # Sex mismatch
        if a.sex and b.sex and a.sex != b.sex:
            conflicts.append(f"sex mismatch: {a.sex} vs {b.sex}")

        # Death before other's birth
        ya_birth = self._extract_year(a.birth_date)
        yb_birth = self._extract_year(b.birth_date)
        ya_death = self._extract_year(a.death_date)
        yb_death = self._extract_year(b.death_date)

        if ya_death and yb_birth and ya_death < yb_birth - 5:
            conflicts.append(
                f"person A died {ya_death} before person B born {yb_birth}"
            )
        if yb_death and ya_birth and yb_death < ya_birth - 5:
            conflicts.append(
                f"person B died {yb_death} before person A born {ya_birth}"
            )

        # Large date gap (> 20 years)
        if ya_birth and yb_birth and abs(ya_birth - yb_birth) > 20:
            conflicts.append(
                f"birth year gap: {abs(ya_birth - yb_birth)} years"
            )

        return conflicts

    @staticmethod
    def _extract_year(date_str: str) -> Optional[int]:
        if not date_str:
            return None
        match = re.search(r"\d{4}", date_str)
        return int(match.group()) if match else None

    @staticmethod
    def _place_tokens(person: Person) -> set:
        """Extract normalized place tokens from birth and death places."""
        tokens = set()
        for place in [person.birth_place, person.death_place]:
            if place:
                for part in place.split(","):
                    cleaned = part.strip().lower()
                    if cleaned and len(cleaned) > 1:
                        tokens.add(cleaned)
        return tokens
