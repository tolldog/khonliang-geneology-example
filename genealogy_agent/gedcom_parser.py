"""
GEDCOM file parser — reads standard genealogy files into a queryable structure.

Handles GEDCOM 5.5/5.5.1 format (the most common export format from
Ancestry, FamilySearch, MyHeritage, etc.)
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Person:
    """A person in the family tree."""

    xref: str  # GEDCOM cross-reference ID (e.g. @I1@)
    given_name: str = ""
    surname: str = ""
    sex: str = ""  # M, F, or ""
    birth_date: str = ""
    birth_place: str = ""
    death_date: str = ""
    death_place: str = ""
    burial_place: str = ""
    occupation: str = ""
    notes: List[str] = field(default_factory=list)
    family_spouse: List[str] = field(default_factory=list)  # FAM xrefs as spouse
    family_child: List[str] = field(default_factory=list)  # FAM xrefs as child

    @property
    def full_name(self) -> str:
        parts = [self.given_name, self.surname]
        return " ".join(p for p in parts if p) or f"[Unknown {self.xref}]"

    @property
    def display(self) -> str:
        """One-line summary."""
        parts = [self.full_name]
        if self.birth_date or self.death_date:
            parts.append(f"({self.birth_date or '?'} - {self.death_date or '?'})")
        if self.birth_place:
            parts.append(f"b. {self.birth_place}")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "xref": self.xref,
            "name": self.full_name,
            "given_name": self.given_name,
            "surname": self.surname,
            "sex": self.sex,
            "birth_date": self.birth_date,
            "birth_place": self.birth_place,
            "death_date": self.death_date,
            "death_place": self.death_place,
            "occupation": self.occupation,
        }


@dataclass
class Family:
    """A family unit linking spouses and children."""

    xref: str
    husband: str = ""  # Person xref
    wife: str = ""  # Person xref
    children: List[str] = field(default_factory=list)  # Person xrefs
    marriage_date: str = ""
    marriage_place: str = ""
    divorce_date: str = ""


class GedcomTree:
    """
    Parsed GEDCOM family tree with lookup methods.

    Usage:
        tree = GedcomTree.from_file("family.ged")
        person = tree.find_person("John Smith")
        ancestors = tree.get_ancestors(person.xref, generations=4)
        summary = tree.get_summary()
    """

    def __init__(self):
        self.persons: Dict[str, Person] = {}
        self.families: Dict[str, Family] = {}
        self.source_file: str = ""

    @classmethod
    def from_file(cls, path: str) -> "GedcomTree":
        """Parse a GEDCOM file into a GedcomTree."""
        tree = cls()
        tree.source_file = path
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        tree._parse(content)
        logger.info(
            f"Parsed {path}: {len(tree.persons)} persons, "
            f"{len(tree.families)} families"
        )
        return tree

    def find_person(self, name: str) -> Optional[Person]:
        """Find a person by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for person in self.persons.values():
            if name_lower in person.full_name.lower():
                return person
        return None

    def search_persons(self, query: str) -> List[Person]:
        """Search persons by name, place, or date."""
        query_lower = query.lower()
        results = []
        for person in self.persons.values():
            searchable = " ".join([
                person.full_name,
                person.birth_place,
                person.death_place,
                person.birth_date,
                person.occupation,
            ]).lower()
            if query_lower in searchable:
                results.append(person)
        return results

    def get_parents(self, xref: str) -> List[Person]:
        """Get parents of a person."""
        person = self.persons.get(xref)
        if not person:
            return []
        parents = []
        for fam_xref in person.family_child:
            fam = self.families.get(fam_xref)
            if fam:
                if fam.husband and fam.husband in self.persons:
                    parents.append(self.persons[fam.husband])
                if fam.wife and fam.wife in self.persons:
                    parents.append(self.persons[fam.wife])
        return parents

    def get_children(self, xref: str) -> List[Person]:
        """Get children of a person."""
        person = self.persons.get(xref)
        if not person:
            return []
        children = []
        for fam_xref in person.family_spouse:
            fam = self.families.get(fam_xref)
            if fam:
                for child_xref in fam.children:
                    if child_xref in self.persons:
                        children.append(self.persons[child_xref])
        return children

    def get_spouses(self, xref: str) -> List[Person]:
        """Get spouses of a person."""
        person = self.persons.get(xref)
        if not person:
            return []
        spouses = []
        for fam_xref in person.family_spouse:
            fam = self.families.get(fam_xref)
            if fam:
                spouse_xref = fam.wife if fam.husband == xref else fam.husband
                if spouse_xref and spouse_xref in self.persons:
                    spouses.append(self.persons[spouse_xref])
        return spouses

    def get_siblings(self, xref: str) -> List[Person]:
        """Get siblings of a person."""
        person = self.persons.get(xref)
        if not person:
            return []
        siblings = []
        for fam_xref in person.family_child:
            fam = self.families.get(fam_xref)
            if fam:
                for child_xref in fam.children:
                    if child_xref != xref and child_xref in self.persons:
                        siblings.append(self.persons[child_xref])
        return siblings

    def get_ancestors(self, xref: str, generations: int = 4) -> List[Person]:
        """Get ancestors up to N generations back."""
        if generations <= 0:
            return []
        ancestors = []
        parents = self.get_parents(xref)
        ancestors.extend(parents)
        for parent in parents:
            ancestors.extend(self.get_ancestors(parent.xref, generations - 1))
        return ancestors

    def get_descendants(self, xref: str, generations: int = 4) -> List[Person]:
        """Get descendants up to N generations forward."""
        if generations <= 0:
            return []
        descendants = []
        children = self.get_children(xref)
        descendants.extend(children)
        for child in children:
            descendants.extend(
                self.get_descendants(child.xref, generations - 1)
            )
        return descendants

    def build_context(self, xref: str, depth: int = 2) -> str:
        """
        Build a text context about a person suitable for LLM injection.

        Includes the person's details, parents, spouses, children, and
        optionally grandparents/grandchildren.
        """
        person = self.persons.get(xref)
        if not person:
            return ""

        lines = [f"Person: {person.display}"]
        if person.sex:
            lines.append(f"Sex: {person.sex}")
        if person.occupation:
            lines.append(f"Occupation: {person.occupation}")

        parents = self.get_parents(xref)
        if parents:
            lines.append(f"Parents: {', '.join(p.display for p in parents)}")

        spouses = self.get_spouses(xref)
        if spouses:
            lines.append(f"Spouses: {', '.join(s.display for s in spouses)}")

        children = self.get_children(xref)
        if children:
            lines.append(f"Children: {', '.join(c.display for c in children)}")

        siblings = self.get_siblings(xref)
        if siblings:
            lines.append(f"Siblings: {', '.join(s.display for s in siblings)}")

        if depth >= 2:
            for parent in parents:
                grandparents = self.get_parents(parent.xref)
                if grandparents:
                    lines.append(
                        f"Grandparents ({parent.full_name}'s parents): "
                        f"{', '.join(g.display for g in grandparents)}"
                    )

        if person.notes:
            lines.append(f"Notes: {'; '.join(person.notes[:3])}")

        return "\n".join(lines)

    def get_summary(self) -> str:
        """Get a text summary of the tree for LLM context."""
        surnames = {}
        for p in self.persons.values():
            if p.surname:
                surnames[p.surname] = surnames.get(p.surname, 0) + 1

        top_surnames = sorted(surnames.items(), key=lambda x: -x[1])[:10]

        lines = [
            f"Family tree: {len(self.persons)} persons, "
            f"{len(self.families)} families",
            f"Surnames: {', '.join(f'{n} ({c})' for n, c in top_surnames)}",
        ]

        # Date range
        years = []
        for p in self.persons.values():
            for date_str in [p.birth_date, p.death_date]:
                match = re.search(r"\d{4}", date_str)
                if match:
                    years.append(int(match.group()))
        if years:
            lines.append(f"Date range: {min(years)} - {max(years)}")

        # Places
        places = set()
        for p in self.persons.values():
            for place in [p.birth_place, p.death_place]:
                if place:
                    places.add(place)
        if places:
            lines.append(f"Places ({len(places)}): {', '.join(sorted(places)[:15])}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # GEDCOM parsing
    # ------------------------------------------------------------------

    def _parse(self, content: str) -> None:
        """Parse raw GEDCOM content."""
        lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        records = self._split_records(lines)

        for record in records:
            if not record:
                continue
            header = record[0]
            if header.startswith("0") and "@" in header:
                xref_match = re.match(r"0\s+(@\S+@)\s+(\w+)", header)
                if xref_match:
                    xref = xref_match.group(1)
                    record_type = xref_match.group(2)
                    if record_type == "INDI":
                        self._parse_individual(xref, record[1:])
                    elif record_type == "FAM":
                        self._parse_family(xref, record[1:])

    def _split_records(self, lines: List[str]) -> List[List[str]]:
        """Split GEDCOM lines into level-0 records."""
        records: List[List[str]] = []
        current: List[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("0 "):
                if current:
                    records.append(current)
                current = [line]
            else:
                current.append(line)
        if current:
            records.append(current)
        return records

    def _parse_individual(self, xref: str, lines: List[str]) -> None:
        """Parse an INDI record."""
        person = Person(xref=xref)
        i = 0
        while i < len(lines):
            line = lines[i]
            parts = line.split(None, 2)
            if len(parts) < 2:
                i += 1
                continue

            level = parts[0]
            tag = parts[1]
            value = parts[2] if len(parts) > 2 else ""

            if level == "1":
                if tag == "NAME":
                    # Parse name: "Given /Surname/"
                    name_match = re.match(r"(.+?)\s*/(.+?)/", value)
                    if name_match:
                        person.given_name = name_match.group(1).strip()
                        person.surname = name_match.group(2).strip()
                    else:
                        person.given_name = value.strip("/").strip()
                elif tag == "SEX":
                    person.sex = value.strip()
                elif tag == "BIRT":
                    person.birth_date, person.birth_place = self._parse_event(
                        lines, i
                    )
                elif tag == "DEAT":
                    person.death_date, person.death_place = self._parse_event(
                        lines, i
                    )
                elif tag == "BURI":
                    _, person.burial_place = self._parse_event(lines, i)
                elif tag == "OCCU":
                    person.occupation = value
                elif tag == "FAMS":
                    person.family_spouse.append(value.strip())
                elif tag == "FAMC":
                    person.family_child.append(value.strip())
                elif tag == "NOTE":
                    note = value
                    # Collect continuation lines
                    j = i + 1
                    while j < len(lines):
                        np = lines[j].split(None, 2)
                        if len(np) >= 2 and int(np[0]) > int(level):
                            if np[1] in ("CONT", "CONC"):
                                note += " " + (np[2] if len(np) > 2 else "")
                            j += 1
                        else:
                            break
                    person.notes.append(note.strip())
            i += 1

        self.persons[xref] = person

    def _parse_family(self, xref: str, lines: List[str]) -> None:
        """Parse a FAM record."""
        family = Family(xref=xref)
        i = 0
        while i < len(lines):
            line = lines[i]
            parts = line.split(None, 2)
            if len(parts) < 2:
                i += 1
                continue

            level = parts[0]
            tag = parts[1]
            value = parts[2] if len(parts) > 2 else ""

            if level == "1":
                if tag == "HUSB":
                    family.husband = value.strip()
                elif tag == "WIFE":
                    family.wife = value.strip()
                elif tag == "CHIL":
                    family.children.append(value.strip())
                elif tag == "MARR":
                    family.marriage_date, family.marriage_place = (
                        self._parse_event(lines, i)
                    )
                elif tag == "DIV":
                    family.divorce_date, _ = self._parse_event(lines, i)
            i += 1

        self.families[xref] = family

    @staticmethod
    def _parse_event(lines: List[str], start: int) -> tuple:
        """Parse DATE and PLAC from an event's sub-records."""
        date = ""
        place = ""
        i = start + 1
        while i < len(lines):
            parts = lines[i].split(None, 2)
            if len(parts) < 2:
                break
            if parts[0] == "1":
                break  # Next level-1 tag
            if parts[1] == "DATE" and len(parts) > 2:
                date = parts[2].strip()
            elif parts[1] == "PLAC" and len(parts) > 2:
                place = parts[2].strip()
            i += 1
        return date, place
