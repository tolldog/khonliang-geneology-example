"""
TreeForest — manages multiple named GedcomTree instances.

Wraps GedcomTree without modifying it. Provides qualified xrefs
(tree_name:@I1@) to avoid collisions across trees, unified search,
and backward-compatible single-tree access via default_tree.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from genealogy_agent.gedcom_parser import GedcomTree, Person

logger = logging.getLogger(__name__)


@dataclass
class QualifiedPerson:
    """A person with tree provenance."""

    tree_name: str
    person: Person

    @property
    def qualified_xref(self) -> str:
        return f"{self.tree_name}:{self.person.xref}"

    @property
    def display(self) -> str:
        return f"[{self.tree_name}] {self.person.display}"

    def to_dict(self) -> dict:
        d = self.person.to_dict()
        d["tree_name"] = self.tree_name
        d["qualified_xref"] = self.qualified_xref
        return d


class TreeForest:
    """Manages multiple named GedcomTree instances."""

    def __init__(self):
        self._trees: Dict[str, GedcomTree] = {}
        self._default: Optional[str] = None

    def load(self, name: str, path: str) -> GedcomTree:
        """Load a GEDCOM file under a name. First loaded becomes default."""
        tree = GedcomTree.from_file(path)
        self._trees[name] = tree
        if self._default is None:
            self._default = name
        logger.info(
            f"Forest: loaded '{name}' from {path} "
            f"({len(tree.persons)} persons, {len(tree.families)} families)"
        )
        return tree

    def unload(self, name: str) -> None:
        """Remove a tree from the forest."""
        self._trees.pop(name, None)
        if self._default == name:
            self._default = next(iter(self._trees), None)

    @property
    def default_tree(self) -> Optional[GedcomTree]:
        """Backward-compatible single-tree access."""
        if self._default:
            return self._trees.get(self._default)
        return None

    @property
    def default_name(self) -> Optional[str]:
        """Name of the default tree."""
        return self._default

    @default_name.setter
    def default_name(self, name: str) -> None:
        if name in self._trees:
            self._default = name

    @property
    def tree_names(self) -> List[str]:
        return list(self._trees.keys())

    def __len__(self) -> int:
        return len(self._trees)

    def __contains__(self, name: str) -> bool:
        return name in self._trees

    def get_tree(self, name: str) -> Optional[GedcomTree]:
        return self._trees.get(name)

    def resolve_xref(self, qualified_xref: str) -> Tuple[str, str]:
        """Parse 'tree_name:@I1@' -> (tree_name, '@I1@').

        If no prefix, assumes default tree.
        """
        if ":" in qualified_xref:
            parts = qualified_xref.split(":", 1)
            return parts[0], parts[1]
        return self._default or "", qualified_xref

    def get_person(self, qualified_xref: str) -> Optional[QualifiedPerson]:
        """Resolve a qualified xref to a QualifiedPerson."""
        tree_name, xref = self.resolve_xref(qualified_xref)
        tree = self._trees.get(tree_name)
        if tree and xref in tree.persons:
            return QualifiedPerson(tree_name=tree_name, person=tree.persons[xref])
        return None

    def find_person(self, name: str, tree_name: str = "") -> Optional[QualifiedPerson]:
        """Find a person by name, optionally scoped to a tree."""
        if tree_name:
            tree = self._trees.get(tree_name)
            if tree:
                person = tree.find_person(name)
                if person:
                    return QualifiedPerson(tree_name=tree_name, person=person)
            return None

        # Search all trees, default first
        search_order = []
        if self._default:
            search_order.append(self._default)
        search_order.extend(n for n in self._trees if n != self._default)

        for tname in search_order:
            person = self._trees[tname].find_person(name)
            if person:
                return QualifiedPerson(tree_name=tname, person=person)
        return None

    def search_all(self, query: str) -> List[QualifiedPerson]:
        """Search all trees for matching persons."""
        results = []
        for tree_name, tree in self._trees.items():
            for person in tree.search_persons(query):
                results.append(QualifiedPerson(tree_name=tree_name, person=person))
        return results

    def get_summary(self) -> str:
        """Summary of all loaded trees."""
        if not self._trees:
            return "No trees loaded."

        lines = [f"Forest: {len(self._trees)} trees loaded\n"]
        for name, tree in self._trees.items():
            default = " (default)" if name == self._default else ""
            lines.append(
                f"  {name}{default}: "
                f"{len(tree.persons)} persons, "
                f"{len(tree.families)} families"
            )
            # Top surnames
            surnames: Dict[str, int] = {}
            for p in tree.persons.values():
                if p.surname:
                    surnames[p.surname] = surnames.get(p.surname, 0) + 1
            if surnames:
                top = sorted(surnames.items(), key=lambda x: -x[1])[:5]
                lines.append(
                    f"    Top surnames: {', '.join(f'{s} ({c})' for s, c in top)}"
                )
        return "\n".join(lines)

    def get_tree_info(self, name: str) -> Optional[dict]:
        """Get structured info about a tree."""
        tree = self._trees.get(name)
        if not tree:
            return None
        return {
            "name": name,
            "is_default": name == self._default,
            "persons": len(tree.persons),
            "families": len(tree.families),
            "source_file": tree.source_file,
        }

    def list_trees(self) -> List[dict]:
        """Get structured info about all trees."""
        return [
            self.get_tree_info(name)
            for name in self._trees
        ]


def load_forest_from_config(config: dict) -> TreeForest:
    """Build a TreeForest from config, with backward compat for single gedcom."""
    forest = TreeForest()

    gedcoms = config.get("app", {}).get("gedcoms", {})
    if gedcoms:
        for name, path in gedcoms.items():
            forest.load(name, path)
    else:
        # Backward compat: single gedcom file
        path = config.get("app", {}).get("gedcom", "")
        if path:
            name = Path(path).stem.lower().replace(" ", "_")
            # Sanitize: remove non-alphanumeric chars except underscore
            name = re.sub(r"[^a-z0-9_]", "_", name)
            forest.load(name, path)

    return forest
