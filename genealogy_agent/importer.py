"""
GEDCOM importer/exporter with agent-driven sanity checking.

Before importing a GEDCOM file into the forest, the importer runs
sanity passes using TreeAnalyzer to catch date anomalies, missing data,
and impossible relationships. High-severity issues block import;
warnings are reported but don't block.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from genealogy_agent.forest import TreeForest
from genealogy_agent.gedcom_parser import GedcomTree
from genealogy_agent.tree_analysis import TreeAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Result of a GEDCOM import attempt."""

    tree_name: str
    persons: int = 0
    families: int = 0
    issues: List[str] = field(default_factory=list)     # blocking issues
    warnings: List[str] = field(default_factory=list)    # non-blocking warnings
    status: str = "ok"  # "ok" | "warnings" | "rejected"

    @property
    def display(self) -> str:
        lines = [f"Import '{self.tree_name}': {self.status}"]
        lines.append(f"  {self.persons} persons, {self.families} families")
        if self.issues:
            lines.append(f"  Issues ({len(self.issues)}):")
            for issue in self.issues[:10]:
                lines.append(f"    - {issue}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings[:10]:
                lines.append(f"    - {w}")
        return "\n".join(lines)


class GedcomImporter:
    """Handles GEDCOM import with validation and optional cross-scan."""

    def __init__(self, forest: TreeForest, cross_matcher=None):
        self.forest = forest
        self.cross_matcher = cross_matcher

    def import_file(
        self,
        path: str,
        name: Optional[str] = None,
        reject_on_issues: bool = True,
    ) -> ImportResult:
        """Load, validate, and optionally add a GEDCOM file to the forest.

        Args:
            path: Path to GEDCOM file
            name: Tree name (derived from filename if not provided)
            reject_on_issues: If True, high-severity issues block import

        Returns:
            ImportResult with status, counts, and issues/warnings
        """
        if not name:
            name = Path(path).stem.lower().replace(" ", "_")
            name = re.sub(r"[^a-z0-9_]", "_", name)

        result = ImportResult(tree_name=name)

        # Parse the GEDCOM
        try:
            tree = GedcomTree.from_file(path)
        except Exception as e:
            result.status = "rejected"
            result.issues.append(f"Failed to parse GEDCOM: {e}")
            return result

        result.persons = len(tree.persons)
        result.families = len(tree.families)

        if result.persons == 0:
            result.status = "rejected"
            result.issues.append("GEDCOM file contains no persons")
            return result

        # Run sanity checks
        issues, warnings = self.sanity_check(tree)
        result.issues = issues
        result.warnings = warnings

        if issues and reject_on_issues:
            result.status = "rejected"
            logger.warning(
                f"Import '{name}' rejected: {len(issues)} issues"
            )
            return result

        # Check for name collision
        if name in self.forest:
            result.warnings.append(
                f"Tree '{name}' already loaded — replacing"
            )
            self.forest.unload(name)

        # Add to forest
        self.forest._trees[name] = tree
        if self.forest._default is None:
            self.forest._default = name

        result.status = "warnings" if warnings else "ok"
        logger.info(
            f"Imported '{name}': {result.persons} persons, "
            f"{result.families} families, {len(warnings)} warnings"
        )
        return result

    def sanity_check(self, tree: GedcomTree) -> tuple:
        """Run TreeAnalyzer checks on a candidate tree.

        Returns:
            (issues: List[str], warnings: List[str])
        """
        analyzer = TreeAnalyzer(tree)
        issues = []
        warnings = []

        # Date anomalies — high severity blocks import
        for gap in analyzer.find_date_anomalies():
            if gap.severity == "high":
                issues.append(f"[{gap.gap_type}] {gap.description}")
            else:
                warnings.append(f"[{gap.gap_type}] {gap.description}")

        # Missing data — informational only
        missing = analyzer.find_missing_data()
        if missing:
            warnings.append(
                f"{len(missing)} persons with missing data fields"
            )

        return issues, warnings

    def export_gedcom(self, tree_name: str, path: str) -> str:
        """Export a tree back to GEDCOM 5.5.1 format.

        Args:
            tree_name: Name of tree in forest
            path: Output file path

        Returns:
            Path to written file
        """
        tree = self.forest.get_tree(tree_name)
        if not tree:
            raise ValueError(f"Tree '{tree_name}' not found")

        lines = []

        # Header
        lines.append("0 HEAD")
        lines.append("1 SOUR GenealogyAgent")
        lines.append("1 GEDC")
        lines.append("2 VERS 5.5.1")
        lines.append("2 FORM LINEAGE-LINKED")
        lines.append("1 CHAR UTF-8")

        # Individuals
        for person in tree.persons.values():
            lines.append(f"0 {person.xref} INDI")
            if person.given_name or person.surname:
                lines.append(
                    f"1 NAME {person.given_name} /{person.surname}/"
                )
                if person.given_name:
                    lines.append(f"2 GIVN {person.given_name}")
                if person.surname:
                    lines.append(f"2 SURN {person.surname}")
            if person.sex:
                lines.append(f"1 SEX {person.sex}")
            if person.birth_date or person.birth_place:
                lines.append("1 BIRT")
                if person.birth_date:
                    lines.append(f"2 DATE {person.birth_date}")
                if person.birth_place:
                    lines.append(f"2 PLAC {person.birth_place}")
            if person.death_date or person.death_place:
                lines.append("1 DEAT")
                if person.death_date:
                    lines.append(f"2 DATE {person.death_date}")
                if person.death_place:
                    lines.append(f"2 PLAC {person.death_place}")
            if person.burial_place:
                lines.append("1 BURI")
                lines.append(f"2 PLAC {person.burial_place}")
            if person.occupation:
                lines.append(f"1 OCCU {person.occupation}")
            for fam_xref in person.family_spouse:
                lines.append(f"1 FAMS {fam_xref}")
            for fam_xref in person.family_child:
                lines.append(f"1 FAMC {fam_xref}")

        # Families
        for family in tree.families.values():
            lines.append(f"0 {family.xref} FAM")
            if family.husband:
                lines.append(f"1 HUSB {family.husband}")
            if family.wife:
                lines.append(f"1 WIFE {family.wife}")
            for child_xref in family.children:
                lines.append(f"1 CHIL {child_xref}")
            if family.marriage_date or family.marriage_place:
                lines.append("1 MARR")
                if family.marriage_date:
                    lines.append(f"2 DATE {family.marriage_date}")
                if family.marriage_place:
                    lines.append(f"2 PLAC {family.marriage_place}")
            if family.divorce_date:
                lines.append("1 DIV")
                lines.append(f"2 DATE {family.divorce_date}")

        # Trailer
        lines.append("0 TRLR")

        output = "\n".join(lines) + "\n"
        Path(path).write_text(output, encoding="utf-8")
        logger.info(f"Exported '{tree_name}' to {path}")
        return path
