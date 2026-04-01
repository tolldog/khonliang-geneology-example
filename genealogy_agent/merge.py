"""
Merge engine — merge matched persons across trees.

Records all merge operations as triples in TripleStore for provenance.
Supports three strategies: prefer_target (fill gaps), prefer_source
(overwrite), and merge_all (keep both, flag conflicts).
"""

import logging
from dataclasses import dataclass, field
from typing import List

from genealogy_agent.forest import TreeForest

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result of a merge operation."""

    merged_fields: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    source_qxref: str = ""
    target_qxref: str = ""

    @property
    def display(self) -> str:
        lines = [f"Merge {self.source_qxref} -> {self.target_qxref}"]
        if self.merged_fields:
            lines.append(f"  Updated fields: {', '.join(self.merged_fields)}")
        if self.conflicts:
            lines.append(f"  Conflicts ({len(self.conflicts)}):")
            for c in self.conflicts:
                lines.append(f"    - {c}")
        return "\n".join(lines)


class MergeEngine:
    """Merge matched persons from source into target tree."""

    def __init__(self, forest: TreeForest, triple_store=None):
        self.forest = forest
        self.triple_store = triple_store

    def merge_person(
        self,
        source_qxref: str,
        target_qxref: str,
        strategy: str = "prefer_target",
    ) -> MergeResult:
        """Merge a source person's data into the target person.

        Strategies:
            prefer_target: Only fill empty fields in target from source
            prefer_source: Overwrite target fields with source values
            merge_all: Keep target values, note conflicts for different values

        Does NOT merge family links — only person-level fields.
        """
        result = MergeResult(
            source_qxref=source_qxref,
            target_qxref=target_qxref,
        )

        qp_source = self.forest.get_person(source_qxref)
        qp_target = self.forest.get_person(target_qxref)

        if not qp_source or not qp_target:
            result.conflicts.append("One or both persons not found")
            return result

        source = qp_source.person
        target = qp_target.person

        merge_fields = [
            ("given_name", source.given_name, target.given_name),
            ("surname", source.surname, target.surname),
            ("sex", source.sex, target.sex),
            ("birth_date", source.birth_date, target.birth_date),
            ("birth_place", source.birth_place, target.birth_place),
            ("death_date", source.death_date, target.death_date),
            ("death_place", source.death_place, target.death_place),
            ("burial_place", source.burial_place, target.burial_place),
            ("occupation", source.occupation, target.occupation),
        ]

        for field_name, source_val, target_val in merge_fields:
            if not source_val:
                continue  # nothing to merge

            if not target_val:
                # Target is empty — always fill
                setattr(target, field_name, source_val)
                result.merged_fields.append(field_name)
            elif source_val != target_val:
                if strategy == "prefer_source":
                    setattr(target, field_name, source_val)
                    result.merged_fields.append(field_name)
                elif strategy == "merge_all":
                    result.conflicts.append(
                        f"{field_name}: target='{target_val}' vs source='{source_val}'"
                    )
                # prefer_target: keep existing value, no action

        # Merge notes
        if source.notes:
            for note in source.notes:
                if note not in target.notes:
                    target.notes.append(note)
                    result.merged_fields.append("notes")

        # Record merge in triple store
        if self.triple_store:
            self._record_merge_triple(source_qxref, target_qxref)

        logger.info(
            f"Merged {source_qxref} -> {target_qxref}: "
            f"{len(result.merged_fields)} fields updated, "
            f"{len(result.conflicts)} conflicts"
        )
        return result

    def _record_merge_triple(
        self, source_qxref: str, target_qxref: str
    ) -> None:
        """Store merge provenance as a triple."""
        self.triple_store.add(
            subject=source_qxref,
            predicate="merged_into",
            obj=target_qxref,
            confidence=1.0,
            source="merge_engine",
        )
