"""Tests for GRA Pipeline — Generator/Reviewer/Adjudicator entity resolution (FR-11)."""

import pytest

from genealogy_agent.cross_matcher import CrossMatcher
from genealogy_agent.forest import QualifiedPerson, TreeForest
from genealogy_agent.gedcom_parser import GedcomTree, Person
from genealogy_agent.gra_pipeline import (
    AdjudicationResult,
    GenealogyAdjudicator,
    GRAPipeline,
    GRAResult,
    MatchReviewerRole,
    ReviewResult,
)
from genealogy_agent.match_agent import MatchAgentRole, MatchAssessment


class FakeModelPool:
    """Minimal model pool stub."""
    pass


def _make_person(xref, given, surname, sex="M", birth_date="", birth_place="",
                 death_date="", death_place=""):
    return Person(
        xref=xref, given_name=given, surname=surname, sex=sex,
        birth_date=birth_date, birth_place=birth_place,
        death_date=death_date, death_place=death_place,
    )


def _make_forest_with_two_trees():
    """Build a forest with two trees containing similar persons."""
    forest = TreeForest()

    # Tree A: John Smith born 1850 Springfield, died 1920 Chicago
    tree_a = GedcomTree.__new__(GedcomTree)
    tree_a.persons = {
        "@I1@": _make_person(
            "@I1@", "John", "Smith", "M",
            "15 MAR 1850", "Springfield, Illinois",
            "22 NOV 1920", "Chicago, Illinois",
        ),
    }
    tree_a.families = {}
    forest._trees["tree_a"] = tree_a

    # Tree B: John Smith born 1850 Springfield (same person, different tree)
    tree_b = GedcomTree.__new__(GedcomTree)
    tree_b.persons = {
        "@I1@": _make_person(
            "@I1@", "John", "Smith", "M",
            "1850", "Springfield, Illinois",
            "1920", "Chicago, Illinois",
        ),
    }
    tree_b.families = {}
    forest._trees["tree_b"] = tree_b

    return forest


def _make_mismatch_forest():
    """Forest with conflicting persons (different sex)."""
    forest = TreeForest()

    tree_a = GedcomTree.__new__(GedcomTree)
    tree_a.persons = {
        "@I1@": _make_person(
            "@I1@", "Alex", "Smith", "M",
            "1850", "Springfield, Illinois",
        ),
    }
    tree_a.families = {}
    forest._trees["tree_a"] = tree_a

    tree_b = GedcomTree.__new__(GedcomTree)
    tree_b.persons = {
        "@I1@": _make_person(
            "@I1@", "Alex", "Smith", "F",
            "1850", "Springfield, Illinois",
        ),
    }
    tree_b.families = {}
    forest._trees["tree_b"] = tree_b

    return forest


def _make_weak_forest():
    """Forest with weakly matching persons."""
    forest = TreeForest()

    tree_a = GedcomTree.__new__(GedcomTree)
    tree_a.persons = {
        "@I1@": _make_person(
            "@I1@", "James", "Smith", "M",
            "1850", "Springfield, Illinois",
        ),
    }
    tree_a.families = {}
    forest._trees["tree_a"] = tree_a

    tree_b = GedcomTree.__new__(GedcomTree)
    tree_b.persons = {
        "@I1@": _make_person(
            "@I1@", "Jacob", "Smith", "M",
            "1890", "Denver, Colorado",
        ),
    }
    tree_b.families = {}
    forest._trees["tree_b"] = tree_b

    return forest


# ─── ReviewResult parsing tests ──────────────────────────────────────


class TestReviewResultParsing:
    """Verify MatchReviewerRole output parsing."""

    def setup_method(self):
        self.forest = TreeForest()
        self.reviewer = MatchReviewerRole(FakeModelPool(), self.forest)

    def test_parse_agrees(self):
        original = MatchAssessment(
            confidence=0.9, verdict="match", reasoning="Strong match"
        )
        response = """
AGREES: yes
VERDICT: match
CONFIDENCE: 0.88
CRITIQUE: Good assessment with thorough evidence.
MISSED_EVIDENCE: none
MISSED_CONFLICTS: none
"""
        result = self.reviewer._parse_review(response, original)
        assert result.agrees is True
        assert result.verdict == "match"
        assert result.confidence == 0.88
        assert len(result.missed_evidence) == 0

    def test_parse_disagrees(self):
        original = MatchAssessment(
            confidence=0.9, verdict="match", reasoning="Strong match"
        )
        response = """
AGREES: no
VERDICT: possible_match
CONFIDENCE: 0.55
CRITIQUE: The assessor missed important date discrepancy.
MISSED_EVIDENCE: none
MISSED_CONFLICTS:
- Birth year differs by 5 years
- Different death places
"""
        result = self.reviewer._parse_review(response, original)
        assert result.agrees is False
        assert result.verdict == "possible_match"
        assert result.confidence == 0.55
        assert len(result.missed_conflicts) == 2

    def test_parse_with_missed_evidence(self):
        original = MatchAssessment(
            confidence=0.5, verdict="possible_match", reasoning="Weak evidence"
        )
        response = """
AGREES: no
VERDICT: match
CONFIDENCE: 0.85
CRITIQUE: The assessor undervalued the family structure evidence.
MISSED_EVIDENCE:
- Matching spouse surname (Jones)
- Same number of children
MISSED_CONFLICTS: none
"""
        result = self.reviewer._parse_review(response, original)
        assert result.agrees is False
        assert result.verdict == "match"
        assert len(result.missed_evidence) == 2

    def test_parse_defaults_to_original(self):
        """When parsing fails, defaults come from original assessment."""
        original = MatchAssessment(
            confidence=0.7, verdict="possible_match", reasoning="Unclear"
        )
        result = self.reviewer._parse_review("Unparseable response", original)
        assert result.verdict == "possible_match"
        assert result.confidence == 0.7


# ─── Adjudicator tests ──────────────────────────────────────────────


class TestGenealogyAdjudicator:
    """Tests for rule-based genealogy adjudicator."""

    def test_strong_match_returns_match(self):
        forest = _make_forest_with_two_trees()
        adj = GenealogyAdjudicator(forest, match_threshold=0.70)

        person_a = QualifiedPerson(
            tree_name="tree_a",
            person=forest.get_tree("tree_a").persons["@I1@"],
        )
        person_b = QualifiedPerson(
            tree_name="tree_b",
            person=forest.get_tree("tree_b").persons["@I1@"],
        )

        gen = MatchAssessment(confidence=0.9, verdict="match", reasoning="test")
        rev = ReviewResult(
            agrees=False, verdict="possible_match", confidence=0.5, critique="test"
        )

        result = adj.adjudicate(person_a, person_b, gen, rev)
        assert result.verdict == "match"
        assert result.heuristic_score >= 0.70
        assert "name" in result.criteria

    def test_sex_mismatch_forces_no_match(self):
        forest = _make_mismatch_forest()
        adj = GenealogyAdjudicator(forest)

        person_a = QualifiedPerson(
            tree_name="tree_a",
            person=forest.get_tree("tree_a").persons["@I1@"],
        )
        person_b = QualifiedPerson(
            tree_name="tree_b",
            person=forest.get_tree("tree_b").persons["@I1@"],
        )

        gen = MatchAssessment(confidence=0.9, verdict="match", reasoning="test")
        rev = ReviewResult(
            agrees=True, verdict="match", confidence=0.9, critique="test"
        )

        result = adj.adjudicate(person_a, person_b, gen, rev)
        assert result.verdict == "no_match"
        assert "sex mismatch" in result.reason

    def test_weak_match_returns_no_match(self):
        forest = _make_weak_forest()
        adj = GenealogyAdjudicator(forest, match_threshold=0.75, possible_threshold=0.50)

        person_a = QualifiedPerson(
            tree_name="tree_a",
            person=forest.get_tree("tree_a").persons["@I1@"],
        )
        person_b = QualifiedPerson(
            tree_name="tree_b",
            person=forest.get_tree("tree_b").persons["@I1@"],
        )

        gen = MatchAssessment(confidence=0.8, verdict="match", reasoning="test")
        rev = ReviewResult(
            agrees=False, verdict="no_match", confidence=0.7, critique="test"
        )

        result = adj.adjudicate(person_a, person_b, gen, rev)
        # 40 year date gap + different places → low score
        assert result.verdict in ("no_match", "possible_match")
        assert result.heuristic_score < 0.75

    def test_result_includes_criteria(self):
        forest = _make_forest_with_two_trees()
        adj = GenealogyAdjudicator(forest)

        person_a = QualifiedPerson(
            tree_name="tree_a",
            person=forest.get_tree("tree_a").persons["@I1@"],
        )
        person_b = QualifiedPerson(
            tree_name="tree_b",
            person=forest.get_tree("tree_b").persons["@I1@"],
        )

        gen = MatchAssessment(confidence=0.9, verdict="match", reasoning="test")
        rev = ReviewResult(
            agrees=False, verdict="no_match", confidence=0.5, critique="test"
        )

        result = adj.adjudicate(person_a, person_b, gen, rev)
        assert "name" in result.criteria
        assert "date" in result.criteria
        assert "place" in result.criteria
        assert "family" in result.criteria

    def test_to_dict(self):
        result = AdjudicationResult(
            verdict="match",
            confidence=0.85,
            reason="test",
            heuristic_score=0.82,
            criteria={"name": 1.0, "date": 0.9},
        )
        d = result.to_dict()
        assert d["verdict"] == "match"
        assert d["heuristic_score"] == 0.82


# ─── ReviewResult dataclass tests ────────────────────────────────────


class TestReviewResult:
    """Tests for ReviewResult dataclass."""

    def test_to_dict(self):
        result = ReviewResult(
            agrees=False,
            verdict="no_match",
            confidence=0.7,
            critique="Poor evidence",
            missed_evidence=["Spouse match"],
            missed_conflicts=["Date gap"],
        )
        d = result.to_dict()
        assert d["agrees"] is False
        assert d["verdict"] == "no_match"
        assert len(d["missed_evidence"]) == 1
        assert len(d["missed_conflicts"]) == 1


# ─── GRAResult tests ────────────────────────────────────────────────


class TestGRAResult:
    """Tests for GRAResult dataclass."""

    def test_consensus_result(self):
        gen = MatchAssessment(confidence=0.9, verdict="match", reasoning="test")
        rev = ReviewResult(agrees=True, verdict="match", confidence=0.85, critique="ok")

        result = GRAResult(
            verdict="match",
            confidence=0.875,
            generator=gen,
            reviewer=rev,
            adjudication=None,
            resolved_by="consensus",
        )
        assert result.resolved_by == "consensus"
        assert result.adjudication is None

    def test_adjudicated_result(self):
        gen = MatchAssessment(confidence=0.9, verdict="match", reasoning="test")
        rev = ReviewResult(
            agrees=False, verdict="no_match", confidence=0.7, critique="test"
        )
        adj = AdjudicationResult(
            verdict="match", confidence=0.85, reason="test",
            heuristic_score=0.82,
        )

        result = GRAResult(
            verdict="match",
            confidence=0.85,
            generator=gen,
            reviewer=rev,
            adjudication=adj,
            resolved_by="adjudicator",
        )
        assert result.resolved_by == "adjudicator"
        assert result.adjudication is not None

    def test_to_dict(self):
        gen = MatchAssessment(confidence=0.9, verdict="match", reasoning="test")
        result = GRAResult(
            verdict="match",
            confidence=0.9,
            generator=gen,
            reviewer=None,
            adjudication=None,
            resolved_by="generator",
        )
        d = result.to_dict()
        assert d["verdict"] == "match"
        assert d["resolved_by"] == "generator"
        assert d["reviewer"] is None
        assert d["adjudication"] is None
