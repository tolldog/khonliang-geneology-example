"""Tests for self-evaluation — date checks, relationship checks, evaluator factory."""

from genealogy_agent.self_eval import (
    DateCheckRule,
    RelationshipCheckRule,
    create_genealogy_evaluator,
)


class TestDateCheckRule:
    """Verify date claims are checked against tree data."""

    def test_correct_date_no_issues(self, tree):
        rule = DateCheckRule(tree)
        # Use unambiguous phrasing — the regex scans for "born/died" near dates
        response = "John Smith was born 1850."
        issues = rule.check(response)
        assert len(issues) == 0

    def test_wrong_birth_date(self, tree):
        rule = DateCheckRule(tree)
        response = "John Smith was born in 1900."
        issues = rule.check(response)
        assert len(issues) == 1
        assert issues[0].issue_type == "date_mismatch"
        assert issues[0].severity == "high"
        assert "1850" in issues[0].detail

    def test_wrong_death_date(self, tree):
        rule = DateCheckRule(tree)
        response = "John Smith died in 1800."
        issues = rule.check(response)
        assert len(issues) == 1
        assert issues[0].issue_type == "date_mismatch"

    def test_close_date_within_tolerance(self, tree):
        rule = DateCheckRule(tree)
        # Within 5-year tolerance
        response = "John Smith was born in 1852."
        issues = rule.check(response)
        assert len(issues) == 0

    def test_unknown_person_ignored(self, tree):
        rule = DateCheckRule(tree)
        response = "Unknown Person was born in 1900."
        issues = rule.check(response)
        assert len(issues) == 0

    def test_extract_year(self):
        assert DateCheckRule._extract_year("15 MAR 1850") == 1850
        assert DateCheckRule._extract_year("1920") == 1920
        assert DateCheckRule._extract_year("") is None
        assert DateCheckRule._extract_year(None) is None
        assert DateCheckRule._extract_year("ABT 1860") == 1860


class TestRelationshipCheckRule:
    """Verify relationship claims are checked against tree data."""

    def test_correct_relationship_no_issues(self, tree):
        rule = RelationshipCheckRule(tree)
        response = "William Smith's father was John Smith."
        issues = rule.check(response)
        assert len(issues) == 0

    def test_wrong_relationship(self, tree):
        rule = RelationshipCheckRule(tree)
        # Robert Brown is NOT William's parent
        response = "William Smith's father was Robert Brown."
        issues = rule.check(response)
        assert len(issues) == 1
        assert issues[0].issue_type == "wrong_relationship"
        assert issues[0].severity == "high"


class TestGenealogyEvaluator:
    """Verify the composed evaluator factory."""

    def test_create_evaluator(self, tree):
        evaluator = create_genealogy_evaluator(tree)
        assert evaluator is not None

    def test_evaluator_passes_clean_response(self, tree):
        evaluator = create_genealogy_evaluator(tree)
        result = evaluator.evaluate(
            "John Smith was born in 1850 in Springfield, Illinois.",
            query="When was John born?",
            role="researcher",
        )
        assert result.passed
        assert result.confidence > 0.5

    def test_evaluator_flags_wrong_date(self, tree):
        evaluator = create_genealogy_evaluator(tree)
        result = evaluator.evaluate(
            "John Smith was born in 1900.",
            query="When was John born?",
            role="researcher",
        )
        assert not result.passed
        assert len(result.issues) > 0

    def test_evaluator_detects_speculation(self, tree):
        evaluator = create_genealogy_evaluator(tree)
        result = evaluator.evaluate(
            "It's possible that perhaps maybe likely John might have "
            "been a soldier, I think it's conceivable he could have served.",
            query="What did John do?",
            role="narrator",
        )
        # SpeculationRule should flag excessive hedging
        speculation_issues = [
            i for i in result.issues if i.issue_type == "excessive_speculation"
        ]
        assert len(speculation_issues) > 0
