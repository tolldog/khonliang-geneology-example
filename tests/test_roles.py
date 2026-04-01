"""Tests for role context building and heuristic injection."""

from unittest.mock import MagicMock

from genealogy_agent.roles import (
    ResearcherRole,
    FactCheckerRole,
    NarratorRole,
    _build_multi_context,
    _build_context_with_session,
)


class TestContextBuilding:
    """Verify smart context assembly from tree data."""

    def test_multi_context_single_person(self, tree):
        ctx = _build_multi_context(tree, "tell me about John Smith")
        assert "John Smith" in ctx
        assert "1850" in ctx

    def test_multi_context_multiple_matches(self, tree):
        ctx = _build_multi_context(tree, "Smith")
        assert "John Smith" in ctx
        assert "William Smith" in ctx

    def test_multi_context_fallback_summary(self, tree):
        ctx = _build_multi_context(tree, "zzzzz nothing matches")
        # Should fall back to tree summary
        assert "8" in ctx or "persons" in ctx.lower()

    def test_multi_context_place_in_query(self, tree):
        ctx = _build_multi_context(tree, "Denver")
        assert "William Smith" in ctx or "Denver" in ctx

    def test_context_with_session_empty(self, tree):
        ctx = _build_context_with_session(tree, "John Smith")
        assert "John Smith" in ctx
        # No session context set, so no SESSION CONTEXT section
        assert "SESSION CONTEXT" not in ctx


class TestRoleHeuristicInjection:
    """Verify heuristic rules are injected into system prompts."""

    def test_researcher_without_heuristics(self, tree):
        pool = MagicMock()
        role = ResearcherRole(pool, tree=tree)
        prompt = role._effective_system_prompt()
        assert "LEARNED PATTERNS" not in prompt

    def test_researcher_with_heuristics(self, tree):
        pool = MagicMock()
        heuristic_pool = MagicMock()
        heuristic_pool.build_prompt_context.return_value = "- Always cite sources"
        role = ResearcherRole(pool, tree=tree, heuristic_pool=heuristic_pool)
        prompt = role._effective_system_prompt()
        assert "LEARNED PATTERNS" in prompt
        assert "Always cite sources" in prompt

    def test_fact_checker_with_heuristics(self, tree):
        pool = MagicMock()
        heuristic_pool = MagicMock()
        heuristic_pool.build_prompt_context.return_value = "- Check dates carefully"
        role = FactCheckerRole(pool, tree=tree, heuristic_pool=heuristic_pool)
        prompt = role._effective_system_prompt()
        assert "Check dates carefully" in prompt

    def test_narrator_with_heuristics(self, tree):
        pool = MagicMock()
        heuristic_pool = MagicMock()
        heuristic_pool.build_prompt_context.return_value = "- No fabrication"
        role = NarratorRole(pool, tree=tree, heuristic_pool=heuristic_pool)
        prompt = role._effective_system_prompt()
        assert "No fabrication" in prompt

    def test_empty_heuristics_no_section(self, tree):
        pool = MagicMock()
        heuristic_pool = MagicMock()
        heuristic_pool.build_prompt_context.return_value = ""
        role = ResearcherRole(pool, tree=tree, heuristic_pool=heuristic_pool)
        prompt = role._effective_system_prompt()
        assert "LEARNED PATTERNS" not in prompt
