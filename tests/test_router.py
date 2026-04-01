"""Tests for genealogy router — keyword-based role dispatch."""

from genealogy_agent.router import GenealogyRouter


class TestGenealogyRouter:
    """Verify keyword routing to correct roles."""

    def setup_method(self):
        self.router = GenealogyRouter()

    def test_fact_checker_keywords(self):
        assert self.router.route("check this date") == "fact_checker"
        assert self.router.route("validate this record") == "fact_checker"
        assert self.router.route("is this accurate?") == "fact_checker"
        assert self.router.route("verify the birth date") == "fact_checker"

    def test_narrator_keywords(self):
        assert self.router.route("tell me a story about John") == "narrator"
        assert self.router.route("write a narrative about the family") == "narrator"
        assert self.router.route("describe the migration journey") == "narrator"
        assert self.router.route("what was life like for them") == "narrator"

    def test_researcher_fallback(self):
        assert self.router.route("who were John's grandparents?") == "researcher"
        assert self.router.route("when was Mary born?") == "researcher"
        assert self.router.route("where did the family live?") == "researcher"

    def test_route_with_reason(self):
        role, reason = self.router.route_with_reason("check this date")
        assert role == "fact_checker"
        assert reason  # should have an explanation
