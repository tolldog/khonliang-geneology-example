"""Tests for personality system — registry, @mention routing, prompt building."""

from khonliang.personalities import extract_mention, build_prompt, format_response

from genealogy_agent.personalities import create_genealogy_registry


class TestPersonalityRegistry:
    """Verify genealogy personality registration."""

    def setup_method(self):
        self.registry = create_genealogy_registry()

    def test_genealogy_personas_registered(self):
        ids = {p.id for p in self.registry.list_enabled()}
        assert "genealogist" in ids
        assert "historian" in ids
        assert "detective" in ids
        assert "skeptic" in ids

    def test_builtin_personas_present(self):
        ids = {p.id for p in self.registry.list_enabled()}
        assert "resolver" in ids
        assert "analyst" in ids
        assert "advocate" in ids

    def test_genealogist_config(self):
        g = self.registry.get("genealogist")
        assert g is not None
        assert g.name == "Genealogy Researcher"
        assert g.voting_weight == 0.30
        assert "records" in g.focus
        assert g.system_prompt  # not empty

    def test_historian_config(self):
        h = self.registry.get("historian")
        assert h is not None
        assert "history" in h.focus
        assert "migration" in h.focus

    def test_detective_config(self):
        d = self.registry.get("detective")
        assert d is not None
        assert "dead_ends" in d.focus

    def test_skeptic_overrides_default(self):
        s = self.registry.get("skeptic")
        assert s is not None
        assert s.name == "Source Critic"  # our override, not default
        assert s.voting_weight == 0.20

    def test_alias_lookup(self):
        assert self.registry.get("research") is not None
        assert self.registry.get("research").id == "genealogist"
        assert self.registry.get("brickwall") is not None
        assert self.registry.get("brickwall").id == "detective"

    def test_weights_sum_reasonable(self):
        genealogy_personas = ["genealogist", "historian", "detective", "skeptic"]
        total = sum(
            self.registry.get(pid).voting_weight
            for pid in genealogy_personas
        )
        assert 0.9 <= total <= 1.1  # roughly sums to 1


class TestMentionExtraction:
    """Verify @mention parsing."""

    def test_extract_known_mention(self):
        assert extract_mention("@skeptic check this record") == "skeptic"

    def test_extract_builtin_mention(self):
        # extract_mention uses the global default registry (4 built-in personas)
        # Custom genealogy personas are resolved via registry.get() in the server
        pid = extract_mention("@analyst check this lineage")
        assert pid == "analyst"

    def test_custom_persona_via_registry(self):
        # Custom personas like @detective resolve through our registry, not extract_mention
        registry = create_genealogy_registry()
        config = registry.get("detective")
        assert config is not None
        assert config.name == "Brick Wall Breaker"

    def test_extract_no_mention(self):
        assert extract_mention("who were John's parents?") is None

    def test_extract_unknown_mention(self):
        # Unknown @mention should return None
        result = extract_mention("@nonexistent do something")
        # May return None or the raw string depending on khonliang impl
        assert result is None or result == "nonexistent"


class TestPromptBuilding:
    """Verify personality prompt construction."""

    def test_build_prompt_includes_question(self):
        prompt = build_prompt("skeptic", "Is this birth record reliable?")
        assert "Is this birth record reliable?" in prompt

    def test_build_prompt_includes_context(self):
        prompt = build_prompt(
            "skeptic",
            "check this",
            context="John Smith born 1850",
        )
        assert "John Smith born 1850" in prompt

    def test_format_response_includes_persona(self):
        formatted = format_response("skeptic", "This record is questionable.")
        assert "This record is questionable." in formatted
        # Should include persona name header
        assert "Skeptic" in formatted or "skeptic" in formatted.lower()
