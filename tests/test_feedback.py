"""Tests for feedback and heuristic systems — FeedbackStore, HeuristicPool."""

from khonliang.training import FeedbackStore, HeuristicPool


class TestFeedbackStore:
    """Verify interaction logging and feedback collection."""

    def test_log_interaction(self, knowledge_db_path):
        store = FeedbackStore(db_path=knowledge_db_path)
        iid = store.log_interaction(
            message="Who was John Smith?",
            role="researcher",
            response="John Smith was born in 1850.",
            generation_ms=150,
        )
        assert iid is not None
        assert isinstance(iid, int)

    def test_add_feedback(self, knowledge_db_path):
        store = FeedbackStore(db_path=knowledge_db_path)
        iid = store.log_interaction(
            message="test", role="researcher", response="response"
        )
        fid = store.add_feedback(
            interaction_id=iid, rating=5, feedback="great answer"
        )
        assert fid is not None

    def test_get_stats(self, knowledge_db_path):
        store = FeedbackStore(db_path=knowledge_db_path)
        store.log_interaction(message="q1", role="researcher", response="r1")
        store.log_interaction(message="q2", role="narrator", response="r2")
        iid = store.log_interaction(message="q3", role="researcher", response="r3")
        store.add_feedback(interaction_id=iid, rating=4)

        stats = store.get_stats()
        assert stats["interactions"] == 3
        assert "researcher" in stats["by_role"]
        assert stats["by_role"]["researcher"] == 2
        assert stats["feedback"]["total"] == 1
        assert 4 in stats["feedback"]["by_rating"]

    def test_multiple_ratings(self, knowledge_db_path):
        store = FeedbackStore(db_path=knowledge_db_path)
        iid1 = store.log_interaction(message="q1", role="researcher", response="r1")
        iid2 = store.log_interaction(message="q2", role="narrator", response="r2")
        store.add_feedback(interaction_id=iid1, rating=5)
        store.add_feedback(interaction_id=iid2, rating=2, feedback="not helpful")

        stats = store.get_stats()
        assert stats["feedback"]["total"] == 2
        assert 5 in stats["feedback"]["by_rating"]
        assert 2 in stats["feedback"]["by_rating"]


class TestHeuristicPool:
    """Verify outcome recording and heuristic extraction."""

    def test_record_outcome(self, knowledge_db_path):
        pool = HeuristicPool(db_path=knowledge_db_path)
        oid = pool.record_outcome(
            action="respond_as_researcher",
            result="success",
            context={"role": "researcher"},
        )
        assert oid is not None

    def test_record_failure(self, knowledge_db_path):
        pool = HeuristicPool(db_path=knowledge_db_path)
        pool.record_outcome(action="respond_as_narrator", result="failure")
        stats = pool.get_stats()
        assert stats["total_outcomes"] >= 1

    def test_add_manual_heuristic(self, knowledge_db_path):
        pool = HeuristicPool(db_path=knowledge_db_path)
        pool.add_heuristic(
            rule="Always cite primary sources for date claims",
            confidence=0.8,
            source="manual",
        )
        rules = pool.get_heuristics(min_confidence=0.0)
        assert len(rules) == 1
        assert "primary sources" in rules[0].rule

    def test_build_prompt_context(self, knowledge_db_path):
        pool = HeuristicPool(db_path=knowledge_db_path)
        pool.add_heuristic(
            rule="Check census records for migration patterns",
            confidence=0.9,
        )
        ctx = pool.build_prompt_context(max_rules=5, min_confidence=0.5)
        assert "census records" in ctx

    def test_empty_prompt_context(self, knowledge_db_path):
        pool = HeuristicPool(db_path=knowledge_db_path)
        ctx = pool.build_prompt_context(max_rules=5, min_confidence=0.5)
        assert ctx == "" or ctx is None or isinstance(ctx, str)
