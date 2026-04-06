"""
GRA Pipeline — Generator-Reviewer-Adjudicator for entity resolution (FR-11).

Implements the GRA collaborative framework from "A Strategic Coordination
Framework of Small LLMs Matches Large LLMs in Data Synthesis" adapted
for genealogical entity resolution.

Roles:
    - Generator (MatchAgentRole): Produces initial match assessments
    - Reviewer (MatchReviewerRole): Critiques the assessment quality
    - Adjudicator (GenealogyAdjudicator): Rule-based tiebreaker using
      heuristic scoring when Generator and Reviewer disagree

Pipeline:
    1. Generator evaluates match candidate → MatchAssessment
    2. Reviewer critiques the assessment → ReviewResult
    3. If they agree → final verdict
    4. If they disagree → Adjudicator applies genealogical criteria

Usage:
    pipeline = GRAPipeline(generator=match_agent, reviewer=reviewer, adjudicator=adj)
    result = await pipeline.evaluate(person_a, person_b)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from khonliang.roles.base import BaseRole

from genealogy_agent.cross_matcher import CrossMatcher
from genealogy_agent.forest import QualifiedPerson, TreeForest
from genealogy_agent.match_agent import MatchAgentRole, MatchAssessment

logger = logging.getLogger(__name__)


# ─── Reviewer ────────────────────────────────────────────────────────────


@dataclass
class ReviewResult:
    """Structured result from the Reviewer."""

    agrees: bool
    verdict: str  # "match" | "possible_match" | "no_match"
    confidence: float
    critique: str
    missed_evidence: List[str] = field(default_factory=list)
    missed_conflicts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agrees": self.agrees,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "critique": self.critique,
            "missed_evidence": self.missed_evidence,
            "missed_conflicts": self.missed_conflicts,
        }


class MatchReviewerRole(BaseRole):
    """Reviews and critiques Generator's match assessments.

    Checks for:
    - Evidence quality and completeness
    - Missed conflicts or contradictions
    - Logical consistency of reasoning
    - Whether confidence level is justified
    """

    def __init__(self, model_pool, forest: TreeForest, **kwargs):
        super().__init__(role="match_reviewer", model_pool=model_pool, **kwargs)
        self.forest = forest
        self._system_prompt = (
            "You are a genealogy quality reviewer. Your job is to critique "
            "a match assessment produced by another agent. Check whether the "
            "assessment is thorough, accurate, and well-justified.\n\n"
            "REVIEW CRITERIA:\n"
            "1. Evidence completeness — did the assessor consider all available "
            "   facts (names, dates, places, family structure)?\n"
            "2. Conflict detection — are there missed conflicts the assessor "
            "   overlooked (sex mismatch, date impossibilities, wrong parents)?\n"
            "3. Confidence calibration — is the confidence level justified by "
            "   the evidence? High confidence needs strong, specific evidence.\n"
            "4. Logical consistency — does the reasoning support the verdict?\n\n"
            "OUTPUT FORMAT (use exactly these labels):\n"
            "AGREES: yes | no\n"
            "VERDICT: match | possible_match | no_match\n"
            "CONFIDENCE: 0.0-1.0\n"
            "CRITIQUE: your assessment of the quality of the original evaluation\n"
            "MISSED_EVIDENCE: any supporting facts the assessor missed (or 'none')\n"
            "MISSED_CONFLICTS: any conflicts the assessor missed (or 'none')"
        )

    async def handle(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Standard role handle."""
        response, elapsed_ms = await self._timed_generate(
            prompt=message, system=self._system_prompt
        )
        return {
            "response": response.strip(),
            "metadata": {"role": self.role, "generation_time_ms": elapsed_ms},
        }

    async def review(
        self,
        person_a: QualifiedPerson,
        person_b: QualifiedPerson,
        assessment: MatchAssessment,
    ) -> ReviewResult:
        """Review a Generator's match assessment."""
        prompt = self._build_review_prompt(person_a, person_b, assessment)
        response, _ = await self._timed_generate(
            prompt=prompt, system=self._system_prompt
        )
        return self._parse_review(response, assessment)

    def _build_review_prompt(
        self,
        person_a: QualifiedPerson,
        person_b: QualifiedPerson,
        assessment: MatchAssessment,
    ) -> str:
        """Build prompt with person data and the Generator's assessment."""
        pa = person_a.person
        pb = person_b.person

        lines = [
            "Review this match assessment for accuracy and completeness.\n",
            f"=== PERSON A (tree '{person_a.tree_name}') ===",
            f"Name: {pa.full_name}",
            f"Sex: {pa.sex or 'unknown'}",
            f"Birth: {pa.birth_date or 'unknown'}, {pa.birth_place or 'unknown'}",
            f"Death: {pa.death_date or 'unknown'}, {pa.death_place or 'unknown'}",
        ]

        # Family context for person A
        tree_a = self.forest.get_tree(person_a.tree_name)
        if tree_a:
            parents = tree_a.get_parents(pa.xref)
            if parents:
                lines.append(f"Parents: {', '.join(p.full_name for p in parents)}")
            spouses = tree_a.get_spouses(pa.xref)
            if spouses:
                lines.append(f"Spouse(s): {', '.join(s.full_name for s in spouses)}")

        lines.extend([
            "",
            f"=== PERSON B (tree '{person_b.tree_name}') ===",
            f"Name: {pb.full_name}",
            f"Sex: {pb.sex or 'unknown'}",
            f"Birth: {pb.birth_date or 'unknown'}, {pb.birth_place or 'unknown'}",
            f"Death: {pb.death_date or 'unknown'}, {pb.death_place or 'unknown'}",
        ])

        tree_b = self.forest.get_tree(person_b.tree_name)
        if tree_b:
            parents = tree_b.get_parents(pb.xref)
            if parents:
                lines.append(f"Parents: {', '.join(p.full_name for p in parents)}")
            spouses = tree_b.get_spouses(pb.xref)
            if spouses:
                lines.append(f"Spouse(s): {', '.join(s.full_name for s in spouses)}")

        lines.extend([
            "",
            "=== GENERATOR'S ASSESSMENT ===",
            f"Verdict: {assessment.verdict}",
            f"Confidence: {assessment.confidence}",
            f"Evidence: {'; '.join(assessment.evidence) or 'none provided'}",
            f"Conflicts: {'; '.join(assessment.conflicts) or 'none'}",
            f"Reasoning: {assessment.reasoning}",
            "",
            "Provide your review:",
        ])
        return "\n".join(lines)

    def _parse_review(
        self, response: str, original: MatchAssessment
    ) -> ReviewResult:
        """Parse structured Reviewer output."""
        agrees = False  # Default to disagree unless explicitly parsed
        verdict = original.verdict
        confidence = original.confidence
        critique = response
        missed_evidence: List[str] = []
        missed_conflicts: List[str] = []

        # AGREES
        m = re.search(r"AGREES:\s*(yes|no)", response, re.I)
        if m:
            agrees = m.group(1).lower() == "yes"

        # VERDICT
        m = re.search(r"VERDICT:\s*(match|possible_match|no_match)", response, re.I)
        if m:
            verdict = m.group(1).lower()

        # CONFIDENCE
        m = re.search(r"CONFIDENCE:\s*([\d.]+)", response)
        if m:
            try:
                confidence = max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass

        # CRITIQUE
        m = re.search(
            r"CRITIQUE:(.*?)(?=MISSED_EVIDENCE:|MISSED_CONFLICTS:|$)",
            response, re.S | re.I,
        )
        if m:
            critique = m.group(1).strip()

        # MISSED_EVIDENCE
        m = re.search(
            r"MISSED_EVIDENCE:(.*?)(?=MISSED_CONFLICTS:|$)",
            response, re.S | re.I,
        )
        if m:
            missed_evidence = [
                line.strip().lstrip("- ").lstrip("* ")
                for line in m.group(1).strip().split("\n")
                if line.strip() and line.strip().lower() not in ("none", "n/a")
            ]

        # MISSED_CONFLICTS
        m = re.search(r"MISSED_CONFLICTS:(.*?)$", response, re.S | re.I)
        if m:
            missed_conflicts = [
                line.strip().lstrip("- ").lstrip("* ")
                for line in m.group(1).strip().split("\n")
                if line.strip() and line.strip().lower() not in ("none", "n/a")
            ]

        return ReviewResult(
            agrees=agrees,
            verdict=verdict,
            confidence=confidence,
            critique=critique[:500],
            missed_evidence=missed_evidence,
            missed_conflicts=missed_conflicts,
        )


# ─── Adjudicator ─────────────────────────────────────────────────────────


@dataclass
class AdjudicationResult:
    """Result from the rule-based genealogy adjudicator."""

    verdict: str  # "match" | "possible_match" | "no_match"
    confidence: float
    reason: str
    heuristic_score: float
    criteria: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "reason": self.reason,
            "heuristic_score": self.heuristic_score,
            "criteria": self.criteria,
        }


class GenealogyAdjudicator:
    """Rule-based adjudicator using CrossMatcher heuristic scoring.

    When Generator and Reviewer disagree on a match verdict, the
    Adjudicator applies the same heuristic criteria used by CrossMatcher
    to make a deterministic ruling. No LLM calls.

    Thresholds:
        - score >= match_threshold → match
        - score >= possible_threshold → possible_match
        - below → no_match

    Conflicts (sex mismatch, impossible dates) force no_match regardless
    of score.
    """

    def __init__(
        self,
        forest: TreeForest,
        match_threshold: float = 0.75,
        possible_threshold: float = 0.50,
    ):
        """
        Initialize the adjudicator.

        Args:
            forest: TreeForest for family context lookups
            match_threshold: Minimum heuristic score for "match" verdict
            possible_threshold: Minimum score for "possible_match"
        """
        self.forest = forest
        self.matcher = CrossMatcher(forest)
        self.match_threshold = match_threshold
        self.possible_threshold = possible_threshold

    def adjudicate(
        self,
        person_a: QualifiedPerson,
        person_b: QualifiedPerson,
        generator_assessment: MatchAssessment,
        reviewer_result: ReviewResult,
    ) -> AdjudicationResult:
        """Apply heuristic criteria to resolve a Generator/Reviewer disagreement.

        Args:
            person_a: First person (from tree A)
            person_b: Second person (from tree B)
            generator_assessment: The Generator's original assessment
            reviewer_result: The Reviewer's critique

        Returns:
            AdjudicationResult with deterministic verdict
        """
        # Get heuristic scores via CrossMatcher
        candidate = self.matcher.compare(
            f"{person_a.tree_name}:{person_a.person.xref}",
            f"{person_b.tree_name}:{person_b.person.xref}",
        )

        if candidate is None:
            # Fallback: score directly if qualified xref lookup fails
            tree_a = self.forest.get_tree(person_a.tree_name)
            tree_b = self.forest.get_tree(person_b.tree_name)
            if tree_a and tree_b:
                candidate = self.matcher.score_pair(
                    person_a.person, person_b.person,
                    tree_a, tree_b,
                    person_a.tree_name, person_b.tree_name,
                )

        if candidate is None:
            return AdjudicationResult(
                verdict="possible_match",
                confidence=0.5,
                reason="Could not retrieve tree data for heuristic scoring",
                heuristic_score=0.0,
            )

        # Hard conflicts force no_match
        if candidate.conflicts:
            hard_conflicts = [
                c for c in candidate.conflicts if "sex mismatch" in c or "died" in c
            ]
            if hard_conflicts:
                return AdjudicationResult(
                    verdict="no_match",
                    confidence=0.9,
                    reason=f"Hard conflicts: {'; '.join(hard_conflicts)}",
                    heuristic_score=candidate.score,
                    criteria={
                        "name": candidate.name_score,
                        "date": candidate.date_score,
                        "place": candidate.place_score,
                        "family": candidate.family_score,
                    },
                )

        # Apply threshold-based verdict
        score = candidate.score
        if score >= self.match_threshold:
            verdict = "match"
            confidence = min(0.95, 0.7 + score * 0.25)
        elif score >= self.possible_threshold:
            verdict = "possible_match"
            confidence = 0.5 + (score - self.possible_threshold) * 0.5
        else:
            verdict = "no_match"
            confidence = 0.6 + (self.possible_threshold - score) * 0.3

        reasons = []
        reasons.append(f"Heuristic score: {score:.3f}")
        reasons.append(f"Name: {candidate.name_score:.2f}")
        reasons.append(f"Date: {candidate.date_score:.2f}")
        reasons.append(f"Place: {candidate.place_score:.2f}")
        reasons.append(f"Family: {candidate.family_score:.2f}")
        if candidate.conflicts:
            reasons.append(f"Soft conflicts: {'; '.join(candidate.conflicts)}")

        logger.info(
            f"Adjudicator ruled {verdict} for "
            f"{person_a.person.full_name} <-> {person_b.person.full_name} "
            f"(score={score:.3f})"
        )

        return AdjudicationResult(
            verdict=verdict,
            confidence=min(1.0, confidence),
            reason=" | ".join(reasons),
            heuristic_score=score,
            criteria={
                "name": candidate.name_score,
                "date": candidate.date_score,
                "place": candidate.place_score,
                "family": candidate.family_score,
            },
        )


# ─── Pipeline ─────────────────────────────────────────────────────────


@dataclass
class GRAResult:
    """Combined result from the full GRA pipeline."""

    verdict: str
    confidence: float
    generator: MatchAssessment
    reviewer: Optional[ReviewResult]
    adjudication: Optional[AdjudicationResult]
    resolved_by: str  # "generator", "consensus", or "adjudicator"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "resolved_by": self.resolved_by,
            "generator": self.generator.to_dict(),
            "reviewer": self.reviewer.to_dict() if self.reviewer else None,
            "adjudication": self.adjudication.to_dict() if self.adjudication else None,
        }


class GRAPipeline:
    """Orchestrates the Generator-Reviewer-Adjudicator entity resolution flow.

    Flow:
        1. Generator (MatchAgent) evaluates the match → MatchAssessment
        2. Reviewer critiques the assessment → ReviewResult
        3. If they agree on verdict → return consensus result
        4. If they disagree → Adjudicator applies heuristic criteria

    Example:
        >>> pipeline = GRAPipeline(generator, reviewer, adjudicator)
        >>> result = await pipeline.evaluate(person_a, person_b)
        >>> print(result.verdict, result.resolved_by)
    """

    def __init__(
        self,
        generator: MatchAgentRole,
        reviewer: MatchReviewerRole,
        adjudicator: GenealogyAdjudicator,
    ):
        self.generator = generator
        self.reviewer = reviewer
        self.adjudicator = adjudicator

    async def evaluate(
        self,
        person_a: QualifiedPerson,
        person_b: QualifiedPerson,
        context: Optional[str] = None,
    ) -> GRAResult:
        """Run the full GRA pipeline on a match candidate.

        Args:
            person_a: First person
            person_b: Second person
            context: Optional additional context for the Generator

        Returns:
            GRAResult with verdict and resolution path
        """
        # Step 1: Generator produces assessment
        assessment = await self.generator.evaluate_match(
            person_a, person_b, context
        )
        logger.info(
            f"Generator: {assessment.verdict} ({assessment.confidence:.2f}) "
            f"for {person_a.person.full_name} <-> {person_b.person.full_name}"
        )

        # Step 2: Reviewer critiques the assessment
        review = await self.reviewer.review(person_a, person_b, assessment)
        logger.info(
            f"Reviewer: {'agrees' if review.agrees else 'disagrees'} "
            f"(verdict={review.verdict}, confidence={review.confidence:.2f})"
        )

        # Step 3: Check for consensus
        if review.agrees and review.verdict == assessment.verdict:
            # Consensus — average confidence
            avg_confidence = (assessment.confidence + review.confidence) / 2
            return GRAResult(
                verdict=assessment.verdict,
                confidence=avg_confidence,
                generator=assessment,
                reviewer=review,
                adjudication=None,
                resolved_by="consensus",
            )

        # Step 4: Disagreement — invoke Adjudicator
        logger.info(
            f"Disagreement: Generator={assessment.verdict}, "
            f"Reviewer={review.verdict}. Invoking adjudicator."
        )
        adjudication = self.adjudicator.adjudicate(
            person_a, person_b, assessment, review
        )

        return GRAResult(
            verdict=adjudication.verdict,
            confidence=adjudication.confidence,
            generator=assessment,
            reviewer=review,
            adjudication=adjudication,
            resolved_by="adjudicator",
        )
