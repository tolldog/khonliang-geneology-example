"""
MatchAgent — dedicated LLM role for evaluating genealogical record matches.

Extends BaseRole with a match-specific system prompt and structured
output parsing. Integrates with consensus via MatchVotingAgent wrapper.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from khonliang.consensus import AgentVote
from khonliang.roles.base import BaseRole

from genealogy_agent.forest import QualifiedPerson, TreeForest

logger = logging.getLogger(__name__)


@dataclass
class MatchAssessment:
    """Structured result from MatchAgent evaluation."""

    confidence: float
    verdict: str         # "match" | "possible_match" | "no_match"
    evidence: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    recommendation: str = "review"  # "link" | "review" | "skip"
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "conflicts": self.conflicts,
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
        }


class MatchAgentRole(BaseRole):
    """Dedicated agent for evaluating genealogical record matches.

    Has its own system prompt focused on comparing person records
    side-by-side and producing structured match assessments.
    """

    def __init__(self, model_pool, forest: TreeForest, triple_store=None,
                 heuristic_pool=None, **kwargs):
        super().__init__(role="match_agent", model_pool=model_pool, **kwargs)
        self.forest = forest
        self.triple_store = triple_store
        self.heuristic_pool = heuristic_pool
        self._system_prompt = (
            "You are a genealogy record matching specialist. Your job is to "
            "evaluate whether two person records from different family trees "
            "refer to the same historical individual.\n\n"
            "EVALUATION CRITERIA:\n"
            "1. Name similarity — exact match, spelling variants (Wm/William, "
            "   Eliz/Elizabeth), maiden vs married names\n"
            "2. Date consistency — birth/death years within reasonable tolerance "
            "   (records often differ by 1-3 years)\n"
            "3. Place overlap — same region, county, or specific location\n"
            "4. Family structure — matching spouse surnames, parent names, "
            "   number of children\n"
            "5. Conflicts — sex mismatch, impossible date overlaps, different "
            "   parents with same surname\n\n"
            "OUTPUT FORMAT (use exactly these labels):\n"
            "VERDICT: match | possible_match | no_match\n"
            "CONFIDENCE: 0.0-1.0\n"
            "EVIDENCE: bullet list of supporting facts\n"
            "CONFLICTS: bullet list of conflicting facts (or 'none')\n"
            "RECOMMENDATION: link | review | skip\n"
            "REASONING: brief explanation of your conclusion"
        )

    def _effective_system_prompt(self) -> str:
        prompt = self.system_prompt
        if self.heuristic_pool:
            rules = self.heuristic_pool.build_prompt_context(
                max_rules=3, min_confidence=0.6
            )
            if rules:
                prompt = f"{prompt}\n\n[LEARNED PATTERNS]\n{rules}"
        return prompt

    async def handle(
        self,
        message: str,
        session_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Standard role handle — interprets message as match request."""
        response, elapsed_ms = await self._timed_generate(
            prompt=message, system=self._effective_system_prompt()
        )
        return {
            "response": response.strip(),
            "metadata": {
                "role": self.role,
                "generation_time_ms": elapsed_ms,
            },
        }

    async def evaluate_match(
        self,
        person_a: QualifiedPerson,
        person_b: QualifiedPerson,
        context: Optional[str] = None,
    ) -> MatchAssessment:
        """Evaluate a specific match candidate with full context."""
        prompt = self._build_comparison_prompt(person_a, person_b, context)
        response, _ = await self._timed_generate(
            prompt=prompt, system=self._effective_system_prompt()
        )
        return self._parse_assessment(response)

    def _build_comparison_prompt(
        self,
        person_a: QualifiedPerson,
        person_b: QualifiedPerson,
        context: Optional[str] = None,
    ) -> str:
        """Build side-by-side comparison prompt with family context."""
        pa = person_a.person
        pb = person_b.person

        lines = [
            "Compare these two person records and determine if they refer "
            "to the same individual.\n",
            f"=== PERSON A (from tree '{person_a.tree_name}') ===",
            f"Name: {pa.full_name}",
            f"Sex: {pa.sex or 'unknown'}",
            f"Birth: {pa.birth_date or 'unknown'}, {pa.birth_place or 'unknown'}",
            f"Death: {pa.death_date or 'unknown'}, {pa.death_place or 'unknown'}",
            f"Occupation: {pa.occupation or 'unknown'}",
        ]

        # Family context for person A
        tree_a = self.forest.get_tree(person_a.tree_name)
        if tree_a:
            parents = tree_a.get_parents(pa.xref)
            if parents:
                lines.append(
                    f"Parents: {', '.join(p.full_name for p in parents)}"
                )
            spouses = tree_a.get_spouses(pa.xref)
            if spouses:
                lines.append(
                    f"Spouse(s): {', '.join(s.full_name for s in spouses)}"
                )
            children = tree_a.get_children(pa.xref)
            if children:
                lines.append(
                    f"Children ({len(children)}): "
                    f"{', '.join(c.full_name for c in children[:5])}"
                )

        lines.append("")
        lines.append(f"=== PERSON B (from tree '{person_b.tree_name}') ===")
        lines.append(f"Name: {pb.full_name}")
        lines.append(f"Sex: {pb.sex or 'unknown'}")
        lines.append(f"Birth: {pb.birth_date or 'unknown'}, {pb.birth_place or 'unknown'}")
        lines.append(f"Death: {pb.death_date or 'unknown'}, {pb.death_place or 'unknown'}")
        lines.append(f"Occupation: {pb.occupation or 'unknown'}")

        tree_b = self.forest.get_tree(person_b.tree_name)
        if tree_b:
            parents = tree_b.get_parents(pb.xref)
            if parents:
                lines.append(
                    f"Parents: {', '.join(p.full_name for p in parents)}"
                )
            spouses = tree_b.get_spouses(pb.xref)
            if spouses:
                lines.append(
                    f"Spouse(s): {', '.join(s.full_name for s in spouses)}"
                )
            children = tree_b.get_children(pb.xref)
            if children:
                lines.append(
                    f"Children ({len(children)}): "
                    f"{', '.join(c.full_name for c in children[:5])}"
                )

        if context:
            lines.append(f"\n[ADDITIONAL CONTEXT]\n{context}")

        lines.append("\nProvide your assessment:")
        return "\n".join(lines)

    def _parse_assessment(self, response: str) -> MatchAssessment:
        """Parse structured LLM output into MatchAssessment."""
        verdict = "possible_match"
        confidence = 0.5
        evidence = []
        conflicts = []
        recommendation = "review"
        reasoning = response

        # Extract VERDICT
        m = re.search(r"VERDICT:\s*(match|possible_match|no_match)", response, re.I)
        if m:
            verdict = m.group(1).lower()

        # Extract CONFIDENCE
        m = re.search(r"CONFIDENCE:\s*([\d.]+)", response)
        if m:
            try:
                confidence = max(0.0, min(1.0, float(m.group(1))))
            except ValueError:
                pass

        # Extract EVIDENCE bullet points
        m = re.search(r"EVIDENCE:(.*?)(?=CONFLICTS:|RECOMMENDATION:|REASONING:|$)",
                       response, re.S | re.I)
        if m:
            evidence = [
                line.strip().lstrip("- ").lstrip("* ")
                for line in m.group(1).strip().split("\n")
                if line.strip() and line.strip() not in ("none", "None")
            ]

        # Extract CONFLICTS bullet points
        m = re.search(r"CONFLICTS:(.*?)(?=RECOMMENDATION:|REASONING:|$)",
                       response, re.S | re.I)
        if m:
            conflicts = [
                line.strip().lstrip("- ").lstrip("* ")
                for line in m.group(1).strip().split("\n")
                if line.strip() and line.strip().lower() not in ("none", "n/a")
            ]

        # Extract RECOMMENDATION
        m = re.search(r"RECOMMENDATION:\s*(link|review|skip)", response, re.I)
        if m:
            recommendation = m.group(1).lower()

        # Extract REASONING
        m = re.search(r"REASONING:(.*?)$", response, re.S | re.I)
        if m:
            reasoning = m.group(1).strip()

        return MatchAssessment(
            confidence=confidence,
            verdict=verdict,
            evidence=evidence,
            conflicts=conflicts,
            recommendation=recommendation,
            reasoning=reasoning[:500],
        )


class MatchVotingAgent:
    """Wraps MatchAgentRole for consensus integration.

    Follows the same protocol as GenealogyVotingAgent so it can
    participate in AgentTeam and DebateOrchestrator.
    """

    def __init__(self, match_role: MatchAgentRole):
        self._role = match_role
        self._agent_id = "match_agent"

    @property
    def agent_id(self) -> str:
        return self._agent_id

    async def analyze(self, subject: str, context: Optional[Dict] = None) -> AgentVote:
        """Evaluate a match and vote approve/reject."""
        context = context or {}

        result = await self._role.handle(subject, session_id="consensus")
        response = result.get("response", "")
        response_lower = response.lower()

        # Check reject first — "no match" contains "match" so order matters
        if any(w in response_lower for w in ["no match", "not a match", "different", "reject", "skip"]):
            action = "reject"
            confidence = 0.8
        elif any(w in response_lower for w in ["match", "same person", "approve", "link"]):
            action = "approve"
            confidence = 0.8
        else:
            action = "defer"
            confidence = 0.5

        return AgentVote(
            agent_id=self.agent_id,
            action=action,
            confidence=confidence,
            reasoning=response[:300],
        )

    async def reconsider(
        self, original_vote: AgentVote, debate_context, round_num: int
    ) -> AgentVote:
        """Reconsider after debate challenge."""
        challenge = debate_context.payload.get("challenge", "")
        prompt = (
            f"You previously voted {original_vote.action} with reasoning:\n"
            f'"{original_vote.reasoning}"\n\n'
            f"Challenge: {challenge}\n\n"
            f"Reconsider your position on whether these records match."
        )

        result = await self._role.handle(prompt, session_id="debate")
        response = result.get("response", "")
        response_lower = response.lower()

        action = original_vote.action
        if "approve" in response_lower and original_vote.action != "approve":
            action = "approve"
        elif "reject" in response_lower and original_vote.action != "reject":
            action = "reject"

        return AgentVote(
            agent_id=self.agent_id,
            action=action,
            confidence=min(1.0, original_vote.confidence + 0.05),
            reasoning=response[:300],
        )
