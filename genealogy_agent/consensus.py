"""
Consensus voting and debate for genealogy responses.

When self-evaluation flags high-severity issues, the consensus system
convenes voting agents (wrappers around existing roles) to evaluate
the response and optionally debate disagreements.
"""

import logging
from typing import Dict, List, Optional

from khonliang.consensus import AgentTeam, AgentVote, ConsensusEngine
from khonliang.debate import DebateConfig, DebateOrchestrator

logger = logging.getLogger(__name__)


class GenealogyVotingAgent:
    """Wraps an existing BaseRole as a voting agent for consensus.

    Implements the protocol expected by AgentTeam (agent_id + analyze)
    and DebateOrchestrator (reconsider).
    """

    def __init__(self, role, role_name: str, tree):
        self._role = role
        self._agent_id = role_name
        self.tree = tree

    @property
    def agent_id(self) -> str:
        return self._agent_id

    async def analyze(self, subject: str, context: Optional[Dict] = None) -> AgentVote:
        """Analyze a response and vote on its quality."""
        context = context or {}
        original_response = context.get("original_response", "")
        query = context.get("query", "")
        issues = context.get("eval_issues", [])

        prompt = (
            f"Original question: {query}\n\n"
            f"Proposed answer: {original_response}\n\n"
            f"Issues found by evaluator: {issues}\n\n"
            f"Evaluate this answer for accuracy. Consider whether the issues "
            f"are genuine errors or false positives. "
            f"Say APPROVE if the answer is acceptable despite the issues, "
            f"or REJECT if it contains real errors that need correction. "
            f"Explain your reasoning."
        )

        result = await self._role.handle(prompt, session_id="consensus")
        response = result.get("response", "")
        response_lower = response.lower()

        if any(w in response_lower for w in ["reject", "incorrect", "wrong", "error", "inaccurate"]):
            action = "reject"
            confidence = 0.8
        elif any(w in response_lower for w in ["approve", "correct", "accurate", "acceptable"]):
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

    async def reconsider(self, original_vote: AgentVote, debate_context, round_num: int) -> AgentVote:
        """Reconsider vote after a debate challenge."""
        challenge = debate_context.payload.get("challenge", "")

        prompt = (
            f"You previously voted {original_vote.action} with reasoning:\n"
            f'"{original_vote.reasoning}"\n\n'
            f"A colleague challenges your position:\n{challenge}\n\n"
            f"Reconsider your position. Vote APPROVE or REJECT with updated reasoning."
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


def create_voting_agents(roles: dict, tree) -> List[GenealogyVotingAgent]:
    """Create voting agent wrappers from existing roles."""
    return [
        GenealogyVotingAgent(role, name, tree)
        for name, role in roles.items()
    ]


def create_consensus_team(roles: dict, tree, config: dict) -> AgentTeam:
    """Create an AgentTeam from existing roles."""
    agents = create_voting_agents(roles, tree)

    engine = ConsensusEngine(
        agent_weights={
            "fact_checker": 0.40,
            "researcher": 0.35,
            "narrator": 0.25,
        },
        veto_blocks=True,
        min_confidence=0.5,
    )

    consensus_cfg = config.get("consensus", {})
    return AgentTeam(
        agents=agents,
        consensus_engine=engine,
        agent_timeout=consensus_cfg.get("timeout", 30),
    )


def create_match_consensus_team(match_agent, fact_checker_agent, config: dict) -> AgentTeam:
    """Create a consensus team specifically for match disputes.

    MatchAgent gets dominant weight since it's the domain specialist.
    """
    engine = ConsensusEngine(
        agent_weights={
            "match_agent": 0.55,
            "fact_checker": 0.45,
        },
        veto_blocks=True,
        min_confidence=0.5,
    )

    consensus_cfg = config.get("consensus", {})
    return AgentTeam(
        agents=[match_agent, fact_checker_agent],
        consensus_engine=engine,
        agent_timeout=consensus_cfg.get("timeout", 30),
    )


def create_debate_orchestrator(
    roles: dict, tree, config: dict
) -> DebateOrchestrator:
    """Create a DebateOrchestrator from existing roles."""
    agents = create_voting_agents(roles, tree)
    agent_map = {a.agent_id: a for a in agents}

    consensus_cfg = config.get("consensus", {})
    debate_config = DebateConfig(
        disagreement_threshold=consensus_cfg.get("disagreement_threshold", 0.6),
        max_rounds=consensus_cfg.get("debate_rounds", 2),
        enabled=consensus_cfg.get("debate_enabled", True),
    )

    return DebateOrchestrator(agents=agent_map, config=debate_config)
