"""Genealogy Agent — LLM-backed genealogy research tool powered by khonliang."""

__version__ = "0.1.0"

from genealogy_agent.gedcom_parser import GedcomTree, Person, Family
from genealogy_agent.config import load_config
from genealogy_agent.router import GenealogyRouter
from genealogy_agent.roles import ResearcherRole, FactCheckerRole, NarratorRole
from genealogy_agent.self_eval import create_genealogy_evaluator
from genealogy_agent.consensus import (
    GenealogyVotingAgent,
    create_consensus_team,
    create_debate_orchestrator,
    create_match_consensus_team,
)
from genealogy_agent.personalities import create_genealogy_registry
from genealogy_agent.forest import TreeForest, QualifiedPerson, load_forest_from_config
from genealogy_agent.cross_matcher import CrossMatcher, MatchCandidate
from genealogy_agent.match_agent import MatchAgentRole, MatchVotingAgent, MatchAssessment
from genealogy_agent.importer import GedcomImporter, ImportResult
from genealogy_agent.merge import MergeEngine, MergeResult

__all__ = [
    # Core
    "GedcomTree",
    "Person",
    "Family",
    "load_config",
    # Multi-tree
    "TreeForest",
    "QualifiedPerson",
    "load_forest_from_config",
    # Roles & routing
    "GenealogyRouter",
    "ResearcherRole",
    "FactCheckerRole",
    "NarratorRole",
    # Evaluation
    "create_genealogy_evaluator",
    # Consensus & debate
    "GenealogyVotingAgent",
    "create_consensus_team",
    "create_debate_orchestrator",
    "create_match_consensus_team",
    # Matching
    "CrossMatcher",
    "MatchCandidate",
    "MatchAgentRole",
    "MatchVotingAgent",
    "MatchAssessment",
    # Import/Export & Merge
    "GedcomImporter",
    "ImportResult",
    "MergeEngine",
    "MergeResult",
    # Personalities
    "create_genealogy_registry",
]
