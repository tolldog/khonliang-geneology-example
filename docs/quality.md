# Quality: Evaluation, Consensus, and Learning

## Self-Evaluation

Every LLM response is checked against the family tree data before being returned to the user. The evaluator uses four rules:

### DateCheckRule

Extracts date claims from the response (regex: `Name born/died YYYY`) and verifies against the GEDCOM tree. A mismatch of more than 5 years flags a `date_mismatch` issue with severity `high`.

### RelationshipCheckRule

Extracts parent claims (regex: `Name's father/mother was OtherName`) and verifies against tree data. Wrong relationships are flagged with severity `high`.

### SpeculationRule (from khonliang)

Detects excessive hedging language ("perhaps", "maybe", "it's possible"). Flags `excessive_speculation` when too many phrases appear.

### UncertaintyRule (from khonliang)

Detects uncertainty indicators ("I don't have information", "unclear"). Flags `uncertainty` and auto-queues background research.

## Consensus Voting

When self-evaluation finds **high-severity** issues (date mismatches, wrong relationships), the consensus system convenes all roles as voting agents:

```text
Response with issues
  → GenealogyVotingAgent(researcher).analyze()  → AgentVote
  → GenealogyVotingAgent(fact_checker).analyze() → AgentVote
  → GenealogyVotingAgent(narrator).analyze()    → AgentVote
  → ConsensusEngine.calculate_consensus()        → ConsensusResult
```

**Weights**: fact_checker 0.40, researcher 0.35, narrator 0.25.

If the consensus action is `reject`, corrections from the rejecting agents are appended to the response. If `approve`, the original caveat is used.

## Debate Orchestration

When consensus votes show high-confidence disagreement, structured debate runs:

1. `DebateOrchestrator.detect_disagreement()` finds the two highest-confidence disagreeing votes
2. `build_challenge()` creates a challenge prompt for the target agent
3. Target agent's `reconsider()` method evaluates the challenge
4. Up to `debate_rounds` (default 2) rounds run
5. `ConsensusEngine.calculate_consensus()` re-aggregates the updated votes

## Match Consensus

For cross-tree match disputes, a specialized consensus team runs:

```text
Match candidate
  → MatchVotingAgent.analyze()       → AgentVote (weight 0.55)
  → GenealogyVotingAgent(fact_checker).analyze() → AgentVote (weight 0.45)
  → ConsensusEngine.calculate_consensus()
```

The MatchAgent gets dominant weight since it's the domain specialist.

## Heuristic Learning

Every evaluation outcome is recorded in `HeuristicPool`:

```python
heuristic_pool.record_outcome(
    action="respond_as_researcher",
    result="success" if evaluation.passed else "failure",
    context={"query_type": "lookup", "role": "researcher"},
    details={"confidence": 0.85, "issues": 0},
)
```

Over time, the pool extracts patterns (e.g., "researcher role succeeds 95% on lookup queries but fails 30% on narrative queries"). These rules are injected into role system prompts via `build_prompt_context()`.

## Feedback Loop

Users rate responses via `/rate 1-5 [feedback]`. All interactions and ratings are stored in `FeedbackStore`:

```text
User query → LLM response → Self-eval → Consensus (if needed)
                                                      ↓
                                         FeedbackStore.log_interaction()
                                                      ↓
                                         /rate N → FeedbackStore.add_feedback()
                                                      ↓
                                         HeuristicPool.record_outcome()
                                                      ↓
                                         Rules injected into future prompts
```

## Auto-Research

When the evaluator detects `uncertainty` (agent said "I don't have info") or `date_mismatch`, background research is automatically queued:

- **Uncertainty**: Queues `web_search` task for the query
- **Date mismatch**: Queues `person_lookup` task to verify
