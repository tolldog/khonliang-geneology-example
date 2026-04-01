# Cross-Tree Matching

## Overview

The matching system identifies the same person across different GEDCOM trees. It uses a two-stage approach: fast heuristic scoring (`CrossMatcher`) followed by LLM-backed evaluation (`MatchAgentRole`) for ambiguous candidates. Results are stored as `TripleStore` triples for persistence.

## Heuristic Scoring (CrossMatcher)

`CrossMatcher.scan()` compares all persons across two trees using weighted scoring:

| Dimension | Weight | Method |
|-----------|--------|--------|
| Name | 40% | Exact surname match (0.6 base) + given name similarity (prefix/initial matching) |
| Date | 25% | Birth year proximity: same year = 1.0, -0.1 per year difference |
| Place | 20% | Token intersection on birth/death places (comma-split, case-insensitive) |
| Family | 15% | Spouse surname overlap, parent surname overlap, child count similarity |

**Performance**: Surname pre-filtering reduces the O(n*m) comparison space. Only persons sharing a surname are compared.

**Conflict detection**: Sex mismatches, impossible date overlaps (death before other's birth), and large birth year gaps (>20 years) are flagged. Each conflict applies a 15% penalty (up to 50% max).

```python
from genealogy_agent.cross_matcher import CrossMatcher
from genealogy_agent.forest import TreeForest

forest = TreeForest()
forest.load("toll", "data/toll.ged")
forest.load("smith", "data/smith.ged")

matcher = CrossMatcher(forest)
candidates = matcher.scan("toll", "smith", min_score=0.6, max_results=50)

for c in candidates:
    print(f"{c.person_a.person.full_name} <-> {c.person_b.person.full_name}")
    print(f"  Score: {c.score:.0%} (name={c.name_score:.0%}, date={c.date_score:.0%})")
    if c.conflicts:
        print(f"  Conflicts: {c.conflicts}")
```

## LLM Evaluation (MatchAgent)

`MatchAgentRole` extends `BaseRole` with a system prompt tuned for genealogical record comparison. It produces structured `MatchAssessment` output:

```
VERDICT: match | possible_match | no_match
CONFIDENCE: 0.0-1.0
EVIDENCE: bullet list of supporting facts
CONFLICTS: bullet list of conflicting facts
RECOMMENDATION: link | review | skip
REASONING: explanation
```

The `_build_comparison_prompt()` method generates a side-by-side comparison with full family context (parents, spouses, children) from both trees.

```python
from genealogy_agent.match_agent import MatchAgentRole

match_role = MatchAgentRole(model_pool, forest=forest, triple_store=triples)
assessment = await match_role.evaluate_match(person_a, person_b)

print(f"Verdict: {assessment.verdict} ({assessment.confidence:.0%})")
print(f"Recommendation: {assessment.recommendation}")
```

## Consensus Integration

For disputed matches (heuristic score 0.5-0.8), the system can convene a consensus team:

```python
from genealogy_agent.consensus import create_match_consensus_team
from genealogy_agent.match_agent import MatchVotingAgent

match_voting = MatchVotingAgent(match_role)
fact_checker_voting = GenealogyVotingAgent(fact_checker_role, "fact_checker", tree)

team = create_match_consensus_team(match_voting, fact_checker_voting, config)
result = await team.evaluate(subject="Compare these records", context={...})
```

Weights: MatchAgent 55%, FactChecker 45%. VETO support enabled.

## Link Storage (TripleStore)

Matches are stored as semantic triples:

```
Subject:    "toll:@I42@"         (qualified xref)
Predicate:  "same_as"            (confirmed) or "possible_match" (candidate)
Object:     "smith:@I17@"        (qualified xref)
Confidence: 0.0 - 1.0
Source:     "cross_matcher" | "match_agent" | "user_confirmed"
```

The `!link` command upgrades `possible_match` to `same_as` with confidence 1.0.

## Merge

Once a match is confirmed, `MergeEngine` can merge person data:

```python
from genealogy_agent.merge import MergeEngine

engine = MergeEngine(forest, triple_store=triples)
result = engine.merge_person("toll:@I42@", "smith:@I17@", strategy="prefer_target")
```

Strategies:
- **prefer_target**: Fill empty fields only (safest)
- **prefer_source**: Overwrite target with source values
- **merge_all**: Keep target values, report conflicts for differing fields

Merge provenance recorded as `merged_into` triple.

## Chat Commands

```
!trees                         — List loaded trees
!load name path.ged            — Import with sanity checking
!scan [tree_a] [tree_b]        — Heuristic scan + top-5 LLM evaluation
!matches [name]                — Show pending and confirmed matches
!link tree:@I1@ tree:@I2@      — Confirm a match
!merge source into target      — Merge person data
!import path.ged [name]        — Import with full pipeline
!export tree_name [path]       — Export to GEDCOM
```

## Import with Sanity Checking

`GedcomImporter.import_file()` validates before adding to the forest:

1. Parse GEDCOM into temporary `GedcomTree`
2. Run `TreeAnalyzer.find_date_anomalies()` — high-severity (death before birth, impossible ages) blocks import
3. Run `TreeAnalyzer.find_missing_data()` — warnings only
4. If passes, add to `TreeForest`

```python
from genealogy_agent.importer import GedcomImporter

importer = GedcomImporter(forest)
result = importer.import_file("data/new_tree.ged", name="new")
print(result.display)  # shows status, counts, issues, warnings
```
