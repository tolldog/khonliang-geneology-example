# Roles and Routing

## Overview

The genealogy agent uses 4 LLM roles, each extending khonliang's `BaseRole`. A `GenealogyRouter` dispatches queries to the right role based on keywords, and a `ModelRouter` selects the appropriate model size based on query complexity.

## Roles

### ResearcherRole

The default role — answers questions about the family tree using GEDCOM data.

- **Model**: `llama3.2:3b` (fast, stays loaded 30m)
- **Context**: Tree data + session history (last 5 exchanges)
- **System prompt**: Precise about records vs uncertainty, cites sources

### FactCheckerRole

Validates genealogy data for contradictions and anomalies.

- **Model**: `qwen2.5:7b` (medium, 5m keep_alive)
- **Context**: Tree data with expanded person limit (15 vs 10)
- **System prompt**: Checks impossible dates, unreasonable age gaps, missing data

### NarratorRole

Generates readable family narratives from dry data.

- **Model**: `llama3.1:8b` (larger, 5m keep_alive)
- **Context**: Tree data + knowledge store (previously researched facts)
- **System prompt**: 6 strict rules against fabrication, labels inferences

### MatchAgentRole

Dedicated cross-tree person matching specialist.

- **Model**: `qwen2.5:7b` (shares with fact_checker)
- **Context**: Side-by-side person comparison with family data from both trees
- **System prompt**: Evaluates name variants, date tolerance, place overlap, family structure
- **Output**: Structured `MatchAssessment` (VERDICT/CONFIDENCE/EVIDENCE/CONFLICTS)

## Routing

`GenealogyRouter` extends `BaseRouter` with keyword dispatch:

| Keywords | Role | Examples |
|----------|------|----------|
| check, validate, verify, contradiction, wrong, error | `fact_checker` | "check this date", "is this accurate?" |
| story, narrative, tell me about, biography, describe | `narrator` | "tell me about the Smith family" |
| *(everything else)* | `researcher` | "who were John's parents?" |

## Model Routing

`ModelRouter` with `ComplexityStrategy` classifies query complexity using the fast researcher model:
- **Simple** lookups stay on `llama3.2:3b`
- **Complex** narratives can escalate to larger models

## Context Building

All roles share `_build_multi_context()` which uses a multi-strategy approach:

1. **Broad search**: Extract name-like words, search for all matching persons
2. **Multi-word names**: Try 2-3 word name combinations
3. **Fallback**: Tree summary if no matches found

Session context is injected via `contextvars` for async safety — each WebSocket connection gets isolated conversation history.

## Heuristic Injection

All roles accept an optional `heuristic_pool` parameter. When present, `_effective_system_prompt()` appends learned rules:

```text
[LEARNED PATTERNS]
- Always cite primary sources for date claims (90% confidence)
- Check census records for migration patterns (85% confidence)
```

Rules are extracted from evaluation outcomes (success/failure by role and query type) via `HeuristicPool.build_prompt_context()`.

## Personality Routing

Users can route queries to specific personas via `@mention`:

| Persona | Focus | Example |
|---------|-------|---------|
| `@genealogist` | Primary sources, vital records | `@genealogist find records for John Smith` |
| `@historian` | Historical context, migrations | `@historian what was life like in 1850s Illinois?` |
| `@detective` | Brick walls, creative strategies | `@detective help with this dead end` |
| `@skeptic` | Source reliability, challenges | `@skeptic is this birth record trustworthy?` |

Personality routing uses the researcher model with the persona's custom system prompt, then formats the response with persona headers.
