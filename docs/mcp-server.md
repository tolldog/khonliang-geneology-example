# MCP Server

## Overview

The genealogy MCP server extends khonliang's `KhonliangMCPServer` with domain-specific tools for tree queries, cross-tree matching, import/export, and training feedback. It exposes the full genealogy research capability to external LLMs like Claude Code.

## Setup

The `.mcp.json` is pre-configured for Claude Code:

```json
{
  "mcpServers": {
    "genealogy": {
      "command": "python3",
      "args": ["-m", "genealogy_agent.mcp_server", "--config", "config.yaml"],
      "cwd": "/path/to/genealogy"
    }
  }
}
```

Manual usage:

```bash
# Stdio transport (for Claude Code)
python -m genealogy_agent.mcp_server

# HTTP transport (for remote access)
python -m genealogy_agent.mcp_server --transport http --port 8080
```

## Tree Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `tree_summary` | — | Tree statistics (person count, families, date range) |
| `tree_search` | `query: str` | Search persons by name |
| `tree_person` | `name: str` | Detailed person info with family (parents, spouses, children, siblings) |
| `tree_ancestors` | `name: str, generations: int = 4` | Ancestor chain |
| `tree_descendants` | `name: str, generations: int = 3` | Descendant chain |
| `tree_migration` | `name: str, generations: int = 10` | Migration timeline through ancestor line |
| `tree_context` | `name: str` | Raw LLM context (what the agent sees for this person) |
| `tree_gaps` | `name: str = ""` | Gap analysis and research opportunities |

## Forest Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `forest_list` | — | List all loaded trees with person/family counts |
| `forest_search` | `query: str, tree_name: str = ""` | Search across all trees (or a specific tree) |
| `match_scan` | `tree_a: str, tree_b: str, min_score: float = 0.6` | Heuristic cross-matching between two trees |
| `match_confirm` | `xref_a: str, xref_b: str` | Confirm a match (stores same_as triple) |
| `import_gedcom` | `path: str, name: str = ""` | Import GEDCOM with sanity checking |
| `export_gedcom` | `tree_name: str, path: str = ""` | Export tree to GEDCOM format |

## Training Tools

| Tool | Parameters | Description |
|------|-----------|-------------|
| `feedback_stats` | — | Interaction and feedback statistics |
| `heuristic_list` | — | Learned rules from evaluation outcomes |
| `personality_list` | — | Available @mention personalities |

## Khonliang Tools (inherited)

| Tool | Description |
|------|-------------|
| `knowledge_search` | Search the knowledge store |
| `knowledge_ingest` | Add content to knowledge |
| `triple_add` | Add a semantic triple |
| `blackboard_post` | Post to shared blackboard |
| `invoke_role` | Invoke a specific role |

## Resources

| Resource URI | Description |
|-------------|-------------|
| `tree://summary` | Family tree summary statistics |
| `knowledge://axioms` | Knowledge axioms (from khonliang) |
| `blackboard://sections` | Blackboard sections (from khonliang) |
