# Getting Started

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally with at least one model pulled
- A GEDCOM family tree file (exported from Ancestry, FamilySearch, MyHeritage, etc.)

## Installation

```bash
# Clone and install
git clone https://github.com/tolldog/khonliang-genealogy-example.git
cd khonliang-genealogy-example

# Install (includes khonliang dependency)
pip install -e .

# If khonliang imports fail, install it directly
pip install -e ../ollama-khonliang
```

## Pull Ollama Models

The default config uses three models:

```bash
ollama pull llama3.2:3b     # Researcher (fast Q&A, intent classification)
ollama pull qwen2.5:7b      # Fact checker + match agent
ollama pull llama3.1:8b     # Narrator (family stories)
```

## Quick Start

### 1. Place Your GEDCOM File

```bash
mkdir -p data
cp /path/to/your/family.ged data/
```

### 2. Start the Chat Server

```bash
# Single tree
GEDCOM_FILE=data/family.ged python -m genealogy_agent.server

# Multiple trees
GEDCOM_FILES="toll=data/toll.ged,smith=data/smith.ged" python -m genealogy_agent.server
```

### 3. Connect

- **Web UI**: Open http://localhost:8766
- **CLI**: `python -m genealogy_agent.chat_client`

### 4. Try Some Commands

```
# Natural language
Who were John Smith's grandparents?
Tell me a story about the Smith family migration

# Research commands
!lookup John Smith
!ancestors John Smith
!migration John Smith

# Tree analysis
!gaps
!dead-ends John Smith
!anomalies

# Personalities
@skeptic Is this birth record reliable?
@historian What was life like in Springfield in the 1850s?

# Multi-tree
!trees
!scan toll smith
!matches

# Feedback
/rate 5 Great answer!
```

## Using as MCP Server (Claude Code)

The `.mcp.json` is pre-configured. Start Claude Code in the project directory and the MCP tools are available automatically.

```bash
# Or run manually
python -m genealogy_agent.mcp_server
```

Available MCP tools:

| Category | Tools |
|----------|-------|
| Tree | `tree_summary`, `tree_search`, `tree_person`, `tree_ancestors`, `tree_descendants`, `tree_migration`, `tree_context`, `tree_gaps` |
| Forest | `forest_list`, `forest_search`, `match_scan`, `match_confirm`, `import_gedcom`, `export_gedcom` |
| Training | `feedback_stats`, `heuristic_list`, `personality_list` |
| Khonliang | `knowledge_search`, `knowledge_ingest`, `triple_add`, `blackboard_post` |

## Using as CLI Tool

```bash
genealogy search data/family.ged "Smith"
genealogy person data/family.ged "John Smith"
genealogy ancestors data/family.ged "John Smith" --generations 4
genealogy migration data/family.ged "John Smith"
genealogy gaps data/family.ged
```

## Configuration

Edit `config.yaml` to customize ports, models, themes, and feature toggles. See `docs/architecture.md` for the full configuration reference.
