# khonliang-genealogy-example

LLM-backed genealogy research tool — example project demonstrating [ollama-khonliang](https://github.com/tolldog/ollama-khonliang) as a dependency.

## Features

- **GEDCOM parser** — loads standard genealogy files (2,067 persons, 605 families tested)
- **3 LLM roles** — researcher, fact checker, narrator with smart context injection
- **Web search** — DDG + Google + Bing in parallel with relevance filtering
- **Research pool** — threaded web + tree researchers with capability routing
- **Knowledge management** — three-tier RAG (axioms, imported, derived from interactions)
- **WebSocket chat server** — web UI + CLI client + tool interface
- **Config-driven** — YAML config for ports, models, themes

## Quick Start

```bash
# Install (requires Ollama running locally)
pip install -e .

# Load and explore a GEDCOM file
genealogy load data/family.ged
genealogy search data/family.ged "Smith"
genealogy chat data/family.ged

# Start the chat server (WebSocket + Web UI)
python -m genealogy_agent.server

# Connect via CLI
python -m genealogy_agent.chat_client

# Or open http://localhost:8766 in a browser
```

## Chat Commands

| Command | Description |
|---------|-------------|
| Regular text | Routes to researcher/narrator/fact_checker |
| `!lookup: name` | Web search across DDG + Google + Bing |
| `!search: query` | General web search |
| `!google: query` | Google search |
| `!fetch: url` | Fetch and extract page content |
| `!fetch ingest url` | Fetch and save to knowledge |
| `!tree: name` | Structured tree data lookup |
| `!ancestors: name` | Ancestor chain |
| `!migration: name` | Migration timeline |
| `!ingest title \| content` | Add to knowledge (Tier 2) |
| `!ingest-file path` | Ingest a file |
| `!knowledge` | Show knowledge store status |
| `!axiom` | List/set axioms (Tier 1) |
| `!prune` | Clean up low-quality knowledge |
| `/search query` | Search knowledge base |
| `/rate 1-5` | Rate last response |

## Tool Interface (for external LLMs)

```bash
python -m genealogy_agent.tool summary
python -m genealogy_agent.tool person "Timothy Toll"
python -m genealogy_agent.tool ancestors "Timothy Toll" --generations 4
python -m genealogy_agent.tool migration "Timothy Toll"
python -m genealogy_agent.tool websearch "Roger Tolle"
python -m genealogy_agent.tool query "Who were Timothy's grandparents?"
```

## Configuration

Edit `config.yaml`:

```yaml
server:
  ws_port: 8765
  web_port: 8766

app:
  title: "Genealogy Agent"
  gedcom: "data/family.ged"

ollama:
  models:
    researcher: "llama3.2:3b"
    fact_checker: "qwen2.5:7b"
    narrator: "llama3.1:8b"

theme:
  primary: "#e94560"
  background: "#1a1a2e"
```

## Architecture

```
User (browser/CLI/tool)
  → WebSocket Chat Server
    → ResearchTrigger (! commands)
    → Router (keyword matching)
    → Specialist Role (LLM inference)
    → Librarian (auto-indexes responses to Tier 3)
    → Research Pool (web search, tree lookups)
```

## License

MIT
