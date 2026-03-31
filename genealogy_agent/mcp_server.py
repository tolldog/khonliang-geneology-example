"""
Genealogy MCP server — extends khonliang's MCP server with tree-specific tools.

Exposes GEDCOM tree queries, ancestor chains, migration timelines, gap analysis,
and web search to external LLMs like Claude Code.

Usage:
    python -m genealogy_agent.mcp_server
    python -m genealogy_agent.mcp_server --transport http --port 8080
"""

import argparse
import json
import logging
import re

from khonliang.gateway.blackboard import Blackboard
from khonliang.knowledge.store import KnowledgeStore
from khonliang.knowledge.triples import TripleStore
from khonliang.mcp import KhonliangMCPServer

from genealogy_agent.config import load_config
from genealogy_agent.gedcom_parser import GedcomTree

logger = logging.getLogger(__name__)


class GenealogyMCPServer(KhonliangMCPServer):
    """MCP server with genealogy-specific tree tools.

    Extends the generic khonliang MCP server with GEDCOM tree queries,
    ancestor/descendant chains, migration timelines, and gap analysis.
    """

    def __init__(self, tree: GedcomTree, **kwargs):
        super().__init__(**kwargs)
        self.tree = tree

    def create_app(self):
        app = super().create_app()
        self._register_tree_tools(app)
        return app

    def _register_tree_tools(self, app) -> None:
        tree = self.tree

        @app.tool()
        def tree_summary() -> str:
            """Get a summary of the family tree (person count, families, date range)."""
            return tree.get_summary()

        @app.tool()
        def tree_search(query: str) -> str:
            """Search for persons in the family tree by name."""
            results = tree.search_persons(query)
            if not results:
                return f"No persons found matching: {query}"
            lines = [f"Found {len(results)} persons:"]
            for p in results:
                lines.append(f"  {p.display}")
            return "\n".join(lines)

        @app.tool()
        def tree_person(name: str) -> str:
            """Get detailed information about a person including family."""
            person = tree.find_person(name)
            if not person:
                return f"Person '{name}' not found in the tree."

            data = person.to_dict()
            data["parents"] = [p.to_dict() for p in tree.get_parents(person.xref)]
            data["spouses"] = [s.to_dict() for s in tree.get_spouses(person.xref)]
            data["children"] = [c.to_dict() for c in tree.get_children(person.xref)]
            data["siblings"] = [s.to_dict() for s in tree.get_siblings(person.xref)]
            return json.dumps(data, indent=2)

        @app.tool()
        def tree_ancestors(name: str, generations: int = 4) -> str:
            """Get the ancestor chain for a person."""
            person = tree.find_person(name)
            if not person:
                return f"Person '{name}' not found."

            ancestors = tree.get_ancestors(person.xref, generations=generations)
            if not ancestors:
                return f"No ancestors found for {person.full_name}."

            lines = [f"Ancestors of {person.full_name} ({len(ancestors)}):"]
            for a in ancestors:
                lines.append(f"  {a.display}")
            return "\n".join(lines)

        @app.tool()
        def tree_descendants(name: str, generations: int = 3) -> str:
            """Get descendants of a person."""
            person = tree.find_person(name)
            if not person:
                return f"Person '{name}' not found."

            descendants = tree.get_descendants(person.xref, generations=generations)
            if not descendants:
                return f"No descendants found for {person.full_name}."

            lines = [f"Descendants of {person.full_name} ({len(descendants)}):"]
            for d in descendants:
                lines.append(f"  {d.display}")
            return "\n".join(lines)

        @app.tool()
        def tree_migration(name: str, generations: int = 10) -> str:
            """Trace migration patterns through a person's ancestor line."""
            person = tree.find_person(name)
            if not person:
                return f"Person '{name}' not found."

            ancestors = tree.get_ancestors(person.xref, generations=generations)
            all_people = [person] + ancestors

            moves = []
            for p in all_people:
                year = None
                for date_str in [p.birth_date, p.death_date]:
                    if date_str:
                        m = re.search(r"\d{4}", date_str)
                        if m:
                            year = int(m.group())
                            break
                place = p.birth_place or p.death_place or ""
                if place:
                    moves.append({
                        "name": p.full_name,
                        "year": year,
                        "birth_place": p.birth_place,
                        "death_place": p.death_place,
                    })

            moves.sort(key=lambda x: x["year"] or 9999)

            lines = [f"Migration timeline for {person.full_name}'s line:\n"]
            for m in moves:
                yr = str(m["year"]) if m["year"] else "????"
                bp = m["birth_place"] or "?"
                dp = m["death_place"] or ""
                line = f"  {yr}  {m['name']}  b. {bp}"
                if dp and dp != bp:
                    line += f"  d. {dp}"
                lines.append(line)
            return "\n".join(lines)

        @app.tool()
        def tree_context(name: str) -> str:
            """Get the raw LLM context for a person (what the agent sees)."""
            person = tree.find_person(name)
            if not person:
                return f"Person '{name}' not found."
            return tree.build_context(person.xref, depth=2)

        @app.tool()
        def tree_gaps(name: str = "") -> str:
            """Analyze the tree for gaps and research opportunities."""
            from genealogy_agent.tree_analysis import TreeAnalyzer

            analyzer = TreeAnalyzer(tree)
            if name:
                return analyzer.summary(root_name=name)
            return analyzer.summary()

        @app.resource("tree://summary")
        def tree_summary_resource() -> str:
            """Family tree summary statistics."""
            return tree.get_summary()


def main():
    parser = argparse.ArgumentParser(description="Genealogy MCP server")
    parser.add_argument(
        "--transport", choices=["stdio", "http"], default="stdio"
    )
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(args.config)

    tree = GedcomTree.from_file(config["app"]["gedcom"])
    store = KnowledgeStore(config["app"].get("knowledge_db", "data/knowledge.db"))
    triples = TripleStore(config["app"].get("knowledge_db", "data/knowledge.db"))
    board = Blackboard()

    server = GenealogyMCPServer(
        tree=tree,
        knowledge_store=store,
        triple_store=triples,
        blackboard=board,
    )

    app = server.create_app()

    if args.transport == "stdio":
        logger.info("Starting genealogy MCP server (stdio)")
        app.run(transport="stdio")
    else:
        logger.info(f"Starting genealogy MCP server (http://{args.host}:{args.port})")
        app.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
