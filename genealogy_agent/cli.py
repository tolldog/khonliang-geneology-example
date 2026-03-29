"""
Genealogy agent CLI — interactive family tree research.

Usage:
    genealogy load family.ged
    genealogy chat family.ged
    genealogy query family.ged "who were John Smith's parents?"
    genealogy summary family.ged
    genealogy search family.ged "Smith"
"""

import argparse
import asyncio
import logging
import os
import sys

from khonliang import ModelPool

from genealogy_agent.gedcom_parser import GedcomTree
from genealogy_agent.roles import FactCheckerRole, NarratorRole, ResearcherRole
from genealogy_agent.router import GenealogyRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def build_service(tree: GedcomTree, ollama_url: str):
    """Build the genealogy agent service."""
    pool = ModelPool(
        {
            "researcher": "llama3.2:3b",
            "fact_checker": "qwen2.5:7b",
            "narrator": "llama3.1:8b",
        },
        base_url=ollama_url,
    )

    roles = {
        "researcher": ResearcherRole(pool, tree=tree),
        "fact_checker": FactCheckerRole(pool, tree=tree),
        "narrator": NarratorRole(pool, tree=tree),
    }

    router = GenealogyRouter()
    return roles, router


async def handle_query(message: str, roles: dict, router: GenealogyRouter) -> str:
    """Route and handle a query."""
    role_name, reason = router.route_with_reason(message)
    logger.info(f"Routing -> {role_name} ({reason})")

    role = roles[role_name]
    result = await role.handle(message, session_id="cli")
    return result["response"]


def cmd_load(args):
    """Load and summarize a GEDCOM file."""
    tree = GedcomTree.from_file(args.file)
    print(tree.get_summary())
    print(f"\nLoaded {len(tree.persons)} persons from {args.file}")


def cmd_summary(args):
    """Print detailed tree summary."""
    tree = GedcomTree.from_file(args.file)
    print(tree.get_summary())

    print("\n--- Persons ---")
    for person in sorted(tree.persons.values(), key=lambda p: p.surname):
        print(f"  {person.display}")


def cmd_search(args):
    """Search for persons in the tree."""
    tree = GedcomTree.from_file(args.file)
    results = tree.search_persons(args.query)

    if not results:
        print(f"No results for '{args.query}'")
        return

    print(f"Found {len(results)} results for '{args.query}':\n")
    for person in results:
        print(f"  {person.display}")
        parents = tree.get_parents(person.xref)
        if parents:
            print(f"    Parents: {', '.join(p.full_name for p in parents)}")


def cmd_query(args):
    """Run a single query against the tree."""
    tree = GedcomTree.from_file(args.file)
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    roles, router = build_service(tree, ollama_url)

    response = asyncio.run(handle_query(args.query, roles, router))
    print(response)


def cmd_chat(args):
    """Interactive chat with the genealogy agent."""
    tree = GedcomTree.from_file(args.file)
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    roles, router = build_service(tree, ollama_url)

    print(f"Genealogy Agent — {len(tree.persons)} persons loaded")
    print(f"Tree: {tree.get_summary().split(chr(10))[0]}")
    print("Commands: 'search <name>', 'summary', 'exit'\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            break

        if not user_input:
            continue

        if user_input.lower() == "summary":
            print(tree.get_summary())
            continue

        if user_input.lower().startswith("search "):
            query = user_input[7:].strip()
            results = tree.search_persons(query)
            if results:
                for p in results:
                    print(f"  {p.display}")
            else:
                print(f"  No results for '{query}'")
            continue

        response = asyncio.run(handle_query(user_input, roles, router))
        print(f"\nAgent: {response}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="genealogy",
        description="LLM-backed genealogy research tool",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # load
    p_load = sub.add_parser("load", help="Load and summarize a GEDCOM file")
    p_load.add_argument("file", help="Path to GEDCOM file")

    # summary
    p_summary = sub.add_parser("summary", help="Print detailed tree summary")
    p_summary.add_argument("file", help="Path to GEDCOM file")

    # search
    p_search = sub.add_parser("search", help="Search for persons")
    p_search.add_argument("file", help="Path to GEDCOM file")
    p_search.add_argument("query", help="Search query")

    # query
    p_query = sub.add_parser("query", help="One-shot query")
    p_query.add_argument("file", help="Path to GEDCOM file")
    p_query.add_argument("query", help="Question to ask")

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat")
    p_chat.add_argument("file", help="Path to GEDCOM file")

    args = parser.parse_args()

    commands = {
        "load": cmd_load,
        "summary": cmd_summary,
        "search": cmd_search,
        "query": cmd_query,
        "chat": cmd_chat,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
