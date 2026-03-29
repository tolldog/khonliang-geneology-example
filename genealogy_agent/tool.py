#!/usr/bin/env python3
"""
Genealogy agent tool — callable interface for external LLMs.

Simple stdin/stdout interface: pass a command as arguments or pipe JSON in,
get JSON or plain text out. Designed to be called from Claude Code or
any other LLM that can run shell commands.

Usage:
    # Plain text query
    python -m genealogy_agent.tool query "Who were Timothy Toll's grandparents?"

    # Search
    python -m genealogy_agent.tool search "Toll"

    # Fact check
    python -m genealogy_agent.tool check "Timothy Toll"

    # Tell me a story
    python -m genealogy_agent.tool narrate "Tell the story of the Tolle migration"

    # Get person details as JSON
    python -m genealogy_agent.tool person "Timothy Toll"

    # Get ancestors
    python -m genealogy_agent.tool ancestors "Timothy Toll" --generations 4

    # Get tree summary
    python -m genealogy_agent.tool summary

    # List all persons matching a surname
    python -m genealogy_agent.tool list "Toll"

    # Raw context (what gets injected into LLM prompts)
    python -m genealogy_agent.tool context "Timothy Toll"

Environment:
    GEDCOM_FILE  Path to GEDCOM file (default: data/Toll Family Tree.ged)
    OLLAMA_URL   Ollama server URL (default: http://localhost:11434)
"""

import asyncio
import json
import os
import sys
from typing import Optional

# Suppress logging to keep output clean for tool use
import logging

logging.disable(logging.WARNING)

_tree = None
_roles = None
_router = None


def _get_tree():
    global _tree
    if _tree is None:
        from genealogy_agent.gedcom_parser import GedcomTree

        gedcom_path = os.environ.get(
            "GEDCOM_FILE", "data/Toll Family Tree.ged"
        )
        _tree = GedcomTree.from_file(gedcom_path)
    return _tree


def _get_service():
    global _roles, _router
    if _roles is None:
        from khonliang import ModelPool

        from genealogy_agent.roles import (
            FactCheckerRole,
            NarratorRole,
            ResearcherRole,
        )
        from genealogy_agent.router import GenealogyRouter

        tree = _get_tree()
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        pool = ModelPool(
            {
                "researcher": "llama3.2:3b",
                "fact_checker": "qwen2.5:7b",
                "narrator": "llama3.1:8b",
            },
            base_url=ollama_url,
        )
        _roles = {
            "researcher": ResearcherRole(pool, tree=tree),
            "fact_checker": FactCheckerRole(pool, tree=tree),
            "narrator": NarratorRole(pool, tree=tree),
        }
        _router = GenealogyRouter()
    return _roles, _router


def cmd_summary() -> str:
    """Tree summary as plain text."""
    return _get_tree().get_summary()


def cmd_search(query: str) -> str:
    """Search persons, return JSON array."""
    tree = _get_tree()
    results = tree.search_persons(query)
    return json.dumps(
        [p.to_dict() for p in results],
        indent=2,
    )


def cmd_list(surname: str) -> str:
    """List all persons with a surname, one per line."""
    tree = _get_tree()
    results = [
        p for p in tree.persons.values()
        if surname.lower() in p.surname.lower()
    ]
    results.sort(key=lambda p: p.birth_date or "9999")
    return "\n".join(p.display for p in results)


def cmd_person(name: str) -> str:
    """Get person details as JSON."""
    tree = _get_tree()
    person = tree.find_person(name)
    if not person:
        return json.dumps({"error": f"Person '{name}' not found"})

    data = person.to_dict()
    data["parents"] = [p.to_dict() for p in tree.get_parents(person.xref)]
    data["spouses"] = [s.to_dict() for s in tree.get_spouses(person.xref)]
    data["children"] = [c.to_dict() for c in tree.get_children(person.xref)]
    data["siblings"] = [s.to_dict() for s in tree.get_siblings(person.xref)]
    return json.dumps(data, indent=2)


def cmd_ancestors(name: str, generations: int = 4) -> str:
    """Get ancestors as JSON."""
    tree = _get_tree()
    person = tree.find_person(name)
    if not person:
        return json.dumps({"error": f"Person '{name}' not found"})

    ancestors = tree.get_ancestors(person.xref, generations=generations)
    return json.dumps(
        {
            "person": person.full_name,
            "generations": generations,
            "ancestors": [a.to_dict() for a in ancestors],
        },
        indent=2,
    )


def cmd_descendants(name: str, generations: int = 4) -> str:
    """Get descendants as JSON."""
    tree = _get_tree()
    person = tree.find_person(name)
    if not person:
        return json.dumps({"error": f"Person '{name}' not found"})

    descendants = tree.get_descendants(person.xref, generations=generations)
    return json.dumps(
        {
            "person": person.full_name,
            "generations": generations,
            "descendants": [d.to_dict() for d in descendants],
        },
        indent=2,
    )


def cmd_migration(name: str, generations: int = 10) -> str:
    """Trace migration patterns through a person's ancestor line."""
    import re

    tree = _get_tree()
    person = tree.find_person(name)
    if not person:
        return json.dumps({"error": f"Person '{name}' not found"})

    ancestors = tree.get_ancestors(person.xref, generations=generations)
    all_people = [person] + ancestors

    # Extract place + year for each person
    moves = []
    for p in all_people:
        year = None
        for date_str in [p.birth_date, p.death_date]:
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

    # Build a readable migration timeline
    lines = [f"Migration timeline for {person.full_name}'s ancestors:\n"]
    seen_places = []
    for m in moves:
        yr = str(m["year"]) if m["year"] else "????"
        bp = m["birth_place"] or "?"
        dp = m["death_place"] or ""
        line = f"  {yr}  {m['name']}"
        line += f"  b. {bp}"
        if dp and dp != bp:
            line += f"  d. {dp}"
        lines.append(line)

        # Track migration hops
        if bp and bp not in [s[1] for s in seen_places]:
            seen_places.append((yr, bp))

    lines.append(f"\nDistinct locations ({len(seen_places)}):")
    for yr, place in seen_places:
        lines.append(f"  {yr}: {place}")

    return "\n".join(lines)


def cmd_context(name: str) -> str:
    """Get raw LLM context for a person (what the agent sees)."""
    tree = _get_tree()
    person = tree.find_person(name)
    if not person:
        return f"Person '{name}' not found"
    return tree.build_context(person.xref, depth=2)


def cmd_query(question: str) -> str:
    """Ask the researcher agent a question."""
    roles, router = _get_service()
    role_name, _ = router.route_with_reason(question)
    role = roles[role_name]
    result = asyncio.run(role.handle(question, session_id="tool"))
    return result["response"]


def cmd_check(question: str) -> str:
    """Ask the fact checker agent."""
    roles, _ = _get_service()
    result = asyncio.run(
        roles["fact_checker"].handle(question, session_id="tool")
    )
    return result["response"]


def cmd_narrate(question: str) -> str:
    """Ask the narrator agent."""
    roles, _ = _get_service()
    result = asyncio.run(
        roles["narrator"].handle(question, session_id="tool")
    )
    return result["response"]


def cmd_websearch(name: str) -> str:
    """Quick web scan for a person — combines tree data with web results."""
    import re

    from genealogy_agent.web_search import GenealogySearcher

    tree = _get_tree()
    person = tree.find_person(name)

    searcher = GenealogySearcher()

    if person:
        # Extract year from birth date
        year_match = re.search(r"\d{4}", person.birth_date)
        birth_year = int(year_match.group()) if year_match else None

        result = searcher.quick_scan(
            person.full_name,
            birth_year=birth_year,
            place=person.birth_place,
        )
        header = f"Tree data:\n{tree.build_context(person.xref, depth=1)}\n\n"
        return header + result
    else:
        # Search by raw name
        return searcher.quick_scan(name)


def cmd_websearch_history(place: str, year: str) -> str:
    """Search for historical context about a place and time."""
    from genealogy_agent.web_search import GenealogySearcher

    searcher = GenealogySearcher()
    results = searcher.search_historical_context(place, int(year))
    return searcher.build_context(results)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m genealogy_agent.tool <command> [args]\n"
            "Commands: summary, search, list, person, ancestors, "
            "descendants, context, query, check, narrate"
        )
        sys.exit(1)

    cmd = sys.argv[1]
    args = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

    commands = {
        "summary": lambda: cmd_summary(),
        "search": lambda: cmd_search(args),
        "list": lambda: cmd_list(args),
        "person": lambda: cmd_person(args),
        "ancestors": lambda: cmd_ancestors(
            args.split("--generations")[0].strip(),
            int(args.split("--generations")[1].strip())
            if "--generations" in args
            else 4,
        ),
        "descendants": lambda: cmd_descendants(
            args.split("--generations")[0].strip(),
            int(args.split("--generations")[1].strip())
            if "--generations" in args
            else 4,
        ),
        "migration": lambda: cmd_migration(
            args.split("--generations")[0].strip(),
            int(args.split("--generations")[1].strip())
            if "--generations" in args
            else 10,
        ),
        "context": lambda: cmd_context(args),
        "query": lambda: cmd_query(args),
        "check": lambda: cmd_check(args),
        "narrate": lambda: cmd_narrate(args),
        "websearch": lambda: cmd_websearch(args),
        "history": lambda: cmd_websearch_history(
            args.rsplit(None, 1)[0] if " " in args else args,
            args.rsplit(None, 1)[1] if " " in args else "1800",
        ),
    }

    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands.keys())}")
        sys.exit(1)

    print(commands[cmd]())


if __name__ == "__main__":
    main()
