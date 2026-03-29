"""
Chat handler — wraps the WebSocket chat server with research and ingestion.

Intercepts messages starting with ! prefixes:
- Research commands: !lookup, !search, !find, !history, !tree, !ancestors, !migration
- Ingestion commands: !ingest, !ingest-file, !ingest-dir
- Knowledge commands: !knowledge, !prune, !promote
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from khonliang.knowledge import Librarian
from khonliang.knowledge.store import Tier
from khonliang.research import ResearchPool, ResearchTrigger

logger = logging.getLogger(__name__)


class ResearchChatHandler:
    """
    Middleware that adds research + ingestion commands to the chat.

    Intercepts messages starting with ! prefixes, queues research,
    ingests content, or manages knowledge. Non-! messages pass through.
    """

    # Commands that this handler owns
    COMMANDS = {
        # Research (delegated to trigger/pool)
        "!lookup", "!search", "!find", "!history",
        "!ancestors", "!migration", "!tree",
        # Ingestion
        "!ingest", "!ingest-file", "!ingest-dir",
        # Direct search
        "!google", "!fetch",
        # Knowledge management
        "!knowledge", "!prune", "!promote", "!demote",
        "!axiom",
    }

    def __init__(
        self,
        pool: ResearchPool,
        trigger: ResearchTrigger,
        librarian: Optional[Librarian] = None,
        poll_interval: float = 0.5,
        poll_timeout: float = 30.0,
    ):
        self.pool = pool
        self.trigger = trigger
        self.librarian = librarian
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

    def is_command(self, message: str) -> bool:
        """Check if a message is any ! command."""
        msg_lower = message.lower().strip()
        return any(msg_lower.startswith(cmd) for cmd in self.COMMANDS)

    async def handle(
        self,
        message: str,
        scope: str = "global",
        source: str = "user",
    ) -> Dict[str, Any]:
        """Route a ! command to the right handler."""
        msg_lower = message.lower().strip()

        # Research commands
        if self.trigger.strip_prefix(message) is not None:
            return await self._handle_research(message, scope, source)

        # Direct search / fetch
        if msg_lower.startswith("!google"):
            return self._handle_google(message)
        if msg_lower.startswith("!fetch"):
            return self._handle_fetch(message, scope)

        # Ingestion commands
        if msg_lower.startswith("!ingest-file"):
            return self._handle_ingest_file(message, scope)
        if msg_lower.startswith("!ingest-dir"):
            return self._handle_ingest_dir(message, scope)
        if msg_lower.startswith("!ingest"):
            return self._handle_ingest_text(message, scope)

        # Knowledge management
        if msg_lower.startswith("!knowledge"):
            return self._handle_knowledge_status()
        if msg_lower.startswith("!prune"):
            return self._handle_prune()
        if msg_lower.startswith("!promote"):
            return self._handle_promote(message)
        if msg_lower.startswith("!demote"):
            return self._handle_demote(message)
        if msg_lower.startswith("!axiom"):
            return self._handle_axiom(message)

        return {"type": "error", "content": f"Unknown command: {message[:30]}"}

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    async def _handle_research(
        self, message: str, scope: str, source: str
    ) -> Dict[str, Any]:
        """Queue research task and wait for result."""
        task_ids = self.trigger.check_message(message, scope=scope, source=source)

        if not task_ids:
            return {"type": "error", "content": "Could not parse research command."}

        results = []
        for task_id in task_ids:
            result = await self._wait_for_result(task_id)
            if result:
                results.append(result)

        if not results:
            return {
                "type": "response",
                "content": "Research queued but timed out. Results will appear in knowledge when complete.",
                "role": "research",
                "reason": "timeout",
            }

        content_parts = []
        all_sources: List[str] = []
        for r in results:
            content_parts.append(f"**{r.title}**\n{r.content}")
            all_sources.extend(r.sources)

        return {
            "type": "response",
            "content": "\n\n".join(content_parts),
            "role": "research",
            "reason": "research_complete",
            "metadata": {
                "task_ids": task_ids,
                "sources": all_sources[:10],
                "result_count": len(results),
            },
        }

    async def _wait_for_result(self, task_id: str) -> Optional[Any]:
        deadline = time.time() + self.poll_timeout
        while time.time() < deadline:
            result = self.pool.get_result(task_id)
            if result:
                return result
            await asyncio.sleep(self.poll_interval)
        return None

    # ------------------------------------------------------------------
    # Google search / Fetch
    # ------------------------------------------------------------------

    def _handle_google(self, message: str) -> Dict[str, Any]:
        """
        Google search. Format: !google query text
        """
        from genealogy_agent.web_search import GenealogySearcher

        query = message.split(None, 1)[1].strip() if " " in message else ""
        if not query:
            return {"type": "error", "content": "Usage: !google <query>"}

        searcher = GenealogySearcher()
        results = searcher.google_search(query, max_results=8)

        if not results:
            # Fallback to DDG
            results = searcher.search(query, max_results=8)

        if not results:
            return {
                "type": "response",
                "content": "No results found.",
                "role": "research",
                "reason": "google",
            }

        return {
            "type": "response",
            "content": searcher.build_context(results),
            "role": "research",
            "reason": "google",
            "metadata": {"sources": [r.url for r in results[:10]]},
        }

    def _handle_fetch(self, message: str, scope: str) -> Dict[str, Any]:
        """
        Fetch a URL, extract text, and optionally ingest into knowledge.
        Format: !fetch <url>
        Format: !fetch ingest <url>  (also saves to knowledge)
        """
        from genealogy_agent.web_search import GenealogySearcher

        text = message.split(None, 1)[1].strip() if " " in message else ""
        if not text:
            return {
                "type": "error",
                "content": "Usage: !fetch <url> or !fetch ingest <url>",
            }

        ingest = False
        if text.lower().startswith("ingest "):
            ingest = True
            text = text[7:].strip()

        url = text
        searcher = GenealogySearcher()
        content = searcher.fetch_page(url, max_chars=6000)

        if not content:
            return {"type": "error", "content": f"Failed to fetch {url}"}

        # Optionally ingest into knowledge
        if ingest and self.librarian:
            domain = url.split("/")[2] if url.count("/") >= 2 else url
            self.librarian.ingest_text(
                content=content,
                title=f"Fetched: {domain}",
                scope=scope,
                source=url,
            )
            return {
                "type": "response",
                "content": f"Fetched and ingested ({len(content)} chars):\n\n"
                + content[:2000],
                "role": "librarian",
                "reason": "fetch_ingest",
            }

        return {
            "type": "response",
            "content": content[:4000],
            "role": "research",
            "reason": "fetch",
            "metadata": {"url": url, "length": len(content)},
        }

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def _handle_ingest_text(self, message: str, scope: str) -> Dict[str, Any]:
        """
        Ingest raw text into knowledge.
        Format: !ingest [scope:name] title | content
        Example: !ingest scope:toll Census note | John Toll listed in 1850 census
        """
        if not self.librarian:
            return {"type": "error", "content": "No librarian configured."}

        text = message.split(None, 1)[1] if " " in message else ""
        if not text:
            return {
                "type": "error",
                "content": "Usage: !ingest [scope:name] title | content",
            }

        # Parse optional scope override
        ingest_scope = scope
        if text.lower().startswith("scope:"):
            scope_part, text = text.split(None, 1)
            ingest_scope = scope_part.split(":", 1)[1]

        # Split title | content
        if "|" in text:
            title, content = text.split("|", 1)
            title = title.strip()
            content = content.strip()
        else:
            title = text[:60]
            content = text

        result = self.librarian.ingest_text(
            content=content, title=title, scope=ingest_scope, source="user_chat"
        )

        return {
            "type": "response",
            "content": f"Ingested: \"{title}\" into scope '{ingest_scope}' "
            f"({result.added} added, {result.skipped} skipped)",
            "role": "librarian",
            "reason": "ingestion",
        }

    def _handle_ingest_file(self, message: str, scope: str) -> Dict[str, Any]:
        """
        Ingest a file.
        Format: !ingest-file [scope:name] /path/to/file.txt
        """
        if not self.librarian:
            return {"type": "error", "content": "No librarian configured."}

        text = message.split(None, 1)[1] if " " in message else ""
        if not text:
            return {
                "type": "error",
                "content": "Usage: !ingest-file [scope:name] /path/to/file",
            }

        ingest_scope = scope
        if text.lower().startswith("scope:"):
            scope_part, text = text.split(None, 1)
            ingest_scope = scope_part.split(":", 1)[1]

        path = text.strip()
        result = self.librarian.ingest_file(path, scope=ingest_scope)

        return {
            "type": "response",
            "content": f"Ingested file: {path} into scope '{ingest_scope}' "
            f"({result.added} added, {result.errors} errors)",
            "role": "librarian",
            "reason": "ingestion",
        }

    def _handle_ingest_dir(self, message: str, scope: str) -> Dict[str, Any]:
        """
        Ingest a directory.
        Format: !ingest-dir [scope:name] /path/to/directory
        """
        if not self.librarian:
            return {"type": "error", "content": "No librarian configured."}

        text = message.split(None, 1)[1] if " " in message else ""
        if not text:
            return {
                "type": "error",
                "content": "Usage: !ingest-dir [scope:name] /path/to/dir",
            }

        ingest_scope = scope
        if text.lower().startswith("scope:"):
            scope_part, text = text.split(None, 1)
            ingest_scope = scope_part.split(":", 1)[1]

        directory = text.strip()
        result = self.librarian.ingest_directory(directory, scope=ingest_scope)

        return {
            "type": "response",
            "content": f"Ingested directory: {directory} into scope '{ingest_scope}' "
            f"({result.added} added, {result.errors} errors)",
            "role": "librarian",
            "reason": "ingestion",
        }

    # ------------------------------------------------------------------
    # Knowledge management
    # ------------------------------------------------------------------

    def _handle_knowledge_status(self) -> Dict[str, Any]:
        """Show knowledge store status."""
        if not self.librarian:
            return {"type": "error", "content": "No librarian configured."}

        stats = self.librarian.get_status()
        by_tier = stats.get("by_tier", {})
        by_scope = stats.get("by_scope", {})

        lines = [
            f"Knowledge Store: {stats.get('total_entries', 0)} entries",
            f"  Axioms (Tier 1): {by_tier.get('axiom', 0)}",
            f"  Imported (Tier 2): {by_tier.get('imported', 0)}",
            f"  Derived (Tier 3): {by_tier.get('derived', 0)}",
        ]
        if by_scope:
            lines.append("  Scopes: " + ", ".join(
                f"{k} ({v})" for k, v in by_scope.items()
            ))

        pool_status = self.pool.get_status()
        lines.append(
            f"\nResearch Pool: {pool_status['completed']} completed, "
            f"{pool_status['queue_size']} queued, "
            f"{pool_status['failed']} failed"
        )

        return {
            "type": "response",
            "content": "\n".join(lines),
            "role": "librarian",
            "reason": "status",
        }

    def _handle_prune(self) -> Dict[str, Any]:
        """Prune low-quality Tier 3 entries."""
        if not self.librarian:
            return {"type": "error", "content": "No librarian configured."}

        pruned = self.librarian.prune()
        promoted = self.librarian.auto_promote()

        return {
            "type": "response",
            "content": f"Maintenance complete: {pruned} pruned, {promoted} promoted.",
            "role": "librarian",
            "reason": "maintenance",
        }

    def _handle_promote(self, message: str) -> Dict[str, Any]:
        """Promote a Tier 3 entry to Tier 2. Format: !promote entry_id"""
        if not self.librarian:
            return {"type": "error", "content": "No librarian configured."}

        entry_id = message.split(None, 1)[1].strip() if " " in message else ""
        if not entry_id:
            return {"type": "error", "content": "Usage: !promote <entry_id>"}

        if self.librarian.store.promote(entry_id):
            return {
                "type": "response",
                "content": f"Promoted {entry_id} from Tier 3 to Tier 2.",
                "role": "librarian",
                "reason": "promote",
            }
        return {"type": "error", "content": f"Could not promote {entry_id}."}

    def _handle_demote(self, message: str) -> Dict[str, Any]:
        """Demote a Tier 2 entry to Tier 3. Format: !demote entry_id"""
        if not self.librarian:
            return {"type": "error", "content": "No librarian configured."}

        entry_id = message.split(None, 1)[1].strip() if " " in message else ""
        if not entry_id:
            return {"type": "error", "content": "Usage: !demote <entry_id>"}

        if self.librarian.store.demote(entry_id):
            return {
                "type": "response",
                "content": f"Demoted {entry_id} from Tier 2 to Tier 3.",
                "role": "librarian",
                "reason": "demote",
            }
        return {"type": "error", "content": f"Could not demote {entry_id}."}

    def _handle_axiom(self, message: str) -> Dict[str, Any]:
        """
        Add/view axioms.
        !axiom — list all axioms
        !axiom key | content — set an axiom
        """
        if not self.librarian:
            return {"type": "error", "content": "No librarian configured."}

        text = message.split(None, 1)[1].strip() if " " in message else ""

        if not text:
            # List axioms
            axioms = self.librarian.get_axioms()
            if not axioms:
                return {
                    "type": "response",
                    "content": "No axioms set.",
                    "role": "librarian",
                    "reason": "axioms",
                }
            lines = ["Axioms (Tier 1):"]
            for a in axioms:
                lines.append(f"  - {a.title}: {a.content}")
            return {
                "type": "response",
                "content": "\n".join(lines),
                "role": "librarian",
                "reason": "axioms",
            }

        # Set axiom: !axiom key | content
        if "|" in text:
            key, content = text.split("|", 1)
            self.librarian.set_axiom(key.strip(), content.strip())
            return {
                "type": "response",
                "content": f"Axiom set: {key.strip()}",
                "role": "librarian",
                "reason": "axiom_set",
            }

        return {
            "type": "error",
            "content": "Usage: !axiom key | content (or !axiom to list)",
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "pool": self.pool.get_status(),
            "researchers": self.pool.list_researchers(),
        }
