from abc import ABC, abstractmethod
from typing import List, Optional


class QueryRewriterInterface(ABC):
    """Interface for rewriting user queries for better retrieval."""

    @abstractmethod
    async def rewrite(self, query: str, history: Optional[List[str]] = None) -> str:
        """Rewrite the query using prior conversation history when available."""
        raise NotImplementedError


class SimpleQueryRewriter(QueryRewriterInterface):
    """A lightweight fallback rewriter used during scaffolding."""

    async def rewrite(self, query: str, history: Optional[List[str]] = None) -> str:
        if not history:
            return query.strip()

        history_text = " ".join(history[-3:])
        return f"{query.strip()} Context: {history_text}".strip()
