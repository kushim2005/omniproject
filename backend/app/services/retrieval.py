from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class RetrieverInterface(ABC):
    """Interface for retrieving documents from a vector database."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant documents for a query."""
        raise NotImplementedError


class SimpleRetriever(RetrieverInterface):
    """A lightweight fallback retriever used during scaffolding."""

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        return [
            {
                "content": f"No external documents were loaded for query: {query}",
                "score": 0.0,
                "metadata": filter or {},
            }
        ][:top_k]
