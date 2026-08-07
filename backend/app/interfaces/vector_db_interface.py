from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class VectorDBInterface(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for documents matching the provided query."""
        raise NotImplementedError
