from abc import ABC, abstractmethod
from typing import Any, Dict, List


class AnswerGeneratorInterface(ABC):
    """Interface for generating grounded answers from retrieved documents."""

    @abstractmethod
    async def generate(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        max_length: int = 500,
    ) -> str:
        """Generate an answer grounded in the supplied documents."""
        raise NotImplementedError


class SimpleAnswerGenerator(AnswerGeneratorInterface):
    """A lightweight fallback implementation used during scaffolding."""

    async def generate(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        max_length: int = 500,
    ) -> str:
        if not documents:
            return (
                "I could not find supporting context for this question. "
                "Please try rephrasing or provide more context."
            )

        excerpt = documents[0].get("content") or documents[0].get("text") or ""
        if excerpt:
            return excerpt[:max_length]

        return f"I found {len(documents)} document(s) related to '{query}'."
