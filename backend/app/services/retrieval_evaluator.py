from abc import ABC, abstractmethod
from typing import Any, Dict, List

from app.models.selfrag_state import RetrievalEvaluationResult


class RetrievalEvaluatorInterface(ABC):
    """Interface for evaluating whether retrieved documents are relevant."""

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        threshold: float = 0.5,
    ) -> RetrievalEvaluationResult:
        """Return a relevance evaluation for the retrieved documents."""
        raise NotImplementedError


class SimpleRetrievalEvaluator(RetrievalEvaluatorInterface):
    """A lightweight fallback evaluator used during scaffolding."""

    async def evaluate(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        threshold: float = 0.5,
    ) -> RetrievalEvaluationResult:
        if not documents:
            return RetrievalEvaluationResult(
                relevant=False,
                confidence=0.0,
                reason="No documents were retrieved.",
            )

        confidence = 0.8 if documents else 0.0
        relevant = confidence >= threshold
        return RetrievalEvaluationResult(
            relevant=relevant,
            confidence=confidence,
            reason="Retrieved documents are available for answer generation." if relevant else "Retrieved documents are below the relevance threshold.",
        )
