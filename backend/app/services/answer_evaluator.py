from abc import ABC, abstractmethod
from typing import Any, Dict, List


class AnswerEvaluationResult:
    """Container for answer-quality evaluation details."""

    def __init__(
        self,
        confidence: float,
        reason: str = "",
        groundedness: float = 0.0,
        completeness: float = 0.0,
        hallucination_score: float = 0.0,
        retry: bool = False,
    ) -> None:
        self.confidence = confidence
        self.reason = reason
        self.groundedness = groundedness
        self.completeness = completeness
        self.hallucination_score = hallucination_score
        self.retry = retry


class AnswerEvaluatorInterface(ABC):
    """Interface for evaluating answer quality and groundedness."""

    @abstractmethod
    async def evaluate(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        answer: str,
        threshold: float = 0.8,
    ) -> AnswerEvaluationResult:
        """Evaluate the final answer quality against the retrieved evidence."""
        raise NotImplementedError


class SimpleAnswerEvaluator(AnswerEvaluatorInterface):
    """A lightweight fallback evaluator used during scaffolding."""

    async def evaluate(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        answer: str,
        threshold: float = 0.8,
    ) -> AnswerEvaluationResult:
        confidence = 0.9 if documents and answer.strip() else 0.2
        if not documents:
            return AnswerEvaluationResult(
                confidence=0.0,
                reason="No supporting documents were retrieved.",
                groundedness=0.0,
                completeness=0.0,
                hallucination_score=1.0,
                retry=True,
            )

        if confidence < threshold:
            return AnswerEvaluationResult(
                confidence=confidence,
                reason="The answer needs more grounding.",
                groundedness=confidence,
                completeness=0.6,
                hallucination_score=0.1,
                retry=True,
            )

        return AnswerEvaluationResult(
            confidence=confidence,
            reason="Answer appears grounded and complete.",
            groundedness=confidence,
            completeness=0.8,
            hallucination_score=0.0,
            retry=False,
        )
