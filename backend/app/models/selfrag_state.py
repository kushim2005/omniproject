from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalEvaluationResult:
    """Stores retrieval quality assessment for one iteration."""

    relevant: bool
    confidence: float
    reason: str = ""


@dataclass
class AnswerEvaluationResult:
    """Stores answer quality assessment for one iteration."""

    confidence: float
    reason: str = ""
    groundedness: float = 0.0
    completeness: float = 0.0
    hallucination_score: float = 0.0
    retry: bool = False


@dataclass
class SelfRAGState:
    """Tracks the iterative Self-RAG process state."""

    original_query: str
    current_query: str
    max_iterations: int
    confidence_threshold: float
    query_history: List[str] = field(default_factory=list)
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_evaluations: List[RetrievalEvaluationResult] = field(default_factory=list)
    generated_answers: List[str] = field(default_factory=list)
    answer_evaluations: List[AnswerEvaluationResult] = field(default_factory=list)
    retry_reason: Optional[str] = None
    final_answer: Optional[str] = None
    final_confidence: float = 0.0
    is_complete: bool = False
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    def add_iteration(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        retrieval_eval: Optional[RetrievalEvaluationResult] = None,
        answer: Optional[str] = None,
        answer_eval: Optional[AnswerEvaluationResult] = None,
    ) -> None:
        self.query_history.append(query)
        self.retrieved_documents.extend(docs)
        if retrieval_eval is not None:
            self.retrieval_evaluations.append(retrieval_eval)
        if answer is not None:
            self.generated_answers.append(answer)
        if answer_eval is not None:
            self.answer_evaluations.append(answer_eval)

    def mark_complete(self, answer: str, confidence: float) -> None:
        self.final_answer = answer
        self.final_confidence = confidence
        self.is_complete = True
        self.end_time = datetime.utcnow()

    @property
    def iteration(self) -> int:
        return max(len(self.retrieval_evaluations), len(self.answer_evaluations), len(self.generated_answers))
