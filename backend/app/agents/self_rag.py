"""
Self-RAG Agent Implementation with Self-Correction Loop.
Orchestrates query rewriting, retrieval, evaluation, and generation.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.models.selfrag_state import SelfRAGState
from app.services.query_rewriter import QueryRewriterInterface
from app.services.retrieval import RetrieverInterface
from app.services.retrieval_evaluator import RetrievalEvaluatorInterface
from app.services.answer_generator import AnswerGeneratorInterface
from app.services.answer_evaluator import AnswerEvaluatorInterface

logger = logging.getLogger(__name__)


class SelfRAGAgent:
    """
    Self-RAG Agent that implements an iterative self-correction loop.
    Repeatedly refines queries, retrieves documents, and evaluates answers
    until confidence threshold is met or max iterations are reached.
    """
    
    def __init__(
        self,
        query_rewriter: QueryRewriterInterface,
        retriever: RetrieverInterface,
        retrieval_evaluator: RetrievalEvaluatorInterface,
        answer_generator: AnswerGeneratorInterface,
        answer_evaluator: AnswerEvaluatorInterface,
        max_iterations: int = 3,
        confidence_threshold: float = 0.8,
        retrieval_threshold: float = 0.5,
    ):
        """
        Initialize the Self-RAG Agent with all required components.
        
        Args:
            query_rewriter: Service to rewrite/enhance queries
            retriever: Service to retrieve documents from vector DB
            retrieval_evaluator: Service to evaluate retrieved documents
            answer_generator: Service to generate grounded answers
            answer_evaluator: Service to evaluate answer quality
            max_iterations: Maximum number of correction iterations
            confidence_threshold: Minimum confidence to stop iteration
            retrieval_threshold: Minimum score for document relevance
        """
        self.query_rewriter = query_rewriter
        self.retriever = retriever
        self.retrieval_evaluator = retrieval_evaluator
        self.answer_generator = answer_generator
        self.answer_evaluator = answer_evaluator
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.retrieval_threshold = retrieval_threshold
        
        logger.info(
            f"SelfRAGAgent initialized with max_iterations={max_iterations}, "
            f"confidence_threshold={confidence_threshold}"
        )
    
    async def run(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        document_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the Self-RAG correction loop.
        
        Args:
            query: User's question
            conversation_id: Optional conversation ID
            document_filter: Optional filter for document retrieval
            
        Returns:
            Dict with final answer and metadata
        """
        logger.info(f"Starting Self-RAG process for query: {query[:100]}...")
        
        # Initialize state
        state = SelfRAGState(
            original_query=query,
            current_query=query,
            max_iterations=self.max_iterations,
            confidence_threshold=self.confidence_threshold,
        )
        
        # Main correction loop
        for iteration in range(self.max_iterations):
            logger.info(f"--- Iteration {iteration + 1}/{self.max_iterations} ---")
            
            # 1. Query Rewriting
            if iteration == 0:
                current_query = query
            else:
                # Use history for context
                current_query = await self.query_rewriter.rewrite(
                    state.current_query,
                    history=state.query_history
                )
            
            state.current_query = current_query
            
            # 2. Document Retrieval
            docs = await self.retriever.retrieve(
                current_query,
                top_k=5,
                filter=document_filter,
            )
            
            # 3. Retrieval Evaluation
            retrieval_eval = await self.retrieval_evaluator.evaluate(
                current_query,
                docs,
                threshold=self.retrieval_threshold,
            )
            
            # If documents are not relevant, rewrite query and retry
            if not retrieval_eval.relevant:
                logger.warning(
                    f"Retrieved documents not relevant (confidence: {retrieval_eval.confidence:.2f})"
                )
                state.retry_reason = retrieval_eval.reason
                state.add_iteration(
                    query=current_query,
                    docs=docs,
                    retrieval_eval=retrieval_eval,
                )
                continue
            
            # 4. Answer Generation
            answer = await self.answer_generator.generate(
                current_query,
                docs,
                max_length=500,  # Configurable
            )
            
            # 5. Answer Evaluation
            answer_eval = await self.answer_evaluator.evaluate(
                current_query,
                docs,
                answer,
                threshold=self.confidence_threshold,
            )
            
            # Record iteration
            state.add_iteration(
                query=current_query,
                docs=docs,
                retrieval_eval=retrieval_eval,
                answer=answer,
                answer_eval=answer_eval,
            )
            
            # Check if answer is good enough
            if not answer_eval.retry and answer_eval.confidence >= self.confidence_threshold:
                logger.info(
                    f"Answer accepted with confidence {answer_eval.confidence:.2f} "
                    f"after {iteration + 1} iterations"
                )
                state.mark_complete(answer, answer_eval.confidence)
                break
            
            # If answer needs improvement, prepare for next iteration
            if answer_eval.retry:
                state.retry_reason = answer_eval.reason
                logger.info(f"Retry needed: {answer_eval.reason}")
                
                # If we're at max iterations, use the best answer we have
                if iteration == self.max_iterations - 1:
                    logger.warning(
                        f"Max iterations reached. Using best answer with confidence "
                        f"{answer_eval.confidence:.2f}"
                    )
                    state.mark_complete(answer, answer_eval.confidence)
                    break
                
                # Otherwise, continue to next iteration
                continue
        
        # If we exit the loop without a complete answer, use the last one
        if not state.is_complete:
            if state.generated_answers:
                last_answer = state.generated_answers[-1]
                last_confidence = (
                    state.answer_evaluations[-1].confidence 
                    if state.answer_evaluations 
                    else 0.5
                )
                state.mark_complete(last_answer, last_confidence)
                logger.warning(f"Using last answer with confidence {last_confidence:.2f}")
            else:
                # No answers generated at all
                state.mark_complete(
                    "I was unable to generate an answer. Please try rephrasing your question.",
                    0.0
                )
        
        # Prepare final response
        result = {
            "answer": state.final_answer,
            "confidence": state.final_confidence,
            "iterations": state.iteration,
            "retrieval_evaluations": [
                {
                    "relevant": eval.relevant,
                    "confidence": eval.confidence,
                    "reason": eval.reason,
                }
                for eval in state.retrieval_evaluations
            ],
            "answer_evaluations": [
                {
                    "confidence": eval.confidence,
                    "reason": eval.reason,
                    "groundedness": eval.groundedness,
                    "completeness": eval.completeness,
                    "hallucination_score": eval.hallucination_score,
                }
                for eval in state.answer_evaluations
            ],
            "documents_retrieved": len(state.retrieved_documents),
            "processing_time": (
                datetime.utcnow() - state.start_time
            ).total_seconds() if state.end_time else None,
        }
        
        logger.info(
            f"Self-RAG completed: confidence={state.final_confidence:.2f}, "
            f"iterations={state.iteration}"
        )
        
        return result