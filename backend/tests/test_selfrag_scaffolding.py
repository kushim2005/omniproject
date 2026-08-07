import asyncio

from app.agents.self_rag import SelfRAGAgent
from app.services.answer_evaluator import SimpleAnswerEvaluator
from app.services.answer_generator import SimpleAnswerGenerator
from app.services.query_rewriter import SimpleQueryRewriter
from app.services.retrieval import SimpleRetriever
from app.services.retrieval_evaluator import SimpleRetrievalEvaluator


def test_selfrag_agent_runs_with_fallback_services():
    agent = SelfRAGAgent(
        query_rewriter=SimpleQueryRewriter(),
        retriever=SimpleRetriever(),
        retrieval_evaluator=SimpleRetrievalEvaluator(),
        answer_generator=SimpleAnswerGenerator(),
        answer_evaluator=SimpleAnswerEvaluator(),
        max_iterations=1,
        confidence_threshold=0.8,
    )

    result = asyncio.run(agent.run("What is the document about?"))

    assert result["answer"]
    assert result["confidence"] >= 0.0
    assert result["iterations"] >= 1
