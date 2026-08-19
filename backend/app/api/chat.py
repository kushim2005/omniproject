import logging
import sys
import os
from uuid import uuid4
from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.agents.self_rag import SelfRAGAgent
from app.services.answer_evaluator import SimpleAnswerEvaluator
from app.services.answer_generator import SimpleAnswerGenerator
from app.services.query_rewriter import SimpleQueryRewriter
from app.services.retrieval import SimpleRetriever
from app.services.retrieval_evaluator import SimpleRetrievalEvaluator
from app.utils.tracing import trace_span

# Add project root to path for guardrails import
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from graph_workflow.guardrails_wrapper import GuardrailsWrapper

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

# Initialize guardrails once at startup
guardrails = GuardrailsWrapper()

rag_agent = SelfRAGAgent(
    query_rewriter=SimpleQueryRewriter(),
    retriever=SimpleRetriever(),
    retrieval_evaluator=SimpleRetrievalEvaluator(),
    answer_generator=SimpleAnswerGenerator(),
    answer_evaluator=SimpleAnswerEvaluator(),
)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with NeMo Guardrails integration.
    Validates input before routing to LangGraph and
    checks output before returning to the user.
    """
    conversation_id = request.conversation_id or uuid4()
    async with trace_span(
        "chat_request",
        conversation_id=str(conversation_id),
        query_length=len(request.query),
    ) as span:
        logger.info(
            "Chat request: conversation_id=%s, query=%s...",
            conversation_id,
            request.query[:50],
        )

        input_status, input_result = guardrails.check_input(request.query)
        if input_status == "blocked":
            logger.warning("Query blocked by guardrails: %s", request.query[:50])
            span["status"] = "blocked_input"
            return ChatResponse(
                answer=input_result,
                conversation_id=conversation_id,
            )

        document_filter = None
        if request.documents:
            document_filter = {"document_ids": [str(document_id) for document_id in request.documents]}

        result = await rag_agent.run(
            query=request.query,
            conversation_id=str(conversation_id),
            document_filter=document_filter,
        )
        raw_response = result["answer"]
        span.update(
            status="completed",
            confidence=result["confidence"],
            iterations=result["iterations"],
        )

        output_status, output_result = guardrails.check_output(raw_response)
        if output_status == "flagged":
            logger.warning("Response flagged by guardrails")
            span["status"] = "flagged_output"
            return ChatResponse(
                answer="The response has been flagged by OmniBrain's safety guardrails. Please rephrase your query.",
                conversation_id=conversation_id,
            )

        return ChatResponse(
            answer=output_result,
            conversation_id=conversation_id,
        )