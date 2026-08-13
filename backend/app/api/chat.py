import logging
import sys
import os
from uuid import uuid4
from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse

# Add project root to path for guardrails import
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from graph_workflow.guardrails_wrapper import GuardrailsWrapper

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

# Initialize guardrails once at startup
guardrails = GuardrailsWrapper()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with NeMo Guardrails integration.
    Validates input before routing to LangGraph and
    checks output before returning to the user.
    """
    conversation_id = request.conversation_id or uuid4()
    logger.info(
        f"Chat request: conversation_id={conversation_id}, "
        f"query={request.query[:50]}..."
    )

    # ── INPUT GUARDRAIL CHECK (Member 4 - Ranjith) ────────────
    input_status, input_result = guardrails.check_input(request.query)
    if input_status == "blocked":
        logger.warning(f"Query BLOCKED by guardrails: {request.query[:50]}")
        return ChatResponse(
            answer=input_result,
            conversation_id=conversation_id,
        )

    # ── LANGGRAPH AGENT ROUTING (connected in later integration) ─
    # This is where the compiled app_graph.invoke() will be wired in.
    # For now, pass through with a placeholder response.
    raw_response = "Backend connected successfully."

    # ── OUTPUT GUARDRAIL CHECK (Member 4 - Ranjith) ───────────
    output_status, output_result = guardrails.check_output(raw_response)
    if output_status == "flagged":
        logger.warning(f"Response FLAGGED by guardrails.")
        return ChatResponse(
            answer="⚠️ The response has been flagged by OmniBrain's safety guardrails. Please rephrase your query.",
            conversation_id=conversation_id,
        )

    return ChatResponse(
        answer=output_result,
        conversation_id=conversation_id,
    )