import logging
from uuid import uuid4
from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Dummy chat endpoint.

    Returns a fixed response; AI logic will be added in later weeks.
    """
    conversation_id = request.conversation_id or uuid4()
    logger.info(
        f"Chat request: conversation_id={conversation_id}, "
        f"query={request.query[:50]}..."
    )
    return ChatResponse(
        answer="Backend connected successfully.",
        conversation_id=conversation_id,
    )