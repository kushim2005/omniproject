from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    query: str
    conversation_id: Optional[UUID] = None
    documents: Optional[List[UUID]] = None  # future use for RAG


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""

    answer: str
    conversation_id: UUID