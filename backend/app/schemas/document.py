from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from typing import List


class DocumentMetadata(BaseModel):
    """Metadata for an uploaded document."""

    document_id: UUID
    filename: str
    upload_date: datetime
    file_size: int
    status: str = "uploaded"


class DocumentListResponse(BaseModel):
    """Response model for listing documents."""

    documents: List[DocumentMetadata]
    total: int