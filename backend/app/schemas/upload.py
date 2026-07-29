from pydantic import BaseModel
from uuid import UUID


class UploadResponse(BaseModel):
    """Response after a successful document upload."""

    document_id: UUID
    filename: str
    status: str = "uploaded"