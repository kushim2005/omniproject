import logging
from uuid import UUID
from fastapi import APIRouter, Depends, status
from app.schemas.document import DocumentListResponse
from app.services.document_service import DocumentService
from app.api.dependencies import get_document_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    doc_service: DocumentService = Depends(get_document_service),
):
    """List all uploaded documents."""
    docs = await doc_service.list_documents()
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    doc_service: DocumentService = Depends(get_document_service),
):
    """Delete a document by its UUID."""
    await doc_service.delete_document(document_id)
    logger.info(f"Deleted document {document_id}")
    return None