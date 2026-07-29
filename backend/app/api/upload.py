import os
import logging
from fastapi import APIRouter, File, UploadFile, Depends, status, HTTPException
from app.schemas.upload import UploadResponse
from app.services.document_service import DocumentService
from app.api.dependencies import get_document_service
from app.config.settings import settings
from app.utils.exceptions import InvalidFileTypeError, FileTooLargeError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["Upload"])

ALLOWED_EXTENSIONS = {".pdf"}


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    doc_service: DocumentService = Depends(get_document_service),
):
    """
    Upload a PDF document.

    Validates file type, size, saves the file, and stores metadata.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is missing")

    # Validate file extension
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidFileTypeError(filename)

    # Read file content and check size
    content = await file.read()
    file_size = len(content)
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise FileTooLargeError(settings.MAX_UPLOAD_SIZE)

    # Save document using service
    doc = await doc_service.create_document(filename, content)
    logger.info(f"Uploaded document {doc.document_id} with size {file_size}")

    return UploadResponse(
        document_id=doc.document_id,
        filename=doc.filename,
        status=doc.status,
    )