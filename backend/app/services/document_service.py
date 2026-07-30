import json
import os
import uuid
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import aiofiles
import logging

from app.config.settings import settings
from app.schemas.document import DocumentMetadata
from app.utils.exceptions import DocumentNotFoundError

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service for managing document metadata and file storage.

    Uses an in-memory cache with JSON file persistence.
    """

    def __init__(self):
        self._documents: Dict[uuid.UUID, DocumentMetadata] = {}
        self._lock = asyncio.Lock()
        self._upload_folder = Path(settings.UPLOAD_FOLDER)
        self._metadata_file = Path(settings.METADATA_FILE)
        self._ensure_upload_folder()

    def _ensure_upload_folder(self) -> None:
        """Create upload directory if it doesn't exist."""
        self._upload_folder.mkdir(parents=True, exist_ok=True)

    async def load_metadata(self) -> None:
        """Load document metadata from JSON file."""
        async with self._lock:
            if not self._metadata_file.exists():
                return
            try:
                async with aiofiles.open(self._metadata_file, "r") as f:
                    content = await f.read()
                    data = json.loads(content)
                    for doc_data in data:
                        doc_data["document_id"] = uuid.UUID(doc_data["document_id"])
                        doc_data["upload_date"] = datetime.fromisoformat(
                            doc_data["upload_date"]
                        )
                        doc = DocumentMetadata(**doc_data)
                        self._documents[doc.document_id] = doc
                logger.info(f"Loaded {len(self._documents)} document(s) metadata")
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")

    async def save_metadata(self) -> None:
        """Save document metadata to JSON file."""
        async with self._lock:
            data = []
            for doc in self._documents.values():
                data.append(
                    {
                        "document_id": str(doc.document_id),
                        "filename": doc.filename,
                        "upload_date": doc.upload_date.isoformat(),
                        "file_size": doc.file_size,
                        "status": doc.status,
                    }
                )
            try:
                async with aiofiles.open(self._metadata_file, "w") as f:
                    await f.write(json.dumps(data, indent=2))
                logger.info(f"Saved {len(data)} document(s) metadata")
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")

    async def create_document(self, filename: str, file_data: bytes) -> DocumentMetadata:
        """
        Create a new document: store the PDF file and its metadata.

        Args:
            filename: Original uploaded filename.
            file_data: Raw file content as bytes.

        Returns:
            DocumentMetadata object.

        Raises:
            FileTooLargeError, InvalidFileTypeError (handled by caller).
        """
        doc_id = uuid.uuid4()
        file_path = self._upload_folder / f"{doc_id}.pdf"
        file_size = len(file_data)

        # Write file asynchronously
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_data)

        doc = DocumentMetadata(
            document_id=doc_id,
            filename=filename,
            upload_date=datetime.now(timezone.utc),
            file_size=file_size,
            status="uploaded",
        )

        async with self._lock:
            self._documents[doc_id] = doc

        await self.save_metadata()
        logger.info(f"Document {doc_id} created: {filename}")
        return doc

    async def get_document(self, document_id: uuid.UUID) -> DocumentMetadata:
        """
        Retrieve metadata for a single document.

        Raises:
            DocumentNotFoundError if the document does not exist.
        """
        async with self._lock:
            doc = self._documents.get(document_id)
            if not doc:
                raise DocumentNotFoundError(str(document_id))
            return doc

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """
        Delete a document: remove its file and metadata.

        Raises:
            DocumentNotFoundError if the document does not exist.
        """
        doc = await self.get_document(document_id)  # ensures existence
        file_path = self._upload_folder / f"{document_id}.pdf"
        if file_path.exists():
            os.remove(file_path)

        async with self._lock:
            del self._documents[document_id]

        await self.save_metadata()
        logger.info(f"Document {document_id} deleted")

    async def list_documents(self) -> List[DocumentMetadata]:
        """Return a list of all document metadata."""
        async with self._lock:
            return list(self._documents.values())