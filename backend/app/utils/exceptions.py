from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base exception for application errors."""

    def __init__(self, message: str, status_code: int = 400, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class DocumentNotFoundError(AppException):
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document with id {document_id} not found",
            status_code=404,
            details={"document_id": document_id},
        )


class InvalidFileTypeError(AppException):
    def __init__(self, filename: str):
        super().__init__(
            message=f"Invalid file type. Only PDF allowed. Received: {filename}",
            status_code=400,
            details={"filename": filename},
        )


class FileTooLargeError(AppException):
    def __init__(self, max_size: int):
        super().__init__(
            message=f"File size exceeds maximum allowed ({max_size} bytes)",
            status_code=413,
            details={"max_size": max_size},
        )


def setup_exception_handlers(app):
    """Register global exception handlers for the FastAPI app."""

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.error(f"AppException: {exc.message}", exc_info=True)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.message,
                "status": exc.status_code,
                "errors": exc.details,
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        logger.error(f"HTTPException: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "status": exc.status_code},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "Validation error",
                "status": 422,
                "errors": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "status": 500},
        )