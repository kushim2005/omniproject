import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, upload, chat, documents
from app.config.logging_config import setup_logging
from app.config.settings import settings
from app.middleware.logging_middleware import LoggingMiddleware
from app.services.document_service import DocumentService
from app.utils.exceptions import setup_exception_handlers

# Setup logging early
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    logger.info("Starting OmniBrain Backend")
    # Initialize and load document metadata
    doc_service = DocumentService()
    await doc_service.load_metadata()
    app.state.document_service = doc_service
    yield
    # Shutdown
    logger.info("Shutting down OmniBrain Backend")
    await doc_service.save_metadata()


app = FastAPI(
    title="OmniBrain Backend",
    description="Agentic Multi-Modal RAG Orchestrator - Backend Foundation",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
app.add_middleware(LoggingMiddleware)

# Global exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(health.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/")
async def root():
    """Root endpoint for basic connectivity check."""
    return {"message": "OmniBrain Backend is running"}