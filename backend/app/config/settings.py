from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    OPENAI_API_KEY: Optional[str] = None
    QDRANT_URL: Optional[str] = None
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    UPLOAD_FOLDER: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    LOG_LEVEL: str = "INFO"
    METADATA_FILE: str = "./metadata.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()