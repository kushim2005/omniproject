from pydantic import BaseModel
from typing import Optional, Dict, Any


class ErrorResponse(BaseModel):
    """Structured error response for API clients."""

    detail: str
    status: int
    errors: Optional[Dict[str, Any]] = None