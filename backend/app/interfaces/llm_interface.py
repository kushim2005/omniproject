from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMInterface(ABC):
    """Abstract interface for LLM-backed generation and rewriting."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from a prompt."""
        raise NotImplementedError

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Embed a single text chunk."""
        raise NotImplementedError
