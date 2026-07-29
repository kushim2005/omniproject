from abc import ABC, abstractmethod
from typing import List


class EmbeddingInterface(ABC):
    """Interface for embedding generation. To be implemented by the AI team."""

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate vector embeddings for a list of text strings.

        Args:
            texts: List of text chunks.

        Returns:
            List of embedding vectors.
        """
        # Placeholder
        return []