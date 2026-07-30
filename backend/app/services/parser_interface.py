from abc import ABC, abstractmethod
from typing import Dict, List


class ParserInterface(ABC):
    """Interface for PDF parsing. To be implemented by the AI team."""

    @abstractmethod
    async def parse(self, pdf_path: str) -> Dict[str, List[str]]:
        """
        Parse a PDF and extract text chunks.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A dict containing, e.g., {"chunks": [...]}.
        """
        # Placeholder implementation
        return {"chunks": []}