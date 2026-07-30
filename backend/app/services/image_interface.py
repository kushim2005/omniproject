from abc import ABC, abstractmethod
from typing import Dict, List


class ImageInterface(ABC):
    """Interface for image extraction/analysis. To be implemented by the AI team."""

    @abstractmethod
    async def extract_images(self, pdf_path: str) -> List[Dict]:
        """
        Extract images from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of image metadata or base64-encoded images.
        """
        # Placeholder
        return []