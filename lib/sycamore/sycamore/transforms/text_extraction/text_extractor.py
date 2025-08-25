import logging
from typing import Union, TYPE_CHECKING, Optional, Any, Generic, Generator, Tuple, TypeVar
from abc import abstractmethod
from sycamore.data import Element, BoundingBox
from io import IOBase
from pathlib import Path

from sycamore.utils.cache import DiskCache

if TYPE_CHECKING:
    from PIL.Image import Image
    from pdfminer.pdfpage import PDFPage


logger = logging.getLogger(__name__)


PageType = TypeVar("PageType")


class TextExtractor(Generic[PageType]):
    @abstractmethod
    def extract_page(self, page: Optional[PageType]) -> list[Element]:
        pass

    @abstractmethod
    def extract_document(
        self, filename: Union[str, IOBase], hash_key: str, use_cache=False, **kwargs
    ) -> list[list[Element]]:
        pass

    def parse_output(self, output: list[dict[str, Any]], width, height) -> list[Element]:
        texts: list[Element] = []
        for obj in output:
            obj_bbox = obj.get("bbox")
            obj_text = obj.get("text")
            if obj_bbox and obj_text and not obj_bbox.is_empty():
                text = Element()
                text.type = "text"
                text.bbox = BoundingBox(
                    obj_bbox.x1 / width,
                    obj_bbox.y1 / height,
                    obj_bbox.x2 / width,
                    obj_bbox.y2 / height,
                )
                text.text_representation = obj_text
                if fs := obj.get("font_size"):
                    text.properties["font_size"] = fs
                if vec := obj.get("vector"):
                    text.data["_vector"] = vec
                texts.append(text)
        return texts

    def __name__(self):
        return "TextExtractor"


pdf_cache = DiskCache(str(Path.home() / ".sycamore/PDFExtractorCache"))


class PdfTextExtractor(TextExtractor[PageType], Generic[PageType]):
    """A base class for text extraction from PDF files."""

    @classmethod
    def _convert_bbox_coordinates(
        cls,
        rect: Tuple[float, float, float, float],
        height: float,
    ) -> Tuple[float, float, float, float]:
        """
        pdf coordinates are different, bottom left is origin, also two diagonal points defining a rectangle is
        (bottom left, upper right), for details, refer
        https://www.leadtools.com/help/leadtools/v19/dh/to/pdf-topics-pdfcoordinatesystem.html
        """
        x1, y2, x2, y1 = rect
        y1 = height - y1
        y2 = height - y2
        return x1, y1, x2, y2

    @classmethod
    @abstractmethod
    def pdf_to_pages(cls, file_name: Union[str, IOBase]) -> Generator[PageType, None, None]:
        """Convert a PDF file to a generator of pages."""
        pass

    def extract_document(
        self, filename: Union[str, IOBase], hash_key: str, use_cache=False, **kwargs
    ) -> list[list[Element]]:
        cached_result = pdf_cache.get(hash_key) if use_cache else None
        if cached_result:
            logger.info(f"Cache Hit for pdf extraction. Cache hit-rate is {pdf_cache.get_hit_rate()}")
            return cached_result
        else:
            pages = []
            for page in self.pdf_to_pages(filename):
                texts = self.extract_page(page)
                pages.append(texts)
            if use_cache:
                logger.info("Cache Miss for pdf extraction. Storing the result to the cache.")
                pdf_cache.set(hash_key, pages)
            return pages
