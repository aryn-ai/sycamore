from sycamore.data import Element, BoundingBox
from sycamore.utils.cache import DiskCache
from typing import BinaryIO, Tuple, cast, Generator, TYPE_CHECKING, Union, Optional, Any
from pathlib import Path
from sycamore.utils.import_utils import requires_modules
from sycamore.utils.time_trace import timetrace
from sycamore.transforms.text_extraction.text_extractor import TextExtractor
import logging

if TYPE_CHECKING:
    from PIL.Image import Image
    from pdfminer.pdfpage import PDFPage

logger = logging.getLogger(__name__)

# TODO: Add cache support for PDFMiner per page
pdf_miner_cache = DiskCache(str(Path.home() / ".sycamore/PDFMinerCache"))


class PdfMinerExtractor(TextExtractor):
    @requires_modules(["pdfminer", "pdfminer.utils"], extra="local-inference")
    def __init__(self):
        from pdfminer.converter import PDFPageAggregator
        from pdfminer.layout import LAParams
        from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager

        self.rm = PDFResourceManager()
        self.param = LAParams()
        self.device = PDFPageAggregator(self.rm, laparams=self.param)
        self.interpreter = PDFPageInterpreter(self.rm, self.device)

    @staticmethod
    @requires_modules(["pdfminer", "pdfminer.utils"], extra="local-inference")
    def pdf_to_pages(file_name: str) -> Generator["PDFPage", None, None]:
        from pdfminer.utils import open_filename
        from pdfminer.pdfpage import PDFPage

        with open_filename(file_name, "rb") as fp:
            fp = cast(BinaryIO, fp)
            pages = PDFPage.get_pages(fp)
            for page in pages:
                yield page

    @staticmethod
    def _convert_bbox_coordinates(
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

    @timetrace("PdfMinerDocEx")
    def extract_document(self, filename: str, hash_key: str, use_cache=False, **kwargs) -> list[list[Element]]:
        cached_result = pdf_miner_cache.get(hash_key) if use_cache else None
        if cached_result:
            logger.info(f"Cache Hit for PdfMiner. Cache hit-rate is {pdf_miner_cache.get_hit_rate()}")
            return cached_result
        else:
            pages = []
            for page in PdfMinerExtractor.pdf_to_pages(filename):
                texts = self.extract_page(page)
                pages.append(texts)
            if use_cache:
                logger.info("Cache Miss for PDFMiner. Storing the result to the cache.")
                pdf_miner_cache.set(hash_key, pages)
            return pages

    @timetrace("PdfMinerPageEx")
    def extract_page(
        self, page: Optional[Union["PDFPage", "Image"]] = None, page_data: list[tuple[Any, Any, Any, Any]] = []
    ) -> list[Element]:
        from pdfminer.pdfpage import PDFPage

        texts: list[Element] = []
        if page:
            assert isinstance(page, PDFPage)
            self.interpreter.process_page(page)
            page_layout = self.device.get_result()
            width = page_layout.width
            height = page_layout.height
            page_data = []
            for obj in page_layout:
                if hasattr(obj, "get_text"):
                    page_data.append((obj.bbox, obj.get_text(), width, height))
        for bbox, bbox_text, width, height in page_data:
            x1, y1, x2, y2 = self._convert_bbox_coordinates(bbox, height)
            text = Element()
            text.type = "text"
            text.bbox = BoundingBox(x1 / width, y1 / height, x2 / width, y2 / height)
            text.text_representation = bbox_text
            if text.text_representation:
                texts.append(text)
        return texts

    def __name__(self):
        return "PdfMinerExtractor"
