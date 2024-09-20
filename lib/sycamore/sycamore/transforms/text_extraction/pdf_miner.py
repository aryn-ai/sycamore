from sycamore.data import Element, BoundingBox
from sycamore.utils.cache import DiskCache
from typing import BinaryIO, Tuple, List, cast, Generator, TYPE_CHECKING, Union
from pathlib import Path
from sycamore.utils.import_utils import requires_modules
from sycamore.utils.time_trace import timetrace
from sycamore.transforms.text_extraction.text_extractor import TextExtractor
import logging

if TYPE_CHECKING:
    from PIL.Image import Image
    from pdfminer.layout import LTPage

logger = logging.getLogger(__name__)

# TODO: Add cache support for PDFMiner per page
pdf_miner_cache = DiskCache(str(Path.home() / ".sycamore/PDFMinerCache"))


class PDFMinerExtractor(TextExtractor):
    @requires_modules(["pdfminer", "pdfminer.utils"], extra="local-inference")
    def __init__(self):
        from pdfminer.converter import PDFPageAggregator
        from pdfminer.layout import LAParams
        from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager

        rm = PDFResourceManager()
        param = LAParams()
        self.device = PDFPageAggregator(rm, laparams=param)
        self.interpreter = PDFPageInterpreter(rm, self.device)

    @staticmethod
    @requires_modules(["pdfminer", "pdfminer.utils"], extra="local-inference")
    def pdf_to_pages(file_name: str) -> Generator["LTPage", None, None]:
        from pdfminer.converter import PDFPageAggregator
        from pdfminer.layout import LAParams
        from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
        from pdfminer.pdfpage import PDFPage
        from pdfminer.utils import open_filename

        rm = PDFResourceManager()
        param = LAParams()
        device = PDFPageAggregator(rm, laparams=param)
        interpreter = PDFPageInterpreter(rm, device)
        with open_filename(file_name, "rb") as fp:
            fp = cast(BinaryIO, fp)
            pages = PDFPage.get_pages(fp)
            for page in pages:
                interpreter.process_page(page)
                page_layout = device.get_result()
                yield page_layout

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

    @timetrace("PDFMiner Document Extraction")
    # TODO: Remove this function once the service is moved off it
    def extract_document(self, filename: str, hash_key: str, use_cache=False, **kwargs) -> List[List[Element]]:
        cached_result = pdf_miner_cache.get(hash_key) if use_cache else None
        if cached_result:
            logger.info(f"Cache Hit for PDFMiner. Cache hit-rate is {pdf_miner_cache.get_hit_rate()}")
            return cached_result
        else:
            pages = []
            for page_layout in PDFMinerExtractor.pdf_to_pages(filename):
                width = page_layout.width
                height = page_layout.height
                texts: List[Element] = []
                for obj in page_layout:
                    x1, y1, x2, y2 = self._convert_bbox_coordinates(obj.bbox, height)

                    if hasattr(obj, "get_text"):
                        text = Element()
                        text.type = "text"
                        text.bbox = BoundingBox(x1 / width, y1 / height, x2 / width, y2 / height)
                        text.text_representation = obj.get_text()
                        if text.text_representation:
                            texts.append(text)
                pages.append(texts)
            if use_cache:
                logger.info("Cache Miss for PDFMiner. Storing the result to the cache.")
                pdf_miner_cache.set(hash_key, pages)
            return pages

    def extract_page(self, page: Union["LTPage", "Image"]) -> List[Element]:
        from pdfminer.layout import LTPage

        assert isinstance(page, LTPage)
        width = page.width
        height = page.height
        texts: List[Element] = []
        for obj in page:
            x1, y1, x2, y2 = self._convert_bbox_coordinates(obj.bbox, height)

            if hasattr(obj, "get_text"):
                text = Element()
                text.type = "text"
                text.bbox = BoundingBox(x1 / width, y1 / height, x2 / width, y2 / height)
                text.text_representation = obj.get_text()
                if text.text_representation:
                    texts.append(text)
        return texts

    def __name__(self):
        return "PDFMinerExtractor"
