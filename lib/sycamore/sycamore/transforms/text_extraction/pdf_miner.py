from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.utils import open_filename
from sycamore.data import Element, BoundingBox
from sycamore.utils.cache import DiskCache
from io import IOBase
from typing import BinaryIO, Tuple, List, Union, cast
from pathlib import Path
from sycamore.utils.import_utils import requires_modules
import logging
from sycamore.transforms.text_extraction import TextExtractor


logger = logging.getLogger(__name__)

pdf_miner_cache = DiskCache(str(Path.home() / ".sycamore/PDFMinerCache"))


class PDFMinerExtractor(TextExtractor):
    @requires_modules(["pdfminer", "pdfminer.utils"], extra="local-inference")
    def __init__(self):
        rm = PDFResourceManager()
        param = LAParams()
        self.device = PDFPageAggregator(rm, laparams=param)
        self.interpreter = PDFPageInterpreter(rm, self.device)

    def _open_pdfminer_pages_generator(self, fp: BinaryIO):
        pages = PDFPage.get_pages(fp)
        for page in pages:
            self.interpreter.process_page(page)
            page_layout = self.device.get_result()
            yield page, page_layout

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

    def extract(self, filename: Union[str, IOBase], hash_key: str, use_cache=False, **kwargs) -> List[List[Element]]:
        # The naming is slightly confusing, but `open_filename` accepts either
        # a filename (str) or a file-like object (IOBase)

        cached_result = pdf_miner_cache.get(hash_key) if use_cache else None
        if cached_result:
            logger.info(f"Cache Hit for PDFMiner. Cache hit-rate is {pdf_miner_cache.get_hit_rate()}")
            return cached_result
        else:
            with open_filename(filename, "rb") as fp:
                fp = cast(BinaryIO, fp)
                pages = []
                for page, page_layout in self._open_pdfminer_pages_generator(fp):
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
