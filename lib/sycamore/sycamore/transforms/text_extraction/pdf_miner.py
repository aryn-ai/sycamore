from io import IOBase
from sycamore.data import Element, BoundingBox
from sycamore.utils.cache import DiskCache
from typing import Any, BinaryIO, Tuple, Iterable, Literal, Optional, cast, Generator, TYPE_CHECKING, Union
from pathlib import Path
from sycamore.utils.import_utils import requires_modules
from sycamore.utils.time_trace import timetrace
from sycamore.transforms.text_extraction.text_extractor import PdfTextExtractor
import logging

if TYPE_CHECKING:
    from PIL.Image import Image
    from pdfminer.pdfpage import PDFPage

logger = logging.getLogger(__name__)

# TODO: Add cache support for PDFMiner per page
pdf_miner_cache = DiskCache(str(Path.home() / ".sycamore/PDFMinerCache"))


@requires_modules(["pdfminer.layout"], extra="local-inference")
def _enumerate_objs(page_layout, target_type: str):
    from pdfminer.layout import LTTextLine

    for obj in page_layout:
        if not hasattr(obj, "get_text"):
            continue
        if target_type == "boxes" or isinstance(obj, LTTextLine):
            yield obj
        elif isinstance(obj, Iterable):
            yield from _enumerate_objs(obj, target_type)


class PdfMinerExtractor(PdfTextExtractor["PDFPage"]):

    # TODO: Switch the default to lines once we are confident there aren't any regressions.
    @requires_modules(["pdfminer", "pdfminer.utils"], extra="local-inference")
    def __init__(self, object_type: Literal["boxes", "lines"] = "boxes"):
        from pdfminer.converter import PDFPageAggregator
        from pdfminer.layout import LAParams
        from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager

        rm = PDFResourceManager()
        param = LAParams(detect_vertical=True)
        self.device = PDFPageAggregator(rm, laparams=param)
        self.interpreter = PDFPageInterpreter(rm, self.device)
        self.object_type = object_type

    @classmethod
    @requires_modules(["pdfminer", "pdfminer.utils"], extra="local-inference")
    def pdf_to_pages(cls, file_name: Union[str, IOBase]) -> Generator["PDFPage", None, None]:
        from pdfminer.utils import open_filename
        from pdfminer.pdfpage import PDFPage

        with open_filename(file_name, "rb") as fp:
            fp = cast(BinaryIO, fp)
            pages = PDFPage.get_pages(fp)
            for page in pages:
                yield page

    @staticmethod
    def _get_font_size(objs) -> float:
        font_size_list = []

        def traverse(objs):
            for obj in objs:
                if isinstance(obj, Iterable):
                    traverse(obj)
                elif hasattr(obj, "fontname"):
                    font_size_list.append(obj.size)

        traverse(objs)
        return sum(font_size_list) / len(font_size_list)

    @timetrace("PdfMinerPageEx")
    def extract_page(self, page: Optional[Union["PDFPage", "Image"]]) -> list[Element]:
        from pdfminer.pdfpage import PDFPage

        assert isinstance(page, PDFPage)
        page_data: list[dict[str, Any]] = []
        self.interpreter.process_page(page)
        page_layout = self.device.get_result()

        for obj in _enumerate_objs(page_layout, self.object_type):
            x1, y1, x2, y2 = self._convert_bbox_coordinates(obj.bbox, page_layout.height)
            page_data.append(
                {
                    "bbox": BoundingBox(x1, y1, x2, y2),
                    "text": obj.get_text(),
                    "font_size": PdfMinerExtractor._get_font_size(obj),
                }
            )

        return self.parse_output(page_data, page_layout.width, page_layout.height)

    def __name__(self):
        return "PdfMinerExtractor"
