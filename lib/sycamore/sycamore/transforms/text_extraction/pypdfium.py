from contextlib import closing
from io import IOBase
from typing import Optional, Union, TYPE_CHECKING, Generator

from PIL import Image

from sycamore.data import BoundingBox, Element
from sycamore.transforms.text_extraction.text_extractor import PdfTextExtractor
from sycamore.utils.import_utils import requires_modules


if TYPE_CHECKING:
    from pypdfium2 import PdfTextPage


# TODO Caching
class PyPdfiumTextExtractor(PdfTextExtractor["PdfTextPage"]):

    @classmethod
    @requires_modules(["pypdfium2"], extra="local-inference")
    def pdf_to_pages(cls, file_name: Union[str, IOBase]) -> Generator["PdfTextPage", None, None]:
        from pypdfium2 import PdfDocument

        with closing(PdfDocument(file_name)) as pdf:
            for page in pdf:
                yield page.get_textpage()

    def extract_page(self, page: Optional["PdfTextPage"]) -> list[Element]:
        assert page is not None, "Page cannot be None for PyPdfiumTextExtractor"

        page_data = []
        width = page.page.get_width()
        height = page.page.get_height()

        for idx in range(page.count_rects()):
            rect = page.get_rect(idx)
            text = page.get_text_bounded(*rect)

            x1, y1, x2, y2 = self._convert_bbox_coordinates(rect, page.page.get_height())

            # TODO Figure out how to get font size. The C api has a method to
            # get the font size for a specific character, but not for the
            # whole text block.

            page_data.append({"bbox": BoundingBox(x1, y1, x2, y2), "text": text})

        return self.parse_output(page_data, width, height)
