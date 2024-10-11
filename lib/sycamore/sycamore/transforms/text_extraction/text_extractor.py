from typing import Union, TYPE_CHECKING, Optional, Any
from abc import abstractmethod
from sycamore.data import Element, BoundingBox
from io import IOBase

if TYPE_CHECKING:
    from PIL.Image import Image
    from pdfminer.pdfpage import PDFPage


class TextExtractor:
    @abstractmethod
    def extract_page(self, filename: Optional[Union["PDFPage", "Image"]]) -> list[Element]:
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
                texts.append(text)
        return texts

    def __name__(self):
        return "TextExtractor"
