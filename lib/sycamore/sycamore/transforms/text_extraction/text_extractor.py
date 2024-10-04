from typing import List, Union, TYPE_CHECKING
from abc import abstractmethod
from sycamore.data import Element
from io import IOBase

if TYPE_CHECKING:
    from PIL.Image import Image
    from pdfminer.layout import LTPage


class TextExtractor:
    @abstractmethod
    def extract_page(self, filename: Union["Image", "LTPage"]) -> List[Element]:
        pass

    @abstractmethod
    def extract_document(
        self, filename: Union[str, IOBase], hash_key: str, use_cache=False, **kwargs
    ) -> List[List[Element]]:
        pass

    def __name__(self):
        return "TextExtractor"
