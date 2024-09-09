from typing import List, Union
from abc import abstractmethod
from sycamore.data import Element
from io import IOBase


class TextExtractor:
    @abstractmethod
    def extract(self, filename: Union[str, IOBase], hash_key: str, use_cache=False, **kwargs) -> List[List[Element]]:
        pass

    def __name__(self):
        return "TextExtractor"
