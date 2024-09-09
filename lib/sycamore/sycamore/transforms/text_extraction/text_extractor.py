from typing import List
from abc import abstractmethod
from sycamore.data import Element


class TextExtractor:
    @abstractmethod
    def extract(self, file_name, hash_key, use_cache) -> List[List[Element]]:
        pass

    def __name__():
        return "TextExtractor"
