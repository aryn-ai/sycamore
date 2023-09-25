from collections import UserDict
from typing import Any, Optional


class Element(UserDict):
    def __init__(self, element=None, /, **kwargs):
        super().__init__(element, **kwargs)
        default = {
            "type": None,
            "text_representation": None,
            "binary_representation": None,
            "properties": {},
        }
        for k, v in default.items():
            if k not in self.data:
                self.data[k] = v

    @property
    def type(self) -> Optional[str]:
        return self.data["type"]

    @type.setter
    def type(self, value: str) -> None:
        self.data["type"] = value

    @property
    def text_representation(self) -> Optional[str]:
        return self.data["text_representation"]

    @text_representation.setter
    def text_representation(self, value: str) -> None:
        self.data["text_representation"] = value

    @property
    def binary_representation(self) -> Optional[str]:
        return self.data["binary_representation"]

    @binary_representation.setter
    def binary_representation(self, value: str) -> None:
        self.data["binary_representation"] = value

    @property
    def properties(self) -> dict[str, Any]:
        return self.data["properties"]

    @properties.deleter
    def properties(self) -> None:
        self.data["properties"] = {}

    def to_dict(self) -> dict[str, Any]:
        return self.data


class TableElement(Element):
    def __init__(
        self,
        element=None,
        title: Optional[str] = None,
        columns: Optional[list[str]] = None,
        rows: Optional[list[Any]] = None,
        **kwargs,
    ):
        super().__init__(element, **kwargs)
        self.data["type"] = "table"
        self.data["properties"]["title"] = title
        self.data["properties"]["columns"] = columns
        self.data["properties"]["rows"] = rows

    @property
    def rows(self) -> Optional[list[Any]]:
        return self.data["properties"]["rows"]

    @rows.setter
    def rows(self, rows: Optional[list[Any]] = None) -> None:
        self.data["properties"]["rows"] = rows

    @property
    def columns(self) -> Optional[list[str]]:
        return self.data["properties"]["columns"]

    @columns.setter
    def columns(self, columns: Optional[list[str]] = None) -> None:
        self.data["properties"]["columns"] = columns


class Document(UserDict):
    def __init__(self, document=None, /, **kwargs):
        super().__init__(document, **kwargs)
        default = {
            "doc_id": None,
            "type": None,
            "text_representation": None,
            "binary_representation": None,
            "elements": {"array": []},
            "embedding": None,
            "parent_id": None,
            "properties": {},
        }
        for k, v in default.items():
            if k not in self.data:
                self.data[k] = v

        elements = [Element(element) for element in self.data["elements"]["array"]]
        self.data["elements"]["array"] = elements

    @property
    def doc_id(self) -> Optional[str]:
        return self.data["doc_id"]

    @doc_id.setter
    def doc_id(self, value: str) -> None:
        self.data["doc_id"] = value

    @property
    def type(self) -> Optional[str]:
        return self.data["type"]

    @type.setter
    def type(self, value: str) -> None:
        self.data["type"] = value

    @property
    def text_representation(self) -> Optional[str]:
        return self.data["text_representation"]

    @text_representation.setter
    def text_representation(self, value: str) -> None:
        self.data["text_representation"] = value

    @property
    def binary_representation(self) -> Optional[bytes]:
        return self.data["binary_representation"]

    @binary_representation.setter
    def binary_representation(self, value: bytes) -> None:
        self.data["binary_representation"] = value

    @binary_representation.deleter
    def binary_representation(self) -> None:
        self.data["binary_representation"] = None

    @property
    def elements(self) -> list[Element]:
        return self.data["elements"]["array"]

    @elements.setter
    def elements(self, elements: list[Element]):
        self.data["elements"] = {"array": elements}

    @elements.deleter
    def elements(self) -> None:
        self.data["elements"] = {"array": []}

    @property
    def embedding(self) -> list[list[float]]:
        return self.data["embedding"]

    @embedding.setter
    def embedding(self, embedding: list[list[float]]) -> None:
        self.data["embedding"] = embedding

    @property
    def parent_id(self) -> Optional[str]:
        return self.data["parent_id"]

    @parent_id.setter
    def parent_id(self, value: str) -> None:
        self.data["parent_id"] = value

    @property
    def properties(self) -> dict[str, Any]:
        return self.data["properties"]

    @properties.deleter
    def properties(self) -> None:
        self.data["properties"] = {}

    def to_dict(self) -> dict[str, Any]:
        dicts = [element.to_dict() for element in self.data["elements"]["array"]]
        self.data["elements"]["array"] = dicts
        return self.data
