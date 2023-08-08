from collections import UserDict
from typing import (Any, Dict, List, Optional, Union)


class Element(UserDict):
    def __init__(self, element=None, /, **kwargs):
        super().__init__(element, **kwargs)
        default = {
            "type": None,
            "content": {
                "binary": None,
                "text": None,
            },
            "properties": {}
        }
        for k, v in default.items():
            if k not in self.data:
                self.data[k] = v

        if "binary" not in self.data["content"]:
            self.data["content"]["binary"] = None

        if "text" not in self.data["content"]:
            self.data["content"]["text"] = None

    @property
    def type(self) -> Optional[str]:
        return self.data["type"]

    @type.setter
    def type(self, value: str) -> None:
        self.data["type"] = value

    @property
    def content(self) -> Dict[str, Union[str, bytes]]:
        return self.data["content"]

    @content.deleter
    def content(self) -> None:
        self.data["content"] = {
            "binary": None,
            "text": None,
        }

    @property
    def properties(self) -> Optional[Dict[str, Any]]:
        return self.data["properties"]

    @properties.deleter
    def properties(self) -> None:
        self.data["properties"] = {}

    def to_dict(self) -> Dict[str, Any]:
        return self.data


class Document(UserDict):
    def __init__(self, document=None, /, **kwargs):
        super().__init__(document, **kwargs)
        default = {
            "doc_id": None,
            "type": None,
            "content": {
                "binary": None,
                "text": None,
            },
            "elements": {"array": []},
            "embedding": {
                "binary": None,
                "text": None,
            },
            "parent_id": None,
            "properties": {}
        }
        for k, v in default.items():
            if k not in self.data:
                self.data[k] = v

        if "binary" not in self.data["content"]:
            self.data["content"]["binary"] = None

        if "text" not in self.data["content"]:
            self.data["content"]["text"] = None

        elements = \
            [Element(element) for element in self.data["elements"]["array"]]
        self.data["elements"]["array"] = elements

        if "binary" not in self.data["embedding"]:
            self.data["embedding"]["binary"] = None

        if "text" not in self.data["embedding"]:
            self.data["embedding"]["text"] = None

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
    def content(self) -> Dict[str, Union[str, bytes]]:
        return self.data["content"]

    @content.deleter
    def content(self) -> None:
        self.data["content"] = {
            "binary": None,
            "text": None,
        }

    @property
    def elements(self) -> Optional[List[Element]]:
        return self.data["elements"]["array"]

    @elements.deleter
    def elements(self) -> None:
        self.data["elements"] = {"array": []}

    @property
    def embedding(self) -> Dict[str, List[List[float]]]:
        return self.data["embedding"]

    @embedding.deleter
    def embedding(self) -> None:
        self.data["embedding"] = {
            "binary": None,
            "text": None,
        }

    @property
    def parent_id(self) -> Optional[str]:
        return self.data["parent_id"]

    @parent_id.setter
    def parent_id(self, value: str) -> None:
        self.data["parent_id"] = value

    @property
    def properties(self) -> Optional[Dict[str, Any]]:
        return self.data["properties"]

    @properties.deleter
    def properties(self) -> None:
        self.data["properties"] = {}

    def to_dict(self) -> Dict[str, Any]:
        dicts = \
            [element.to_dict() for element in self.data["elements"]["array"]]
        self.data["elements"]["array"] = dicts
        return self.data
