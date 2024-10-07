from collections import UserDict
from io import BytesIO
import json
from typing import Any, Optional

from PIL import Image

from sycamore.data.bbox import BoundingBox
from sycamore.data.table import Table


class Element(UserDict):
    """
    It is often useful to process different parts of a document separately. For example, you might want to process
    tables differently than text paragraphs, and typically small chunks of text are embedded separately for vector
    search. In Sycamore, these chunks are called elements. Like documents, elements contain a text or binary
    representations and collection of properties that can be set by the user or by built-in transforms.
    """

    def __init__(self, element=None, /, **kwargs):
        super().__init__(element, **kwargs)
        if "properties" not in self.data:
            self.data["properties"] = {}

    @property
    def element_index(self) -> Optional[int]:
        """A unique identifier for the element within a Document. Represents an order within the document"""
        return self.data.get("properties", {}).get("_element_index")

    @element_index.setter
    def element_index(self, value: int) -> None:
        """Set the unique identifier of the element within a Document."""
        self.data["properties"]["_element_index"] = value

    @property
    def type(self) -> Optional[str]:
        return self.data.get("type")

    @type.setter
    def type(self, value: str) -> None:
        self.data["type"] = value

    @property
    def text_representation(self) -> Optional[str]:
        return self.data.get("text_representation")

    @text_representation.setter
    def text_representation(self, value: str) -> None:
        self.data["text_representation"] = value

    @property
    def binary_representation(self) -> Optional[bytes]:
        return self.data.get("binary_representation")

    @binary_representation.setter
    def binary_representation(self, value: bytes) -> None:
        self.data["binary_representation"] = value

    @property
    def bbox(self) -> Optional[BoundingBox]:
        return None if self.data.get("bbox") is None else BoundingBox(*self.data["bbox"])

    @bbox.setter
    def bbox(self, bbox: BoundingBox) -> None:
        self.data["bbox"] = bbox.coordinates

    @property
    def properties(self) -> dict[str, Any]:
        return self.data.get("properties", None)

    @properties.setter
    def properties(self, properties: dict[str, Any]):
        self.data["properties"] = properties

    @properties.deleter
    def properties(self) -> None:
        self.data["properties"] = {}

    def __str__(self) -> str:
        """Return a pretty-printed string representing this Element."""
        d = {
            "type": self.type,
            "text_representation": self.text_representation[0:40] + "..." if self.text_representation else None,
            "binary_representation": (
                f"<{len(self.binary_representation)} bytes>" if self.binary_representation else None
            ),
            "bbox": str(self.bbox),
            "properties": {k: str(v) for k, v in self.properties.items()},
        }
        return json.dumps(d, indent=2)

    def field_to_value(self, field: str) -> Any:
        """
        Extracts the value for a particular element field.

        Args:
            field: The field in dotted notation to indicate nesting, e.g. properties.schema

        Returns:
            The value associated with the document field.
            Returns None if field does not exist in document.
        """
        from sycamore.utils.nested import dotted_lookup

        return dotted_lookup(self, field)


class ImageElement(Element):
    def __init__(
        self,
        element=None,
        image_size: Optional[tuple[int, int]] = None,
        image_mode: Optional[str] = None,
        image_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(element, **kwargs)
        self.data["type"] = "Image"
        self.data["properties"]["image_size"] = image_size
        self.data["properties"]["image_mode"] = image_mode
        self.data["properties"]["image_format"] = image_format

    def as_image(self) -> Optional[Image.Image]:
        if self.binary_representation is None:
            return None
        if self.image_format is None:
            if self.image_mode is None or self.image_size is None:
                return None
            # Image is stored in uncompressed PIL format.
            return Image.frombytes(mode=self.image_mode, size=self.image_size, data=self.binary_representation)
        else:
            # Image is stored in format like JPG/PNG.
            return Image.open(BytesIO(self.binary_representation))

    @property
    def image_size(self) -> Optional[tuple[int, int]]:
        if (properties := self.data.get("properties")) is None:
            return None
        else:
            return properties.get("image_size")

    @image_size.setter
    def image_size(self, image_size: Optional[tuple[int, int]]) -> None:
        self.data["properties"]["image_size"] = image_size

    @property
    def image_mode(self) -> Optional[str]:
        if (properties := self.data.get("properties")) is None:
            return None
        else:
            return properties.get("image_mode")

    @image_mode.setter
    def image_mode(self, image_mode: Optional[str]) -> None:
        self.data["properties"]["image_mode"] = image_mode

    @property
    def image_format(self) -> Optional[str]:
        if (properties := self.data.get("properties")) is None:
            return None
        else:
            return properties.get("image_format")

    @image_format.setter
    def image_format(self, image_format: Optional[str]) -> None:
        self.data["properties"]["image_format"] = image_format


class TableElement(Element):
    def __init__(
        self,
        element=None,
        title: Optional[str] = None,
        columns: Optional[list[str]] = None,
        rows: Optional[list[Any]] = None,
        table: Optional[Table] = None,
        tokens: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(element, **kwargs)
        self.data["type"] = "table"
        self.data["properties"]["title"] = title
        self.data["properties"]["columns"] = columns
        self.data["properties"]["rows"] = rows
        self.data["table"] = table
        self.data["tokens"] = tokens

    @property
    def rows(self) -> Optional[list[Any]]:
        if (properties := self.data.get("properties")) is None:
            return None
        else:
            return properties.get("rows")

    @rows.setter
    def rows(self, rows: Optional[list[Any]] = None) -> None:
        self.data["properties"]["rows"] = rows

    @property
    def columns(self) -> Optional[list[str]]:
        if (properties := self.data.get("properties")) is None:
            return None
        else:
            return properties.get("columns")

    @columns.setter
    def columns(self, columns: Optional[list[str]] = None) -> None:
        self.data["properties"]["columns"] = columns

    @property
    def table(self) -> Optional[Table]:
        return self.data.get("table", None)

    @table.setter
    def table(self, value: Optional[Table]) -> None:
        self.data["table"] = value
        self.data["text_representation"] = None  # Invalidate cache

    @property
    def tokens(self) -> Optional[list[dict[str, Any]]]:
        return self.data.get("tokens", None)

    @tokens.setter
    def tokens(self, tokens: list[dict[str, Any]]) -> None:
        self.data["tokens"] = tokens

    @property
    def text_representation(self) -> Optional[str]:
        tr = self.data.get("text_representation")
        if not isinstance(tr, str) and (tbl := self.data.get("table")):
            tr = tbl.to_csv()
            self.data["text_representation"] = tr
        return tr

    @text_representation.setter
    def text_representation(self, text_representation: str) -> None:
        self.data["text_representation"] = text_representation


def create_element(element_index: Optional[int] = None, **kwargs) -> Element:
    element: Element
    if "type" in kwargs and kwargs["type"].lower() == "table":
        if "properties" in kwargs:
            props = kwargs["properties"]
            kwargs["title"] = props.get("title")
            kwargs["columns"] = props.get("columns")
            kwargs["rows"] = props.get("rows")
        if "table" in kwargs and isinstance(kwargs["table"], dict):
            table = Table.from_dict(kwargs["table"])
            kwargs["table"] = table

        element = TableElement(**kwargs)

    elif "type" in kwargs and kwargs["type"].lower() in {"picture", "image", "figure"}:
        if "properties" in kwargs:
            props = kwargs["properties"]
            kwargs["image_size"] = props.get("image_size")
            kwargs["image_mode"] = props.get("image_mode")
            kwargs["image_format"] = props.get("image_format")

        element = ImageElement(**kwargs)

    else:
        element = Element(**kwargs)
    if element_index is not None:
        element.element_index = element_index
    return element
