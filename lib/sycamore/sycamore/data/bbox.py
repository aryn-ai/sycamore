from abc import ABC
from collections.abc import Iterable
import logging


class BoundingBox(ABC):
    """
    Defines a bounding box by top left and bottom right coordinates, coordinates units are ratio over the whole
     document width or height.
        (x1, y1) ------
        |             |
        |             |
        -------(x2, y2)
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        # TODO: Make this as an assertion and fix any bugs in TableTransformers that violate it
        if x1 > x2 or y1 > y2:
            logging.warning(f"x1 ({x1}) must be <= x2 ({x2}) and y1 ({y1}) must be <= y2 ({y2})")

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        if self.x1 != other.x1 or self.x2 != other.x2 or self.y1 != other.y1 or self.y2 != other.y2:
            return False
        return True

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2))

    @classmethod
    def from_union(cls, boxes: Iterable["BoundingBox"]) -> "BoundingBox":
        """Returns the BoundingBox formed by unioning the specified sequence of BoundingBoxes."""

        bbox = EMPTY_BBOX.copy()
        for new_box in iter(boxes):
            bbox.union_self(new_box)

        return bbox

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def coordinates(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def copy(self):
        return BoundingBox(*self.coordinates)

    def iou(self, other: "BoundingBox") -> float:
        """
        Intersection over union, we usually use this for detecting if two bbox cover the same entity.
        """
        i = self.intersect(other).area

        # Area of the union is sum of areas minus area of the intersection.
        u = (self.area + other.area) - i

        return i / u

    def iob(self, other: "BoundingBox") -> float:
        """Intersection over bounding box. This is another metric used for comparing overlap."""

        if self.area > 0:
            return self.intersect(other).area / self.area
        return 0

    def contains(self, other: "BoundingBox") -> bool:
        return self.x1 <= other.x1 and self.x2 >= other.x2 and self.y1 <= other.y1 and self.y2 >= other.y2

    def intersect(self, other: "BoundingBox") -> "BoundingBox":
        x1 = max(self.x1, other.x1)
        x2 = min(self.x2, other.x2)
        y1 = max(self.y1, other.y1)
        y2 = min(self.y2, other.y2)

        if x1 >= x2 or y1 >= y2:
            return EMPTY_BBOX

        return BoundingBox(x1, y1, x2, y2)

    def translate(self, x, y) -> "BoundingBox":
        """Translates (moves) this BoundingBox by the specified x and y values.

        x and y may be negative.
        """
        # TODO: Do we want to allow the coordinates to be negative or truncate in the case that one or
        # more of the coordinates goes < 0?
        return self.copy().translate_self(x, y)

    def translate_self(self, x, y) -> "BoundingBox":
        self.x1 += x
        self.y1 += y
        self.x2 += x
        self.y2 += y
        return self

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return self.copy().union_self(other)

    def union_self(self, other: "BoundingBox") -> "BoundingBox":
        """Updates this BoundingBox in place to include the specified BoundingBox.

        Note that A union B == B union A == A if B is empty,
        regardless of the precise x and y values in B.
        """

        if self.is_empty():
            self.x1, self.y1, self.x2, self.y2 = other.coordinates
        elif other.is_empty():
            pass
        else:
            self.x1 = min(self.x1, other.x1)
            self.x2 = max(self.x2, other.x2)
            self.y1 = min(self.y1, other.y1)
            self.y2 = max(self.y2, other.y2)
        return self

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_relative(self, width, height) -> "BoundingBox":
        """Converts this bounding box to be relative to the corresponding width and height."""

        return self.copy().to_relative_self(width, height)

    def to_relative_self(self, width, height) -> "BoundingBox":
        """Converts this bounding box to be relative to the corresponding width and height."""

        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0. Got ({width}, {height})")

        self.x1 = self.x1 / width
        self.y1 = self.y1 / height
        self.x2 = self.x2 / width
        self.y2 = self.y2 / height

        return self

    def to_absolute(self, width, height) -> "BoundingBox":
        return self.copy().to_absolute_self(width, height)

    def to_absolute_self(self, width, height) -> "BoundingBox":
        self.x1 = self.x1 * width
        self.y1 = self.y1 * height
        self.x2 = self.x2 * width
        self.y2 = self.y2 * height
        return self

    def is_empty(self):
        return self.x1 >= self.x2 or self.y1 >= self.y2

    def __repr__(self):
        return f"BoundingBox({self.x1}, {self.y1}, {self.x2}, {self.y2})"


EMPTY_BBOX = BoundingBox(0, 0, 0, 0)
