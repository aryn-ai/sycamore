from abc import ABC
from typing import Tuple


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

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def coordinates(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def iou(self, other: "BoundingBox") -> float:
        """
        Intersection over union, we usually use this for detecting if two bbox cover the same entity.
        """
        x1 = max(self.x1, other.x1)
        x2 = min(self.x2, other.x2)
        y1 = max(self.y1, other.y1)
        y2 = min(self.y2, other.y2)
        i = 0 if x1 > x2 or y1 > y2 else (x2 - x1) * (y2 - y1)
        u = (self.y2 - self.y1) * (self.x2 - self.x1) + (other.y2 - other.y1) * (other.x2 - other.x1) - i
        return i / u

    def contains(self, other: "BoundingBox") -> bool:
        return self.x1 <= other.x1 and self.x2 >= other.x2 and self.y1 <= other.y1 and self.y2 >= other.y2
