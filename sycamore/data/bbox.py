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
