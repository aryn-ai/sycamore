"""
Utilities to deal with vectors, rotation, coordinates, images, transformation
matrices, etc.

A lot of this deal with a "quad" which is a 90-degree rotation, usually
counterclockwise, as in math class.

Keep in mind that our coordinate system puts 0,0 at the top left and 1,1 at
the bottom right.
"""

import math
import cmath
from typing import Any, Iterable

from PIL import Image

from sycamore.data import BoundingBox


g_quad_to_method = {
    1: Image.Transpose.ROTATE_90,
    2: Image.Transpose.ROTATE_180,
    3: Image.Transpose.ROTATE_270,
}


def rot_image(img: Image.Image, quad: int) -> Image.Image:
    method = g_quad_to_method.get(quad % 4)
    if not method:
        return img
    return img.transpose(method=method)


def rot_xy(x: float, y: float, quad: int) -> tuple[float, float]:
    """Rotate counterclockwise quad * 90 degrees about (0.5, 0.5)"""
    if quad:
        quad %= 4
        if quad == 1:
            return (y, 1.0 - x)
        elif quad == 2:
            return (1.0 - x, 1.0 - y)
        elif quad == 3:
            return (1.0 - y, x)
    return (x, y)


def rot_bbox(bbox: BoundingBox, quad: int) -> BoundingBox:
    if not quad:
        return bbox
    x1, y1 = rot_xy(bbox.x1, bbox.y1, quad)
    x2, y2 = rot_xy(bbox.x2, bbox.y2, quad)
    return BoundingBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def rot_dict(bbox: dict[str, float], quad: int) -> dict[str, float]:
    if not quad:
        return bbox
    x1, y1 = rot_xy(bbox["x1"], bbox["y1"], quad)
    x2, y2 = rot_xy(bbox["x2"], bbox["y2"], quad)
    return {"x1": min(x1, x2), "y1": min(y1, y2), "x2": max(x1, x2), "y2": max(y1, y2)}


def rot_tuple(bbox: tuple[float, float, float, float], quad: int) -> tuple[float, float, float, float]:
    if not quad:
        return bbox
    x1, y1 = rot_xy(bbox[0], bbox[1], quad)
    x2, y2 = rot_xy(bbox[2], bbox[3], quad)
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def quad_rotation(vec: complex, thres: float = 0.8) -> int:
    """Returns number of quadrants counterclockwise that a vector is rotated."""
    if abs(vec) < thres:
        return 0
    rad = cmath.phase(vec)
    quad = round(rad * 2.0 / math.pi) % 4
    return quad


def vnorm(vec: complex) -> complex:
    """Normalize to unit length."""
    ln = abs(vec)
    return vec / ln if ln else vec


class VectorMean:
    """Sums vectors and divides by the count, yielding an average vector."""

    def __init__(self) -> None:
        self.vec = complex(0.0, 0.0)
        self.cnt = 0

    def add(self, v: complex) -> None:
        self.vec += v
        self.cnt += 1

    def get(self) -> complex:
        if cnt := self.cnt:
            return self.vec / cnt
        return self.vec


def vector_mean_attr_norm_recurse(obj: Any, attr: str, vm: VectorMean) -> None:
    if isinstance(obj, Iterable):
        for sub in obj:
            vector_mean_attr_norm_recurse(sub, attr, vm)
    elif (mat := getattr(obj, attr, None)) is not None:
        if isinstance(mat, complex):
            vm.add(mat)
        else:
            vm.add(vnorm(complex(mat[0], mat[1])))  # useful assumption


def vector_mean_attr_norm(obj: Any, attr: str) -> VectorMean:
    vm = VectorMean()
    vector_mean_attr_norm_recurse(obj, attr, vm)
    return vm
