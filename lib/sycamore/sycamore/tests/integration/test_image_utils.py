import pytest

import pdf2image
from PIL import Image
from copy import deepcopy


import sycamore
from sycamore.data import BoundingBox
from sycamore.utils.image_utils import try_draw_boxes
from sycamore.tests.config import TEST_DIR
from sycamore.transforms.partition import SycamorePartitioner


path = str(TEST_DIR / "resources/data/pdfs/Ray_page11.pdf")


@pytest.fixture(scope="module")
def image_boxes() -> list[BoundingBox]:
    context = sycamore.init()
    docs = (
        context.read.binary(paths=[path], binary_format="pdf")
        .partition(partitioner=SycamorePartitioner())
        .explode()
        .take_all()
    )
    return [d.bbox for d in docs if d.bbox is not None]


@pytest.fixture(scope="module")
def source_image_module_scope() -> Image.Image:
    images = pdf2image.convert_from_path(path)
    return images[0].convert(mode="RGBA")


@pytest.fixture(scope="function")
def source_image(source_image_module_scope) -> Image.Image:
    return deepcopy(source_image_module_scope)


# Checks that the image contains blue pixels. This is of course an imperfect check, but
# it at least will tell us if we drew some bounding boxes. on the image. Won't work
# if the image contains blue pixels to begin with. Image must have mode RGBA.
def check_image(image: Image.Image, expected_color=(0, 0, 255, 255)) -> None:
    raw_colors = image.getcolors(64_000)
    assert raw_colors is not None, "Image has too many colors to count"
    assert expected_color in set((color_tup[1] for color_tup in raw_colors)), "Did not draw boxes"


def test_draw_boxes_bbox(source_image, image_boxes) -> None:
    output: Image.Image = try_draw_boxes(source_image, image_boxes)
    check_image(output)


def test_draw_boxes_object_bbox(source_image, image_boxes) -> None:
    class BBoxHolder:
        def __init__(self, bbox):
            self._bbox = bbox

        @property
        def bbox(self):
            return self._bbox

    boxes = [BBoxHolder(b) for b in image_boxes]
    output: Image.Image = try_draw_boxes(source_image, boxes)
    check_image(output)


def test_draw_boxes_coord_list(source_image, image_boxes) -> None:
    boxes = [b.coordinates for b in image_boxes]
    output: Image.Image = try_draw_boxes(source_image, boxes)
    check_image(output)


def test_draw_boxes_point_list(source_image, image_boxes) -> None:
    boxes = [[(b.x1, b.y1), (b.x2, b.y1), (b.x2, b.y2), (b.x1, b.y2)] for b in image_boxes]
    output: Image.Image = try_draw_boxes(source_image, boxes)
    check_image(output)


def test_draw_boxes_two_points(source_image, image_boxes) -> None:
    boxes = [((b.x1, b.y1), (b.x2, b.y2)) for b in image_boxes]
    output: Image.Image = try_draw_boxes(source_image, boxes)
    check_image(output)


def test_draw_boxes_dict(source_image, image_boxes) -> None:
    boxes = [{"bbox": b.coordinates} for b in image_boxes]
    output: Image.Image = try_draw_boxes(source_image, boxes)
    check_image(output)


def test_invalid_list(source_image, image_boxes) -> None:
    boxes = [[b.coordinates] for b in image_boxes]
    output: Image.Image = try_draw_boxes(source_image, boxes)
    with pytest.raises(AssertionError, match=r".*Did not draw boxes.*"):
        check_image(output)


def test_invalid_dict(source_image, image_boxes) -> None:
    boxes = [{"bboxes": b.coordinates} for b in image_boxes]
    output: Image.Image = try_draw_boxes(source_image, boxes)
    with pytest.raises(AssertionError, match=r".*Did not draw boxes.*"):
        check_image(output)
