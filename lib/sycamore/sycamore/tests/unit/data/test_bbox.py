from sycamore.data import BoundingBox
from sycamore.data.bbox import EMPTY_BBOX
from math import isclose

# Test bounding boxes. Box 1 and 2 intersect. Box 3 is intersects with Box 1 but not Box 2. Box 4 and 5
# are disjoint from all others and are empty.
bbox1 = BoundingBox(20.0, 20.0, 60.0, 60.0)
bbox2 = BoundingBox(10.0, 40.0, 40.0, 80.0)
bbox3 = BoundingBox(50.0, 10.0, 90.0, 50.0)
bbox4 = BoundingBox(100.0, 100.0, 100.0, 120.0)
bbox5 = BoundingBox(100.0, 100.0, 100.0, 100.0)


def test_intersect():
    assert isclose(bbox1.intersect(bbox2).area, 400.0)
    assert isclose(bbox2.intersect(bbox1).area, 400.0)

    assert isclose(bbox1.intersect(bbox3).area, 300.0)
    assert isclose(bbox2.intersect(bbox3).area, 0.0)


def test_union():
    assert isclose(bbox1.union(bbox1).area, bbox1.area)
    assert isclose(bbox1.union(bbox2).area, 3000.0)

    assert isclose(EMPTY_BBOX.union(bbox1).area, bbox1.area)
    assert isclose(bbox1.union(EMPTY_BBOX).area, bbox1.area)


def test_absolute_relative():
    assert bbox1.area == bbox1.to_relative(100, 100).to_absolute(100, 100).area
    assert isclose(bbox1.to_relative(100, 100).area, 0.16)


def test_empty_bbox():
    assert bbox5.is_empty()
    assert bbox4.is_empty()
    assert bbox4.area == 0 and bbox5.area == 0
    assert bbox4.union(bbox5).is_empty() and bbox4.union(bbox5).area == 0
    assert bbox4.intersect(bbox5).is_empty() and bbox4.intersect(bbox5) == EMPTY_BBOX
