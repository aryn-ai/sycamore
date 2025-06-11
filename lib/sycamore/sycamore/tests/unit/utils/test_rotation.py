from pytest import approx

from sycamore.data import BoundingBox
from sycamore.utils.rotation import VectorMean, quad_rotation, rot_bbox, rot_dict, rot_tuple, rot_xy, vnorm


def test_rot_xy() -> None:
    z = (0.4, 0.2)
    a = rot_xy(*z, 0)
    assert a == z
    b = rot_xy(*z, 1)
    assert b == approx((0.2, 0.6))
    c = rot_xy(*z, 2)
    assert c == approx((0.6, 0.8))
    cc = rot_xy(*b, 1)
    assert c == cc
    d = rot_xy(*z, 3)
    assert d == approx((0.8, 0.4))
    dd = rot_xy(*c, 1)
    assert d == dd


def test_rot_tuple() -> None:
    bb0 = (0.1, 0.2, 0.4, 0.8)
    bb1 = rot_tuple(bb0, 1)
    assert bb1 == approx((0.2, 0.6, 0.8, 0.9))
    bb2 = rot_tuple(bb0, 2)
    assert bb2 == approx((0.6, 0.2, 0.9, 0.8))
    bb3 = rot_tuple(bb0, 3)
    assert bb3 == approx((0.2, 0.1, 0.8, 0.4))


def test_rot_dict() -> None:
    dd0 = {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.8}
    dd1 = rot_dict(dd0, 1)
    assert 0.2 == approx(dd1["x1"])
    assert 0.6 == approx(dd1["y1"])
    assert 0.8 == approx(dd1["x2"])
    assert 0.9 == approx(dd1["y2"])


def test_rot_bbox() -> None:
    bb0 = BoundingBox(0.1, 0.2, 0.4, 0.8)
    bb1 = rot_bbox(bb0, 2)
    assert 0.6 == approx(bb1.x1)
    assert 0.2 == approx(bb1.y1)
    assert 0.9 == approx(bb1.x2)
    assert 0.8 == approx(bb1.y2)


def test_quad_rotation() -> None:
    assert quad_rotation(0.0 + 0.0j) == 0
    assert quad_rotation(0.1 + 0.1j) == 0
    assert quad_rotation(0.9 + 0.1j) == 0
    assert quad_rotation(0.1 + 0.9j) == 1
    assert quad_rotation(-0.9 + 0.1j) == 2
    assert quad_rotation(-0.1 + -0.9j) == 3


def test_vnorm() -> None:
    x = 4.0 + 3.0j
    assert abs(x) == approx(5.0)
    n = vnorm(x)
    assert n == approx(0.8 + 0.6j)


def test_vector_mean() -> None:
    vm = VectorMean()
    assert vm.get() == approx(0.0 + 0.0j)
    vm.add(1.0 + 0.0j)
    assert vm.get() == approx(1.0 + 0.0j)
    vm.add(0.0 + 1.0j)
    assert vm.get() == approx(0.5 + 0.5j)
