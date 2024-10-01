import pytest

from sycamore.utils.ray_utils import check_serializable


def test_non_serializable():
    import threading

    lock = threading.Lock()
    with pytest.raises(ValueError):
        check_serializable(lock)


def test_serializable():
    check_serializable("a")
