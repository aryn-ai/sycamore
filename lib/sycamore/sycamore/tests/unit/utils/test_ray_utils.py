import pytest

from sycamore.utils.ray_utils import check_serializable, handle_serialization_exception


def test_non_serializable():
    import threading

    lock1 = threading.Lock()
    lock2 = threading.Lock()
    with pytest.raises(ValueError):
        # Make sure this works with passing multiple objects.
        check_serializable(lock1, lock2)


def test_serializable():
    check_serializable("a")


def test_decorator_non_serializable():
    import threading

    class Dummy:
        def __init__(self):
            self.lock1 = threading.Lock()
            self.lock2 = threading.Lock()

        @handle_serialization_exception("lock1", "lock2")
        def test_func(self):
            raise TypeError("Not serializable")

    with pytest.raises(ValueError) as error_info:
        dummy = Dummy()
        dummy.test_func()

    print(error_info.value)
