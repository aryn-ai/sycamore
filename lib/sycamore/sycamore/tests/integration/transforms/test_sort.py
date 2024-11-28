from sycamore.context import ExecMode
import sycamore.tests.unit.transforms.test_sort as unit


class TestSort(unit.TestSort):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exec_mode = ExecMode.RAY
