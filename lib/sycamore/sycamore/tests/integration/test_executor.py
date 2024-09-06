from sycamore.context import ExecMode
import sycamore.tests.unit.test_executor as unit


class TestPrepare(unit.TestPrepare):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exec_mode = ExecMode.RAY
