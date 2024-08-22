from sycamore.context import ExecMode
import sycamore.tests.unit.test_materialize as test_materialize


class TestMaterializeRead(test_materialize.TestMaterializeRead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exec_mode = ExecMode.RAY
