from concurrent.futures import ThreadPoolExecutor
import tempfile
import sycamore
from sycamore.context import ExecMode
from sycamore.tests.config import TEST_DIR
import sycamore.tests.unit.test_executor as unit


class TestPrepare(unit.TestPrepare):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exec_mode = ExecMode.RAY

def test_multiple_ray_init():
    import ray

    with tempfile.TemporaryDirectory() as tempdir:
        context = sycamore.init(exec_mode=ExecMode.RAY)

        def write():
            (
                context.read.materialize(path=TEST_DIR / "resources/data/materialize/json_writer")
                    .write.json(tempdir)
            )

        num_workers = 10
        executor = ThreadPoolExecutor(max_workers=num_workers)
        futures = [executor.submit(write) for _ in range(num_workers)]
        got = 0
        for future in futures:
            e = future.exception()
            if e is not None:
                assert False, e
            future.result()
            got += 1

        assert got == num_workers
        executor.shutdown()