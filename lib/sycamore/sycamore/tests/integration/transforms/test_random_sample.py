import sycamore
from sycamore.tests.unit.transforms.test_random_sample import TestRandomSample


class TestRandomSampleRay(TestRandomSample):
    exec_mode = sycamore.EXEC_RAY
