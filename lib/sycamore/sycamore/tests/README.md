# Test

We use [pytest](https://docs.pytest.org/) as test framework, and
[pytest-mock](https://pytest-mock.readthedocs.io/en/latest/usage.html) as
mocking.

Tests are split into integration/unit based on the criteria whether external
resources are required. We organize integration and unit tests into two sub
folders. This avoids marking tests with attributes or @pytest.mark manually,
also tests are not limited to a specific test runner.

## Run Tests

To run all integration tests:
```
python -m pytest sycamore/tests/integration
```
To run all unit tests:
```
python -m pytest sycamore/tests/unit
```
To run all tests in a single test file
```
python -m pytest sycamore/tests/unit/test_writer.py
```
To run all tests in a single test class
```
python -m pytest sycamore/tests/unit/execution/transforms/test_partition.py::TestPartition
```
To run a single test method of a test class
```
python -m pytest sycamore/tests/unit/execution/transforms/test_partition.py::TestPartition::test_pdf_partition
```
