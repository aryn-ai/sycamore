import pytest
from pyarrow.fs import LocalFileSystem

from sycamore import ExecMode
from sycamore.data.document import Document


@pytest.fixture
def read_local_binary(request) -> Document:
    local = LocalFileSystem()
    path = str(request.param)
    input_stream = local.open_input_stream(path)
    document = Document()
    document.binary_representation = input_stream.readall()
    document.properties["path"] = path
    return document


@pytest.fixture(params=(exec_mode for exec_mode in ExecMode if exec_mode == ExecMode.LOCAL))
def exec_mode(request):
    """
    Use this to run a test against all available execution modes. You will need to pass this as a parameter to
    the Context initialization. e.g.

    Example:
        .. code-block:: python

            def test_example(exec_mode):
                context = sycamore.init(exec_mode=exec_mode)
                ...
    """
    return request.param
