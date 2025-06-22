import shutil

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


@pytest.fixture(params=(exec_mode for exec_mode in ExecMode if exec_mode != ExecMode.UNKNOWN))
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

@pytest.fixture(scope="function", autouse=True)
def check_huggingface_hub(request):
    """
    Use this to find tests that download a model from Huggingface.
    """

    import os
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    curr_test = request.node.name
    if os.path.exists(hf_cache_dir):
        # try2 = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
        print(f"!!!!!! BEFORE: {curr_test} Hugging Face Hub cache exists.")
        shutil.rmtree(hf_cache_dir)

    yield

    if os.path.exists(hf_cache_dir):
        print(f"!!!!!! AFTER: {curr_test} Hugging Face Hub cache exists.")
        shutil.rmtree(hf_cache_dir)
