import pytest
from pyarrow.fs import LocalFileSystem

from sycamore.data.document import Document


@pytest.fixture
def read_local_binary(request) -> Document:
    local = LocalFileSystem()
    input_stream = local.open_input_stream(str(request.param))
    document = Document()
    document.binary_representation = input_stream.readall()
    return document
