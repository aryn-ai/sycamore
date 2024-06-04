import pytest
from pyarrow.fs import LocalFileSystem

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
