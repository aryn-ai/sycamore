from typing import Dict

from pyarrow.fs import LocalFileSystem
import pytest

from shannon.data.document import Document


@pytest.fixture
def read_local_binary(request) -> Dict[str, bytes]:
    local = LocalFileSystem()
    input_stream = local.open_input_stream(str(request.param))
    document = Document()
    document.content.update({"binary": input_stream.readall()})
    return document.to_dict()
