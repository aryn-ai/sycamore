from pyarrow.fs import LocalFileSystem
import pytest
from typing import Dict


@pytest.fixture
def read_local_binary(request) -> Dict[str, bytes]:
    local = LocalFileSystem()
    input_stream = local.open_input_stream(str(request.param))
    return {"bytes": input_stream.readall()}
