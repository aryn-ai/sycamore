from pathlib import Path

import pytest
from typing import Dict, List, Optional, TYPE_CHECKING, Union

from sycamore.materialize import Materialize

if TYPE_CHECKING:
    from ray.data import Dataset

from sycamore import DocSet, Context
from sycamore.connectors.opensearch.opensearch_reader import (
    OpenSearchReader,
    OpenSearchReaderClientParams,
    OpenSearchReaderQueryParams,
)
from sycamore.context import context_params
from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.reader import DocSetReader

MOCK_SCAN_NUM_DOCUMENTS = 20


class MockOpenSearchReader(OpenSearchReader):
    """Mock out OpenSearchReader for tests."""

    def read_docs(self) -> List[Document]:
        return get_mock_docs()

    def execute(self, **kwargs) -> "Dataset":
        from ray.data import from_items

        return from_items(items=[{"doc": doc.serialize()} for doc in self.read_docs()])


class MockMaterialize(Materialize):
    """Mock out OpenSearchReader for tests."""

    def execute(self, **kwargs) -> "Dataset":
        from ray.data import from_items

        return from_items(items=[{"doc": doc.serialize()} for doc in get_mock_docs()])


class MockDocSetReader(DocSetReader):
    """Mock out DocSetReader for tests."""

    def __init__(self, context: Context, plan: Optional[Node] = None):
        super().__init__(context, plan)
        self.context = context
        self.plan = plan

    @context_params
    def opensearch(self, os_client_args: dict, index_name: str, query: Optional[Dict] = None, **kwargs) -> DocSet:
        client_params = OpenSearchReaderClientParams(os_client_args=os_client_args)
        query_params = (
            OpenSearchReaderQueryParams(index_name=index_name, query=query)
            if query is not None
            else OpenSearchReaderQueryParams(index_name=index_name, query={"query": {"match_all": {}}})
        )
        mock_osr = MockOpenSearchReader(client_params=client_params, query_params=query_params)
        return DocSet(self._context, mock_osr)

    def materialize(self, path: Union[Path, str], **kwargs) -> DocSet:
        m = MockMaterialize(child=None, context=self._context, path=path)
        return DocSet(self._context, m)


@pytest.fixture
def mock_sycamore_docsetreader():
    return MockDocSetReader


@pytest.fixture
def mock_opensearch_num_docs():
    return MOCK_SCAN_NUM_DOCUMENTS


@pytest.fixture
def mock_docs() -> List[Document]:
    return get_mock_docs()


def get_mock_docs() -> List[Document]:
    docs = []
    for i in range(MOCK_SCAN_NUM_DOCUMENTS):
        docs += [Document({"foo": "bar", "properties": {"counter": i, "_original_elements": "value"}})]
    return docs
