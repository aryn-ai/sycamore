import json
import pytest
from unittest.mock import MagicMock
from typing import Any, Optional, List, Dict, Tuple

from ray.data import Dataset, from_items
from sycamore import DocSet, Context
from sycamore.connectors.base_reader import BaseDBReader
from sycamore.data import Document
from sycamore.plan_nodes import Node
from sycamore.reader import DocSetReader
from sycamore.query.execution.sycamore_operator import SycamoreOperator
from sycamore.query.operators.logical_operator import LogicalOperator
from sycamore.query.execution.physical_operator import get_var_name

MOCK_SCAN_NUM_DOCUMENTS = 20


class MockOpenSearchReader(BaseDBReader):
    """Mock out OpenSearchReader for tests."""

    def __init__(self, **kwargs):
        super().__init__(client_params=BaseDBReader.ClientParams, query_params=BaseDBReader.QueryParams, **kwargs)

    def read_docs(self) -> List[Document]:
        return get_mock_docs()


class MockDocSetReader(DocSetReader):
    """Mock out DocSetReader for tests."""

    def __init__(self, context: Context, plan: Optional[Node] = None):
        super().__init__(context, plan)
        self.context = context
        self.plan = plan

    def opensearch(self, *args, **kwargs) -> DocSet:
        return DocSet(self._context, MockOpenSearchReader())


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
        docs += [Document({"foo": "bar", "properties": {"counter": i}})]
    return docs
