from unittest.mock import patch
from typing import Dict

import pytest
import sycamore

from sycamore.query.execution.sycamore_executor import SycamoreExecutor
from sycamore.query.logical_plan import LogicalPlan, Node
from sycamore.query.operators.count import Count
from sycamore.query.operators.loaddata import LoadData


@pytest.fixture
def test_count_docs_query_plan() -> LogicalPlan:
    """A simple query plan which only counts the number of documents."""
    load_node = LoadData(node_id=0, description="Load data", index="test_index")
    count_node = Count(
        node_id=1, description="Count number of documents", input=[load_node.node_id]
    )

    # pylint: disable=protected-access
    load_node._downstream_nodes = [count_node]
    # pylint: disable=protected-access
    count_node._dependencies = [load_node]
    nodes: Dict[int, Node] = {
        load_node.node_id: load_node,
        count_node.node_id: count_node,
    }
    plan = LogicalPlan(result_node=count_node, nodes=nodes, query="Test query plan")
    assert plan.result_node == count_node
    assert plan.nodes == nodes
    return plan


def test_count_docs(test_count_docs_query_plan, mock_sycamore_docsetreader, mock_opensearch_num_docs):
    with patch("sycamore.reader.DocSetReader", new=mock_sycamore_docsetreader):
        context = sycamore.init()

        os_client_args = {
            "hosts": [{"host": "localhost", "port": 9200}],
            "http_compress": True,
            "http_auth": ("admin", "admin"),
            "use_ssl": True,
            "verify_certs": False,
            "ssl_assert_hostname": False,
            "ssl_show_warn": False,
            "timeout": 120,
        }

        executor = SycamoreExecutor(context, os_client_args=os_client_args, s3_cache_path="s3://sycamore-cache")
        result = executor.execute(test_count_docs_query_plan)
        assert result == mock_opensearch_num_docs
