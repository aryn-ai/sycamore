import unittest
from unittest.mock import patch, ANY, Mock

import sycamore
from sycamore.query.operators.join import Join
from sycamore.query.operators.llmextract import LlmExtract
from sycamore import DocSet

from sycamore.query.operators.count import Count
from sycamore.query.operators.limit import Limit
from sycamore.query.execution.sycamore_operator import (
    SycamoreJoin,
    SycamoreLoadData,
    SycamoreLlmGenerate,
    SycamoreLlmFilter,
    SycamoreFilter,
    SycamoreCount,
    SycamoreLlmExtract,
    SycamoreSort,
    SycamoreTopK,
    SycamoreLimit,
)
from sycamore.query.operators.filter import Filter
from sycamore.query.operators.sort import Sort
from sycamore.query.operators.llmfilter import LlmFilter
from sycamore.query.operators.llmgenerate import LlmGenerate
from sycamore.query.operators.loaddata import LoadData
from sycamore.query.operators.topk import TopK


def test_load_data(mock_sycamore_docsetreader, mock_opensearch_num_docs):
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
        logical_node = LoadData("load", {"description": "Load data", "index": "test_index", "id": 0})
        sycamore_operator = SycamoreLoadData(
            context=context, logical_node=logical_node, query_id="test", os_client_args=os_client_args
        )
        result = sycamore_operator.execute()
        # Validate result type
        assert isinstance(result, DocSet)

        # Validate result
        assert result.count() == mock_opensearch_num_docs


def test_llm_generate():
    with (
        patch("sycamore.query.execution.sycamore_operator.llm_generate_operation") as mock_impl,
        patch("sycamore.query.execution.sycamore_operator.OpenAI"),  # disable OpenAI client initialization
    ):
        # Define the mock return value
        mock_impl.return_value = "success"
        context = sycamore.init()
        load_node = LoadData("load", {"description": "Load data", "index": "test_index", "id": 0})
        logical_node = LlmGenerate("node_id", {"question": "who?", "description": "describe me", "id": 0})
        sycamore_operator = SycamoreLlmGenerate(context, logical_node, query_id="test", inputs=[load_node])
        result = sycamore_operator.execute()

        assert result == "success"
        mock_impl.assert_called_once_with(
            client=ANY,
            question=logical_node.data.get("question"),
            result_description=logical_node.data.get("description"),
            result_data=[load_node],
            **sycamore_operator.get_execute_args(),
        )


def test_llm_filter():
    with (
        patch("sycamore.query.execution.sycamore_operator.llm_filter_operation") as mock_impl,
        patch("sycamore.query.execution.sycamore_operator.OpenAI"),  # disable OpenAI client initialization
    ):
        # Define the mock return value
        mock_impl.return_value = "success"

        doc_set = Mock(spec=DocSet)
        context = sycamore.init()
        logical_node = LlmFilter("node_id", {"question": "who?", "field": "name", "id": 0})
        sycamore_operator = SycamoreLlmFilter(context, logical_node, query_id="test", inputs=[doc_set])
        result = sycamore_operator.execute()

        assert result == "success"
        mock_impl.assert_called_once_with(
            client=ANY,
            docset=doc_set,
            filter_question=logical_node.data.get("question"),
            field=logical_node.data.get("field"),
            messages=None,
            threshold=3,
            name=logical_node.node_id,
        )


def test_filter_range(mock_docs):
    context = sycamore.init()
    doc_set = context.read.document(mock_docs)
    logical_node = Filter(
        "node_id", {"rangeFilter": True, "field": "properties.counter", "start": 1, "end": 2, "id": 0}
    )
    sycamore_operator = SycamoreFilter(context, logical_node, query_id="test", inputs=[doc_set])
    result = sycamore_operator.execute().take_all()

    assert len(result) == 2
    for doc in result:
        assert doc.properties.get("counter") >= 1
        assert doc.properties.get("counter") <= 2


def test_filter_exact_match(mock_docs):
    context = sycamore.init()
    doc_set = context.read.document(mock_docs)
    logical_node = Filter("node_id", {"query": 2, "field": "properties.counter", "id": 0})
    sycamore_operator = SycamoreFilter(context, logical_node, query_id="test", inputs=[doc_set])
    result = sycamore_operator.execute().take_all()

    assert len(result) == 1
    for doc in result:
        assert doc.properties.get("counter") == 2


def test_count():
    with patch("sycamore.query.execution.sycamore_operator.count_operation") as mock_impl:
        # Define the mock return value
        mock_impl.return_value = "success"

        doc_set = Mock(spec=DocSet)
        context = sycamore.init()
        logical_node = Count("node_id", {"question": "who?", "field": "name", "id": 0})
        sycamore_operator = SycamoreCount(context, logical_node, query_id="test", inputs=[doc_set])
        result = sycamore_operator.execute()

        assert result == "success"
        mock_impl.assert_called_once_with(
            docset=doc_set,
            field=logical_node.data.get("field"),
            primary_field=logical_node.data.get("primaryField"),
            **sycamore_operator.get_execute_args(),
        )


def test_join():
    with patch("sycamore.query.execution.sycamore_operator.join_operation") as mock_impl:
        # Define the mock return value
        mock_impl.return_value = "success"

        doc_set1 = Mock(spec=DocSet)
        doc_set2 = Mock(spec=DocSet)
        context = sycamore.init()
        logical_node = Join("node_id", {"fieldOne": "field1", "fieldTwo": "field2", "id": 0})
        sycamore_operator = SycamoreJoin(context, logical_node, query_id="test", inputs=[doc_set1, doc_set2])
        result = sycamore_operator.execute()

        assert result == "success"
        mock_impl.assert_called_once_with(
            docset1=doc_set1,
            docset2=doc_set2,
            field1=logical_node.data.get("fieldOne"),
            field2=logical_node.data.get("fieldTwo"),
        )


def test_sort():
    context = sycamore.init()
    doc_set = Mock(spec=DocSet)
    return_doc_set = Mock(spec=DocSet)
    doc_set.sort.return_value = return_doc_set
    logical_node = Sort("node_id", {"descending": True, "field": "properties.counter", "defaultValue": 0, "id": 0})
    sycamore_operator = SycamoreSort(context, logical_node, query_id="test", inputs=[doc_set])
    result = sycamore_operator.execute()

    doc_set.sort.assert_called_once_with(
        descending=logical_node.data.get("descending"),
        field=logical_node.data.get("field"),
        default_val=logical_node.data.get("defaultValue"),
    )
    assert result == return_doc_set


def test_top_k():
    with (
        patch("sycamore.query.execution.sycamore_operator.top_k_operation") as mock_impl,
        patch("sycamore.query.execution.sycamore_operator.OpenAI"),  # disable OpenAI client initialization
    ):
        # Define the mock return value
        mock_impl.return_value = "success"

        doc_set = Mock(spec=DocSet)
        context = sycamore.init()
        logical_node = TopK(
            "node_id",
            {
                "descending": True,
                "K": 10,
                "field": "name",
                "id": 0,
                "description": "some description",
                "useLLM": True,
                "primaryField": "id",
            },
        )
        sycamore_operator = SycamoreTopK(context, logical_node, query_id="test", inputs=[doc_set])
        result = sycamore_operator.execute()

        assert result == "success"
        mock_impl.assert_called_once_with(
            client=ANY,
            docset=doc_set,
            field=logical_node.data.get("field"),
            k=logical_node.data.get("K"),
            description=logical_node.data.get("description"),
            descending=logical_node.data.get("descending"),
            use_llm=logical_node.data.get("useLLM"),
            unique_field=logical_node.data.get("primaryField"),
            **sycamore_operator.get_execute_args(),
        )


def test_limit(mock_docs):
    context = sycamore.init()
    k = 2
    doc_set = Mock(spec=DocSet)
    doc_set.limit.return_value = mock_docs[0:k]
    logical_node = Limit("node_id", {"query": 2, "K": k, "id": 0})
    sycamore_operator = SycamoreLimit(context, logical_node, query_id="test", inputs=[doc_set])
    result = sycamore_operator.execute()

    doc_set.limit.assert_called_once_with(k)
    assert len(result) == 2


class ValidationTests(unittest.TestCase):
    def test_load_data_validation(self):
        context = sycamore.init()
        logical_node = LoadData("load", {"description": "Load data", "id": 0})
        sycamore_operator = SycamoreLoadData(
            context=context, logical_node=logical_node, query_id="test", os_client_args={}
        )
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_llm_generate_validation(self):
        context = sycamore.init()
        logical_node = LlmGenerate("generate", {})
        sycamore_operator = SycamoreLlmGenerate(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_llm_filter_validation(self):
        context = sycamore.init()
        logical_node = LlmGenerate("llm_filter", {})
        sycamore_operator = SycamoreLlmFilter(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreLlmFilter(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_filter_validation(self):
        context = sycamore.init()
        logical_node = Filter("filter", {})

        # assert 1 input
        sycamore_operator = SycamoreFilter(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)
        sycamore_operator = SycamoreFilter(context, logical_node, query_id="test", inputs=[Mock(DocSet), Mock(DocSet)])
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreFilter(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_count_validation(self):
        context = sycamore.init()
        logical_node = Count("count", {})
        sycamore_operator = SycamoreCount(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)
        sycamore_operator = SycamoreCount(context, logical_node, query_id="test", inputs=[Mock(DocSet), Mock(DocSet)])
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreCount(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_sort(self):
        context = sycamore.init()
        logical_node = Sort("count", {})
        sycamore_operator = SycamoreSort(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)
        sycamore_operator = SycamoreSort(context, logical_node, query_id="test", inputs=[Mock(DocSet), Mock(DocSet)])
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreSort(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_topk(self):
        context = sycamore.init()
        logical_node = TopK("count", {})
        sycamore_operator = SycamoreTopK(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)
        sycamore_operator = SycamoreTopK(context, logical_node, query_id="test", inputs=[Mock(DocSet), Mock(DocSet)])
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreTopK(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_limit(self):
        context = sycamore.init()
        logical_node = Limit("count", {})
        sycamore_operator = SycamoreLimit(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)
        sycamore_operator = SycamoreLimit(context, logical_node, query_id="test", inputs=[Mock(DocSet), Mock(DocSet)])
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreLimit(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)
