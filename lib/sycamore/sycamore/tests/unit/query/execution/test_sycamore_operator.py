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
        logical_node = LoadData(node_id=0, description="Load data", index="test_index")
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
        load_node = LoadData(node_id=0, description="Load data", index="test_index")
        logical_node = LlmGenerate(node_id=1, question="who?", description="describe me")
        sycamore_operator = SycamoreLlmGenerate(context, logical_node, query_id="test", inputs=[load_node])
        result = sycamore_operator.execute()

        assert result == "success"
        mock_impl.assert_called_once_with(
            client=ANY,
            question=logical_node.question,
            result_description=logical_node.description,
            result_data=[load_node],
            **sycamore_operator.get_execute_args(),
        )


def test_llm_filter():
    with (
        patch("sycamore.query.execution.sycamore_operator.OpenAI"),  # disable OpenAI client initialization
        patch("sycamore.query.execution.sycamore_operator.LLMFilterMessagesPrompt") as MockLLMFilterMessagesPrompt,
    ):
        context = sycamore.init()
        doc_set = Mock(spec=DocSet)
        return_doc_set = Mock(spec=DocSet)
        doc_set.llm_filter.return_value = return_doc_set
        logical_node = LlmFilter(node_id=0, question="who?", field="name")
        sycamore_operator = SycamoreLlmFilter(context, logical_node, query_id="test", inputs=[doc_set])

        result = sycamore_operator.execute()

        # assert LLMFilterMessagesPrompt called with expected arguments
        MockLLMFilterMessagesPrompt.assert_called_once_with(
            filter_question=logical_node.question,
        )

        doc_set.llm_filter.assert_called_once_with(
            llm=ANY,
            new_field="_autogen_LLMFilterOutput",
            prompt=ANY,
            field=logical_node.field,
            threshold=3,
            name=str(logical_node.node_id),
        )

        assert result == return_doc_set


def test_filter_range(mock_docs):
    context = sycamore.init()
    doc_set = context.read.document(mock_docs)
    logical_node = Filter(node_id=0, range_filter=True, field="properties.counter", start=1, end=2)
    sycamore_operator = SycamoreFilter(context, logical_node, query_id="test", inputs=[doc_set])
    result = sycamore_operator.execute().take_all()

    assert len(result) == 2
    for doc in result:
        assert doc.properties.get("counter") >= 1
        assert doc.properties.get("counter") <= 2


def test_filter_exact_match(mock_docs):
    context = sycamore.init()
    doc_set = context.read.document(mock_docs)
    logical_node = Filter(node_id=0, query=2, field="properties.counter")
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
        logical_node = Count(node_id=0, question="who?", field="name")
        sycamore_operator = SycamoreCount(context, logical_node, query_id="test", inputs=[doc_set])
        result = sycamore_operator.execute()

        assert result == "success"
        mock_impl.assert_called_once_with(
            docset=doc_set,
            field=logical_node.field,
            primary_field=logical_node.primary_field,
            **sycamore_operator.get_execute_args(),
        )


def test_join():
    with patch("sycamore.query.execution.sycamore_operator.join_operation") as mock_impl:
        # Define the mock return value
        mock_impl.return_value = "success"

        doc_set1 = Mock(spec=DocSet)
        doc_set2 = Mock(spec=DocSet)
        context = sycamore.init()
        logical_node = Join(node_id=0, field_one="field1", field_two="field2")
        sycamore_operator = SycamoreJoin(context, logical_node, query_id="test", inputs=[doc_set1, doc_set2])
        result = sycamore_operator.execute()

        assert result == "success"
        mock_impl.assert_called_once_with(
            docset1=doc_set1,
            docset2=doc_set2,
            field1=logical_node.field_one,
            field2=logical_node.field_two,
        )


def test_llm_extract():
    with (
        patch("sycamore.query.execution.sycamore_operator.OpenAI"),
        patch(
            "sycamore.query.execution.sycamore_operator.EntityExtractorMessagesPrompt"
        ) as MockEntityExtractorMessagesPrompt,
        patch("sycamore.query.execution.sycamore_operator.OpenAIEntityExtractor") as MockOpenAIEntityExtractor,
    ):

        context = sycamore.init()
        doc_set = Mock(spec=DocSet)
        return_doc_set = Mock(spec=DocSet)
        doc_set.extract_entity.return_value = return_doc_set

        logical_node = LlmExtract(
            node_id=0, question="who?", field="properties.counter", new_field="new", new_field_type="str", discrete=True
        )
        sycamore_operator = SycamoreLlmExtract(context, logical_node, query_id="test", inputs=[doc_set])
        result = sycamore_operator.execute()

        # assert EntityExtractorMessagesPrompt called with expected arguments
        MockEntityExtractorMessagesPrompt.assert_called_once_with(
            question=logical_node.question,
            field=logical_node.field,
            format=logical_node.new_field_type,
            discrete=logical_node.discrete,
        )

        # assert OpenAIEntityExtractor called with expected arguments
        MockOpenAIEntityExtractor.assert_called_once_with(
            entity_name=logical_node.new_field,
            llm=ANY,
            use_elements=False,
            prompt=ANY,
            field=logical_node.field,
        )

        # assert extract_entity called with expected arguments
        doc_set.extract_entity.assert_called_once_with(
            entity_extractor=MockOpenAIEntityExtractor(),
            name=str(logical_node.node_id),
        )
        assert result == return_doc_set


def test_sort():
    context = sycamore.init()
    doc_set = Mock(spec=DocSet)
    return_doc_set = Mock(spec=DocSet)
    doc_set.sort.return_value = return_doc_set
    logical_node = Sort(node_id=0, descending=True, field="properties.counter", default_value=0)
    sycamore_operator = SycamoreSort(context, logical_node, query_id="test", inputs=[doc_set])
    result = sycamore_operator.execute()

    doc_set.sort.assert_called_once_with(
        descending=logical_node.descending,
        field=logical_node.field,
        default_val=logical_node.default_value,
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
            node_id=0,
            descending=True,
            K=10,
            field="name",
            description="some description",
            use_llm=True,
            primary_field="id",
        )
        sycamore_operator = SycamoreTopK(context, logical_node, query_id="test", inputs=[doc_set])
        result = sycamore_operator.execute()

        assert result == "success"
        mock_impl.assert_called_once_with(
            client=ANY,
            docset=doc_set,
            field=logical_node.field,
            k=logical_node.K,
            description=logical_node.description,
            descending=logical_node.descending,
            use_llm=logical_node.use_llm,
            unique_field=logical_node.primary_field,
            **sycamore_operator.get_execute_args(),
        )


def test_limit(mock_docs):
    context = sycamore.init()
    k = 2
    doc_set = Mock(spec=DocSet)
    doc_set.limit.return_value = mock_docs[0:k]
    logical_node = Limit(node_id=0, num_records=k)
    sycamore_operator = SycamoreLimit(context, logical_node, query_id="test", inputs=[doc_set])
    result = sycamore_operator.execute()

    doc_set.limit.assert_called_once_with(k)
    assert len(result) == 2


class ValidationTests(unittest.TestCase):
    def test_load_data_validation(self):
        context = sycamore.init()
        logical_node = LoadData(node_id=0, description="Load data", index="test_index")
        sycamore_operator = SycamoreLoadData(
            context=context, logical_node=logical_node, query_id="test", os_client_args={}
        )
        _ = sycamore_operator.execute()

    def test_llm_generate_validation(self):
        context = sycamore.init()
        logical_node = LlmGenerate(node_id=0, question="generate")
        sycamore_operator = SycamoreLlmGenerate(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_llm_filter_validation(self):
        context = sycamore.init()
        logical_node = LlmGenerate(node_id=0, question="llm_filter")
        sycamore_operator = SycamoreLlmFilter(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreLlmFilter(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_filter_validation(self):
        context = sycamore.init()
        logical_node = Filter(node_id=0, field="filter_field")

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
        logical_node = Count(node_id=0, field="count_field")
        sycamore_operator = SycamoreCount(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)
        sycamore_operator = SycamoreCount(context, logical_node, query_id="test", inputs=[Mock(DocSet), Mock(DocSet)])
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreCount(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_sort(self):
        context = sycamore.init()
        logical_node = Sort(node_id=0, field="sort_field", default_value=0)
        sycamore_operator = SycamoreSort(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)
        sycamore_operator = SycamoreSort(context, logical_node, query_id="test", inputs=[Mock(DocSet), Mock(DocSet)])
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreSort(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_llm_extract(self):
        context = sycamore.init()
        logical_node = LlmExtract(
            node_id=0, field="input_field", question="question", new_field="output_field", new_field_type="str"
        )
        sycamore_operator = SycamoreLlmExtract(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)
        sycamore_operator = SycamoreLlmExtract(
            context, logical_node, query_id="test", inputs=[Mock(DocSet), Mock(DocSet)]
        )
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreLlmExtract(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_topk(self):
        context = sycamore.init()
        logical_node = TopK(node_id=0, field="count", K=10)
        sycamore_operator = SycamoreTopK(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)
        sycamore_operator = SycamoreTopK(context, logical_node, query_id="test", inputs=[Mock(DocSet), Mock(DocSet)])
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreTopK(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)

    def test_limit(self):
        context = sycamore.init()
        logical_node = Limit(node_id=0, num_records=10)
        sycamore_operator = SycamoreLimit(context, logical_node, query_id="test", inputs=[])
        self.assertRaises(AssertionError, sycamore_operator.execute)
        sycamore_operator = SycamoreLimit(context, logical_node, query_id="test", inputs=[Mock(DocSet), Mock(DocSet)])
        self.assertRaises(AssertionError, sycamore_operator.execute)

        # non-DocSet input
        sycamore_operator = SycamoreLimit(context, logical_node, query_id="test", inputs=[1])
        self.assertRaises(AssertionError, sycamore_operator.execute)
