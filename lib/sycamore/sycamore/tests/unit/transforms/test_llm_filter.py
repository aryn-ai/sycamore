import pytest
from typing import Optional, Union
from unittest.mock import MagicMock

import sycamore
from sycamore.context import Context, OperationTypes, ExecMode
from sycamore.data import Document, Element
from sycamore.functions import Tokenizer
from sycamore.llms import LLM
from sycamore.tests.unit.test_docset import MockLLM, TestSimilarityScorer
from sycamore.transforms.extract_entity import EntityExtractor


class TestLLMFilter:
    def test_llm_filter_with_doc_structure(self):
        doc_list = [
            Document(
                doc_id="doc_1",
                elements=[
                    Element(text_representation="test1"),  # llm_filter result = 4
                    Element(text_representation="test1"),  # llm_filter result = 4
                ],
            ),
            Document(
                doc_id="doc_2",
                elements=[
                    Element(text_representation="test2", element_index=1),  # llm_filter result = 2,
                    Element(text_representation="test1", element_index=2),  # llm_filter result = 4
                ],
            ),
            Document(
                doc_id="doc_3",
                elements=[
                    Element(text_representation="test2"),  # llm_filter result = 2
                ],
            ),
            Document(doc_id="doc_4", text_representation="empty elements, maybe an exploded doc", elements=[]),
        ]
        mock_llm = MockLLM()
        mock_llm.generate = MagicMock(wraps=mock_llm.generate)
        context = sycamore.init(params={OperationTypes.BINARY_CLASSIFIER: {"llm": mock_llm}}, exec_mode=ExecMode.LOCAL)
        docset = context.read.document(doc_list)
        new_field = "_autogen_LLMFilterOutput"

        filtered_docset = docset.llm_filter(
            new_field=new_field, prompt=[], field="text_representation", threshold=4, use_elements=True
        )

        taken = filtered_docset.take()
        assert len(taken) == 2
        assert taken[0].doc_id == "doc_1"
        assert taken[1].doc_id == "doc_2"
        assert mock_llm.generate.call_count == 4

        # doc level field checks
        assert taken[0].properties[new_field] == 4
        assert taken[1].properties[new_field] == 4
        assert taken[0].properties[new_field + "_source_element_index"] is None  # no index
        assert taken[1].properties[new_field + "_source_element_index"] == 2

        filtered_docset = docset.llm_filter(
            new_field=new_field, prompt=[], field="text_representation", threshold=2, use_elements=True
        )

        taken = filtered_docset.take()
        assert mock_llm.generate.call_count == (4 + 3)
        assert len(taken) == 3
        assert taken[0].doc_id == "doc_1"
        assert taken[1].doc_id == "doc_2"
        assert taken[2].doc_id == "doc_3"

    def test_llm_filter_with_doc_structure_with_similarity_sorting(self):
        doc_list = [
            Document(
                doc_id="doc_1",
                elements=[
                    Element(properties={"_element_index": 1}, text_representation="test1"),  # llm_filter result = 4
                    Element(properties={"_element_index": 2}, text_representation="test1"),  # llm_filter result = 4
                ],
            ),
            Document(
                doc_id="doc_2",
                elements=[
                    Element(properties={"_element_index": 4}, text_representation="test2"),  # llm_filter result = 2,
                    Element(properties={"_element_index": 9}, text_representation="test1"),  # llm_filter result = 4
                ],
            ),
            Document(
                doc_id="doc_3",
                elements=[
                    Element(properties={"_element_index": 1}, text_representation="test2"),  # llm_filter result = 2
                ],
            ),
            Document(doc_id="doc_4", text_representation="empty elements, maybe an exploded doc", elements=[]),
        ]
        mock_llm = MockLLM()
        similarity_scorer = TestSimilarityScorer()
        mock_llm.generate = MagicMock(wraps=mock_llm.generate)
        context = sycamore.init(
            params={
                OperationTypes.BINARY_CLASSIFIER: {"llm": mock_llm},
                OperationTypes.TEXT_SIMILARITY: {"similarity_scorer": similarity_scorer},
            },
            exec_mode=ExecMode.LOCAL,
        )
        docset = context.read.document(doc_list)
        new_field = "_autogen_LLMFilterOutput"

        filtered_docset = docset.llm_filter(
            new_field=new_field,
            prompt=[],
            field="text_representation",
            threshold=4,
            use_elements=True,
            similarity_scorer=similarity_scorer,
            similarity_query="this is an unused query because unit test",
        )

        """
        "test2" elements will be in front, resulting in 2 llm calls for doc_2 (first element threshold miss),
        1 each for other 2.
        """
        taken = filtered_docset.take()
        assert len(taken) == 2
        assert taken[0].doc_id == "doc_1"
        assert taken[1].doc_id == "doc_2"
        assert mock_llm.generate.call_count == 4

        filtered_docset = docset.llm_filter(
            new_field=new_field,
            prompt=[],
            field="text_representation",
            threshold=2,
            use_elements=True,
            similarity_scorer=similarity_scorer,
            similarity_query="this is an unused query because unit test",
        )

        """
        "test2" elements will be in front, resulting in 1 llm calls for doc_2 (threshold matches), 1 for other 2
        """
        taken = filtered_docset.take()
        assert mock_llm.generate.call_count == (4 + 3)
        assert len(taken) == 3
        assert taken[0].doc_id == "doc_1"
        assert taken[1].doc_id == "doc_2"
        assert taken[2].doc_id == "doc_3"

    def test_llm_filter(self):
        doc_list = [Document(text_representation="test1"), Document(text_representation="test2")]
        context = sycamore.init(params={OperationTypes.BINARY_CLASSIFIER: {"llm": MockLLM()}}, exec_mode=ExecMode.LOCAL)
        docset = context.read.document(doc_list)
        new_field = "_autogen_LLMFilterOutput"

        filtered_docset = docset.llm_filter(new_field=new_field, prompt=[], field="text_representation", threshold=3)

        assert filtered_docset.count() == 1
        for doc in filtered_docset.take():
            assert doc.text_representation == "test1"
            assert int(doc.properties[new_field]) == 4

        filtered_docset = docset.llm_filter(new_field=new_field, prompt=[], field="text_representation", threshold=2)

        assert filtered_docset.count() == 2

        for doc in filtered_docset.take():
            if doc.text_representation == "test1":
                assert int(doc.properties[new_field]) == 4
            elif doc.text_representation == "test2":
                assert int(doc.properties[new_field]) == 2

    def test_llm_filter_with_keep_none(self):
        doc_list = [Document(text_representation="test1"), Document(text_representation="test2")]
        context = sycamore.init(params={OperationTypes.BINARY_CLASSIFIER: {"llm": MockLLM()}}, exec_mode=ExecMode.LOCAL)
        docset = context.read.document(doc_list)
        new_field = "_autogen_LLMFilterOutput"

        filtered_docset = docset.llm_filter(
            new_field=new_field, prompt=[], field="missing_field", threshold=5, keep_none=True
        ).take()

        assert len(filtered_docset) == 2
        assert filtered_docset[0].text_representation == "test1"
        assert filtered_docset[1].text_representation == "test2"

    def test_llm_filter_with_tokenizer_and_max_tokens(self):
        # Create a mock tokenizer that simply counts tokens
        class MockTokenizer:
            def tokenize(self, text):
                return text.split()  # Simple tokenization by splitting on whitespace

        doc_list = [
            Document(
                doc_id="doc_1",
                elements=[
                    Element(
                        properties={"_element_index": 0}, text_representation="first short element"
                    ),  # llm_filter result = 4
                    Element(properties={"_element_index": 1}, text_representation=None),
                    Element(
                        properties={"_element_index": 2}, text_representation="second longer element with more words"
                    ),
                ],
            ),
            Document(
                doc_id="doc_2",
                elements=[
                    Element(
                        properties={"_element_index": 1}, text_representation="third element"
                    ),  # llm_filter result = 2
                    Element(
                        properties={"_element_index": 2},
                        text_representation="very long element with many words that might exceed token limit",
                    ),  # llm_filter result = 5
                ],
            ),
        ]
        mock_llm = MockLLM()
        mock_tokenizer = MockTokenizer()
        mock_llm.generate = MagicMock(wraps=mock_llm.generate)

        context = sycamore.init(params={OperationTypes.BINARY_CLASSIFIER: {"llm": mock_llm}}, exec_mode=ExecMode.LOCAL)
        docset = context.read.document(doc_list)
        new_field = "_autogen_LLMFilterOutput"

        filtered_docset = docset.llm_filter(
            new_field=new_field,
            prompt=[],
            field="text_representation",
            threshold=3,
            use_elements=True,
            tokenizer=mock_tokenizer,
            max_tokens=10,  # Low token limit to test windowing
        )

        taken = filtered_docset.take()

        assert len(taken) == 2
        assert taken[0].doc_id == "doc_1"
        assert mock_llm.generate.call_count == 3

        # Check the properties of the filtered document
        assert taken[0].properties[new_field] == 4
        assert taken[1].properties[new_field] == 5
        assert taken[0].elements[0]["properties"]["_autogen_LLMFilterOutput_source_element_index"] == {0, 1, 2}
        assert taken[0].elements[1]["properties"]["_autogen_LLMFilterOutput_source_element_index"] == {0, 1, 2}
        assert taken[0].elements[2]["properties"]["_autogen_LLMFilterOutput_source_element_index"] == {0, 1, 2}
        assert taken[1].elements[0]["properties"]["_autogen_LLMFilterOutput_source_element_index"] == {1}
        assert taken[1].elements[1]["properties"]["_autogen_LLMFilterOutput_source_element_index"] == {2}
        assert taken[0].elements[0]["properties"]["_autogen_LLMFilterOutput"] == 4
        assert taken[0].elements[1]["properties"]["_autogen_LLMFilterOutput"] == 4
        assert taken[0].elements[2]["properties"]["_autogen_LLMFilterOutput"] == 4
        assert taken[1].elements[0]["properties"]["_autogen_LLMFilterOutput"] == 2
        assert taken[1].elements[1]["properties"]["_autogen_LLMFilterOutput"] == 5


class BadEntityExtractor(EntityExtractor):
    def __init__(self, entity_name, bad_val):
        super().__init__(entity_name)
        self.bad_val = bad_val

    def extract_entity(
        self, document: Document, context: Optional[Context] = None, llm: Optional[LLM] = None
    ) -> Document:
        document.properties[self._entity_name] = self.bad_val
        return document


class FakeTokenizer(Tokenizer):
    def tokenize(self, text: str, as_ints: bool = False) -> Union[list[int], list[str]]:
        return ["a"]


class TestTolerateFailedEntityExtract:
    def make_document(self):
        return Document(docid=1, properties={"fake": 1}, elements=[Element(properties={"fake": 2})])

    @pytest.mark.parametrize("bad_val", ["", "abcdef"])
    def test_document(self, bad_val):
        from sycamore.transforms.llm_filter import document_threshold_llm_filter

        bee = BadEntityExtractor("extract_prop", bad_val)

        document_threshold_llm_filter(self.make_document(), "properties.fake", bee, 0, True)

    @pytest.mark.parametrize("bad_val", ["", "abcdef"])
    def test_tokenized(self, bad_val):
        from sycamore.transforms.llm_filter import tokenized_threshold_llm_filter

        bee = BadEntityExtractor("extract_prop", bad_val)

        tokenized_threshold_llm_filter(
            self.make_document(), "properties.fake", bee, 0, True, lambda d: d, 10000, FakeTokenizer()
        )

    @pytest.mark.parametrize("bad_val", ["", "abcdef"])
    def test_untokenized(self, bad_val):
        from sycamore.transforms.llm_filter import untokenized_threshold_llm_filter

        bee = BadEntityExtractor("extract_prop", bad_val)

        untokenized_threshold_llm_filter(self.make_document(), "properties.fake", bee, 0, True, lambda d: d)
