import random
import string

from sycamore.data import Document, Element
from sycamore.llms import LLM
from sycamore.llms.llms import LLMMode
from sycamore.functions.tokenizer import CharacterTokenizer
from sycamore.transforms.summarize import (
    LLMElementTextSummarizer,
    OneStepDocumentSummarizer,
    MultiStepDocumentSummarizer,
    EtCetera,
    SummaryDocument,
)
from sycamore.transforms.standardizer import USStateStandardizer


class TestSummarize:
    def test_summarize_text_does_not_call_llm(self, mocker):
        llm = mocker.Mock(spec=LLM)
        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        doc.elements = [element1]

        text_summarizer = LLMElementTextSummarizer(llm, filter_elements_on_length)
        doc = text_summarizer.summarize(doc)

        assert doc.elements[0].properties == {}

    def test_summarize_text_calls_llm(self, mocker):
        llm = TestOneStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "this is the summary"
        doc = Document()
        element1 = Element()
        element1.text_representation = "".join(random.choices(string.ascii_letters, k=10))
        element2 = Element()
        element2.text_representation = "".join(random.choices(string.ascii_letters, k=20))
        doc.elements = [element1, element2]

        text_summarizer = LLMElementTextSummarizer(llm, filter_elements_on_length)
        doc = text_summarizer.summarize(doc)

        assert doc.elements[0].properties == {}
        assert doc.elements[1].properties == {"summary": "this is the summary"}


class TestMultiStepSummarize:
    doc = Document(
        elements=[
            Element(text_representation="aaaaaaaa", properties={"key": "m"}),
            Element(text_representation="bbbbbbbb", properties={"key": "n"}),
            Element(text_representation="cccccccc", properties={"key": "o"}),
            Element(text_representation="dddddddd", properties={"key": "p"}),
            Element(text_representation="eeeeeeee", properties={"key": "q"}),
        ]
    )

    @staticmethod
    def llm(mocker):
        llm = mocker.Mock(spec=LLM)
        mode = mocker.patch.object(llm, "default_mode")
        mode.return_value = LLMMode.SYNC
        return llm

    def test_base(self, mocker):
        llm = TestMultiStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = MultiStepDocumentSummarizer(llm=llm)
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        for e in self.doc.elements:
            assert f"Text: {e.text_representation}" in usermessage
            assert f"properties.key: {e.properties['key']}" not in usermessage

    def test_multistep_set_fields(self, mocker):
        llm = TestMultiStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = MultiStepDocumentSummarizer(llm=llm, fields=["properties.key"])
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        for e in self.doc.elements:
            assert f"Text: {e.text_representation}" in usermessage
            assert f"properties.key: {e.properties['key']}" in usermessage

    def test_multistep_all_fields(self, mocker):
        llm = TestMultiStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = MultiStepDocumentSummarizer(llm=llm, fields=[EtCetera])
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        for e in self.doc.elements:
            assert f"Text: {e.text_representation}" in usermessage
            assert f"key: {e.properties['key']}" in usermessage

    def test_multistep_set_question(self, mocker):
        llm = TestMultiStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = MultiStepDocumentSummarizer(llm=llm, question="loser says what?")
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert "loser says what?" in usermessage
        for e in self.doc.elements:
            assert f"Text: {e.text_representation}" in usermessage
            assert f"properties.key: {e.properties['key']}" not in usermessage

    def test_small_token_limit(self, mocker):
        llm = TestMultiStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = MultiStepDocumentSummarizer(
            llm=llm, tokenizer=CharacterTokenizer(max_tokens=570)
        )  # 1 element -> ~500 chars
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 3
        first_call = generate.call_args_list[0]
        second_call = generate.call_args_list[1]
        third_call = generate.call_args_list[2]

        prompt = first_call.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        for e in self.doc.elements[:3]:
            assert f"Text: {e.text_representation}" in usermessage
        for e in self.doc.elements[3:]:
            assert f"Text: {e.text_representation}" not in usermessage

        prompt = second_call.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        for e in self.doc.elements[:3]:
            assert f"Text: {e.text_representation}" not in usermessage
        for e in self.doc.elements[3:]:
            assert f"Text: {e.text_representation}" in usermessage

        prompt = third_call.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert occurrences(usermessage, ": sum") == 2

    def test_no_sub_docs(self, mocker):
        llm = TestMultiStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "no summary"

        single_element_doc = SummaryDocument()

        summarizer = MultiStepDocumentSummarizer(llm=llm, question="What is this about?")
        d = summarizer.summarize(single_element_doc)

        assert d.properties["summary"] == "Empty Summary Document, nothing to summarize"
        assert generate.call_count == 0  # No sub-documents to summarize

    def test_summary_document(self, mocker):
        llm = TestMultiStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = MultiStepDocumentSummarizer(
            llm=llm, tokenizer=CharacterTokenizer(max_tokens=1000)
        )  # 1 element -> ~500 chars
        d = summarizer.summarize(TestOneStepSummarize.doc)
        assert d.properties["summary"] == "sum"

    def test_summary_document_serde(self, mocker):
        serde_doc = Document.deserialize(TestOneStepSummarize.doc.serialize())
        llm = TestMultiStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = MultiStepDocumentSummarizer(
            llm=llm, tokenizer=CharacterTokenizer(max_tokens=1000)
        )  # 1 element -> ~500 chars
        d = summarizer.summarize(serde_doc)
        assert d.properties["summary"] == "sum"


class TestOneStepSummarize:
    doc = SummaryDocument(
        sub_docs=[
            Document(
                text_representation="Something very long",
                properties={
                    f"state_{i}": s
                    for i, s in enumerate(list(USStateStandardizer.state_abbreviations.values()) + ["Canada"])
                }
                | {"title": "Title A"},
                elements=[
                    Element(text_representation="subelement 1"),
                    Element(text_representation="subelement 2"),
                ],
            ),
            Document(
                text_representation="Something very long part 2",
                properties={
                    f"state_{i}": s
                    for i, s in enumerate(
                        sorted(
                            list(USStateStandardizer.state_abbreviations.values()) + ["Canada"],
                            key=lambda state: state[1:],
                        )
                    )
                }
                | {"title": "Title B"},
                elements=[
                    Element(text_representation="subelement 1"),
                    Element(text_representation="subelement 2"),
                    Element(text_representation="subelement 3"),
                    Element(text_representation="subelement 4"),
                    Element(text_representation="subelement 5"),
                ]
                * 10,
            ),
            Document(
                text_representation="Something very long part 3",
                properties={
                    f"state_{i}": s
                    for i, s in enumerate(
                        sorted(
                            list(USStateStandardizer.state_abbreviations.values()) + ["Canada"],
                            key=lambda state: state[3:],
                        )
                    )
                }
                | {"title": "Title C"},
            ),
            Document(
                text_representation="Something very long part 4",
                properties={
                    f"state_{i}": s
                    for i, s in enumerate(reversed(list(USStateStandardizer.state_abbreviations.values()) + ["Canada"]))
                }
                | {"title": "Title D"},
            ),
        ]
    )

    @staticmethod
    def llm(mocker):
        llm = mocker.Mock(spec=LLM)
        mode = mocker.patch.object(llm, "default_mode")
        mode.return_value = LLMMode.SYNC
        return llm

    def test_basic(self, mocker):
        llm = TestOneStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = OneStepDocumentSummarizer(llm, question="say what?", fields=[])
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert occurrences(usermessage, "subelement") == 52
        assert "say what?" in usermessage
        for e in self.doc.elements:
            for p in e.properties:
                assert f"properties.{p}: {e.properties[p]}" in usermessage

    def test_title_first(self, mocker):
        llm = TestOneStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = OneStepDocumentSummarizer(
            llm,
            question="say what?",
            fields=["properties.title", EtCetera],
            tokenizer=CharacterTokenizer(max_tokens=1_000_000),
        )
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert occurrences(usermessage, "subelement") == 52
        # Intro to each element is "Entry i:"
        # other properties are "property name: property value"
        # if last nonwhitespace before properties.title is ':',
        # then properties.title was the first property
        before_title = usermessage.split(sep="properties.title")[:-1]
        assert all(b.strip().endswith(":") for b in before_title)
        assert "say what?" in usermessage
        for e in self.doc.elements:
            for p in e.properties:
                assert f"properties.{p}: {e.properties[p]}" in usermessage

    def test_only_title(self, mocker):
        llm = TestOneStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = OneStepDocumentSummarizer(llm, question="say what?", fields=["properties.title"])
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert occurrences(usermessage, "subelement") == 52
        assert occurrences(usermessage, "properties.title") == 4
        assert "say what?" in usermessage
        assert "properties.state" not in usermessage

    def test_too_many_tokens(self, mocker):
        llm = TestOneStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = OneStepDocumentSummarizer(
            llm,
            question="say what?",
            fields=["properties.title"],
            tokenizer=CharacterTokenizer(max_tokens=1850),
        )
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert occurrences(usermessage, "subelement") == 34
        assert occurrences(usermessage, "properties.title") == 4
        assert "say what?" in usermessage
        assert "properties.state" not in usermessage

    def test_too_many_tokens_takes_evenly(self, mocker):
        llm = TestOneStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = OneStepDocumentSummarizer(
            llm,
            question="say what?",
            fields=["properties.title"],
            tokenizer=CharacterTokenizer(max_tokens=600),
        )
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert occurrences(usermessage, "subelement") == 2
        assert "subelement 2" not in usermessage
        assert occurrences(usermessage, "properties.title") == 4
        assert "say what?" in usermessage
        assert "properties.state" not in usermessage

    def test_no_sub_docs(self, mocker):
        llm = TestOneStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"

        doc = SummaryDocument()

        summarizer = OneStepDocumentSummarizer(llm, question="What is this about?")
        summarized_doc = summarizer.summarize(doc)

        assert summarized_doc.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert "What is this about?" in usermessage

    def test_basic_with_serde(self, mocker):
        serde_doc = Document.deserialize(self.doc.serialize())
        llm = TestOneStepSummarize.llm(mocker)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = OneStepDocumentSummarizer(llm, question="say what?", fields=[])
        d = summarizer.summarize(serde_doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert occurrences(usermessage, "subelement") == 52
        assert "say what?" in usermessage
        for e in self.doc.elements:
            for p in e.properties:
                assert f"properties.{p}: {e.properties[p]}" in usermessage


def filter_elements_on_length(element: Element) -> bool:
    return False if element.text_representation is None else len(element.text_representation) > 10


def occurrences(superstring: str, substring: str) -> int:
    return len(superstring.split(sep=substring)) - 1
