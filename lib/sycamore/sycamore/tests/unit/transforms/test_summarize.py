import random
import string

from sycamore.data import Document, Element
from sycamore.llms import LLM
from sycamore.llms.prompts.default_prompts import SummarizeDataHeirarchicalPrompt
from sycamore.transforms.summarize import (
    HeirarchicalDocumentSummarizer,
    LLMElementTextSummarizer,
    MaxTokensHeirarchicalDocumentSummarizer,
    RoundRobinOneshotDocumentSummarizer,
    EtCetera,
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
        llm = mocker.Mock(spec=LLM)
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


class TestHeirarchicalSummarize:
    doc = Document(
        elements=[
            Element(text_representation="a", properties={"key": "m"}),
            Element(text_representation="b", properties={"key": "n"}),
            Element(text_representation="c", properties={"key": "o"}),
            Element(text_representation="d", properties={"key": "p"}),
            Element(text_representation="e", properties={"key": "q"}),
        ]
    )

    def test_heirarchical_calls_noargs(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = HeirarchicalDocumentSummarizer(llm=llm)
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        for e in self.doc.elements:
            assert f"Text: {e.text_representation}" in usermessage
            assert f"properties.key: {e.properties['key']}" not in usermessage

    def test_heirarchical_set_fields(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = HeirarchicalDocumentSummarizer(llm=llm, fields=["properties.key"])
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        for e in self.doc.elements:
            assert f"Text: {e.text_representation}" in usermessage
            assert f"properties.key: {e.properties['key']}" in usermessage

    def test_heirarchical_all_fields(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = HeirarchicalDocumentSummarizer(llm=llm, fields="*")
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        for e in self.doc.elements:
            assert f"Text: {e.text_representation}" in usermessage
            assert f"key: {e.properties['key']}" in usermessage

    def test_heirarchical_set_question(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = HeirarchicalDocumentSummarizer(llm=llm, question="say what?")
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert "say what?" in usermessage
        for e in self.doc.elements:
            assert f"Text: {e.text_representation}" in usermessage

    def test_heirarchical_num_elements(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = HeirarchicalDocumentSummarizer(llm=llm, element_batch_size=3)
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
        assert "Summary: sum" in usermessage

    def test_heirarchical_set_prompt(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = HeirarchicalDocumentSummarizer(llm=llm, prompt=SummarizeDataHeirarchicalPrompt.set(data_description="FINDME"))  # type: ignore
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert "FINDME" in usermessage
        for e in self.doc.elements:
            assert f"Text: {e.text_representation}" in usermessage


class TestMaxTokensHeirarchicalSummarize:
    doc = Document(
        elements=[
            Element(text_representation="aaaaa", properties={"key": "m"}),
            Element(text_representation="bbbbb", properties={"key": "n"}),
            Element(text_representation="ccccc", properties={"key": "o"}),
            Element(text_representation="ddddd", properties={"key": "p"}),
            Element(text_representation="eeeee", properties={"key": "q"}),
        ]
    )

    def test_base(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = MaxTokensHeirarchicalDocumentSummarizer(llm=llm)
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        for e in self.doc.elements:
            assert f"Text: {e.text_representation}" in usermessage
            assert f"properties.key: {e.properties['key']}" not in usermessage

    def test_small_token_limit(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = MaxTokensHeirarchicalDocumentSummarizer(llm=llm, max_tokens=310)  # 310 chars = first 3 elements
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
        assert "Summary: sum" in usermessage


class TestRoundRobinOneshotSummarize:
    doc = Document(
        elements=[
            Element(
                text_representation="Something very long",
                properties={
                    f"state_{i}": s
                    for i, s in enumerate(list(USStateStandardizer.state_abbreviations.values()) + ["Canada"])
                }
                | {"title": "Title A"},
                elements=[Element(text_representation="subelement 1"), Element(text_representation="subelement 2")],
            ),
            Element(
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
            Element(
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
            Element(
                text_representation="Something very long part 4",
                properties={
                    f"state_{i}": s
                    for i, s in enumerate(reversed(list(USStateStandardizer.state_abbreviations.values()) + ["Canada"]))
                }
                | {"title": "Title D"},
            ),
        ]
    )

    def test_basic(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = RoundRobinOneshotDocumentSummarizer(llm, question="say what?")
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
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = RoundRobinOneshotDocumentSummarizer(
            llm, question="say what?", fields=["properties.title", EtCetera]
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
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = RoundRobinOneshotDocumentSummarizer(llm, question="say what?", fields=["properties.title"])
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
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = RoundRobinOneshotDocumentSummarizer(
            llm, question="say what?", fields=["properties.title"], token_limit=1000
        )
        d = summarizer.summarize(self.doc)

        assert d.properties["summary"] == "sum"
        assert generate.call_count == 1
        prompt = generate.call_args.kwargs["prompt"]
        usermessage = prompt.messages[-1].content
        assert occurrences(usermessage, "subelement") == 32
        assert occurrences(usermessage, "properties.title") == 4
        assert "say what?" in usermessage
        assert "properties.state" not in usermessage

    def test_too_many_tokens_takes_evenly(self, mocker):
        llm = mocker.Mock(spec=LLM)
        generate = mocker.patch.object(llm, "generate")
        generate.return_value = "sum"
        summarizer = RoundRobinOneshotDocumentSummarizer(
            llm, question="say what?", fields=["properties.title"], token_limit=500
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


def filter_elements_on_length(element: Element) -> bool:
    return False if element.text_representation is None else len(element.text_representation) > 10


def occurrences(superstring: str, substring: str) -> int:
    return len(superstring.split(sep=substring)) - 1
