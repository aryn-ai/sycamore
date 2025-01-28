from sycamore.data import Document, Element
from sycamore.llms.llms import LLM
from sycamore.llms.prompts import RenderedPrompt, SycamorePrompt
from sycamore.llms.prompts.prompts import RenderedMessage
from sycamore.transforms.base_llm import LLMMap, LLMMapElements
import pytest
from typing import Optional


class FakeLLM(LLM):
    def __init__(self):
        super().__init__(model_name="dummy")

    def is_chat_mode(self) -> bool:
        return True

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        return "".join(m.content for m in prompt.messages)


class FakeDocPrompt(SycamorePrompt):
    def render_document(self, doc: Document) -> RenderedPrompt:
        return RenderedPrompt(messages=[RenderedMessage(role="system", content=doc.text_representation or "None")])


class FakeEltPrompt(SycamorePrompt):
    def render_element(self, elt: Element, doc: Document) -> RenderedPrompt:
        return RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content=doc.text_representation or "None"),
                RenderedMessage(role="user", content=elt.text_representation or "None"),
            ]
        )


class TestLLMMap:
    def test_wrong_prompt_fails_fast(self):
        prompt = FakeEltPrompt()
        llm = FakeLLM()
        with pytest.raises(NotImplementedError) as einfo:
            _ = LLMMap(None, prompt, "out", llm)
        assert "FakeEltPrompt" in str(einfo.value)

    def test_happy_path(self):
        prompt = FakeDocPrompt()
        llm = FakeLLM()
        doc1 = Document({"text_representation": "ooga"})
        doc2 = Document({"text_representation": "booga"})
        map = LLMMap(None, prompt, "out", llm)
        outdocs = map.llm_map([doc1, doc2])

        assert outdocs[0].text_representation == "ooga"
        assert outdocs[0].properties["out"] == "ooga"
        assert outdocs[1].text_representation == "booga"
        assert outdocs[1].properties["out"] == "booga"

    def test_postprocess(self):
        prompt = FakeDocPrompt()
        llm = FakeLLM()
        doc1 = Document({"text_representation": "ooga"})
        doc2 = Document({"text_representation": "booga"})
        count = 0

        def ppfn(d: Document, i: int) -> Document:
            nonlocal count
            count += 1
            return d

        map = LLMMap(None, prompt, "out", llm, postprocess_fn=ppfn)
        _ = map.llm_map([doc1, doc2])

        assert count == 2


class TestLLMMapElements:
    def test_wrong_prompt_fails_fast(self):
        prompt = FakeDocPrompt()
        llm = FakeLLM()
        with pytest.raises(NotImplementedError) as einfo:
            _ = LLMMapElements(None, prompt, "out", llm)
        assert "FakeDocPrompt" in str(einfo.value)

    def test_happy_path(self):
        prompt = FakeEltPrompt()
        llm = FakeLLM()
        doc1 = Document(
            {
                "doc_id": "1",
                "text_representation": "ooga",
                "elements": [{"text_representation": "yo"}, {"text_representation": "ho"}],
            }
        )
        doc2 = Document({"doc_id": "2", "elements": [{"text_representation": "booga"}, {}]})
        map = LLMMapElements(None, prompt, "out", llm)
        outdocs = map.llm_map_elements([doc1, doc2])

        assert outdocs[0].elements[0].properties["out"] == "oogayo"
        assert outdocs[0].elements[1].properties["out"] == "oogaho"
        assert outdocs[1].elements[0].properties["out"] == "Nonebooga"
        assert outdocs[1].elements[1].properties["out"] == "NoneNone"

    def test_postprocess(self):
        prompt = FakeEltPrompt()
        llm = FakeLLM()
        doc1 = Document(
            {
                "doc_id": "1",
                "text_representation": "ooga",
                "elements": [{"text_representation": "yo"}, {"text_representation": "ho"}],
            }
        )
        doc2 = Document({"doc_id": "2", "elements": [{"text_representation": "booga"}, {}]})
        count = 0

        def ppfn(e: Element, i: int) -> Element:
            nonlocal count
            count += 1
            return e

        map = LLMMapElements(None, prompt, "out", llm, postprocess_fn=ppfn)
        _ = map.llm_map_elements([doc1, doc2])

        assert count == 4
