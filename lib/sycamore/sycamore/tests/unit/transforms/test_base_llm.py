from sycamore.data import Document, Element
from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts import RenderedPrompt, SycamorePrompt
from sycamore.llms.prompts.prompts import RenderedMessage
from sycamore.transforms.base_llm import LLMMap, LLMMapElements
import pytest
from typing import Any, Optional


class FakeLLM(LLM):
    def __init__(self, default_mode: LLMMode = LLMMode.SYNC):
        super().__init__(model_name="dummy", default_mode=default_mode)
        self.async_calls = 0
        self.used_llm_kwargs: dict[str, Any] = {}

    def is_chat_mode(self) -> bool:
        return True

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        self.used_llm_kwargs = self._merge_llm_kwargs(llm_kwargs)
        return "".join(m.content for m in prompt.messages)

    async def generate_async(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        self.async_calls += 1
        return self.generate(prompt=prompt, llm_kwargs=llm_kwargs)


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

    @pytest.mark.parametrize("mode", [LLMMode.SYNC, LLMMode.ASYNC])
    def test_happy_path(self, mode):
        prompt = FakeDocPrompt()
        llm = FakeLLM()
        doc1 = Document({"text_representation": "ooga"})
        doc2 = Document({"text_representation": "booga"})
        map = LLMMap(None, prompt, "out", llm, llm_mode=mode)
        outdocs = map.llm_map([doc1, doc2])

        assert outdocs[0].text_representation == "ooga"
        assert outdocs[0].properties["out"] == "ooga"
        assert outdocs[1].text_representation == "booga"
        assert outdocs[1].properties["out"] == "booga"

    @pytest.mark.parametrize("mode", [LLMMode.SYNC, LLMMode.ASYNC])
    def test_mode_from_llm(self, mode):
        prompt = FakeDocPrompt()
        llm = FakeLLM(default_mode=mode)
        doc1 = Document({"text_representation": "ooga"})
        doc2 = Document({"text_representation": "booga"})
        map = LLMMap(None, prompt, "out", llm)
        outdocs = map.llm_map([doc1, doc2])

        assert outdocs[0].text_representation == "ooga"
        assert outdocs[0].properties["out"] == "ooga"
        assert outdocs[1].text_representation == "booga"
        assert outdocs[1].properties["out"] == "booga"
        if mode == LLMMode.SYNC:
            assert llm.async_calls == 0
        if mode == LLMMode.ASYNC:
            assert llm.async_calls == 2

    @pytest.mark.parametrize("mode", [LLMMode.SYNC, LLMMode.ASYNC])
    def test_validate(self, mode):
        prompt = FakeDocPrompt()
        llm = FakeLLM()
        doc1 = Document({"text_representation": "ooga"})
        doc2 = Document({"text_representation": "booga"})
        count = 0

        def valfn(d: Document) -> bool:
            nonlocal count
            count += 1
            return count > 1

        map = LLMMap(None, prompt, "out", llm, validate=valfn, llm_mode=mode)
        _ = map.llm_map([doc1, doc2])

        assert count == 2


class TestLLMMapElements:
    def test_wrong_prompt_fails_fast(self):
        prompt = FakeDocPrompt()
        llm = FakeLLM()
        with pytest.raises(NotImplementedError) as einfo:
            _ = LLMMapElements(None, prompt, "out", llm)
        assert "FakeDocPrompt" in str(einfo.value)

    @pytest.mark.parametrize("mode", [LLMMode.SYNC, LLMMode.ASYNC])
    def test_happy_path(self, mode):
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
        map = LLMMapElements(None, prompt, "out", llm, llm_mode=mode)
        outdocs = map.llm_map_elements([doc1, doc2])

        assert outdocs[0].elements[0].properties["out"] == "oogayo"
        assert outdocs[0].elements[1].properties["out"] == "oogaho"
        assert outdocs[1].elements[0].properties["out"] == "Nonebooga"
        assert outdocs[1].elements[1].properties["out"] == "NoneNone"

    @pytest.mark.parametrize("mode", [LLMMode.SYNC, LLMMode.ASYNC])
    def test_mode_from_llm(self, mode):
        prompt = FakeEltPrompt()
        llm = FakeLLM(default_mode=mode)
        doc1 = Document(
            {
                "doc_id": "1",
                "text_representation": "ooga",
                "elements": [{"text_representation": "yo"}, {"text_representation": "ho"}],
            }
        )
        doc2 = Document({"doc_id": "2", "elements": [{"text_representation": "booga"}, {}]})
        map = LLMMapElements(None, prompt, "out", llm, llm_mode=mode)
        outdocs = map.llm_map_elements([doc1, doc2])

        assert outdocs[0].elements[0].properties["out"] == "oogayo"
        assert outdocs[0].elements[1].properties["out"] == "oogaho"
        assert outdocs[1].elements[0].properties["out"] == "Nonebooga"
        assert outdocs[1].elements[1].properties["out"] == "NoneNone"
        if mode == LLMMode.SYNC:
            assert llm.async_calls == 0
        if mode == LLMMode.ASYNC:
            assert llm.async_calls == 4

    @pytest.mark.parametrize("mode", [LLMMode.SYNC, LLMMode.ASYNC])
    def test_postprocess(self, mode):
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

        def valfn(e: Element) -> bool:
            nonlocal count
            count += 1
            return count > 1

        map = LLMMapElements(None, prompt, "out", llm, validate=valfn, llm_mode=mode)
        _ = map.llm_map_elements([doc1, doc2])

        assert count == 4
