from typing import Optional

from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts.prompts import SycamorePrompt, RenderedPrompt
from sycamore.plan_nodes import Node
from sycamore.transforms.map import MapBatch
from sycamore.data import Document, Element


def _infer_prompts(prompts: list[RenderedPrompt], llm: LLM, llm_mode: LLMMode) -> list[str]:
    if llm_mode == LLMMode.SYNC:
        return [llm.generate(p) for p in prompts]
    elif llm_mode == LLMMode.ASYNC:
        raise NotImplementedError("Haven't done async yet")
    elif llm_mode == LLMMode.BATCH:
        raise NotImplementedError("Haven't done batch yet")
    else:
        raise NotImplementedError("Unknown LLM Mode")


class LLMMap(MapBatch):
    def __init__(
        self,
        child: Optional[Node],
        prompt: SycamorePrompt,
        output_field: str,
        llm: LLM,
        llm_mode: LLMMode = LLMMode.SYNC,
        **kwargs,
    ):
        self._prompt = prompt
        self._validate_prompt()
        self._output_field = output_field
        self._llm = llm
        self._llm_mode = llm_mode
        super().__init__(child, f=self.llm_map, **kwargs)

    def llm_map(self, documents: list[Document]) -> list[Document]:
        rendered = [self._prompt.render_document(d) for d in documents]
        results = _infer_prompts(rendered, self._llm, self._llm_mode)
        for d, r in zip(documents, results):
            d.properties[self._output_field] = r
        return documents

    def _validate_prompt(self):
        doc = Document()
        try:
            _ = self._prompt.render_document(doc)
        except NotImplementedError as e:
            raise e
        except Exception:
            pass


class LLMMapElements(MapBatch):
    def __init__(
        self,
        child: Optional[Node],
        prompt: SycamorePrompt,
        output_field: str,
        llm: LLM,
        llm_mode: LLMMode = LLMMode.SYNC,
        **kwargs,
    ):
        self._prompt = prompt
        self._validate_prompt()
        self._output_field = output_field
        self._llm = llm
        self._llm_mode = llm_mode
        super().__init__(child, f=self.llm_map_elements, **kwargs)

    def llm_map_elements(self, documents: list[Document]) -> list[Document]:
        rendered = [(e, self._prompt.render_element(e, d)) for d in documents for e in d.elements]
        results = _infer_prompts([p for _, p in rendered], self._llm, self._llm_mode)
        for r, (e, _) in zip(results, rendered):
            e.properties[self._output_field] = r
        return documents

    def _validate_prompt(self):
        doc = Document()
        elt = Element()
        try:
            _ = self._prompt.render_element(elt, doc)
        except NotImplementedError as e:
            raise e
        except Exception:
            pass
