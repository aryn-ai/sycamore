from typing import Optional, Sequence, Callable, Union

from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts.prompts import SycamorePrompt, RenderedPrompt
from sycamore.plan_nodes import Node
from sycamore.transforms.map import MapBatch
from sycamore.data import Document, Element


def _infer_prompts(
    prompts: list[Sequence[RenderedPrompt]],
    llm: LLM,
    llm_mode: LLMMode,
    is_done: Callable[[str], bool] = lambda s: True,
) -> list[tuple[str, int]]:
    if llm_mode == LLMMode.SYNC:
        res = []
        for piter in prompts:
            s = ""
            i = -1
            for p in piter:
                i += 1
                s = llm.generate(prompt=p)
                if is_done(s):
                    break
            res.append((s, i))
        return res
    elif llm_mode == LLMMode.ASYNC:
        raise NotImplementedError("Haven't done async yet")
    elif llm_mode == LLMMode.BATCH:
        raise NotImplementedError("Haven't done batch yet")
    else:
        raise NotImplementedError("Unknown LLM Mode")


class LLMMap(MapBatch):
    """The LLMMap transform renders each Document in a docset into
    a prompt for an LLM, calls the LLM, and attaches the output to
    the document.

    Args:

        child: Child node in the sycamore execution graph
        prompt: The SycamorePrompt to use to render each document.
            Must implement the ``render_document`` method.
        output_field: The name of the field in doc.properties in which
            to store the llm output
        llm: The llm to use for inference.
        llm_mode: How to call the llm - sync/async/batch. All LLMs do not
            necessarily implement all options.
        postprocess_fn: function to call on documents after performing the
            llm inference. If the prompt rendered into multiple RenderedPrompts,
            ``i`` is the index of the RenderedPrompt that succeeded; if the
            prompt rendered into an empty list, ``i`` is -1; and otherwise
            ``i`` is 0

    Example:
         .. code-block:: python

            prompt = EntityExtractorZeroShotGuidancePrompt.set(entity="title")

            docset.llm_map(
                prompt=prompt,
                output_field="title",
                llm=OpenAI(OpenAIModels.GPT_4O_MINI)
            )
    """

    def __init__(
        self,
        child: Optional[Node],
        prompt: SycamorePrompt,
        output_field: str,
        llm: LLM,
        llm_mode: LLMMode = LLMMode.SYNC,
        postprocess_fn: Callable[[Document, int], Document] = lambda d, i: d,
        **kwargs,
    ):
        self._prompt = prompt
        self._validate_prompt()
        self._output_field = output_field
        self._llm = llm
        self._llm_mode = llm_mode
        self._postprocess_fn = postprocess_fn
        super().__init__(child, f=self.llm_map, **kwargs)

    def llm_map(self, documents: list[Document]) -> list[Document]:
        rendered = [self._prompt.render_document(d) for d in documents]
        rendered = _as_sequences(rendered)
        results = _infer_prompts(rendered, self._llm, self._llm_mode, self._prompt.is_done)
        postprocessed = []
        for d, (r, i) in zip(documents, results):
            d.properties[self._output_field] = r
            new_d = self._postprocess_fn(d, i)
            postprocessed.append(new_d)
        return postprocessed

    def _validate_prompt(self):
        doc = Document()
        try:
            _ = self._prompt.render_document(doc)
        except NotImplementedError as e:
            raise e
        except Exception:
            pass


class LLMMapElements(MapBatch):
    """The LLMMapElements transform renders each Element for each
    Document in a docset into a prompt for an LLM, calls the LLM,
    and attaches the output to the document.

    Args:
        child: Child node in the sycamore execution graph
        prompt: The SycamorePrompt to use to render each element.
            Must implement the ``render_element`` method.
        output_field: The name of the field in elt.properties in which
            to store the llm output.
        llm: The llm to use for inference.
        llm_mode: How to call the llm - sync/async/batch. All LLMs do not
            necessarily implement all options.
        postprocess_fn: function to call on documents after performing the
            llm inference. If the prompt rendered into multiple RenderedPrompts,
            ``i`` is the index of the RenderedPrompt that succeeded; if the
            prompt rendered into an empty list, ``i`` is -1; and otherwise
            ``i`` is 0

    Example:
         .. code-block:: python

            prompt = TextSummarizerGuidancePrompt

            docset.llm_map_elements(
                prompt = prompt,
                output_field = "summary",
                llm = OpenAI(OpenAIModels.GPT_4O)
    """

    def __init__(
        self,
        child: Optional[Node],
        prompt: SycamorePrompt,
        output_field: str,
        llm: LLM,
        llm_mode: LLMMode = LLMMode.SYNC,
        postprocess_fn: Callable[[Element, int], Element] = lambda e, i: e,
        **kwargs,
    ):
        self._prompt = prompt
        self._validate_prompt()
        self._output_field = output_field
        self._llm = llm
        self._llm_mode = llm_mode
        self._postprocess_fn = postprocess_fn
        super().__init__(child, f=self.llm_map_elements, **kwargs)

    def llm_map_elements(self, documents: list[Document]) -> list[Document]:
        rendered = [(d, e, self._prompt.render_element(e, d)) for d in documents for e in d.elements]
        results = _infer_prompts(
            _as_sequences([p for _, _, p in rendered]), self._llm, self._llm_mode, self._prompt.is_done
        )
        new_elts = []
        last_doc = None
        for (r, i), (d, e, _) in zip(results, rendered):
            if last_doc is not None and last_doc.doc_id != d.doc_id:
                last_doc.elements = new_elts
                new_elts = []
            e.properties[self._output_field] = r
            new_elts.append(self._postprocess_fn(e, i))
            last_doc = d
        if last_doc is not None:
            last_doc.elements = new_elts
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


def _as_sequences(ls: list[Union[RenderedPrompt, Sequence[RenderedPrompt]]]) -> list[Sequence[RenderedPrompt]]:
    return [[p] if isinstance(p, RenderedPrompt) else p for p in ls]
