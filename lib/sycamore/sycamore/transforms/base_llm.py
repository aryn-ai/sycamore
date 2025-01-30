from typing import Optional, Sequence, Callable, Union

from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts.prompts import SycamorePrompt, RenderedPrompt
from sycamore.plan_nodes import Node
from sycamore.transforms.map import MapBatch
from sycamore.data import Document, Element


def _infer_prompts(
    prompts: list[RenderedPrompt],
    llm: LLM,
    llm_mode: LLMMode,
) -> list[str]:
    if llm_mode == LLMMode.SYNC:
        res = []
        for p in prompts:
            s = llm.generate(prompt=p)
            res.append(s)
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
        iteration_var: Name of the document property to increment with every
            invalid response. Default is None, which means no re-try.
        validate: Function to determine whether an LLM response is valid.
            Default is 'everything is valid'
        max_tries: Hard limit on the number of LLM calls per document. Default
            is 5

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
        iteration_var: Optional[str] = None,
        validate: Callable[[Document], bool] = lambda d: True,
        max_tries: int = 5,
        **kwargs,
    ):
        self._prompt = prompt
        self._validate_prompt()
        self._output_field = output_field
        self._llm = llm
        self._llm_mode = llm_mode
        self._iteration_var = iteration_var
        self._validate = validate
        self._max_tries = max_tries
        super().__init__(child, f=self.llm_map, **kwargs)

    def llm_map(self, documents: list[Document]) -> list[Document]:
        if self._iteration_var is not None:
            for d in documents:
                d.properties[self._iteration_var] = 0

        valid = [False] * len(documents)
        tries = 0
        while not all(valid) and tries < self._max_tries:
            tries += 1
            rendered = [self._prompt.render_document(d) for v, d in zip(valid, documents) if not v]
            if sum([0, *(len(r.messages) for r in rendered)]) == 0:
                break
            results = _infer_prompts(rendered, self._llm, self._llm_mode)
            ri = 0
            for i in range(len(documents)):
                if valid[i]:
                    continue
                documents[i].properties[self._output_field] = results[ri]
                valid[i] = self._validate(documents[i])
                ri += 1
                if self._iteration_var is not None and not valid[i]:
                    documents[i].properties[self._iteration_var] += 1
            if self._iteration_var is None:
                break

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
    """The LLMMapElements transform renders each Element for each
    Document in a docset into a prompt for an LLM, calls the LLM,
    and attaches the output to the element.

    Args:
        child: Child node in the sycamore execution graph
        prompt: The SycamorePrompt to use to render each element.
            Must implement the ``render_element`` method.
        output_field: The name of the field in elt.properties in which
            to store the llm output.
        llm: The llm to use for inference.
        llm_mode: How to call the llm - sync/async/batch. All LLMs do not
            necessarily implement all options.
        iteration_var: Name of the element property to increment with every
            invalid response. Default is None, which means no re-try.
        validate: Function to determine whether an LLM response is valid.
            Default is 'everything is valid'
        max_tries: Hard limit on the number of LLM calls per element. Default
            is 5

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
        iteration_var: Optional[str] = None,
        validate: Callable[[Element], bool] = lambda d: True,
        max_tries: int = 5,
        **kwargs,
    ):
        self._prompt = prompt
        self._validate_prompt()
        self._output_field = output_field
        self._llm = llm
        self._llm_mode = llm_mode
        self._iteration_var = iteration_var
        self._validate = validate
        self._max_tries = max_tries
        super().__init__(child, f=self.llm_map_elements, **kwargs)

    def llm_map_elements(self, documents: list[Document]) -> list[Document]:
        elt_doc_pairs = [(e, d) for d in documents for e in d.elements]
        if self._iteration_var is not None:
            for e, _ in elt_doc_pairs:
                e.properties[self._iteration_var] = 0

        valid = [False] * len(elt_doc_pairs)
        tries = 0
        while not all(valid) and tries < self._max_tries:
            tries += 1
            rendered = [self._prompt.render_element(e, d) for v, (e, d) in zip(valid, elt_doc_pairs) if not v]
            if sum([0, *(len(r.messages) for r in rendered)]) == 0:
                break
            results = _infer_prompts(rendered, self._llm, self._llm_mode)
            ri = 0
            for i in range(len(elt_doc_pairs)):
                if valid[i]:
                    continue
                print(ri)
                elt, doc = elt_doc_pairs[i]
                elt.properties[self._output_field] = results[ri]
                valid[i] = self._validate(elt)
                ri += 1
                if self._iteration_var is not None:
                    elt.properties[self._iteration_var] += 1
            if self._iteration_var is None:
                break

        last_doc = None
        new_elts = []
        for e, d in elt_doc_pairs:
            if last_doc is not None and last_doc.doc_id != d.doc_id:
                last_doc.elements = new_elts
                new_elts = []
            new_elts.append(e)
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
