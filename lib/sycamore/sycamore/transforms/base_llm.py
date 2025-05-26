import threading
from typing import Optional, Sequence, Callable, Union

from sycamore.llms.llms import LLM, LLMMode
from sycamore.llms.prompts.prompts import SycamorePrompt, RenderedPrompt
from sycamore.plan_nodes import Node
from sycamore.transforms.map import MapBatch
from sycamore.data import Document, Element
import asyncio


async def _infer_prompts_async(prompts: list[RenderedPrompt], llm: LLM) -> list[str]:
    el = asyncio.get_running_loop()
    awaitables = [llm.generate_async(prompt=p, llm_kwargs={}) for p in prompts]
    tasks = [el.create_task(aw) for aw in awaitables]
    return await asyncio.gather(*tasks)


def _run_new_thread(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _infer_prompts(
    prompts: list[RenderedPrompt],
    llm: LLM,
    llm_mode: LLMMode,
) -> list[str]:
    if llm_mode == LLMMode.SYNC:
        res = []
        for p in prompts:
            if len(p.messages) == 0:
                res.append("")
                continue
            s = llm.generate(prompt=p)
            res.append(s)
        return res
    elif llm_mode == LLMMode.ASYNC:
        nonempty = [(i, p) for i, p in enumerate(prompts) if len(p.messages) > 0]
        res = [""] * len(prompts)

        # Previously we would use asyncio.run here, but that causes issues in
        # environments like Jupyter notebooks where an event loop is already
        # running. To workaround this we create a separate event loop on a new
        # thread and run the tasks there.
        new_loop = asyncio.new_event_loop()
        t = threading.Thread(target=_run_new_thread, args=(new_loop,), daemon=True)
        t.start()

        fut = asyncio.run_coroutine_threadsafe(_infer_prompts_async([p for _, p in nonempty], llm), new_loop)

        responses = fut.result()

        new_loop.call_soon_threadsafe(new_loop.stop)
        t.join()
        new_loop.close()

        for (i, _), rs in zip(nonempty, responses):
            res[i] = rs
        return res
    elif llm_mode == LLMMode.BATCH:
        return llm.generate_batch(prompts=prompts)
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
        llm_mode: Optional[LLMMode] = None,
        iteration_var: Optional[str] = None,
        validate: Callable[[Document], bool] = lambda d: True,
        max_tries: int = 5,
        filter: Callable[[Document], bool] = lambda d: True,
        **kwargs,
    ):
        self._prompt = prompt
        self._validate_prompt()
        self._output_field = output_field
        self._llm = llm
        self._llm_mode = llm_mode if llm_mode is not None else llm.default_mode()
        self._iteration_var = iteration_var
        self._validate = validate
        self._max_tries = max_tries
        self._filter = filter
        super().__init__(child, f=self.llm_map, **kwargs)

    def llm_map(self, documents: list[Document]) -> list[Document]:
        if self._iteration_var is not None:
            for d in documents:
                d.properties[self._iteration_var] = 0

        skips = [not self._filter(d) for d in documents]
        tries = 0
        while not all(skips) and tries < self._max_tries:
            tries += 1
            rendered_and_index = [
                (self._prompt.render_document(d), i) for sk, d, i in zip(skips, documents, range(len(skips))) if not sk
            ]
            rendered = []
            for r, i in rendered_and_index:
                if len(r.messages) == 0:
                    skips[i] = True
                else:
                    rendered.append(r)
            if len(rendered) == 0:
                break
            results = _infer_prompts(rendered, self._llm, self._llm_mode)
            ri = 0
            for i in range(len(documents)):
                if skips[i]:
                    continue
                documents[i].properties[self._output_field] = results[ri]
                skips[i] = self._validate(documents[i])
                ri += 1
                if self._iteration_var is not None and not skips[i]:
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
        llm_mode: Optional[LLMMode] = None,
        iteration_var: Optional[str] = None,
        validate: Callable[[Element], bool] = lambda e: True,
        max_tries: int = 5,
        filter: Callable[[Element], bool] = lambda e: True,
        **kwargs,
    ):
        self._prompt = prompt
        self._validate_prompt()
        self._output_field = output_field
        self._llm = llm
        self._llm_mode = llm_mode if llm_mode is not None else llm.default_mode()
        self._iteration_var = iteration_var
        self._validate = validate
        self._max_tries = max_tries
        self._filter = filter
        super().__init__(child, f=self.llm_map_elements, **kwargs)

    def llm_map_elements(self, documents: list[Document]) -> list[Document]:
        elt_doc_pairs = [(e, d) for d in documents for e in d.elements]
        if self._iteration_var is not None:
            for e, _ in elt_doc_pairs:
                e.properties[self._iteration_var] = 0

        skips = [not self._filter(e) for e, _ in elt_doc_pairs]
        tries = 0
        while not all(skips) and tries < self._max_tries:
            tries += 1
            rendered_and_index = [
                (self._prompt.render_element(e, d), i)
                for sk, (e, d), i in zip(skips, elt_doc_pairs, range(len(skips)))
                if not sk
            ]
            rendered = []
            for r, i in rendered_and_index:
                if len(r.messages) == 0:
                    skips[i] = True
                else:
                    rendered.append(r)
            if len(rendered) == 0:
                break
            results = _infer_prompts(rendered, self._llm, self._llm_mode)
            ri = 0
            for i in range(len(elt_doc_pairs)):
                if skips[i]:
                    continue
                elt, doc = elt_doc_pairs[i]
                elt.properties[self._output_field] = results[ri]
                skips[i] = self._validate(elt)
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
