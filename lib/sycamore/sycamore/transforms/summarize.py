import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Literal, Union


from sycamore.data import Element, Document
from sycamore.functions.tokenizer import Tokenizer
from sycamore.llms.prompts.default_prompts import (
    SummarizeBranchingFactorJinjaPrompt,
    SummarizeDataMessagesPrompt,
    TextSummarizerJinjaPrompt,
)
from sycamore.llms.prompts.prompts import JinjaElementPrompt, RenderedPrompt, RenderedMessage
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Node
from sycamore.llms import LLM
from sycamore.transforms.map import Map
from sycamore.transforms.base import CompositeTransform
from sycamore.transforms.base_llm import LLMMapElements

NUM_DOCS_GENERATE = 60
NUM_TEXT_CHARS_GENERATE = 2500
BASE_PROPS = [
    "filename",
    "filetype",
    "page_number",
    "page_numbers",
    "links",
    "element_id",
    "parent_id",
    "_schema",
    "_schema_class",
    "entity",
]


class Summarizer(ABC):
    def summarize(self, document: Document) -> Document:
        map = self.as_llm_map(None)
        assert hasattr(map, "_local_process")
        return map._local_process([document])[0]

    @abstractmethod
    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        pass


class LLMElementTextSummarizer(Summarizer):
    """
    LLMElementTextSummarizer uses a specified LLM to summarize text data within elements of a document.

    Args:
        llm: An instance of an LLM class to use for text summarization.
        element_operator: A callable function that operates on the document and returns a list of elements to be
            summarized. Default is None.

    Example:
         .. code-block:: python

            llm_model = OpenAILanguageModel("gpt-3.5-turbo")
            element_operator = my_element_selector  # A custom element selection function
            summarizer = LLMElementTextSummarizer(llm_model, element_operator)

            context = sycamore.init()
            pdf_docset = context.read.binary(paths, binary_format="pdf")
                .partition(partitioner=UnstructuredPdfPartitioner())
                .summarize(summarizer=summarizer)
    """

    def __init__(self, llm: LLM, element_operator: Optional[Callable[[Element], bool]] = None):
        self._llm = llm
        self._element_operator = element_operator

    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        if self._element_operator is not None:
            return LLMMapElements(
                child, TextSummarizerJinjaPrompt, output_field="summary", llm=self._llm, filter=self._element_operator
            )
        else:
            return LLMMapElements(child, TextSummarizerJinjaPrompt, output_field="summary", llm=self._llm)


class QuestionAnsweringSummarizer:
    def __init__(self, llm: LLM, question: str):
        self.llm = llm
        self.question = question

    def __call__(self, text: str) -> str:
        messages = SummarizeDataMessagesPrompt(question=self.question, text=text).as_messages()
        prompt = RenderedPrompt(messages=[RenderedMessage(role=m["role"], content=m["content"]) for m in messages])

        t0 = time.time()
        # call to LLM
        summary = self.llm.generate(prompt=prompt, llm_kwargs={"temperature": 0})
        t1 = time.time()
        logging.info(f"Summarizer took {t1 - t0} seconds to generate summary.")

        return summary


def collapse(text: str, tokens_per_chunk: int, tokenizer: Tokenizer, summarizer_fn: Callable[[str], str]) -> str:
    """
    Collapses text iteratively, summarizing the first chunk and incorporating it in the summary for the next chunk.

    Args:
        text: Text to collapse.
        chunk_size: Size of each chunk.
        tokenizer: Tokenizer to use for counting against max_tokens.

    Returns:
        List of chunks.
    """
    tokens = tokenizer.tokenize(text)
    total = len(tokens)
    if total <= tokens_per_chunk:
        return text
    done = False
    i = 0
    additional = i + tokens_per_chunk
    cur_summary = ""
    while not done:
        input = ""
        if cur_summary:
            input = f"{cur_summary}\n"
        input += "".join([str(tk) for tk in tokens[i : i + additional]])  # make mypy happy
        print(f"input size: {len(input)}")
        cur_summary = summarizer_fn(input)
        assert (
            len(cur_summary) <= tokens_per_chunk
        ), f"Summarizer output is longer than input chunk {len(cur_summary)} > {tokens_per_chunk} !!!"
        print(f"summary to chunk ratio: {len(cur_summary) / tokens_per_chunk}")
        i += additional
        remaining = tokens_per_chunk - len(cur_summary)
        additional = min(remaining, total - i)
        if additional == 0:
            break

    return cur_summary


class DocumentSummarizer(Summarizer):
    def __init__(
        self,
        llm: LLM,
        question: Optional[str] = None,
        prompt: Optional[JinjaElementPrompt] = None,
        fields: Union[None, Literal["*"], list[str]] = None,
        element_batch_size: Optional[int] = None,
    ):
        self.llm = llm
        self.question = question
        self.prompt = prompt or SummarizeBranchingFactorJinjaPrompt
        self.fields = fields
        self.element_batch_size = element_batch_size

    def get_const_vars(self) -> dict[str, str]:
        return {
            "iteration_var": "_summarize_round",
            "num_elements_key": "_num_total_elements",
            "index_key": "_summarizer_element_index",
            "intermediate_summary_key": "_partial_summary",
        }

    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        vars = self.get_const_vars()

        prompt = self.prompt.set(
            iteration_var=vars["iteration_var"],
            intermediate_summary_key=vars["intermediate_summary_key"],
            index_key=vars["index_key"],
        )
        if self.element_batch_size is not None:
            prompt = prompt.set(branching_factor=self.element_batch_size)
        if self.fields is not None:
            prompt = prompt.set(fields=self.fields)
        if self.question is not None:
            prompt = prompt.set(question=self.question)

        def preprocess(doc: Document) -> Document:
            for i, e in enumerate(doc.elements):
                e.properties[vars["num_elements_key"]] = len(doc.elements)
                e.properties[vars["index_key"]] = i
            return doc

        def validate(elt: Element) -> bool:
            iv = elt.properties[vars["iteration_var"]]
            num_elts = elt.properties[vars["num_elements_key"]]
            assert isinstance(prompt, JinjaElementPrompt)
            branching_factor = prompt.kwargs["branching_factor"]
            return branching_factor ** (iv + 1) > num_elts

        def cleanup(doc: Document) -> Document:
            if len(doc.elements) == 0:
                doc.properties["summary"] = ""
                return doc
            doc.properties["summary"] = doc.elements[0].properties[vars["intermediate_summary_key"]]
            for e in doc.elements:
                for k in vars:
                    if k in e.properties:
                        del e.properties[k]
            return doc

        premap = Map(child, f=preprocess)
        llm_map = LLMMapElements(
            child=premap,
            prompt=prompt,
            output_field=vars["intermediate_summary_key"],
            llm=self.llm,
            iteration_var=vars["iteration_var"],
            validate=validate,
            max_tries=20,  # If you hit this then your document has at least 2^20 elements in it.
        )
        postmap = Map(llm_map, f=cleanup)
        comptransform = CompositeTransform(child, [])  # type: ignore
        comptransform.nodes = [premap, llm_map, postmap]
        return comptransform


class Summarize(NonCPUUser, NonGPUUser, Map):
    """
    The summarize transform generates summaries of documents or elements.
    """

    def __init__(self, child: Node, summarizer: Summarizer, **kwargs):
        super().__init__(child, f=summarizer.summarize, **kwargs)
