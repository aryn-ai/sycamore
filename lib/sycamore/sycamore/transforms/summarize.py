import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Literal, Union, Type
import copy


from sycamore.data import Element, Document
from sycamore.functions.tokenizer import Tokenizer, CharacterTokenizer
from sycamore.llms.prompts.default_prompts import (
    MaxTokensHeirarchicalSummarizerPrompt,
    SummarizeBranchingFactorJinjaPrompt,
    SummarizeDataMessagesPrompt,
    TextSummarizerJinjaPrompt,
    RoundRobinSummarizerPrompt,
)
from sycamore.llms.prompts.prompts import JinjaElementPrompt, RenderedPrompt, RenderedMessage, SycamorePrompt
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Node
from sycamore.llms import LLM
from sycamore.transforms.map import Map
from sycamore.transforms.base import CompositeTransform
from sycamore.transforms.base_llm import LLMMapElements, LLMMap

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


class HeirarchicalDocumentSummarizer(Summarizer):
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
            found_summary = False
            for e in doc.elements:
                if not found_summary and vars["intermediate_summary_key"] in e.properties:
                    doc.properties["summary"] = doc.elements[0].properties[vars["intermediate_summary_key"]]
                    found_summary = True
                for k in vars:
                    if k in e.properties:
                        del e.properties[k]
            if not found_summary:
                doc.properties["summary"] = ""
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


class MaxTokensHeirarchicalDocumentSummarizer(Summarizer):
    def __init__(
        self,
        llm: LLM,
        question: str,
        prompt: SycamorePrompt = MaxTokensHeirarchicalSummarizerPrompt,
        fields: Union[None, Literal["*"], list[str]] = None,
        max_tokens: int = 10 * 1000,
        tokenizer: Tokenizer = CharacterTokenizer(),
        rounds: int = 4,
    ):
        self.llm = llm
        self.prompt = prompt.set(**self.get_const_vars())
        self.fields = fields
        self.question = question
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.rounds = 4

    @staticmethod
    def get_const_vars() -> dict[str, str]:
        return {
            "skip_me_key": "_skip_me",
            "batch_key": "_batch",
            "round_key": "_round",
        }

    def prep_batches(self, doc: Document, round: int = 0) -> Document:
        vars = self.get_const_vars()
        for i, elt in enumerate(doc.elements):
            elt.properties[vars["round_key"]] = round
            if vars["skip_me_key"] not in elt.properties:
                elt.properties[vars["skip_me_key"]] = False
            if elt.properties[vars["skip_me_key"]]:
                continue
            this_batch = [i]
            elt.properties[vars["batch_key"]] = this_batch
            for j, e2 in doc.elements[i + 1 :]:
                if e2.properties.get(vars["skip_me_key"], False):
                    continue
                this_batch.append(j)
                tks = self.prompt.render_element(elt, doc).token_count(self.tokenizer)
                if tks > self.max_tokens:
                    this_batch.pop()
                    break
                e2.properties[vars["skip_me_key"]] = True
        return doc

    def cleanup(self, doc: Document) -> Document:
        if len(doc.elements) == 0:
            return doc
        doc.properties["summary"] = doc.elements[0].properties["summary"]
        vars = self.get_const_vars()
        for e in doc.elements:
            for v in vars:
                if v in e.properties:
                    del e.properties[v]
        return doc

    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        if self.fields is not None:
            self.prompt = self.prompt.set(fields=self.fields)
        if self.question is not None:
            self.prompt = self.prompt.set(question=self.question)
        nodes = []
        last = child
        for round in range(self.rounds):
            prep_round = Map(child=last, f=self.prep_batches, f_kwargs={"round": round})
            llm_round = LLMMapElements(child=prep_round, prompt=self.prompt, output_field="summary", llm=self.llm)
            nodes.extend([prep_round, llm_round])
            last = llm_round
        cleanup = Map(child=last, f=self.cleanup)
        nodes.append(cleanup)
        ct = CompositeTransform(child, [])  # type: ignore
        ct.nodes = nodes
        return ct


class CollapseDocumentSummarizer(Summarizer):
    def __init__(
        self,
        llm: LLM,
        question: str,
        chunk_size: int = 10 * 1000,
        tokenizer: Tokenizer = CharacterTokenizer(),
        chunk_overlap: int = 0,
        use_elements: bool = False,
        num_elements: int = 5,
    ):
        self.llm = llm
        self.question = question
        self.chunk_size = chunk_size
        self.tokenizer = tokenizer
        self.chunk_overlap = chunk_overlap
        self.use_elements = use_elements
        self.num_elements = num_elements

    def as_llm_map(self, child: Optional[Node], **kwargs):
        return Map(child, f=self.summarize)  # type: ignore

    def summarize(self, document: Document) -> Document:
        text = self.get_text(document)
        summary = collapse(text, self.chunk_size, self.tokenizer, QuestionAnsweringSummarizer(self.llm, self.question))
        document.properties["summary"] = summary
        return document

    def get_text(self, doc: Document) -> str:
        doc_text = ""
        props_dict = doc.properties.get("entity", {})
        props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})
        for k, v in props_dict.items():
            doc_text += f"{k}: {v}\n"

        doc_text_representation = ""
        if not self.use_elements:
            if doc.text_representation is not None:
                doc_text_representation += doc.text_representation[:NUM_TEXT_CHARS_GENERATE]
        else:
            for element in doc.elements[: self.num_elements]:
                # Greedy fill doc level text length
                if len(doc_text_representation) >= NUM_TEXT_CHARS_GENERATE:
                    break
                doc_text_representation += (element.text_representation or "") + "\n"
        doc_text += f"Text contents:\n{doc_text_representation}\n"

        return doc_text


class EtCetera:
    """Sentinel value to sit at the end of a list of fields, signifying 'add as
    many additional properties as you can within the token limit'"""


class RoundRobinOneshotDocumentSummarizer(Summarizer):
    def __init__(
        self,
        llm: LLM,
        question: str,
        token_limit: int = 10 * 1000,
        tokenizer: Tokenizer = CharacterTokenizer(),
        fields: list[Union[str, Type[EtCetera]]] = [],
    ):
        self.llm = llm
        self.question = question
        self.token_limit = token_limit
        self.tokenizer = tokenizer
        assert EtCetera not in fields[:-1], "EtCetera must be at the end of the list of fields if provided"
        self.fields = fields
        self.prompt = RoundRobinSummarizerPrompt.set(**self.get_const_vars())

    @staticmethod
    def get_const_vars() -> dict[str, str]:
        return {
            "fields_key": "_fields",
            "numel_key": "_num_elements",
        }

    def preprocess(self, doc: Document) -> Document:
        vars = self.get_const_vars()
        fields = copy.deepcopy(self.fields)
        etc = False
        if len(fields) > 0 and fields[-1] is EtCetera:
            etc = True
            fields = fields[:-1]
        doc.properties[vars["fields_key"]] = fields
        doc.properties[vars["numel_key"]] = 0
        last = self.prompt.render_document(doc)
        if len(fields) == 0 or etc:
            for p in doc.properties:
                if p in fields:
                    continue
                fields.append(p)
                last = self.prompt.render_document(doc)
                ntk = last.token_count(self.tokenizer)
                if ntk > self.token_limit:
                    fields.pop()
                    return doc
        doc.properties[vars["numel_key"]] += 1
        while last != (this := self.prompt.render_document(doc)):
            ntk = this.token_count(self.tokenizer)
            if ntk > self.token_limit:
                doc.properties[vars["numel_key"]] -= 1
                return doc
            last = this
        return doc

    def cleanup(self, doc: Document) -> Document:
        vars = self.get_const_vars()
        if vars["fields_key"] in doc.properties:
            del doc.properties[vars["fields_key"]]
        if vars["numel_key"] in doc.properties:
            del doc.properties[vars["numel_key"]]
        return doc

    def as_llm_map(self, child: Optional[Node], **kwargs):
        preprocess = Map(child, f=self.preprocess)
        llm_map = LLMMap(preprocess, prompt=self.prompt, output_field="summary", llm=self.llm, **kwargs)
        postprocess = Map(llm_map, f=self.cleanup)
        comptransform = CompositeTransform(child, [])  # type: ignore
        comptransform.nodes = [preprocess, llm_map, postprocess]
        return comptransform


class Summarize(NonCPUUser, NonGPUUser, Map):
    """
    The summarize transform generates summaries of documents or elements.
    """

    def __init__(self, child: Node, summarizer: Summarizer, **kwargs):
        super().__init__(child, f=summarizer.summarize, **kwargs)
