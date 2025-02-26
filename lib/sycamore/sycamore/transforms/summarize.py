import logging
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Literal, Union, Type
import copy
import textwrap


from sycamore.data import Element, Document
from sycamore.functions.tokenizer import Tokenizer, CharacterTokenizer
from sycamore.llms.prompts.default_prompts import (
    SummarizeDataMessagesPrompt,
    TextSummarizerJinjaPrompt,
)
from sycamore.llms.prompts.prompts import (
    JinjaElementPrompt,
    RenderedPrompt,
    RenderedMessage,
    SycamorePrompt,
    JinjaPrompt,
)
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


J_GET_ELEMENT_TEXT_MACRO = """
{#-
    get_text macro: returns text for an element. If this is the first
    round of summarization:
        If `fields` is provided to the template, add a list of key-value
        pairs to the text (if fields is the string "*", use all properties).
        Always include the text representation
    If this is after the first round of summarization:
        use only the element's summary field
-#}
{%- macro get_text(element, itvarname) %}
    {%- if elt.properties[itvarname] == 0 -%}
        {%- if fields is defined -%}
            {%- if fields == "*" %}{% for p in element.properties %}{% if p.startswith('_') %}{% continue %}{% endif %}
    {{ p }}: {{ element.properties[p] }}
            {%- endfor -%}
            {%- else %}{% for f in fields %}
    {{ f }}: {{ element.field_to_value(f) }}
            {%- endfor %}{% endif -%}
        {%- endif -%}
    Text: {{ element.text_representation }}
    {%- else -%}
    Summary: {{ element.properties[intermediate_summary_key] }}
    {% endif -%}
{% endmacro -%}
"""

MaxTokensHeirarchyPrompt = JinjaElementPrompt(
    system=textwrap.dedent(
        """
        {% if question is defined %}You are a helpful research assistant. You answer questions based on text you are presented with.
        {% else %}You are a helpful data summarizer. You concisely summarize text you are presented with, including as much detail as possible.
        {% endif %}
        """
    ),
    user=J_GET_ELEMENT_TEXT_MACRO
    + textwrap.dedent(
        """
        {%- macro get_data_description() -%}
            {%- if data_description is defined -%}
        {{ data_description }}
            {%- else -%}
        some documents
            {%- endif -%}
        {%- endmacro -%}

        {%- if elt.properties[skip_me_key] -%}{{ norender() }}{%- endif -%}
        {%- if batch_key in elt.properties and elt.properties[batch_key]|count == 1 and intermediate_summary_key in elt.properties -%}{{ norender() }}{%- endif -%}
        {%- if batch_key not in elt.properties -%}{{ norender() }}{%- endif -%}

        {% if elt.properties[round_key] == 0 -%}
        You are given {{ get_data_description() }}. Please use only the information found in these elements to determine an answer
        to the question "{{ question }}". If you cannot answer the question based on the data provided, instead respond with any data
        that might be relevant to the question.
        Elements:
        {% else %}
        You are given a list of partial answers to the question "{{ question }}" based on {{ get_data_description() }}.
        Please combine these partial answers into a coherent single answer to the question "{{ question }}". Some answers may not
        be particularly relevent, so don't pay them too much mind.
        Answers:
        {%- endif -%}
        {%- for idx in elt.properties[batch_key] %}
        {{ loop.index }}: {{ get_text(doc.elements[idx], round_key) }}
        {% endfor %}
        """
    ),
    question="What is the summary of this data?",
)


class MultiStepDocumentSummarizer(Summarizer):
    """
    Summarizes a document by constructing a tree of summaries. Each leaf contains as many consecutive
    elements as possible within the token limit, and each vertex of the tree contains as many sub-
    summaries as possible within the token limit. e.g with max_tokens=10
    Elements: (3 tokens) - (3 tokens) - (5 tokens) - (8 tokens)
                   |            |            |            |
                 (4 token summary) - (3 token summary) - (2 token summary)
                               \\             |            /
                                      (5 token summary)

    Args:
        llm: LLM to use for summarization
        question: Optional question to use as context for the summarization. If set, the llm will
            attempt to answer the question with the data provided
        prompt: Prompt to use for each summarization. Caution: The default (MaxTokensHeirarchicalSummarizerPrompt)
            has some fairly complicated logic encoded in it to make the tree construction work correctly.
        fields: List of fields to include in each element's representation in the prompt. Specify
            with dotted notation (e.g. properties.title), or use "*" to capture everything. If None,
            will include no fields.
        max_tokens: token limit for each summarization. Default is 10k (default tokenizer is by character).
        tokenizer: tokenizer to use when computing how many tokens a prompt will take. Default is
            CharacterTokenizer
        rounds: number of rounds of heirarchical summarization to perform. The number of elements that can be
            included in the summary is O(e^rounds), so rounds can be small. Default is 4.
    """

    def __init__(
        self,
        llm: LLM,
        question: Optional[str] = None,
        prompt: SycamorePrompt = MaxTokensHeirarchyPrompt,
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
            "intermediate_summary_key": "_summary",
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
            for j in range(i + 1, len(doc.elements)):
                e2 = doc.elements[j]
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
        vars = self.get_const_vars()
        doc.properties["summary"] = doc.elements[0].properties[vars["intermediate_summary_key"]]
        for e in doc.elements:
            for v in vars:
                if v in e.properties:
                    del e.properties[v]
        return doc

    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        vars = self.get_const_vars()
        if self.fields is not None:
            self.prompt = self.prompt.set(fields=self.fields)
        if self.question is not None:
            self.prompt = self.prompt.set(question=self.question)
        nodes = []
        last = child
        for round in range(self.rounds):
            prep_round = Map(child=last, f=self.prep_batches, kwargs={"round": round})
            llm_round = LLMMapElements(
                child=prep_round,
                prompt=self.prompt,
                output_field=vars["intermediate_summary_key"],
                llm=self.llm,
            )
            nodes.extend([prep_round, llm_round])
            last = llm_round
        cleanup = Map(child=last, f=self.cleanup)
        nodes.append(cleanup)
        ct = CompositeTransform(child, [])  # type: ignore
        ct.nodes = nodes
        return ct


class CollapseDocumentSummarizer(Summarizer):
    """
    Summarizes a document by converting it all to text, then iteratively summarizing chunks
    of the text + the existing summary to build up a full summary.

    Args:
        llm: LLM to use for summarization
        question: Question to use as context for the summarization. The llm will attempt to
            answer the question using the data in the document.
        chunk_size: Size of the chunks to add in each round of summarization
        tokenizer: Tokenizer to use to compute chunk sizes
        use_elements: If True, will include data from the elements of the document as well
            as the document itself. Default is False
        num_elements: Limit on the number of elements to include if use_elements is true (take
            the first num_elements elements). Default is 5
    """

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


OneStepSummarizerPrompt = JinjaPrompt(
    system="You are a helpful text summarizer",
    user=textwrap.dedent(
        """
        You are given a series of database entries that answer the question "{{ question }}".
        Generate a concise, conversational summary of the data to answer the question.
        {%- for elt in doc.elements %}
        Entry {{ loop.index }}:
            {% for f in doc.properties[fields_key] %}{#{% if f.startswith("_") %}{% continue %}{% endif %}#}
            {{ f }}: {{ elt.field_to_value(f) }}
            {% endfor -%}
            {%- if doc.properties[numel_key] is not none and doc.properties[numel_key] > 0 %}    Text:
            {% endif -%}
            {%- for subel in elt.data.get("elements", [])[:doc.properties[numel_key]] -%}
                {{ subel.text_representation }}
            {% endfor %}
        {% endfor %}
        """
    ),
)


class EtCetera:
    """Sentinel value to sit at the end of a list of fields, signifying 'add as
    many additional properties as you can within the token limit'"""


class OneStepDocumentSummarizer(Summarizer):
    """
    Summarizes a document in a single LLM call by taking as much data as possible
    from every element, spread across them evenly. Intended for use with summarize_data,
    where a summarizer is used to summarize an entire docset.

    Args:
        llm: LLM to use for summarization
        question: Question to use as context for the summary. The llm will attempt to
            use the data provided to answer the question.
        token_limit: Token limit for the prompt. Default is 10k (default tokenizer is
            by character)
        tokenizer: Tokenizer to use to count tokens (to not exceed the token limit).
            Default is CharacterTokenizer
        fields: List of fields to include from every element. To include any additional
            fields (after the ones specified), end the list with `EtCetera`. Default is
            empty list, which stands for 'as many fields as fit within the token limit'
            and is equivalent to `[EtCetera]`

    """

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
        self.prompt = OneStepSummarizerPrompt.set(**self.get_const_vars())

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
        all_element_property_names = {f"properties.{k}" for e in doc.elements for k in e.properties}
        doc.properties[vars["fields_key"]] = fields
        doc.properties[vars["numel_key"]] = 0
        last = self.prompt.render_document(doc)
        if len(fields) == 0 or etc:
            for p in all_element_property_names:
                if p in fields:
                    continue
                fields.append(p)
                last = self.prompt.render_document(doc)
                ntk = last.token_count(self.tokenizer)
                if ntk > self.token_limit:
                    fields.pop()
                    return doc
        doc.properties[vars["numel_key"]] += 1
        this = self.prompt.render_document(doc)
        while last != this:
            ntk = this.token_count(self.tokenizer)
            print(ntk)
            if ntk > self.token_limit:
                doc.properties[vars["numel_key"]] -= 1
                return doc
            last = this
            doc.properties[vars["numel_key"]] += 1
            this = self.prompt.render_document(doc)
        return doc

    def cleanup(self, doc: Document) -> Document:
        vars = self.get_const_vars()
        if vars["fields_key"] in doc.properties:
            del doc.properties[vars["fields_key"]]
        if vars["numel_key"] in doc.properties:
            del doc.properties[vars["numel_key"]]
        return doc

    def as_llm_map(self, child: Optional[Node], **kwargs):
        prompt = self.prompt
        if self.question is not None:
            prompt = prompt.set(question=self.question)
        preprocess = Map(child, f=self.preprocess)
        llm_map = LLMMap(preprocess, prompt=prompt, output_field="summary", llm=self.llm, **kwargs)
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
