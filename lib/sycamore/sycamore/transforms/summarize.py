from abc import ABC, abstractmethod
from typing import Callable, Optional, Literal, Union, Type
import copy
import textwrap


from sycamore.data import Element, Document
from sycamore.functions.tokenizer import Tokenizer, CharacterTokenizer
from sycamore.llms.prompts.default_prompts import (
    TextSummarizerJinjaPrompt,
)
from sycamore.llms.prompts.prompts import (
    JinjaElementPrompt,
    SycamorePrompt,
    JinjaPrompt,
    RenderedPrompt,
)
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Node
from sycamore.llms import LLM
from sycamore.llms.llms import LLMMode
from sycamore.transforms.map import Map, MapBatch
from sycamore.transforms.base import CompositeTransform, BaseMapTransform
from sycamore.transforms.base_llm import LLMMapElements, LLMMap, _infer_prompts


class Summarizer(ABC):
    def summarize(self, document: Document) -> Document:
        map = self.as_llm_map(None)
        assert isinstance(map, (BaseMapTransform, CompositeTransform))
        ds = map.local_execute([document], drop_metadata=True)
        assert len(ds) == 1, f"Found more than one Document after summmarizing just one: {ds}"
        return ds[0]

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

    def __init__(self, llm: LLM, element_filter: Optional[Callable[[Element], bool]] = None):
        self._llm = llm
        self._element_filter = element_filter

    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        filter = self._element_filter or (lambda e: True)
        return LLMMapElements(child, TextSummarizerJinjaPrompt, output_field="summary", llm=self._llm, filter=filter)


MaxTokensHierarchyPrompt = JinjaElementPrompt(
    system=textwrap.dedent(
        """
        {% if question is defined %}You are a helpful research assistant. You answer questions based on
        text you are presented with.
        {% else %}You are a helpful data summarizer. You concisely summarize text you are presented with,
        including as much detail as possible.
        {% endif %}
        """
    ),
    user=textwrap.dedent(
        """
        {%- macro get_text_properties(element) %}
            {% for p in element.properties %}
                {%- if p.startswith('_') %}{% continue %}{% endif %}
            {{ p }}: {{ element.properties[p] }}
            {%- endfor -%}
        {% endmacro -%}

        {%- macro get_text_fields(element, fields) %}
            {% for f in fields %}
            {{ f }}: {{ element.field_to_value(f) }}
            {%- endfor %}
        {% endmacro -%}

        {%- macro get_text_base(element) %}
            {%- if fields is defined -%}
                {%- if fields == "*" %}
            {{ get_text_properties(element) }}
                {%- else %}
            {{ get_text_fields(element, fields) }}
                {% endif -%}
            {%- endif -%}
            Text: {{ element.text_representation }}
        {% endmacro -%}

        {%- macro get_text(element) %}
                {%- if round == 0 -%}
            {{ get_text_base(element) }}
                {%- else -%}
            Summary: {{ element.properties[intermediate_summary_key] }}
                {% endif -%}
        {% endmacro -%}

        {%- macro get_data_description() -%}
            {%- if data_description is defined -%}
        {{ data_description }}
            {%- else -%}
        a set of documents with properties for each document
            {%- endif -%}
        {%- endmacro -%}

        {% if round == 0 -%}
        You are given {{ get_data_description() }}. Please use only the information found in these elements
        to determine an answer to the question "{{ question }}". If you cannot answer the question based on
        the data provided, instead respond with any data that might be relevant to the question.
        Elements:
        {% else %}
        You are given a list of partial answers to the question "{{ question }}" based on {{ get_data_description() }}.
        Please combine these partial answers into a coherent single answer to the question "{{ question }}".
        Some answers may not be particularly relevent, so don't pay them too much mind.
        Answers:
        {%- endif -%}
        {%- for idx in elt.properties[batch_key] %}
        {{ loop.index }}: {{ get_text(doc.elements[idx]) }}
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
        llm_mode: How to call the LLM - SYNC, ASYNC, BATCH. Async is faster but not all llms support it.
        question: Optional question to use as context for the summarization. If set, the llm will
            attempt to answer the question with the data provided
        data_description: Optional string describing the input documents.
        prompt: Prompt to use for each summarization. Caution: The default (MaxTokensHeirarchicalSummarizerPrompt)
            has some fairly complicated logic encoded in it to make the tree construction work correctly.
        fields: List of fields to include in each element's representation in the prompt. Specify
            with dotted notation (e.g. properties.title), or use "*" to capture everything. If None,
            will include no fields.
        max_tokens: token limit for each summarization. Default is 10k (default tokenizer is by character).
        tokenizer: tokenizer to use when computing how many tokens a prompt will take. Default is
            CharacterTokenizer
        max_rounds: max number of rounds of heirarchical summarization to perform. The number of elements that
            can be included in the summary is O(e^max_rounds), so can be small. Default is 4.
    """

    def __init__(
        self,
        llm: LLM,
        llm_mode: LLMMode = LLMMode.SYNC,
        question: Optional[str] = None,
        data_description: Optional[str] = None,
        prompt: SycamorePrompt = MaxTokensHierarchyPrompt,
        fields: Union[None, Literal["*"], list[str]] = None,
        max_tokens: int = 10 * 1000,
        tokenizer: Tokenizer = CharacterTokenizer(),
        max_rounds: int = 4,
    ):
        self.llm = llm
        self.llm_mode = llm_mode
        self.prompt = prompt.fork(**self.get_const_vars())
        self.fields = fields
        self.question = question
        self.data_description = data_description
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.max_rounds = max_rounds

    @staticmethod
    def get_const_vars() -> dict[str, str]:
        return {
            "batch_key": "_batch",
            "intermediate_summary_key": "_summary",
        }

    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        # MultiStepDocumentSummarizer doesn't use LLMMap - it doesn't work very cleanly
        return MapBatch(child, f=self.summarize_many)

    def summarize(self, document: Document) -> Document:
        return self.summarize_many([document])[0]

    def summarize_many(self, docs: list[Document]) -> list[Document]:
        vars = self.get_const_vars()
        prompts_per_doc = [-1] * len(docs)
        # Cache this bc we'll be updating the element list for each doc by dropping the ones that
        # don't get sub-summaries
        original_elements_per_doc = [doc.elements for doc in docs]
        for round in range(self.max_rounds):
            # Each round, render all the prompts for the round and then send them to the LLM
            if all(ppd == 1 for ppd in prompts_per_doc):
                break
            to_infer = []
            for i, doc in enumerate(docs):
                if prompts_per_doc[i] == 1:
                    continue
                prompts = self._doc_to_prompts(doc, round)
                doc.elements = [e for e, _ in prompts]
                prompts_per_doc[i] = len(prompts)
                to_infer.extend(prompts)
            results = _infer_prompts(prompts=[p for _, p in to_infer], llm=self.llm, llm_mode=self.llm_mode)
            for (e, _), r in zip(to_infer, results):
                e.properties[vars["intermediate_summary_key"]] = r
        # Re-attach elements, summaries, cleanup temp properties
        for doc, oelts in zip(docs, original_elements_per_doc):
            doc.properties["summary"] = doc.elements[0].properties[vars["intermediate_summary_key"]]
            doc.elements = oelts
            for e in doc.elements:
                for v in vars:
                    e.properties.pop(v, None)
        return docs

    def _doc_to_prompts(self, doc: Document, round: int) -> list[tuple[Element, RenderedPrompt]]:
        vars = self.get_const_vars()
        prompt = self.prompt.fork(
            ignore_none=True,
            question=self.question,
            data_description=self.data_description,
            fields=self.fields,
            round=round,
        )
        result = []
        curr_tks = 0
        curr_batch: list[int] = []
        for i, elt in enumerate(doc.elements):
            # For a SummarizeDocument, doc.elements is itertools.chain(subdoc.elements for each subdoc)
            # until we override it.
            elt.properties[vars["batch_key"]] = [i]
            etks = prompt.render_element(elt, doc).token_count(self.tokenizer)
            if curr_tks + etks > self.max_tokens:
                if etks > self.max_tokens:
                    raise ValueError(
                        "Element was too big to fit within the specified max tokens. "
                        "Please run `docset.split_elements` to break it up or limit the"
                        f" properties used in the prompt.\n\nElement: {elt}"
                    )
                first_elt = doc.elements[curr_batch[0]]
                first_elt.properties[vars["batch_key"]] = curr_batch
                result.append((first_elt, prompt.render_element(first_elt, doc)))
                curr_batch = [i]
                curr_tks = etks
            else:
                curr_tks += etks
                curr_batch.append(i)
                del elt.properties[vars["batch_key"]]

        first_elt = doc.elements[curr_batch[0]]
        first_elt.properties[vars["batch_key"]] = curr_batch
        result.append((first_elt, prompt.render_element(first_elt, doc)))
        return result


OneStepSummarizerPrompt = JinjaPrompt(
    system="You are a helpful text summarizer",
    user=textwrap.dedent(
        """
        You are given a series of database entries that answer the question "{{ question }}".
        Generate a concise, conversational summary of the data to answer the question.
        {%- for elt in doc.data.get("sub_docs", doc.elements) %}
        Entry {{ loop.index }}:
            {% for f in doc.properties[fields_key] %}{% if f.startswith("_") %}{% continue %}{% endif %}
            {{ f }}: {{ elt.field_to_value(f) }}
            {% endfor -%}
            {%- if doc.properties[numel_key] is not none and doc.properties[numel_key] > 0 %}    Text:
            {% endif -%}
            {%- set start = doc.properties[startel_key] -%}
            {%- set end = doc.properties[startel_key] + doc.properties[numel_key] -%}
            {%- for subel in elt.data.get("elements", [])[start:end] -%}
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
        self.prompt = OneStepSummarizerPrompt.fork(**self.get_const_vars())

    @staticmethod
    def get_const_vars() -> dict[str, str]:
        return {
            "fields_key": "_fields",
            "numel_key": "_num_elements",
            "startel_key": "_start_element",
        }

    def preprocess(self, doc: Document) -> Document:
        vars = self.get_const_vars()
        prompt = self.prompt.fork(ignore_none=True, question=self.question)
        fields = copy.deepcopy(self.fields)
        etc = False
        if len(fields) > 0 and fields[-1] is EtCetera:
            etc = True
            fields = fields[:-1]
        all_element_property_names = {f"properties.{k}" for e in doc.elements for k in e.properties}
        # Compute baseline 'fluff' tokens by setting fields and elements to 'no fields'
        # and 'no elements'. Use this later to figure out how many tokens adding a field adds
        doc.properties[vars["fields_key"]] = []
        doc.properties[vars["numel_key"]] = 0
        doc.properties[vars["startel_key"]] = 0
        data_independent_ntk = prompt.render_document(doc).token_count(self.tokenizer)
        # If fields is specified these are always included, so this will be our starting token total
        doc.properties[vars["fields_key"]] = fields
        curr_ntks = prompt.render_document(doc).token_count(self.tokenizer)
        if etc:
            for p in all_element_property_names:
                if p in fields:
                    continue
                doc.properties[vars["fields_key"]] = [p]
                ntk = prompt.render_document(doc).token_count(self.tokenizer) - data_independent_ntk
                if curr_ntks + ntk < self.token_limit:
                    fields.append(p)
                    curr_ntks += ntk
                else:
                    doc.properties[vars["fields_key"]] = fields
                    return doc
        # We added all the fields, now add as many elements as possible
        final_numel = 0
        doc.properties[vars["numel_key"]] = 1
        doc.properties[vars["fields_key"]] = []
        # This is complicated bc we might get a SummarizeDocument or a Document
        max_numel = max(len(d.data.get("elements", [])) for d in doc.data.get("sub_docs", doc.elements))
        # If elements can fit there's a little additional fluff added, so recompute baseline tokens
        # with no elements (but the element introduction fluff)
        doc.properties[vars["startel_key"]] = max_numel + 10
        data_independent_ntk_with_fluff = prompt.render_document(doc).token_count(self.tokenizer)
        curr_ntks += data_independent_ntk_with_fluff - data_independent_ntk
        for i in range(max_numel):
            doc.properties[vars["startel_key"]] = i
            ntk = prompt.render_document(doc).token_count(self.tokenizer) - data_independent_ntk_with_fluff
            if curr_ntks + ntk < self.token_limit:
                final_numel += 1
                curr_ntks += ntk
            else:
                break

        doc.properties[vars["numel_key"]] = final_numel
        doc.properties[vars["startel_key"]] = 0
        doc.properties[vars["fields_key"]] = fields
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
            prompt = prompt.fork(question=self.question)
        preprocess = Map(child, f=self.preprocess)
        llm_map = LLMMap(preprocess, prompt=prompt, output_field="summary", llm=self.llm, **kwargs)
        postprocess = Map(llm_map, f=self.cleanup)
        comptransform = CompositeTransform(child, nodes=[preprocess, llm_map, postprocess])  # type: ignore
        return comptransform


class Summarize(NonCPUUser, NonGPUUser, Map):
    """
    The summarize transform generates summaries of documents or elements.
    """

    def __init__(self, child: Node, summarizer: Summarizer, **kwargs):
        super().__init__(child, f=summarizer.summarize, **kwargs)
