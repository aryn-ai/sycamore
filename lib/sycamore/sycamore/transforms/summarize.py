from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, Type
import copy
import textwrap
import itertools
import logging


from sycamore.data import Element, Document
from sycamore.functions.tokenizer import Tokenizer, CharacterTokenizer
from sycamore.llms.prompts.default_prompts import (
    TextSummarizerJinjaPrompt,
)
from sycamore.llms.prompts.prompts import (
    SycamorePrompt,
    JinjaPrompt,
)
from sycamore.plan_nodes import NonCPUUser, NonGPUUser, Node
from sycamore.llms import LLM
from sycamore.llms.llms import LLMMode
from sycamore.transforms.map import Map
from sycamore.transforms.base import CompositeTransform, BaseMapTransform
from sycamore.transforms.base_llm import LLMMapElements, LLMMap, _infer_prompts


class SummaryDocument(Document):
    def __init__(self, document=None, **kwargs):
        if "elements" in kwargs:
            raise ValueError("Cannot set elements directly in a SummarizeDocument")
        super().__init__(document, **kwargs)
        if self.data.get("sub_docs") is None:
            self.data["sub_docs"] = []
        elif not isinstance(sd := self.data["sub_docs"], list):
            raise ValueError(f"sub_docs must be a list of Document, found {sd}")
        else:
            subdocs = self.data["sub_docs"]
            for sd in subdocs:
                if not isinstance(sd, Document):
                    raise ValueError(f"sub_docs must be a list of Documents. Found nonmatching {sd}")
            self.data["sub_docs"] = [Document(sd) for sd in subdocs]

    @property
    def sub_docs(self) -> list[Document]:
        return self.data["sub_docs"]

    @sub_docs.setter
    def sub_docs(self, sub_docs: list[Document]):
        self.data["sub_docs"] = sub_docs

    @sub_docs.deleter
    def sub_docs(self) -> None:
        self.data["sub_docs"] = []

    @property
    def elements(self) -> list[Element]:
        """A list of elements belonging to this document. A document does not necessarily always have
        elements, for instance, before a document is chunked."""
        return self.data.get(
            "_elements",
            list(itertools.chain(*(d.elements for d in self.data["sub_docs"]))),
        )

    @elements.setter
    def elements(self, elements: list[Element]):
        """Set the elements for this document."""
        self.data["_elements"] = elements

    @elements.deleter
    def elements(self) -> None:
        """Delete the elements of this document."""
        self.data.pop("_elements", None)


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
        return LLMMapElements(
            child,
            TextSummarizerJinjaPrompt,
            output_field="summary",
            llm=self._llm,
            filter=filter,
        )


class EtCetera:
    """Sentinel value to sit at the end of a list of fields, signifying 'add as
    many additional properties as you can within the token limit'"""


def _partition_fields(document: Document, fields: list[Union[str, Type[EtCetera]]]) -> tuple[list[str], list[str]]:
    """
    Split a list of fields into document and element fields - any fields in the list that
    are in the document properties are document fields, and everything else is an element field.
    EtCetera turns the list into 'every field' (with a prefix if early field order matters)
    """
    # TODO: If property values are varied between document and elements we might
    # not want to drop them from the elements.
    doc_fields: list[str] = []
    elt_fields: list[str] = []
    if len(fields) == 0:
        return doc_fields, elt_fields
    for f in fields:
        if f is EtCetera:
            continue
        assert not isinstance(f, type)
        property_name = f[len("properties.") :]
        if property_name in document.properties:
            assert isinstance(f, str), "mypy thinks f could be EtCetera"
            doc_fields.append(f)
        else:
            assert isinstance(f, str), "mypy thinks f could be EtCetera"
            elt_fields.append(f)
    fieldset = set(fields)
    if fields[-1] is EtCetera:
        for f in document.properties:
            if f"properties.{f}" not in fieldset:
                doc_fields.append(f"properties.{f}")
        docfieldset = set(doc_fields) | fieldset
        eltfieldset = {f"properties.{k}" for e in document.elements for k in e.properties if k not in docfieldset}
        elt_fields.extend(list(eltfieldset))
    return doc_fields, elt_fields


MaxTokensHierarchyPrompt = JinjaPrompt(
    system=textwrap.dedent(
        """
        {%- if element_testing is not defined -%}{# element_testing means only render an element, to get token count #}
        {% if question is defined %}You are a helpful research assistant. You answer questions based on
        text you are presented with.
        {% else %}You are a helpful data summarizer. You concisely summarize text you are presented with,
        including as much detail as possible.
        {% endif %}{% endif %}
        """
    ),
    user=textwrap.dedent(
        """
        {%- macro get_text_fields(element, fields) %}
            {% for f in fields %}
            {{ f }}: {{ element.field_to_value(f) }}
            {%- endfor %}
        {% endmacro -%}

        {%- macro get_text_base(element) %}
            {{ get_text_fields(element, elt_fields) }}
            Text: {{ element.text_representation }}
        {% endmacro -%}

        {%- macro get_text(element) %}
                {%- if round == 0 -%}
            {{ get_text_base(element) }}
                {%- else -%}
            {{ element.properties["summary"] }}
                {% endif -%}
        {% endmacro -%}

        {%- macro get_data_description() -%}
            {%- if data_description is defined -%}
        {{ data_description }}
            {%- else -%}
        a set of documents with properties for each document
            {%- endif -%}
        {%- endmacro -%}

        {%- if element_testing is not defined -%}{# element_testing means only render an element, to get token count #}
        {% if round == 0 -%}
        You are given {{ get_data_description() }}. Please use only the information found in these elements
        to determine an answer to the question "{{ question }}". If you cannot answer the question based on
        the data provided, instead respond with any data that might be relevant to the question.
        {% else %}
        You are given a list of partial answers to the question "{{ question }}" based on {{ get_data_description() }}.
        Please combine these partial answers into a coherent single answer to the question "{{ question }}".
        Include the parts of the partial answers that are relevant, ignore irrelevant parts.
        {%- endif %}

        {% if doc_fields|count > 0 -%}
        Shared Properties:
        {{ get_text_fields(doc, doc_fields) }}
        {%- endif %}

        {% if round == 0 -%}
        Elements:
        {% elif round > 0 -%}
        Answers:
        {% endif %}
        {%- endif -%}{# end of element_testing check. Stuff inside this block was constant across elements #}

        {%- for e in doc.elements %}
        {{ loop.index }}: {{ get_text(e) }}
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

    .. code-block::

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
            with dotted notation (e.g. properties.title). End the list with `EtCetera` to add all fields
            (previously specified fields go first). Default is [] which includes no fields.
        tokenizer: tokenizer to use when computing how many tokens a prompt will take. Default is
            CharacterTokenizer
    """

    def __init__(
        self,
        llm: LLM,
        llm_mode: Optional[LLMMode] = None,
        question: Optional[str] = None,
        data_description: Optional[str] = None,
        prompt: SycamorePrompt = MaxTokensHierarchyPrompt,
        fields: list[Union[str, Type[EtCetera]]] = [],
        tokenizer: Tokenizer = CharacterTokenizer(),
    ):
        self.llm = llm
        self.llm_mode = llm_mode if llm_mode is not None else llm.default_mode()
        self.prompt = prompt
        assert EtCetera not in fields[:-1], "EtCetera must be at the end of the list of fields if provided"
        self.fields = fields
        self.question = question
        self.data_description = data_description
        self.max_tokens = tokenizer.max_tokens or 10_000
        self.tokenizer = tokenizer

    def as_llm_map(self, child: Optional[Node], **kwargs) -> Node:
        # MultiStepDocumentSummarizer doesn't use LLMMap - it doesn't work very cleanly
        return Map(child, f=self.summarize)

    def summarize(self, document: Document) -> Document:
        """Summarize a document by summarizing groups of elements iteratively
        in rounds until only one element remains; that's our new summary"""
        doc_fields, elt_fields = _partition_fields(document, self.fields)
        base_prompt = self.prompt.fork(
            ignore_none=True,
            doc_fields=doc_fields,
            elt_fields=elt_fields,
            question=self.question,
            data_description=self.data_description,
        )
        etk_prompt = base_prompt.fork(element_testing=True)

        dummy_doc = document.copy()
        remaining_elements = dummy_doc.elements

        round = 0
        last_elt_len = len(remaining_elements)
        while len(remaining_elements) > 1 or round == 0:
            round_prompt = base_prompt.fork(round=round)
            round_etk_prompt = etk_prompt.fork(round=round)
            remaining_elements = self.summarize_one_round(dummy_doc, remaining_elements, round_prompt, round_etk_prompt)
            if len(remaining_elements) == last_elt_len and round > 0:
                logging.warning("Detected likely infinite summary loop. Exiting with incomplete summary")
                break
            last_elt_len = len(remaining_elements)
            round += 1
        if remaining_elements:
            document.properties["summary"] = remaining_elements[0].properties["summary"]
        else:
            document.properties["summary"] = "Empty Summary Document, nothing to summarize"
        for e in document.elements:
            e.properties.pop("summary", None)
        return document

    def summarize_one_round(
        self,
        document: Document,
        elements: list[Element],
        base_prompt: SycamorePrompt,
        etk_prompt: SycamorePrompt,
    ) -> list[Element]:
        """Perform a 'round' of element summarization: Assemble batches of maximal amounts
        of elements and summarize them, attaching the resulting summaries to the first
        element of each batch and returning only those elements."""
        # Compute token costs for the base stuff and each element individually
        document.elements = []
        baseline_tks = base_prompt.render_document(document).token_count(self.tokenizer)

        # Batch elements and make prompts out of them
        elt_batches = self.batch_elements(baseline_tks, elements, etk_prompt, document)
        final_elements = []
        to_infer = []
        for eb in elt_batches:
            if eb:
                document.elements = eb
                final_elements.append(eb[0])
                to_infer.append(base_prompt.render_document(document))

        # Invoke the llm and attach summaries
        summaries = _infer_prompts(prompts=to_infer, llm=self.llm, llm_mode=self.llm_mode)
        for e, s in zip(final_elements, summaries):
            e.properties["summary"] = s
        return final_elements

    def batch_elements(
        self,
        baseline_tokens: int,
        elements: list[Element],
        etk_prompt: SycamorePrompt,
        document: Document,
    ) -> list[list[Element]]:
        """Return a list of lengths of consecutive batches of elements keeping total
        token counts below my token limit"""
        limit = self.max_tokens
        result = []
        curr_tks = baseline_tokens
        curr_batch: list[Element] = []
        for e in elements:
            document.elements = [e]
            etks = etk_prompt.render_document(document).token_count(self.tokenizer)
            if etks + curr_tks > limit:
                if etks + baseline_tokens > limit:
                    raise ValueError(
                        "An element was too big to fit within the specified max tokens. "
                        "Please run `docset.split_elements` to break it up or limit the"
                        f" properties used in the prompt.\n\nElement: {e}"
                    )
                result.append(curr_batch)
                curr_batch = [e]
                curr_tks = baseline_tokens + etks
            else:
                curr_batch.append(e)
                curr_tks += etks
        result.append(curr_batch)
        return result


OneStepSummarizerPrompt = JinjaPrompt(
    system="You are a helpful text summarizer",
    user=textwrap.dedent(
        """
        You are given a series of database entries that answer the question "{{ question }}".
        Generate a concise, conversational summary of the data to answer the question.
        {%- for subdoc in doc.data.get("sub_docs", [doc]) %}
        Entry {{ loop.index }}:
            {% for f in doc.properties[doc_fields_key] %}{% if f.startswith("_") %}{% continue %}{% endif %}
            {{ f }}: {{ subdoc.field_to_value(f) }}
            {% endfor -%}
            {%- if doc.properties[numel_key] is not none and doc.properties[numel_key] > 0 %}
            Elements:
                {%- set start = doc.properties[startel_key] -%}
                {%- set end = doc.properties[startel_key] + doc.properties[numel_key] -%}
                {%- for subel in subdoc.elements[start:end] -%}
                {#- Removed {loop.index} from here because it blows up the token count. For an element token count, the index is 0 but when we count the tokens for all the elements included, it becomes like (0,1,2...) which results in a different tokenization from how we tokenize 1 element at a time. -#}
                    {%- for f in doc.properties[elt_fields_key] %}
                    {{ f }}: {{ subel.field_to_value(f) }}
                    {%- endfor %}
                    Text: {{ subel.text_representation }}
                {% endfor %}
            {% endif -%}
        {% endfor %}
        """
    ),
)


class OneStepDocumentSummarizer(Summarizer):
    """
    Summarizes a document in a single LLM call by taking as much data as possible
    from every element, spread across them evenly. Intended for use with summarize_data,
    where a summarizer is used to summarize an entire docset.

    Args:
        llm: LLM to use for summarization
        question: Question to use as context for the summary. The llm will attempt to
            use the data provided to answer the question.
        tokenizer: Tokenizer to use to count tokens (to not exceed the token limit).
            Default is CharacterTokenizer
        fields: List of fields to include from every element. To include any additional
            fields (after the ones specified), end the list with `EtCetera`. Default is
            empty list, which stands for 'no properties'

    """

    def __init__(
        self,
        llm: LLM,
        question: str,
        tokenizer: Tokenizer = CharacterTokenizer(),
        fields: list[Union[str, Type[EtCetera]]] = [],
    ):
        self.llm = llm
        self.question = question
        self.token_limit = tokenizer.max_tokens or 10_000
        self.tokenizer = tokenizer
        assert EtCetera not in fields[:-1], "EtCetera must be at the end of the list of fields if provided"
        self.fields = fields
        self.prompt = OneStepSummarizerPrompt.fork(**self.get_const_vars())

    @staticmethod
    def get_const_vars() -> dict[str, str]:
        return {
            "doc_fields_key": "_doc_fields",
            "elt_fields_key": "_elt_fields",
            "numel_key": "_num_elements",
            "startel_key": "_start_element",
        }

    def _maximize_fields(
        self,
        doc: Document,
        data_independent_ntk: int,
        curr_ntk: int,
        partitioned_fields: list[str],
        initial_fieldset: set[Union[str, Type[EtCetera]]],
        field_key: str,
        prompt: SycamorePrompt,
    ) -> tuple[bool, int, list[str]]:
        """Stuff as many fields into the plan as can fit in the token limit.

        Args:
            doc: The document to operate on
            data_independent_ntk: How many tokens are in the prompt regardless of the data
            curr_ntk: Current token count before adding stuff
            partitioned_fields: list of fields from _partition_fields - either the element or document fields
            initial_fieldset: the set of fields specified by the user
            field_key: either "doc_fields_key" or "elt_fields_key", depending on whether we're adding doc or elt fields
            prompt: the sycamore prompt to use to render and count tokens

        Returns:
            (bool, int, list[str]): Whether we filled up the token limit, the total tokens after adding fields,
                the finalized list of fields to add
        """
        vars = self.get_const_vars()
        final_fields = [f for f in partitioned_fields if f in initial_fieldset]
        for f in partitioned_fields:
            if f in initial_fieldset:
                continue
            doc.properties[vars[field_key]] = [f]
            ntk = prompt.render_document(doc).token_count(self.tokenizer) - data_independent_ntk
            if curr_ntk + ntk < self.token_limit:
                final_fields.append(f)
                curr_ntk += ntk
            else:
                doc.properties[vars[field_key]] = final_fields
                return True, curr_ntk, final_fields
        return False, curr_ntk, final_fields

    def maximize_elements(
        self,
        doc: Document,
        data_independent_ntk: int,
        curr_ntk: int,
        prompt: SycamorePrompt,
    ) -> tuple[bool, int, int]:
        """Stuff as many elements as possible into the prompt.

        Args:
            doc: The document to operate on
            data_independent_ntk: How many tokens are in the prompt regardless of data
            curr_ntk: Current token count before adding elements
            prompt: the sycamore prompt to use to render and count tokens

        Returns:
            (bool, int, int): Whether we filled up the token limit, the total tokens after adding fields,
                the number of elements to use
        """
        vars = self.get_const_vars()
        # This is complicated bc we might get a SummarizeDocument or a Document
        max_numel = (
            max(len(d.data.get("elements", [])) for d in doc.data.get("sub_docs", doc.elements))
            if doc.data.get("sub_docs")
            else 0
        )
        # If elements can fit there's a little additional fluff added, so recompute baseline tokens
        # with no elements (but the element introduction fluff)
        doc.properties[vars["numel_key"]] = 1
        doc.properties[vars["startel_key"]] = max_numel + 1
        data_independent_ntk_with_fluff = prompt.render_document(doc).token_count(self.tokenizer)
        curr_ntk += data_independent_ntk_with_fluff - data_independent_ntk

        final_numel = 0
        for i in range(max_numel):
            doc.properties[vars["startel_key"]] = i
            ntk = prompt.render_document(doc).token_count(self.tokenizer) - data_independent_ntk_with_fluff
            if curr_ntk + ntk < self.token_limit:
                final_numel += 1
                curr_ntk += ntk
            else:
                return True, curr_ntk, final_numel
        return False, curr_ntk, final_numel

    def preprocess(self, doc: Document) -> Document:
        """Compute which fields and how many elements to include in the prompt.

        First: If specified fields has an EtCetera, add as many fields as possible.
        Second: Add as many elements as possible, taking evenly from each document.
        Third: If we can add all the elements and specified fields has an EtCetera,
            add as many element fielse as possible
        """
        vars = self.get_const_vars()
        prompt = self.prompt.fork(ignore_none=True, question=self.question)
        fields = copy.deepcopy(self.fields)
        if isinstance(doc, SummaryDocument):
            doc.properties = {k: True for d in doc.sub_docs for k in d.properties.keys()}
        doc_fields, elt_fields = _partition_fields(doc, fields)
        fieldset = {f for f in fields if f is not EtCetera}
        etc = len(fields) > 0 and fields[-1] is EtCetera

        # Compute baseline 'fluff' tokens by setting fields and elements to 'no fields'
        # and 'no elements'. Use this later to figure out how many tokens adding a field adds
        doc.properties[vars["doc_fields_key"]] = []
        doc.properties[vars["elt_fields_key"]] = []
        doc.properties[vars["numel_key"]] = 0
        doc.properties[vars["startel_key"]] = 0
        data_independent_ntk = prompt.render_document(doc).token_count(self.tokenizer)
        # If fields is specified these are always included, so this will be our starting token total
        final_docfields = [f for f in doc_fields if f in fieldset]
        doc.properties[vars["doc_fields_key"]] = final_docfields
        curr_ntks = prompt.render_document(doc).token_count(self.tokenizer)
        finished = False
        if etc:
            finished, curr_ntks, final_docfields = self._maximize_fields(
                doc,
                data_independent_ntk,
                curr_ntks,
                doc_fields,
                fieldset,
                "doc_fields_key",
                prompt,
            )

        # We added all the fields, now add as many elements as possible
        final_eltfields = [f for f in elt_fields if f in fieldset]
        if not finished:
            doc.properties[vars["doc_fields_key"]] = []
            doc.properties[vars["elt_fields_key"]] = final_eltfields

            finished, curr_ntks, final_numel = self.maximize_elements(doc, data_independent_ntk, curr_ntks, prompt)
            doc.properties[vars["numel_key"]] = final_numel
            doc.properties[vars["startel_key"]] = 0

        # If we're supposed to add as many fields as possible and we still have room,
        # try adding element fields until we run out of space. This feels computationally
        # expensive but I think it's just a 'for each element and for each field' which
        # seems like the optimum for the intended behavior.
        if etc and not finished:
            doc.properties[vars["elt_fields_key"]] = []
            total_ntk_with_no_fields = prompt.render_document(doc).token_count(self.tokenizer)
            finished, curr_ntks, final_eltfields = self._maximize_fields(
                doc,
                total_ntk_with_no_fields,
                curr_ntks,
                elt_fields,
                fieldset,
                "elt_fields_key",
                prompt,
            )

        doc.properties[vars["doc_fields_key"]] = final_docfields
        doc.properties[vars["elt_fields_key"]] = final_eltfields
        return doc

    def cleanup(self, doc: Document) -> Document:
        vars = self.get_const_vars()
        for v in vars:
            doc.properties.pop(vars[v], None)
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
