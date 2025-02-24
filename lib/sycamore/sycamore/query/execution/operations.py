import math
from typing import Any, List, Union, Optional

import structlog

from sycamore import DocSet
from sycamore import ExecMode
from sycamore.context import context_params, Context
from sycamore.data import MetadataDocument, Document, Element
from sycamore.functions import CharacterTokenizer, Tokenizer
from sycamore.llms.llms import LLM
from sycamore.llms.prompts import RenderedPrompt, RenderedMessage
from sycamore.llms.prompts.default_prompts import (
    SummarizeDataHeirarchicalPrompt,
    SummarizeDataMessagesPrompt,
)
from sycamore.transforms.summarize import (
    HeirarchicalDocumentSummarizer,
    collapse,
    QuestionAnsweringSummarizer,
)

log = structlog.get_logger(__name__)
DEFAULT_DOCSET_SUMMARIZER = HeirarchicalDocumentSummarizer


def math_operation(val1: int, val2: int, operator: str) -> Union[int, float]:
    """
    Basic arithmetic operations on integers.

    Args:
        val1: First integer in operation.
        val2: Second integer in operation.
        operator: Type of operation; "add", "subtract", "divide", or "multiply"

    Returns:
        An integer or floating point number.
    """
    if operator == "add":
        return val1 + val2
    elif operator == "subtract":
        return val1 - val2
    elif operator == "divide":
        try:
            return val1 / val2
        except ZeroDivisionError:
            return math.nan
    elif operator == "multiply":
        return val1 * val2
    else:
        raise ValueError("Invalid math operator " + operator)


@context_params
def summarize_data(
    llm: LLM,
    question: Optional[str],
    result_description: str,
    result_data: List[Any],
    summaries_as_text: bool = False,
    context: Optional[Context] = None,
    **kwargs,
) -> str:
    """
    Provides an English response to a question given relevant information. Uses a default maximum for 120k characters,
    that should loosely translate to 30k tokens (1 token ~= 4 chars).

    Args:
        llm: LLM to use for summarization.
        question: Question to answer.
        result_description: Description of each of the inputs in result_data.
        result_data: List of inputs.
        use_elements: Use text contents from document.elements instead of document.text_representation.
        num_elements: Number of elements whose text to use from each document.
        max_tokens: Maximum number of tokens allowed in the summary to send to the LLM.
        tokenizer: Tokenizer to use for counting against max_tokens.
        context: Optional Context object to get default parameters from.
        **kwargs: Additional keyword arguments.

    Returns:
        Conversational response to question.
    """
    if all(isinstance(d, DocSet) for d in result_data):
        return summarize_data_docsets(
            llm, question, result_data, data_description=result_description, summaries_as_text=summaries_as_text
        )

    # If data is not DocSets, text is this list here
    # TODO: Jinjify.
    text = f"Data description: {result_description}\n"
    for i, d in enumerate(result_data):
        text += f"Input {i + 1}: {str(d)}\n"

    messages = SummarizeDataMessagesPrompt(question=question or "", text=text).as_messages()
    prompt = RenderedPrompt(messages=[RenderedMessage(role=m["role"], content=m["content"]) for m in messages])
    completion = llm.generate(prompt=prompt)
    return completion


def summarize_data_docsets(
    llm: LLM,
    question: Optional[str],
    result_data: List[DocSet],
    data_description: Optional[str] = None,
    summaries_as_text: bool = False,
) -> str:
    if summaries_as_text:

        def sum_to_text(d: Document) -> Document:
            if "summary" in d.properties:
                d.text_representation = d.properties.pop("summary")
            return d

        result_data = [ds.summarize(HeirarchicalDocumentSummarizer(llm)).map(sum_to_text) for ds in result_data]

    main_prompt = SummarizeDataHeirarchicalPrompt
    if data_description is not None:
        main_prompt = main_prompt.set(data_description=data_description)  # type: ignore
    single_docs = [_docset_to_singledoc(ds) for ds in result_data]
    agged_ds = (
        result_data[0]
        .context.read.document(single_docs)
        .summarize(HeirarchicalDocumentSummarizer(llm, question, prompt=main_prompt))  # type: ignore
    )
    texts = [d.properties["summary"] for d in agged_ds.take_all()]
    return "\n".join(texts)


def _docset_to_singledoc(ds: DocSet) -> Document:
    """
    Converts a docset into a single document by turning every Document
    into an Element of a global parent document. Essentially a reverse
    explode.
    """
    return Document(elements=[Element(**d.data) for d in ds.take_all()])
    if ds.context.exec_mode == ExecMode.RAY:
        return _ray_docset_to_singledoc(ds)
    else:
        docs = ds.take_all()
        new_doc = Document()
        new_doc.elements = [Element(**doc.data) for doc in docs]
        return new_doc


def _ray_docset_to_singledoc(ds: DocSet) -> Document:
    """
    See _docset_to_singledoc, except do it in ray.
    """
    from ray.data.aggregate import AggregateFn

    def accumulate(collector, doc):
        ddoc = Document.deserialize(doc["doc"])
        if isinstance(ddoc, MetadataDocument):
            return collector
        collector.elements.append(Element(**ddoc.data))
        return collector

    def merge(a, b):
        a.elements.extend(b.elements)
        return a

    agg = AggregateFn(  # type: ignore
        init=lambda c: Document(),
        merge=merge,
        accumulate_row=accumulate,
        name="doc",
    )
    doc = ds.plan.execute().aggregate(agg)
    ds.plan.traverse(visit=lambda n: n.finalize())
    return doc["doc"]


@context_params
def summarize_map_reduce(
    llm: LLM,
    question: str,
    result_description: str,
    result_data: List[Any],
    use_elements: bool = False,
    num_elements: int = 5,
    max_tokens: int = 10 * 1000,
    tokenizer: Tokenizer = CharacterTokenizer(),
) -> str:
    """ """
    text = f"Data description: {result_description}\n"
    for i, result in enumerate(result_data):
        if isinstance(result, DocSet):
            docs = (
                result.filter(lambda d: isinstance(d, MetadataDocument) is False)
                .summarize(
                    summarizer=DocumentSummarizer(llm, question)
                )  # document-level summarization can be parallelized (per DocSet)
                .take_all()
            )
            for doc in docs:
                text += doc.properties["summary"] + "\n"

        else:
            text += str(result) + "\n"

    final_summary = collapse(text, max_tokens, tokenizer, QuestionAnsweringSummarizer(llm, question))
    return final_summary
