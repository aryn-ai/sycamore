import json
from typing import Any, Callable, List, Optional, Union

from sycamore import DocSet, Execution
from sycamore.data import Document, MetadataDocument
from sycamore.llms.llms import LLM
from sycamore.llms.openai import OpenAI
from sycamore.llms.prompts.default_prompts import (
    SummarizeDataMessagesPrompt,
    LlmClusterEntityAssignGroupsMessagesPrompt,
    LlmClusterEntityFormGroupsMessagesPrompt,
)
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.utils.extract_json import extract_json

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

NUM_DOCS_GENERATE = 60
NUM_TEXT_CHARS_GENERATE = 2500


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
        return val1 / val2
    elif operator == "multiply":
        return val1 * val2
    else:
        raise ValueError("Invalid math operator " + operator)


def llm_generate_operation(
    client: OpenAI, question: str, result_description: str, result_data: List[Any], **kwargs
) -> str:
    """
    Provides an English response to a question given relevant information.

    Args:
        client: LLM client.
        question: Question to answer.
        result_description: Description of each of the inputs in result_data.
        result_data: List of inputs.
        **kwargs

    Returns:
        Conversational response to question.
    """
    text = f"Description: {result_description}\n"

    for i, result in enumerate(result_data):
        text += f"Input {i + 1}:\n"

        # consolidates relevant properties to give to LLM
        if isinstance(result, DocSet):
            for doc in result.take(NUM_DOCS_GENERATE, **kwargs):
                if isinstance(doc, MetadataDocument):
                    continue
                props_dict = doc.properties.get("entity", {})
                props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})
                props_dict["text_representation"] = (
                    doc.text_representation[:NUM_TEXT_CHARS_GENERATE] if doc.text_representation is not None else None
                )

                text += json.dumps(props_dict, indent=2) + "\n"

        else:
            text += str(result_data) + "\n"

    messages = SummarizeDataMessagesPrompt(question=question, text=text).get_messages_dict()
    prompt_kwargs = {"messages": messages}

    # call to LLM
    completion = client.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})

    # LLM response
    return completion


def make_filter_fn_join(field: str, join_set: set) -> Callable[[Document], bool]:
    """
    Creates a filter function that can be called on a DocSet. Document
    will be kept if the value corresponding to document field is contained
    in join_set.

    Args:
        field: Document field to filter based on
        join_set: Set that contains valid field values.

    Returns:
        Function that can be called inside of DocSet.filter
    """

    def filter_fn_join(doc: Document) -> bool:
        value = doc.field_to_value(field)
        return value in join_set

    return filter_fn_join


def inner_join_operation(docset1: DocSet, docset2: DocSet, field1: str, field2: str) -> DocSet:
    """
    Joins two docsets based on specified fields; docset1 filtered based on values of docset2.

    SQL Equivalent:
    SELECT docset1.*
    FROM docset1
    INNER JOIN docset2
    ON docset1.field1 = docset2.field2

    Args:
        docset1: DocSet to filter based on.
        docset2: DocSet to filter.
        field1: Field in docset1 to filter based on.
        field2: Field in docset2 to filter.

    Returns:
        A joined DocSet.
    """
    execution = Execution(docset2.context, docset2.plan)
    dataset = execution.execute(docset2.plan)

    # identifies unique values of field1 in docset1
    unique_vals = set()
    for row in dataset.iter_rows():
        doc = Document.from_row(row)
        if isinstance(doc, MetadataDocument):
            continue
        value = doc.field_to_value(field2)
        unique_vals.add(value)

    # filters docset2 based on matches of field2 with unique values
    filter_fn_join = make_filter_fn_join(field1, unique_vals)
    joined_docset = docset1.filter(lambda doc: filter_fn_join(doc))

    return joined_docset


def top_k_operation(
    docset: DocSet,
    llm: LLM,
    field: str,
    k: Optional[int],
    description: str,
    descending: bool = True,
    llm_cluster: bool = False,
    unique_field: Optional[str] = None,
    **kwargs,
) -> DocSet:
    """
    Determines the top k occurrences for a document field.

    Args:
        client: LLM client.
        docset: DocSet to use.
        field: Field to determine top k occurrences of.
        k: Number of top occurrences.
        description: Description of operation purpose.
        descending: Indicates whether to return most or least frequent occurrences.
        llm_cluster: Indicates whether an LLM should be used to normalize values of document field.
        unique_field: Determines what makes a unique document.
        **kwargs

    Returns:
        A DocSet with "properties.key" (unique values of document field)
        and "properties.count" (frequency counts for unique values) which is
        sorted based on descending and contains k records.
    """

    if llm_cluster:
        docset = llm_cluster_entity(docset, llm, description, field)
        field = "properties._autogen_ClusterAssignment"

    docset = docset.groupby_count(field, unique_field, **kwargs)

    # uses 0 as default value -> end of docset
    docset = docset.sort(descending, "properties.count", 0)
    if k is not None:
        docset = docset.limit(k)
    return docset


def llm_cluster_entity(docset: DocSet, llm: LLM, description: str, field: str) -> DocSet:
    """
    Normalizes a particular field of a DocSet. Identifies and assigns each document to a "group".

    Args:
        client: LLM client.
        docset: DocSet to form groups for.
        description: Description of purpose of this operation.
        field: Field to make/assign groups based on.

    Returns:
        A DocSet with an additional field "properties._autogen_ClusterAssignment".
    """
    text = ""
    for i, doc in enumerate(docset.take_all()):
        if i != 0:
            text += ", "
        text += str(doc.field_to_value(field))

    # sets message
    messages = LlmClusterEntityFormGroupsMessagesPrompt(
        field=field, description=description, text=text
    ).get_messages_dict()

    prompt_kwargs = {"messages": messages}

    # call to LLM
    completion = llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})

    groups = extract_json(completion)

    assert isinstance(groups, dict)

    # sets message
    messagesForExtract = LlmClusterEntityAssignGroupsMessagesPrompt(
        field=field, groups=groups["groups"]
    ).get_messages_dict()

    entity_extractor = OpenAIEntityExtractor(
        entity_name="_autogen_ClusterAssignment",
        llm=llm,
        use_elements=False,
        prompt=messagesForExtract,
        field=field,
    )
    docset = docset.extract_entity(entity_extractor=entity_extractor)

    # LLM response
    return docset
