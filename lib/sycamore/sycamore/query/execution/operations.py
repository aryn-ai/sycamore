import json
from typing import Any, Callable, List, Optional, Union

from datetime import datetime
from dateutil import parser
from ray.data import Dataset

from sycamore import DocSet, Execution
from sycamore.data import Document, MetadataDocument
from sycamore.llms.openai import OpenAI
from sycamore.plan_nodes import Node, Transform
from sycamore.transforms.extract_entity import OpenAIEntityExtractor
from sycamore.utils.extract_json import extract_json
from sycamore.plan_nodes import Scan

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

# prompts
SC_FORM_GROUPS_PROMPT = """You are given a list of values corresponding to the database field
"{field}". Categorize the occurrences of "{field}" and create relevant non-overlapping group.
Return ONLY JSON with the various categorized groups of "{field}" that can be used to determine
the answer to the following question "{description}", so form groups accordingly. Return your
answer in the following JSON format and check your work: {{"groups": ["string"]}}. For example,
if the question is "What are the most common types of food in this dataset?" and the values are
"banana, milk, yogurt, chocolate, oranges", you would return something like
    {{"groups": ['fruit', 'dairy', 'dessert', 'other]}}.
Form groups to encompass as many entries as possible and don't create multiple groups with
the same meaning. Here is the list values corresponding to "{field}": "{text}"."""
SC_ASSIGN_GROUPS_PROMPT = """Categorize the database entry you are given corresponding to "{field}"
into one of the following groups: "{groups}". Perform your best work to assign the group. Return
ONLY the string corresponding to the selected group. Here is the database entry you will use: """


def field_to_value(doc: Document, field: str) -> Any:
    """
    Extracts the value for a particular document field.

    Args:
        doc: The document
        field: The field in dotted notation to indicate nesting, e.g. doc.properties.schema.

    Returns:
        The value associated with the document field.
    """
    fields = field.split(".")
    value = getattr(doc, fields[0])
    if len(fields) > 1:
        assert fields[0] == "properties"
        for f in fields[1:]:
            value = value[f]
    return value


def convert_string_to_date(date_string: str) -> datetime:
    """
    Creates datetime object given a date string.

    Args:
        date_string: The string that contains the date in any format.

    Returns:
        A datetime object (without timezone info).
    """
    return parser.parse(date_string).replace(tzinfo=None)


def threshold_filter(doc: Document, threshold) -> bool:
    try:
        return_value = int(doc.properties["_autogen_LlmFilterOutput"]) >= threshold
    except Exception:
        # accounts for llm output errors
        return_value = False

    return return_value


def llm_filter_operation(
    client: OpenAI,
    docset: DocSet,
    filter_question: Optional[str] = None,
    field: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    threshold: int = 3,
    **resource_args,
) -> DocSet:
    """
    Filters DocSet to only keep documents that score (determined by LLM) greater
    than or equal to the inputted threshold value.

    Args:
        client: LLM client to use.
        docset: DocSet to filter.
        filter_question: Question used for filtering during LLM call
        field: Document field to filter based on.
        messages: Custom LLM prompt.
        threshold: Cutoff that determines whether or not to keep document.
        **resource_args

    Returns:
        A filtered DocSet.
    """
    if filter_question is None and messages is None:
        raise Exception("Filter question must be specified for default value of messages.")

    if messages is None:
        # sets prompt
        messages = [
            {
                "role": "system",
                "content": "You are a helpful classifier that generously filters database entries based on questions.",
            },
            {
                "role": "user",
                "content": f"""Given an entry and a question, you will answer the question relating to the entry. 
                    You only respond with 0, 1, 2, 3, 4, or 5 based on your confidence level. 0 is the most negative 
                    answer and 5 is the most positive answer. Question: {filter_question}; Entry: """,
            },
        ]

    if field is None:
        field = "text_representation"

    entity_extractor = OpenAIEntityExtractor(
        entity_name="_autogen_LlmFilterOutput", llm=client, use_elements=False, messages=messages, field=field
    )
    docset = docset.extract_entity(entity_extractor=entity_extractor, **resource_args)
    docset = docset.filter(lambda doc: threshold_filter(doc, threshold), **resource_args)

    return docset


def match_filter_operation(doc: Document, query: Any, field: str, ignore_case: bool = True) -> bool:
    """
    Only keep documents that match the query on the specified field.
    Performs substring matching for strings.

    Args:
        doc: Document to filter.
        query: Query to match for.
        field: Document field that is used for filtering.
        ignore_case: Determines case sensitivity for strings.

    Returns:
        A filtered document.

    Example:
        .. code-block:: python

        def wrapper(doc: Document) -> bool:
            query = "Cessna"
            field = "properties.entity.aircraft"
            return match_filter_operation(doc, query, field)

        docset = docset.filter(wrapper)
    """
    value = field_to_value(doc, field)

    # substring matching
    if isinstance(query, str) or isinstance(value, str):
        query = str(query)
        value = str(value)
        if ignore_case:
            value = value.lower()
            query = query.lower()

        return query in value

    # if not string, exact match
    return query == value


def range_filter_operation(
    doc: Document,
    field: str,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    date: Optional[bool] = False,
) -> bool:
    """
    Only keep documents for which the value of the
    specified field is within the start:end range.

    Args:
        doc: Document to filter.
        field: Document field to filter based on.
        start: Value for start of range.
        end: Value for end of range.
        date: Indicates whether start:end is a date range.

    Returns:
        A filtered document.

    Example:
        .. code-block:: python

        def wrapper(doc: Document) -> bool:
            field = "properties.date"
            start = "July 1, 2020"
            end = "July 30, 2020"
            return range_filter_operation(doc, field, start, end, True)

        docset = docset.filter(wrapper)
    """
    value = field_to_value(doc, field)
    if value is None:
        raise ValueError(f"field {field} must be present in the document")

    if date:
        if not isinstance(value, str):
            raise ValueError("value must be a string for date filtering")
        value_comp = convert_string_to_date(value)
        if start and not isinstance(start, str):
            raise ValueError("start must be a string for date filtering")
        start_comp = convert_string_to_date(start) if start else None
        if end and not isinstance(end, str):
            raise ValueError("end must be a string for date filtering")
        end_comp = convert_string_to_date(end) if end else None
    else:
        value_comp = value
        start_comp = start
        end_comp = end

    if start_comp is None:
        if end_comp is None:
            raise ValueError("At least one of start or end must be specified")
        return value_comp <= end_comp
    if end_comp is None:
        if start_comp is None:
            raise ValueError("At least one of start or end must be specified")
        return value_comp >= start_comp
    return value_comp >= start_comp and value_comp <= end_comp


def count_operation(docset: DocSet, field: Optional[str] = None, primary_field: Optional[str] = None, **kwargs) -> int:
    """
    Counts the number of document in a DocSet. Counts by field or primary_field if specified.

    Args:
        docset: DocSet to count.
        field: Field to count based on. Takes precedence over primary_field if both are specified.
        primary_field: Primary field for a document to be considered different.
        **kwargs

    Returns:
        An integer.
    """
    # normal count
    if field is None and primary_field is None:
        return docset.count(**kwargs)

    else:
        if field is not None:
            unique_field = field
        else:
            if primary_field is None:
                raise ValueError("Must specify either 'field' or 'primary_field'")
            unique_field = primary_field
        unique_docs = set()
        execution = Execution(docset.context, docset.plan)
        dataset = execution.execute(docset.plan, **kwargs)
        for row in dataset.iter_rows():
            doc = Document.from_row(row)
            if isinstance(doc, MetadataDocument):
                continue
            value = field_to_value(doc, unique_field)
            unique_docs.add(value)
        return len(unique_docs)


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

    # sets message
    messages = [
        {
            "role": "system",
            "content": """You are a helpful conversational English response generator for queries 
                regarding database entries.""",
        },
        {
            "role": "user",
            "content": f"""The following question and answer are in regards to database entries. 
                Respond ONLY with a conversational English response WITH JUSTIFICATION to the question
                 \"{question}\" given the answer \"{text}\". Include as much detail/evidence as possible""",
        },
    ]
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
        value = field_to_value(doc, field)
        return value in join_set

    return filter_fn_join


def join_operation(docset1: DocSet, docset2: DocSet, field1: str, field2: str) -> DocSet:
    """
    Joins two docsets based on specified fields; docset2 filtered based on values of docset1.

    Args:
        docset1: DocSet to filter based on.
        docset2: DocSet to filter.
        field1: Field in docset1 to filter based on.
        field2: Field in docset2 to filter.

    Returns:
        A joined DocSet.
    """
    execution = Execution(docset1.context, docset1.plan)
    dataset = execution.execute(docset1.plan)

    # identifies unique values of field1 in docset1
    unique_vals = set()
    for row in dataset.iter_rows():
        doc = Document.from_row(row)
        if isinstance(doc, MetadataDocument):
            continue
        value = field_to_value(doc, field1)
        unique_vals.add(value)

    # filters docset2 based on matches of field2 with unique values
    filter_fn_join = make_filter_fn_join(field2, unique_vals)
    joined_docset = docset2.filter(lambda doc: filter_fn_join(doc))

    return joined_docset


def count_aggregate_operation(docset: DocSet, field, unique_field, **kwargs) -> DocSet:
    """
    Performs a count aggregation on a DocSet.

    Args:
        docset: DocSet to aggregate.
        field: Field to aggregate based on.
        unique_field: Determines what makes a unique document.
        **kwargs

    Returns:
        A DocSet with "properties.key" (unique values of document field)
        and "properties.count" (frequency counts for unique values).
    """
    dataset = CountAggregate(docset.plan, field, unique_field).execute(**kwargs)
    return DocSet(docset.context, DatasetScan(dataset))


def top_k_operation(
    client: OpenAI,
    docset: DocSet,
    field: str,
    k: Optional[int],
    description: str,
    descending: bool = True,
    use_llm: bool = False,
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
        use_llm: Indicates whether an LLM should be used to normalize values of document field.
        unique_field: Determines what makes a unique document.
        **kwargs

    Returns:
        A DocSet with "properties.key" (unique values of document field)
        and "properties.count" (frequency counts for unique values) which is
        sorted based on descending and contains k records.
    """

    if use_llm:
        docset = semantic_cluster(client, docset, description, field)
        field = "properties._autogen_ClusterAssignment"

    docset = count_aggregate_operation(docset, field, unique_field, **kwargs)

    # the names of the fields being "key" and "count" could cause problems down the line?
    # uses 0 as default value -> end of docset
    docset = docset.sort(descending, "properties.count", 0)
    if k is not None:
        docset = docset.limit(k)
    return docset


def semantic_cluster(client: OpenAI, docset: DocSet, description: str, field: str) -> DocSet:
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
        text += field_to_value(doc, field)

    # sets message
    messages = [
        {
            "role": "user",
            "content": SC_FORM_GROUPS_PROMPT.format(field=field, description=description, text=text),
        }
    ]

    prompt_kwargs = {"messages": messages}

    # call to LLM
    completion = client.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})

    groups = extract_json(completion)

    assert isinstance(groups, dict)

    messagesForExtract = [
        {"role": "user", "content": SC_ASSIGN_GROUPS_PROMPT.format(field=field, groups=groups["groups"])}
    ]

    entity_extractor = OpenAIEntityExtractor(
        entity_name="_autogen_ClusterAssignment",
        llm=client,
        use_elements=False,
        messages=messagesForExtract,
        field=field,
    )
    docset = docset.extract_entity(entity_extractor=entity_extractor)

    # LLM response
    return docset


def make_map_fn_count(field: str, unique_field: Optional[str] = None) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Creates a map function that can be called on a Ray Dataset
    based on a DocSet. Adds a column to the Dataset based on
    field and unique_field in DocSet documents.

    Args:
        field: Document field to add as a column.
        unique_field: Unique document field to as a column.

    Returns:
        Function that can be called inside of DocSet.filter
    """

    def ray_callable(input_dict: dict[str, Any]) -> dict[str, Any]:
        doc = Document.from_row(input_dict)

        try:
            val = field_to_value(doc, field)
            # updates row to include new col
            new_doc = doc.to_row()
            new_doc["key"] = val

            if unique_field is not None:
                val = field_to_value(doc, unique_field)
                # updates row to include new col
                new_doc["unique"] = val
            return new_doc

        except Exception:
            if unique_field is not None:
                return {"doc": None, "key": None, "unique": None}
            else:
                return {"doc": None, "key": None}

    return ray_callable


def filterOutNone(row: dict[str, Any]) -> bool:
    """
    Filters out Dataset rows where all values are None.

    Args:
        row: Input Dataset row.

    Returns:
        Boolean that indicates whether or not to keep row.
    """
    return_value = row["doc"] is not None and row["key"] is not None

    if "unique" in row:
        return_value = return_value and row["unique"] is not None

    return return_value


def add_doc_column(row: dict[str, Any]) -> dict[str, Any]:
    """
    Adds a doc column with serialized document to Ray Dataset.

    Args:
        row: Input Dataset row.

    Returns:
        Row with added doc column.
    """
    row["doc"] = Document(text_representation="", properties={"key": row["key"], "count": row["count()"]}).serialize()
    return row


class CountAggregate(Transform):
    """
    Count aggregation that allows you to aggregate by document field(s).
    """

    def __init__(self, child: Node, field: str, unique_field: Optional[str] = None):
        super().__init__(child)
        self._field = field
        self._unique_field = unique_field

    def execute(self, **kwargs) -> "Dataset":
        # creates dataset
        ds = self.child().execute(**kwargs)

        # adds a "key" column containing desired field
        map_fn = make_map_fn_count(self._field, self._unique_field)
        ds = ds.map(map_fn)
        ds = ds.filter(lambda row: filterOutNone(row))
        # lazy grouping + count aggregation

        if self._unique_field is not None:
            ds = ds.groupby(["key", "unique"]).count()

        ds = ds.groupby("key").count()

        # Add the new column to the dataset
        ds = ds.map(add_doc_column)

        return ds


class DatasetScan(Scan):
    """
    Scans a dataset.
    """

    def __init__(self, dataset: Dataset, **resource_args):
        super().__init__(**resource_args)
        self._dataset = dataset

    def execute(self, **kwargs) -> Dataset:
        return self._dataset

    def format(self):
        return "dataset"
