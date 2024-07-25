import json
from typing import Any, List, Optional, Union

from datetime import datetime
from dateutil import parser
from ray.data import Dataset

from sycamore import DocSet, Execution
from sycamore.data import Document, MetadataDocument
from sycamore.llms.openai import OpenAI
from sycamore.plan_nodes import Node, Transform
from sycamore.utils.extract_json import extract_json
from sycamore.plan_nodes import Scan


################## HELPERS #################

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


def field_to_value(doc: Document, field: str) -> Any:
    fields = field.split(".")
    value = getattr(doc, fields[0])
    if len(fields) > 1:
        assert fields[0] == "properties"
        for f in fields[1:]:
            value = value[f]
    return value


def convert_string_to_date(date_string: str) -> datetime:
    return parser.parse(date_string).replace(tzinfo=None)


######################################################################################################################################
def threshold_filter(doc: Document, threshold) -> bool:
    try:
        return_value = int(doc.properties["LlmFilterOutput"]) >= threshold
    except Exception:
        # accounts for llm output errors
        return_value = False

    return return_value


# USAGE EXAMPLE:
# def wrapper(doc: Document) -> bool:
#     return llm_filter_operation(client, doc, filter_question)
# docset = (docset.filter(wrapper))
def llm_filter_operation(
    client: OpenAI,
    docset: DocSet,
    filter_question: Optional[str] = None,
    field: Optional[str] = None,
    messages: Optional[List[dict]] = None,
    threshold: int = 3,
    **resource_args,
) -> DocSet:
    """This operation filters your DocSet to only keep documents that score greater
    than or equal to the inputted threshold value from an LLM call that returns an int.

    Inputs:
    - client: The Sycamore OpenAI client to use, e.g. OpenAI(OpenAIModels.GPT_4O.value)
    - doc: The document in the DocSet
    - field: The field to filter based on, default is doc.text_representation
    - filter_question: Question for filter, e.g. Was this incident caused by environmental factors?
    - filter_prompt: Custom prompt that you can use for filtering
    - system_prompt: Custom prompt to the system
    - threshold: Threshold for success, e.g. 3 (default scale is 0-5)
    """
    if filter_question is None and messages is None:
        raise Exception("Filter question must be specified for default value of messages.")

    if field is None:
        field = "text_representation"

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

    docset = docset.map(
        lambda doc: llm_extract_operation(
            client=client, doc=doc, new_field="LlmFilterOutput", field=field, messages=messages
        )
    )
    docset = docset.filter(lambda doc: threshold_filter(doc, threshold), **resource_args)

    return docset


######################################################################################################################################


# USAGE EXAMPLE:
# def wrapper(doc: Document) -> bool:
# query = "Cessna"
# field = doc.properties["entity"]["aircraft"]
# return operations.match_filter_operation(query, field)
# docset = docset.filter(wrapper)
def match_filter_operation(doc: Document, query: Any, field: str, ignore_case: bool = True) -> bool:
    """This operation filters your Docset to only keep documents that match the inputted
    query on the specified field. If the query/inputted field are strings, it looks for
    a substring match. For any type other than strings, it looks for an exact match.

    Inputs:
    - query: The query to filter based on
    - field: The field to search for a match
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


######################################################################################################################################


# USAGE EXAMPLE
# def wrapper(doc: Document) -> bool:
#     start_date = datetime.datetime(2023, 1, 1)
#     end_date = datetime.datetime(2023, 1, 20)
#     return operations.range_filter_operation(start_date, end_date, doc.properties["dateTimeType"])
# docset = docset.filter(wrapper)
def range_filter_operation(
    doc: Document,
    field: str,
    start: Optional[Any] = None,
    end: Optional[Any] = None,
    date: Optional[bool] = False,
) -> bool:
    """This operation filters your Docset to only keep documents for which the value of the
    specified field is within the start:end range.

    Inputs:
    - field: The field to run the range filter on
    - start: The start value for the range
    - end: The end value for the range
    - date: Interpret the values for start, end, and value as strings containing date values.
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


######################################################################################################################################


# USAGE EXAMPLE
# def wrapper(doc: Document) -> Document:
#     return operations.llm_extract_operation(doc, """Return ONLY the date in the form YYYY/MM/DD. e.g. You are
#                                      given \"January 20, 2024\" and you return \"2024/01/20\"""",
#                                      doc.properties['entity']["dateAndTime"], "formattedDate")
# docset = docset.map(wrapper)
def llm_extract_operation(
    client: OpenAI,
    doc: Document,
    new_field: str,
    field: Optional[str] = None,
    question: Optional[str] = None,
    format: Optional[str] = None,
    discrete: Optional[bool] = None,
    messages: Optional[List[dict]] = None,
) -> Document:
    """This operation adds a new property to your Docset based on a question to the LLM
    on a particular field.

    Inputs:
    - doc: the Document
    - question: The question to extract the entity based on, e.g. What was the cause of this incident?
    - field: The database field for reference, e.g. doc.text_representation
    - new_field: The new database field created with the extracted entity (in properties)
    - (optional) model: OpenAI model to use, e.g. "gpt-4o"
    """

    if messages is None and (question is None or format is None or discrete is None):
        raise ValueError('"question", "format", and "discrete" must be specified for default messages')

    if field is None:
        field = "text_representation"

    value = field_to_value(doc, field)

    if messages is None:

        format_string = f"""The format of your resopnse should be {format}. Use standard convention
        to determine the style of your response. Do not include any abbreviations."""

        # sets message
        messages = [
            {
                "role": "system",
                "content": """You are a helpful entity extractor that creates a new field in a
                database from your reponse to a question on an existing field.""",
            }
        ]

        if discrete:
            messages.append(
                {
                    "role": "user",
                    "content": f"""Question: {question} Use this existing related database field
                    "{field}" to answer the question: {value}. {format_string}""",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": """The following sentence should be valid: The answer to the
                    question based on the existing field is {answer}. Your response should ONLY
                    contain the answer. If you are not able to extract the new field given the
                    information, respond ONLY with None""",
                }
            )

        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"""Question: {question} Use this existing related database field
                    "{field}" to answer the question: {value}. Include as much relevant detail as
                    possible that is related to/could help answer this question. Respond in
                    sentences, not just a single word or phrase.""",
                }
            )

    else:
        messages.append(
            {
                "role": "user",
                "content": f"{value}",
            }
        )

    prompt_kwargs = {"messages": messages}
    # call to LLM
    completion = client.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})

    # adds new property
    doc.properties.update({new_field: completion})

    return doc


######################################################################################################################################
# USAGE EXAMPLE
# num_unique_dates = operations.count_operation(docset, True, "dateTimeType")
# num_total_dates = operations.count_operation(docset)
def count_operation(docset: DocSet, field: Optional[str] = None, primaryField: Optional[str] = None, **kwargs) -> int:
    """Returns a DocSet count. When count_unique is false, the number of entries in the DocSet are
    returned. When count_unique is true, "field" must be specified and must be present in
    doc.properties. When count_unique is true, this function returns the number of rows in the
    DocSet with a unique value corresponding to "field"."""

    # none are specified -> normal count
    if field is None and primaryField is None:
        return docset.count(**kwargs)

    else:
        if field is not None:
            unique_field = field
        else:
            if primaryField is None:
                raise ValueError("Must specify either 'field' or 'primaryField'")
            unique_field = primaryField
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


######################################################################################################################################


# USAGE EXAMPLE
# operations.math_operation(1, 2, "add")
# operations.math_operation(1, 2, "subtract")
# operations.math_operation(1, 2, "divide")
# operations.math_operation(1, 2, "multiply")
def math_operation(val1: int, val2: int, operator: str) -> Union[int, float]:
    """Basic arithmetic operations on integers."""
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


######################################################################################################################################
# USAGE EXAMPLE
# operations.llm_generate_operation("database of airplane incidents (NTSB)",
# "How many incidents occurred between 1/1/2022 and 1/20/2022?", 13)
def llm_generate_operation(client: OpenAI, question: str, result_description: str, result_data: Any, **kwargs) -> str:
    text = f"Description: {result_description}\n"

    for i, result in enumerate(result_data):
        text += f"Input {i + 1}:\n"

        if isinstance(result, DocSet):
            for doc in result.take(60, **kwargs):
                if isinstance(doc, MetadataDocument):
                    continue
                if "entity" in doc.properties:
                    props_dict = doc.properties["entity"]
                else:
                    props_dict = {}

                props_dict.update({p: doc.properties[p] for p in set(doc.properties) - set(BASE_PROPS)})
                props_dict["text_representation"] = (
                    doc.text_representation[:2500] if doc.text_representation is not None else None
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


def make_filter_fn_join(field: str, join_set=set):

    def filter_fn_join(doc: Document) -> bool:
        value = field_to_value(doc, field)
        return value in join_set

    return filter_fn_join


def join_operation(docset1: DocSet, docset2: DocSet, field1: str, field2: str) -> DocSet:
    unique_vals = set()
    execution = Execution(docset1.context, docset1.plan)
    dataset = execution.execute(docset1.plan)
    for row in dataset.iter_rows():
        doc = Document.from_row(row)
        if isinstance(doc, MetadataDocument):
            continue
        value = field_to_value(doc, field1)
        unique_vals.add(value)

    for doc in docset1.take_all():
        value = field_to_value(doc, field1)
        unique_vals.add(value)

    filter_fn_join = make_filter_fn_join(field2, unique_vals)

    joined_docset = docset2.filter(lambda doc: filter_fn_join(doc))

    return joined_docset


######################################################################################################################################


def count_aggregate_operation(docset: DocSet, field, unique_field, **kwargs) -> DocSet:
    dataset = CountAggregate(docset.plan, field, unique_field).execute(**kwargs)
    return DocSet(docset.context, DatasetScan(dataset))


######################################################################################################################################


def top_k_operation(
    client: OpenAI,
    docset: DocSet,
    field: str,
    k: Union[int, None],
    description: str,
    descending: bool = True,
    use_llm: bool = False,
    unique_field: Optional[str] = None,
    **kwargs,
) -> DocSet:

    if use_llm:
        docset = semantic_cluster(client, docset, description, field)
        field = "properties.ClusterAssignment"

    docset = count_aggregate_operation(docset, field, unique_field, **kwargs)

    # the names of the fields being "key" and "count" could cause problems down the line?
    # uses 0 as default value -> end of docset
    docset = docset.sort(descending, "properties.count", 0)
    if k is not None:
        docset = docset.limit(k)
    return docset


sc_form_groups_prompt = """You are given a list of values corresponding to the database field
"{field}". Categorize the occurrences of "{field}" and create relevant non-overlapping group.
Return ONLY JSON with the various categorized groups of "{field}" that can be used to determine
the answer to the following question "{description}", so form groups accordingly. Return your
answer in the following JSON format and check your work: {{"groups": ["string"]}}. For example,
if the question is "What are the most common types of food in this dataset?" and the values are
"banana, milk, yogurt, chocolate, oranges", you would return something like
    {{"groups": ['fruit', 'dairy', 'dessert', 'other]}}.
Form groups to encompass as many entries as possible and don't create multiple groups with
the same meaning. Here is the list values corresponding to "{field}": "{text}"."""

sc_assign_groups_prompt = """Categorize the database entry you are given corresponding to "{field}"
into one of the following groups: "{groups}". Perform your best work to assign the group. Return
ONLY the string corresponding to the selected group. Here is the database entry you will use: """


def semantic_cluster(client: OpenAI, docset: DocSet, description: str, field: str) -> DocSet:

    text = ""
    for i, doc in enumerate(docset.take_all()):
        if i != 0:
            text += ", "
        text += field_to_value(doc, field)

    # sets message
    messages = [
        {
            "role": "user",
            "content": sc_form_groups_prompt.format(field=field, description=description, text=text),
        }
    ]

    prompt_kwargs = {"messages": messages}

    # call to LLM
    completion = client.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})

    groups = extract_json(completion)

    assert isinstance(groups, dict)

    messagesForExtract = [
        {"role": "user", "content": sc_assign_groups_prompt.format(field=field, groups=groups["groups"])}
    ]

    docset = docset.map(
        lambda doc: llm_extract_operation(
            client=client, doc=doc, new_field="ClusterAssignment", field=field, messages=messagesForExtract
        )
    )

    # LLM response
    return docset


############## NODES ################


def make_map_fn_count(field: str, default_val: Any, unique_field: Optional[str] = None):
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


def filterOutNone(row) -> bool:
    return_value = row["doc"] is not None and row["key"] is not None

    if "unique" in row:
        return_value = return_value and row["unique"] is not None

    return return_value


def add_doc_column(row):
    row["doc"] = Document(text_representation="", properties={"key": row["key"], "count": row["count()"]}).serialize()
    return row


class CountAggregate(Transform):
    """
    Aggregation function that allows you to aggregate by field in doc.properties
    """

    def __init__(self, child: Node, field: str, unique_field: Optional[str] = None):
        super().__init__(child)
        self._field = field
        self._unique_field = unique_field

    def execute(self, **kwargs) -> "Dataset":
        # creates dataset
        ds = self.child().execute(**kwargs)

        # adds a "key" column containing desired field
        map_fn = make_map_fn_count(self._field, None, self._unique_field)
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
    def __init__(self, dataset: Dataset, **resource_args):
        super().__init__(**resource_args)
        self._dataset = dataset

    def execute(self, **kwargs) -> Dataset:
        return self._dataset

    def format(self):
        return "dataset"
