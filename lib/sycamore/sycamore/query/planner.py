from dataclasses import dataclass
import logging
import typing
from typing import Any, List, Optional, Tuple, Type


from sycamore.llms.llms import LLM
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.query.logical_plan import LogicalPlan, Node
from sycamore.query.operators.count import Count
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.basic_filter import BasicFilter
from sycamore.query.operators.llm_extract_entity import LlmExtractEntity
from sycamore.query.operators.query_database import QueryDatabase, QueryVectorDatabase
from sycamore.query.operators.math import Math
from sycamore.query.operators.sort import Sort
from sycamore.query.operators.summarize_data import SummarizeData
from sycamore.query.operators.top_k import TopK
from sycamore.query.operators.limit import Limit
from sycamore.query.schema import OpenSearchSchema, OpenSearchSchemaField
from sycamore.utils.extract_json import extract_json

if typing.TYPE_CHECKING:
    from opensearchpy import OpenSearch


# All operators that are allowed for construction of a query plan.
# If a class is not in this list, it will not be used.
OPERATORS: List[Type[Node]] = [
    QueryDatabase,
    QueryVectorDatabase,
    BasicFilter,
    LlmFilter,
    LlmExtractEntity,
    Count,
    SummarizeData,
    Math,
    Sort,
    TopK,
    Limit,
]


# This is the base prompt for the planner.
PLANNER_SYSTEM_PROMPT = """You are a helpful agent that translates the user's question into a
query plan, using a predefined set of query operators. Please adhere to the following
guidelines when generating a plan:

        1. Return your answer as a JSON dictionary containing a query plan in the format shown below.
        2. Do not return any information except a single JSON object. This means not repeating the question
            or providing any text outside the json block.
        3. Only use the query operators described below.
        4. Only use EXACT field names from the DATA_SCHEMA described below and fields created
            from *LlmExtractEntity*. Any new fields created by *LlmExtractEntity* will be nested in properties.
            e.g. if a new field called "state" is added, when referencing it in another operation,
            you should use "properties.state". A database returned from *TopK* operation only has
            "properties.key" or "properties.count"; you can only reference one of those fields.
            Other than those, DO NOT USE ANY OTHER FIELD NAMES.
        5. If an optional field does not have a value in the query plan, return null in its place.
        6. The first step of each plan MUST be a **QueryDatabase** or **QueryVectorDatabase" operation that returns a 
           database. Whenever possible, include all possible filtering operations in the QueryDatabase step.
           That is, you should strive to construct an OpenSearch query that filters the data as
           much as possible, reducing the need for further query operations. Use a QueryVectorDatabase step instead of
           an LLMFilter if the query looks reasonable.
"""

# Variants on the last step in the query plan, based on whether the user has requested raw data
# or a natural language response.
PLANNER_RAW_DATA_PROMPT = "7. The last step of each plan should return the raw data associated with the response."
PLANNER_NATURAL_LANGUAGE_PROMPT = (
    "7. The last step of each plan *MUST* be a **SummarizeData** operation that returns a natural language response."
)


def process_json_plan(json_plan: str) -> LogicalPlan:
    """Deserialize the query plan returned by the LLM."""

    parsed_plan = extract_json(json_plan)
    if not isinstance(parsed_plan, dict):
        raise ValueError(f"Expected LLM query plan to contain a dict, got f{type(parsed_plan)}")
    return LogicalPlan.model_validate(parsed_plan)


@dataclass
class PlannerExample:
    """Represents an example query and query plan for the planner."""

    schema: OpenSearchSchema
    plan: LogicalPlan


# Example schema and planner examples for the NTSB and financial datasets.
EXAMPLE_NTSB_SCHEMA = OpenSearchSchema(
    fields={
        "properties.path": OpenSearchSchemaField(
            field_type="str", examples=["/docs/incident1.pdf", "/docs/incident2.pdf", "/docs/incident3.pdf"]
        ),
        "properties.entity.date": OpenSearchSchemaField(field_type="date", examples=["2023-07-01", "2024-09-01"]),
        "properties.entity.accidentNumber": OpenSearchSchemaField(field_type="str", examples=["1234", "5678", "91011"]),
        "properties.entity.location": OpenSearchSchemaField(
            field_type="str", examples=["Atlanta, Georgia", "Miami, Florida", "San Diego, California"]
        ),
        "properties.entity.aircraft": OpenSearchSchemaField(field_type="str", examples=["Cessna", "Boeing", "Airbus"]),
        "properties.entity.city": OpenSearchSchemaField(field_type="str", examples=["Atlanta", "Savannah", "Augusta"]),
        "text_representation": OpenSearchSchemaField(
            field_type="str", examples=["Can be assumed to have all other details"]
        ),
    }
)

EXAMPLE_FINANCIAL_SCHEMA = OpenSearchSchema(
    fields={
        "properties.path": OpenSearchSchemaField(field_type="str", examples=["doc1.pdf", "doc2.pdf", "doc3.pdf"]),
        "properties.entity.date": OpenSearchSchemaField(
            field_type="str", examples=["2022-01-01", "2022-12-31", "2023-01-01"]
        ),
        "properties.entity.revenue": OpenSearchSchemaField(
            field_type="float", examples=["1000000.0", "2000000.0", "3000000.0"]
        ),
        "properties.entity.firmName": OpenSearchSchemaField(
            field_type="str",
            examples=["Dewey, Cheatem, and Howe", "Saul Goodman & Associates", "Wolfram & Hart"],
        ),
        "text_representation": OpenSearchSchemaField(
            field_type="str", examples=["Can be assumed to have all other details"]
        ),
    }
)

PLANNER_EXAMPLES: List[PlannerExample] = [
    PlannerExample(
        schema=EXAMPLE_NTSB_SCHEMA,
        plan=LogicalPlan(
            query="Were there any incidents in Georgia?",
            result_node=1,
            nodes={
                0: QueryDatabase(
                    node_id=0,
                    description="Get all the incident reports",
                    index="ntsb",
                ),
                1: LlmFilter(
                    node_id=1,
                    description="Filter to only include incidents in Georgia",
                    question="Did this incident occur in Georgia?",
                    field="properties.entity.location",
                    inputs=[0],
                ),
            },
        ),
    ),
    PlannerExample(
        schema=EXAMPLE_NTSB_SCHEMA,
        plan=LogicalPlan(
            query="Count the number of incidents containing 'foo' in the filename.",
            result_node=1,
            nodes={
                0: QueryDatabase(
                    node_id=0,
                    description="Get all the incident reports matching 'foo' in the filename",
                    index="ntsb",
                    query={"match": {"properties.path.keyword": "*foo*"}},
                ),
                1: Count(
                    node_id=1,
                    description="Count the number of incidents",
                    distinct_field="properties.entity.accidentNumber",
                    inputs=[0],
                ),
            },
        ),
    ),
    PlannerExample(
        schema=EXAMPLE_NTSB_SCHEMA,
        plan=LogicalPlan(
            query="""Show incidents between July 1, 2023 and September 1, 2024 with an accident
 .         number containing 'K1234N' that occurred in Georgia.""",
            result_node=0,
            nodes={
                0: QueryDatabase(
                    node_id=0,
                    description="Get all the incident reports in the specified date range matching the accident number",
                    index="ntsb",
                    query={
                        "bool": {
                            "must": [
                                {
                                    "range": {
                                        "properties.entity.isoDateTime": {
                                            "gte": "2023-07-01T00:00:00",
                                            "lte": "2024-09-30T23:59:59",
                                            "format": "strict_date_optional_time",
                                        }
                                    }
                                },
                                {"match": {"properties.entity.accidentNumber.keyword": "*K1234N*"}},
                                {"match": {"properties.entity.location": "Georgia"}},
                            ]
                        }
                    },
                )
            },
        ),
    ),
    PlannerExample(
        schema=EXAMPLE_NTSB_SCHEMA,
        plan=LogicalPlan(
            query="How many cities did Cessna aircrafts have incidents in?",
            result_node=1,
            nodes={
                0: QueryDatabase(
                    node_id=0,
                    description="Get all the incident reports involving Cessna aircrafts",
                    index="ntsb",
                    query={"match": {"properties.entity.aircraft": "Cessna"}},
                ),
                1: Count(
                    node_id=1,
                    description="Count the number of cities that accidents occured in",
                    distinct_field="properties.entity.city",
                    inputs=[0],
                ),
            },
        ),
    ),
    PlannerExample(
        schema=EXAMPLE_NTSB_SCHEMA,
        plan=LogicalPlan(
            query="Which 5 pilots were responsible for the most incidents?",
            result_node=2,
            nodes={
                0: QueryDatabase(
                    node_id=0,
                    description="Get all the NTSB incident reports",
                    index="ntsb",
                ),
                1: LlmExtractEntity(
                    node_id=1,
                    description="Extract the pilot",
                    question="Who was the pilot of this aircraft?",
                    field="text_representation",
                    new_field="pilot",
                    new_field_type="str",
                    discrete=True,
                    inputs=[0],
                ),
                2: TopK(
                    node_id=2,
                    description="Return top 5 pilot names",
                    field="properties.pilot",
                    primary_field="properties.entity.accidentNumber",
                    K=5,
                    descending=True,
                    llm_cluster=False,
                    inputs=[1],
                ),
            },
        ),
    ),
    PlannerExample(
        schema=EXAMPLE_NTSB_SCHEMA,
        plan=LogicalPlan(
            query="What percent of incidents occurred in 2023?",
            result_node=4,
            nodes={
                0: QueryDatabase(
                    node_id=0,
                    description="Get all the incident reports",
                    index="ntsb",
                ),
                1: Count(
                    node_id=1,
                    description="Count the number of total incidents",
                    distinct_field="properties.entity.accidentNumber",
                    inputs=[0],
                ),
                2: BasicFilter(
                    node_id=2,
                    description="Filter to only include incidents in 2023",
                    range_filter=True,
                    query=None,
                    start="01-01-2023",
                    end="12-31-2023",
                    field="properties.entity.date",
                    is_date=True,
                    inputs=[0],
                ),
                3: Count(
                    node_id=3,
                    description="Count the number of incidents in 2023",
                    distinct_field="properties.entity.accidentNumber",
                    inputs=[2],
                ),
                4: Math(
                    node_id=4,
                    description="Divide the number of incidents in 2023 by the total number",
                    operation="divide",
                    inputs=[3, 1],
                ),
            },
        ),
    ),
    PlannerExample(
        schema=EXAMPLE_NTSB_SCHEMA,
        plan=LogicalPlan(
            query="Were there any incidents because of sudden weather changes?",
            result_node=0,
            nodes={
                0: QueryVectorDatabase(
                    node_id=0,
                    description="Get all the incidents relating to sudden weather changes",
                    index="ntsb",
                    query_phrase="sudden weather changes",
                )
            },
        ),
    ),
    PlannerExample(
        schema=EXAMPLE_FINANCIAL_SCHEMA,
        plan=LogicalPlan(
            query="Which 2 law firms had the highest revenue in 2022?",
            result_node=2,
            nodes={
                0: QueryDatabase(
                    node_id=0,
                    description="Get all the financial documents from 2022",
                    index="finance",
                    query={
                        "range": {
                            "properties.entity.isoDateTime": {
                                "gte": "2022-01-01T00:00:00",
                                "lte": "2022-12-31T23:59:59",
                                "format": "strict_date_optional_time",
                            }
                        }
                    },
                ),
                1: Sort(
                    node_id=1,
                    description="Sort in descending order by revenue",
                    descending=True,
                    field="properties.entity.revenue",
                    default_value=0,
                    inputs=[0],
                ),
                2: Limit(
                    node_id=2,
                    description="Get the 2 law firms with highest revenue",
                    num_records=2,
                    inputs=[1],
                ),
            },
        ),
    ),
]


class LlmPlanner:
    """The top-level query planner for SycamoreQuery. This class is responsible for generating
    a logical query plan from a user query using the OpenAI LLM.

    Args:
        index: The name of the index to query.
        data_schema: A dictionary mapping field names to their types.
        os_config: The OpenSearch configuration.
        os_client: The OpenSearch client.
        operators: A list of operators to use in the query plan.
        llm_client: The LLM client.
        examples: Query examples to assist the LLM planner in few-shot learning.
            You may override this to customize the few-shot examples provided to the planner.
        natural_language_response: Whether to generate a natural language response. If False,
            the response will be raw data.
    """

    def __init__(
        self,
        index: str,
        data_schema: OpenSearchSchema,
        os_config: dict[str, str],
        os_client: "OpenSearch",
        operators: Optional[List[Type[Node]]] = None,
        llm_client: Optional[LLM] = None,
        examples: Optional[List[PlannerExample]] = None,
        natural_language_response: bool = False,
    ) -> None:
        super().__init__()
        self._index = index
        self._data_schema = data_schema
        self._operators = operators if operators else OPERATORS
        self._os_config = os_config
        self._os_client = os_client
        self._llm_client = llm_client or OpenAI(OpenAIModels.GPT_4O.value)
        self._examples = PLANNER_EXAMPLES if examples is None else examples
        self._natural_language_response = natural_language_response

    def make_operator_prompt(self, operator: Type[Node]) -> str:
        """Generate the prompt fragment for the given Node."""

        prompt = operator.usage() + "\n\nInput schema:\n"
        schema = operator.input_schema()
        for field_name, value in schema.items():
            prompt += f"- {field_name} ({value.type_hint}): {value.description}\n"
        prompt += "\n------------\n"
        return prompt

    def make_schema_prompt(self, schema: OpenSearchSchema) -> str:
        """Generate the prompt fragment for the provided schema."""
        return schema.model_dump_json(indent=2)

    def make_examples_prompt(self) -> str:
        """Generate the prompt fragment for the query examples."""
        prompt = "\n\nThe following examples demonstrate how to construct query plans.\n\n-- BEGIN EXAMPLES --\n\n"
        schemas_shown = set()
        for example_index, example in enumerate(self._examples):
            # Avoid showing schema multiple times.
            schema_prompt = self.make_schema_prompt(example.schema)
            if schema_prompt not in schemas_shown:
                prompt += f"""
                The following is the data schema for the example queries below:
                -- Begin example schema --\n
                {schema_prompt}
                \n-- End of example schema --\n
                """
                schemas_shown.add(schema_prompt)

            prompt += f"EXAMPLE {example_index + 1}:\n"

            # Get the index name for the example from the first query node that references it.
            index_name_options = [
                example.plan.nodes[x].index  # type: ignore
                for x in example.plan.nodes.keys()
                if hasattr(example.plan.nodes[x], "index")
            ]
            if len(index_name_options) > 0:
                index_name = index_name_options[0]
                prompt += f"INDEX NAME: {index_name}\n"
            prompt += f"USER QUESTION: {example.plan.query}\n"
            prompt += f"Answer:\n{example.plan.model_dump_json(indent=2)}\n\n"
        prompt += "-- END EXAMPLES --\n\n"
        return prompt

    def generate_system_prompt(self, _query: str) -> str:
        """Generate the LLM system prompt for the given query."""

        # Initial prompt.
        prompt = PLANNER_SYSTEM_PROMPT

        if self._natural_language_response:
            prompt += PLANNER_NATURAL_LANGUAGE_PROMPT
        else:
            prompt += PLANNER_RAW_DATA_PROMPT
        prompt += "\n\n"

        # Few-shot examples.
        if self._examples:
            prompt += self.make_examples_prompt()

        # Operator definitions.
        prompt += """
        You may use the following operators to construct your query plan:

        OPERATORS:
        """
        for operator in self._operators:
            prompt += self.make_operator_prompt(operator)

        # Data schema.
        prompt += f"""\n\nThe following represents the schema of the data you should return a query plan for:

        INDEX_NAME: {self._index}
        DATA_SCHEMA:\n\n{self.make_schema_prompt(self._data_schema)}
        """

        return prompt

    def generate_user_prompt(self, query: str) -> str:
        """Generate the LLM user prompt for the given query."""

        prompt = f"""
        INDEX_NAME: {self._index}
        USER QUESTION: {query}
        Answer: """
        return prompt

    def generate_from_llm(self, question: str) -> Tuple[Any, str]:
        """Use LLM to generate a query plan for the given question.

        Returns the prompt sent to the LLM, and the plan.
        """

        messages = [
            {
                "role": "system",
                "content": self.generate_system_prompt(question),
            },
            {
                "role": "user",
                "content": self.generate_user_prompt(question),
            },
        ]

        prompt_kwargs = {"messages": messages}
        chat_completion = self._llm_client.generate(
            prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0}
        )
        return prompt_kwargs, chat_completion

    def plan(self, question: str) -> LogicalPlan:
        """Given a question from the user, generate a logical query plan."""
        llm_prompt, llm_plan = self.generate_from_llm(question)
        try:
            plan = process_json_plan(llm_plan)
        except Exception as e:
            logging.error(f"Error processing LLM-generated query plan: {e}\nPlan is:\n{llm_plan}")
            raise

        plan.query = question
        plan.llm_prompt = llm_prompt
        plan.llm_plan = llm_plan

        logging.debug(f"Query plan: {plan}")
        return plan
