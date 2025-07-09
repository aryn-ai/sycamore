from sycamore.llms.prompts.prompts import (
    SycamorePrompt,
    RenderedPrompt,
    RenderedMessage,
)
from sycamore.schema import Schema, SchemaField
from sycamore.query.schema import OpenSearchSchema
from sycamore.query.logical_plan import LogicalPlan, Node

from sycamore.query.operators.query_database import QueryVectorDatabase, QueryDatabase
from sycamore.query.operators.count import Count
from sycamore.query.operators.llm_extract_entity import LlmExtractEntity
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.basic_filter import BasicFilter
from sycamore.query.operators.top_k import TopK
from sycamore.query.operators.math import Math
from sycamore.query.operators.limit import Limit
from sycamore.query.operators.sort import Sort

from typing import Optional, Union, List, Type
from dataclasses import dataclass


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
        6. The first step of each plan MUST be a **QueryDatabase** or **QueryVectorDatabase" operation.
            Whenever possible, include all possible filtering operations in the first step.
           That is, you should strive to construct an OpenSearch query that filters the data as
           much as possible, reducing the need for further query operations. If using a QueryVectorDatabase, always
              follow it with an LlmFilter operation to ensure the final results are accurate.
"""

# Variants on the last step in the query plan, based on whether the user has requested raw data
# or a natural language response.
PLANNER_RAW_DATA_PROMPT = "7. The last step of each plan should return the raw data associated with the response."
PLANNER_NATURAL_LANGUAGE_PROMPT = (
    "7. The last step of each plan *MUST* be a **SummarizeData** operation that returns a natural language response."
)


@dataclass
class PlannerExample:
    """Represents an example query and query plan for the planner."""

    def __init__(self, schema: Union[OpenSearchSchema, Schema], plan: LogicalPlan) -> None:
        super().__init__()
        self.plan = plan
        self.schema: Schema = schema.to_schema() if isinstance(schema, OpenSearchSchema) else schema


# Example schema and planner examples for the NTSB and financial datasets.
EXAMPLE_NTSB_SCHEMA = Schema(
    fields=[
        SchemaField(
            name="properties.path",
            field_type="str",
            examples=["/docs/incident1.pdf", "/docs/incident2.pdf", "/docs/incident3.pdf"],
        ),
        SchemaField(name="properties.entity.date", field_type="date", examples=["2023-07-01", "2024-09-01"]),
        SchemaField(name="properties.entity.accidentNumber", field_type="str", examples=["1234", "5678", "91011"]),
        SchemaField(
            name="properties.entity.location",
            field_type="str",
            examples=["Atlanta, Georgia", "Miami, Florida", "San Diego, California"],
        ),
        SchemaField(name="properties.entity.aircraft", field_type="str", examples=["Cessna", "Boeing", "Airbus"]),
        SchemaField(name="properties.entity.city", field_type="str", examples=["Atlanta", "Savannah", "Augusta"]),
        SchemaField(
            name="text_representation", field_type="str", examples=["Can be assumed to have all other details"]
        ),
    ]
)

EXAMPLE_FINANCIAL_SCHEMA = Schema(
    fields=[
        SchemaField(name="properties.path", field_type="str", examples=["doc1.pdf", "doc2.pdf", "doc3.pdf"]),
        SchemaField(
            name="properties.entity.date", field_type="str", examples=["2022-01-01", "2022-12-31", "2023-01-01"]
        ),
        SchemaField(
            name="properties.entity.revenue", field_type="float", examples=["1000000.0", "2000000.0", "3000000.0"]
        ),
        SchemaField(
            name="properties.entity.firmName",
            field_type="str",
            examples=["Dewey, Cheatem, and Howe", "Saul Goodman & Associates", "Wolfram & Hart"],
        ),
        SchemaField(
            name="text_representation", field_type="str", examples=["Can be assumed to have all other details"]
        ),
    ]
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
                    query={"match_phrase": {"properties.entity.location": "Georgia"}},
                )
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
 .         number containing 'K1234N' that occurred in New Mexico.""",
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
                                {"match_phrase": {"properties.entity.location": "New Mexico"}},
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
                    query={"match_phrase": {"properties.entity.aircraft": "Cessna"}},
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
                    description="Get some incidents relating to sudden weather changes",
                    index="ntsb",
                    query_phrase="sudden weather changes",
                ),
                1: LlmFilter(
                    node_id=1,
                    description="Filter to only include incidents caused due to sudden weather changes",
                    question="Did this incident occur due to sudden weather changes?",
                    field="text_representation",
                    inputs=[0],
                ),
            },
        ),
    ),
    PlannerExample(
        schema=EXAMPLE_NTSB_SCHEMA,
        plan=LogicalPlan(
            query="Show me some incidents relating to water causes",
            result_node=0,
            nodes={
                0: QueryVectorDatabase(
                    node_id=0,
                    description="Get some incidents relating to water causes",
                    index="ntsb",
                    query_phrase="water causes",
                ),
                1: LlmFilter(
                    node_id=1,
                    description="Filter to only include incidents that occur due to water causes",
                    question="Did this incident occur because of water causes",
                    field="text_representation",
                    inputs=[0],
                ),
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


class PlannerPrompt(SycamorePrompt):
    def __init__(
        self,
        query: Optional[str] = None,
        examples: List[PlannerExample] = [],
        natural_language_response: bool = True,
        operators: List[Type[Node]] = [],
        index: Optional[str] = None,
        data_schema: Optional[Schema] = None,
        planner_system_prompt: str = PLANNER_SYSTEM_PROMPT,
        planner_natural_language_prompt: str = PLANNER_NATURAL_LANGUAGE_PROMPT,
        planner_raw_data_prompt: str = PLANNER_RAW_DATA_PROMPT,
        planner_tail_prompt: Optional[str] = None,
    ):
        self.query = query
        self.examples = examples
        self.natural_language_response = natural_language_response
        self.operators = operators
        self.index = index
        self.data_schema = data_schema
        self.planner_system_prompt = planner_system_prompt
        self.planner_natural_language_prompt = planner_natural_language_prompt
        self.planner_raw_data_prompt = planner_raw_data_prompt
        self.planner_tail_prompt = planner_tail_prompt

    @staticmethod
    def make_operator_prompt(operator: type[Node]) -> str:
        """Generate the prompt fragment for the given Node."""

        prompt = operator.usage() + "\n\nInput schema:\n"
        schema = operator.input_schema()
        for field_name, value in schema.items():
            prompt += f"- {field_name} ({value.type_hint}): {value.description}\n"
        prompt += "\n------------\n"
        return prompt

    @staticmethod
    def make_schema_prompt(schema: Schema) -> str:
        """Generate the prompt fragment for the provided schema."""
        return schema.model_dump_json(indent=2)

    def make_examples_prompt(self) -> str:
        """Generate the prompt fragment for the query examples."""
        prompt = "\n\nThe following examples demonstrate how to construct query plans.\n\n-- BEGIN EXAMPLES --\n\n"
        schemas_shown = set()
        for example_index, example in enumerate(self.examples):
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
        assert self.data_schema is not None, "data_schema is not set yet, cannot generate prompt"

        # Initial prompt.
        prompt = self.planner_system_prompt

        if self.natural_language_response:
            prompt += self.planner_natural_language_prompt
        else:
            prompt += self.planner_raw_data_prompt
        prompt += "\n\n"

        # Few-shot examples.
        if self.examples:
            prompt += self.make_examples_prompt()

        # Operator definitions.
        prompt += """
        You may use the following operators to construct your query plan:

        OPERATORS:
        """
        for operator in self.operators:
            prompt += self.make_operator_prompt(operator)

        # Data schema.
        prompt += f"""\n\nThe following represents the schema of the data you should return a query plan for:

        INDEX_NAME: {self.index}
        DATA_SCHEMA:\n\n{self.make_schema_prompt(self.data_schema)}
        """

        if self.planner_tail_prompt:
            prompt += f"\n{self.planner_tail_prompt}"

        return prompt

    def generate_user_prompt(self, query: str) -> str:
        """Generate the LLM user prompt for the given query."""

        prompt = f"""
        INDEX_NAME: {self.index}
        USER QUESTION: {query}
        Answer: """
        return prompt

    def render(self) -> RenderedPrompt:
        assert self.query is not None, "Query is not set. Please set it with prompt.fork(query=...)"
        sys = self.generate_system_prompt(self.query)
        usr = self.generate_user_prompt(self.query)
        return RenderedPrompt(
            messages=[
                RenderedMessage(role="system", content=sys),
                RenderedMessage(role="user", content=usr),
            ]
        )
