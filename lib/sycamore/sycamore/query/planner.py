from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Type

from opensearchpy import OpenSearch

from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.count import Count
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.basic_filter import BasicFilter
from sycamore.query.operators.summarize_data import SummarizeData
from sycamore.query.operators.query_database import QueryDatabase
from sycamore.query.operators.llm_extract_entity import LlmExtractEntity
from sycamore.query.operators.math import Math
from sycamore.query.operators.sort import Sort
from sycamore.query.operators.top_k import TopK
from sycamore.query.operators.limit import Limit
from sycamore.query.operators.logical_operator import LogicalOperator
from sycamore.utils.extract_json import extract_json


# All operators that are allowed for construction of a query plan.
# If a class is not in this list, it will not be used.
OPERATORS: List[Type[LogicalOperator]] = [
    QueryDatabase,
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


class LlmPlanner:
    """The top-level query planner for SycamoreQuery. This class is responsible for generating
    a logical query plan from a user query using the OpenAI LLM.

    Args:
        index: The name of the index to query.
        data_schema: A dictionary mapping field names to their types.
        os_config: The OpenSearch configuration.
        os_client: The OpenSearch client.
        operators: A list of operators to use in the query plan.
        openai_client: The OpenAI client.
        use_examples: Whether to include examples in the prompt.
    """

    def __init__(
        self,
        index: str,
        data_schema: dict[str, Any],
        os_config: dict[str, str],
        os_client: OpenSearch,
        operators: Optional[List[Type[LogicalOperator]]] = None,
        openai_client: Optional[OpenAI] = None,
        use_examples: bool = True,
    ) -> None:
        super().__init__()
        self._index = index
        self._data_schema = data_schema
        self._operators = operators if operators else OPERATORS
        self._os_config = os_config
        self._os_client = os_client
        self._openai_client = openai_client or OpenAI(OpenAIModels.GPT_4O.value)
        self._use_examples = use_examples

    def make_operator_prompt(self, operator: LogicalOperator) -> str:
        """Generate the prompt fragment for the given LogicalOperator."""

        prompt = operator.usage() + "\n\nInput schema:\n"
        schema = operator.input_schema()
        for field_name, value in schema.items():
            prompt += f"- {field_name} ({value.type_hint}): {value.description}\n"
        prompt += "\n------------\n"
        return prompt

    def generate_prompt(self, query):
        """Generate the LLM prompt for the given query."""

        prompt = """
        1. Return your answer as a standard JSON list of operators. Make sure to include each
            operation as a separate step.
        2. Do not return any information except the standard JSON objects.
        3. Only use operators described below.
        4. Only use EXACT field names from the DATA_SCHEMA described below and fields created
            from *LlmExtractEntity*. Any new fields created by *LlmExtractEntity* will be nested in properties.
            e.g. if a new field called "state" is added, when referencing it in another operation,
            you should use "properties.state". A database returned from *TopK* operation only has
            "properties.key" or "properties.count"; you can only reference one of those fields.
            Other than those, DO NOT USE ANY OTHER FIELD NAMES.
        5. If an optional field does not have a value in the query plan, return null in its place.
        6. If you cannot generate a plan to answer a question, return an empty list.
        7. The first step of each plan MUST be a **QueryDatabase** operation that returns a database.
        8. The last step of each plan MUST be a **SummarizeData** operation to generate an English
            answer.
        """

        # data schema
        prompt += f"""
        INDEX_NAME: {self._index}
        """
        prompt += f"""
        DATA_SCHEMA:
        {self._data_schema}
        """

        # operator definitions
        prompt += """
        OPERATORS:
        """
        for operator in self._operators:
            prompt += self.make_operator_prompt(operator)

        # examples
        if self._use_examples:
            prompt += """
            EXAMPLE 1:

            Data description: Database of aircraft incidents
            Schema: {
                        'properties.entity.date': "(<class 'str'>) e.g. (2023-01-14), (2023-01-14), (2023-01-29),
                        'properties.entity.aircraft': "(<class 'int'>) e.g. (Boeing 123), (Cessna Mini 5), (Piper 0.5),
                        'properties.entity.location': "(<class 'str'>) e.g. (Atlanta, GA), (Phoenix, Arizona), 
                            (Boise, Idaho),
                        'properties.entity.accidentNumber': "(<class 'str'>) e.g. (3589), (5903), (7531L),
                        'text_representation': '(<class 'str'>) Can be assumed to have all other details'
                    }
            Question: Were there any incidents in Georgia?
            Answer:
            [
                {
                    "operatorName": "QueryDatabase",
                    "description": "Get all the incident reports",
                    "index": "ntsb",
                    "query": "aircraft incident reports"
                    "node_id": 0
                },
                {
                    "operatorName": "LlmFilter",
                    "description": "Filter to only include incidents in Georgia",
                    "question": "Did this incident occur in Georgia?",
                    "field": "properties.entity.location",
                    "input": [0],
                    "node_id": 1
                },
                {
                    "operatorName": "SummarizeData",
                    "description": "Generate an English response to the original question.
                        Input 1 is a database that contains incidents in Georgia.",
                    "question": "Were there any incidents in Georgia?",
                    "input": [1],
                    "node_id": 2
                }
            ]

            EXAMPLE 2:
            Data description: Database of aircraft incidents
            Schema: {
                        'properties.entity.date': "(<class 'str'>) e.g. (2023-01-14), (2023-01-14), (2023-01-29),
                        'properties.entity.aircraft': "(<class 'int'>) e.g. (Boeing 123), (Cessna Mini 5), (Piper 0.5),
                        'properties.entity.city': "(<class 'str'>) e.g. (Orlando, FL), (Palo Alto, CA), (Orlando, FL),
                        'properties.entity.accidentNumber': "(<class 'str'>) e.g. (3589), (5903), (7531L),
                        'text_representation': '(<class 'str'>) Can be assumed to have all other details'
                    }
            Question: How many cities did Cessna aircrafts have incidents in?
            Answer:
            [
                {
                    "operatorName": "QueryDatabase",
                    "description": "Get all the incident reports",
                    "index": "ntsb",
                    "query": "aircraft incident reports",
                    "node_id": 0
                },
                {
                    "operatorName": "BasicFilter",
                    "description": "Filter to only include Cessna aircraft incidents",
                    "rangeFilter": false,
                    "query": "Cessna",
                    "start": null,
                    "end": null,
                    "field": "properties.entity.aircraft",
                    "date": false,
                    "input": [0],
                    "node_id": 1,
                },
                {
                    "operatorName": "Count",
                    "description": "Count the number of cities that accidents occured in",
                    "field": "properties.entity.city",
                    "primary_field": "properties.entity.accidentNumber",
                    "input": [1],
                    "node_id": 2
                },
                {
                    "operatorName": "SummarizeData",
                    "description": "description": "Generate an English response to the
                        question. Input 1 is a number that corresponds to the number of
                        cities that accidents occurred in.",
                    "question": "How many cities did Cessna aircrafts have incidents in?",
                    "input": [2],
                    "node_id": 3
                }
            ]

            EXAMPLE 3:
            Data description: Database of financial documents for different law firms
            Schema: {
                        'properties.entity.date': "(<class 'str'>) e.g. (2023-01-14), (2023-01-14), (2023-01-29),
                        'properties.entity.revenue': "(<class 'int'>) e.g. (12304), (7978234), (2938903),
                        'properties.entity.firmName': "(<class 'str'>) e.g. (East West), (Brody), (Hunter & Hunter),
                        'text_representation': '(<class 'str'>) Can be assumed to have all other details'
                    }
            Question: Which 2 law firms had the highest revenue in 2022?
            Answer:
            [
                {
                    "operatorName": "QueryDatabase",
                    "description": "Get all the financial documents",
                    "index": "finance",
                    "query": "law firm financial documents",
                    "node_id": 0
                },
                {
                    "operatorName": "BasicFilter",
                    "description": "Filter to only include documents in 2022",
                    "rangeFilter": true,
                    "query": null,
                    "start": "01-01-2022",
                    "end": "12-31-2022",
                    "field": "properties.entity.date",
                    "date": true,
                    "input": [0],
                    "node_id": 1,
                },
                {
                    "operatorName": "Sort",
                    "description": "Sort in descending order by revenue",
                    "descending": true,
                    "field": "properties.entity.revenue",
                    "default_value": 0
                    "input": [1],
                    "node_id": 2,
                },
                {
                    "operatorName": "Limit",
                    "description": "Get the 2 law firms with highest revenue",
                    "K": 2
                    "input": [2],
                    "node_id": 3,
                }
                {
                    "operatorName": "SummarizeData",
                    "description": "description": "Generate an English response to
                        the question. Input 1 is a database that contains information
                        about the 2 law firms with the highest revenue.",
                    "question": "Which 2 law firms had the highest revenue in 2022?",
                    "input": [3],
                    "node_id": 4
                }
            ]

            EXAMPLE 4:
            Data description: Database of shipwreck records and their respective properties
            Schema: {
                        'properties.entity.date': "(<class 'str'>) e.g. (2023-01-14), (2023-01-14), (2023-01-29),
                        'properties.entity.captain': "(<class 'str'>) e.g. (John D. Moore), (Terry Roberts), 
                            (Alex Clark),
                        'properties.entity.shipwreck_id': "(<class 'str'>) e.g. (ABFUHEU), (FUIHWHD), (FGHIOWB),
                        'text_representation': '(<class 'str'>) Can be assumed to have all other details'
                    }
            Question: Which 5 countries were responsible for the most shipwrecks?
            Answer:
            [
                {
                    "operatorName": "QueryDatabase",
                    "description": "Get all the shipwreck records",
                    "index": "shipwrecks",
                    "query": "shipwreck records",
                    "node_id": 0
                },
                {
                    "operatorName": "LlmExtractEntity",
                    "description": "Extract the country",
                    "question": "What country was responsible for this ship?",
                    "field": "text_representation",
                    "new_field": "country",
                    "format": "string",
                    "discrete": true,
                    "input": [0],
                    "node_id": 1
                },
                {
                    "operatorName": '"TopK"',
                    "description": "Gets top 5 water bodies based on shipwrecks",
                    "field": "properties.country",
                    "primary_field": "properties.entity.shipwreck_id",
                    "K": 5,
                    "descending": true,
                    "useLLM": false,
                    "input": [1],
                    "node_id": 2,
                },
                {
                    "operatorName": "SummarizeData",
                    "description": "description": "Generate an English response to the
                        question. Input 1 is a database that the top 5 water bodies shipwrecks
                        occurred in and their corresponding frequency counts.",
                    "question": "Which 5 countries were responsible for the most shipwrecks?",
                    "input": [2],
                    "node_id": 3
                }
            ]

            EXAMPLE 5:
            Data description: Database of shipwreck records and their respective properties
            Schema: {
                        'properties.entity.date': "(<class 'str'>) e.g. (2023-01-14), (2023-01-14), (2023-01-29),
                        'properties.entity.shipwreck_id': "(<class 'str'>) e.g. (ABFUHEU), (FUIHWHD), (FGHIOWB),
                        'text_representation': '(<class 'str'>) Can be assumed to have all other details'
                    }
            Question: What percent of shipwrecks occurred in 2023?
            Answer:
            [
                {
                    "operatorName": "QueryDatabase",
                    "description": "Get all the shipwreck records",
                    "index": "shipwrecks",
                    "query": "shipwreck records",
                    "node_id": 0
                },
                {
                    "operatorName": "Count",
                    "description": "Count the number of total shipwrecks",
                    "field": null,
                    "primary_field": "properties.entity.shipwreck_id",
                    "input": [0],
                    "node_id": 1
                },
                {
                    "operatorName": "BasicFilter",
                    "description": "Filter to only include documents in 2023",
                    "rangeFilter": true,
                    "query": null,
                    "start": "01-01-2023",
                    "end": "12-31-2023",
                    "field": "properties.entity.date",
                    "date": true,
                    "input": [0],
                    "node_id": 2,
                },
                {
                    "operatorName": "Count",
                    "description": "Count the number of shipwrecks in 2023",
                    "field": null,
                    "primary_field": "properties.entity.shipwreck_id",
                    "input": [2],
                    "node_id": 3
                },
                {
                    "operatorName": "Math",
                    "description": "Divide the number of shipwrecks in 2023 by the total number",
                    "type": "divide",
                    "input": [3, 1],
                    "node_id": 4
                }
                {
                    "operatorName": "SummarizeData",
                    "description": "Generate an English response to the question. Input 1 is a
                        number that is the fraction of shipwrecks that occurred in 2023.",
                    "question": "What percent of shipwrecks occurred in 2023?",
                    "input": [4],
                    "node_id": 5
                }
            ]

            EXAMPLE 5:
            Data description: Database of hospital patients
            Schema: {
                        'text_representation': '(<class 'str'>) Can be assumed to have all other details'
                    }
            Question: How many total patients?
            Answer:
            [
                {
                    "operatorName": "QueryDatabase",
                    "description": "Get all the patient records",
                    "index": "patients",
                    "query": "patient records",
                    "id": 0
                },
                {
                    "operatorName": "Count",
                    "description": "Count the number of total patients",
                    "field": null,
                    "primaryField": null,
                    "input": [0],
                    "id": 1
                },
                {
                    "operatorName": "SummarizeData",
                    "description": "Generate an English response to the question. Input 1 is a
                        number of patients.",
                    "question": "How many total patients?",
                    "input": [1],
                    "id": 2
                }
            ]
            """

        # input
        prompt += f"""
        USER QUESTION: {query}
        """
        return prompt

    def generate_from_openai(self, question: str) -> str:
        """Use OpenAI LLM to generate a query plan for the given question."""

        messages = [
            {
                "role": "user",
                "content": self.generate_prompt(question),
            }
        ]

        prompt_kwargs = {"messages": messages}
        chat_completion = self._openai_client.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})
        return chat_completion

    def process_llm_json_plan(self, llm_json_plan: str) -> Tuple[LogicalOperator, Mapping[int, LogicalOperator]]:
        """Given the query plan provided by the LLM, return a tuple of (result_node, list of nodes)."""

        classes = globals()
        parsed_plan = extract_json(llm_json_plan)
        assert isinstance(parsed_plan, list), f"Expected LLM query plan to contain a list, got f{type(parsed_plan)}"

        nodes: MutableMapping[int, LogicalOperator] = {}
        downstream_dependencies: Dict[int, List[int]] = {}

        # 1. Build nodes
        for step in parsed_plan:
            node_id = step["node_id"]
            cls = classes.get(step["operatorName"])
            if cls is None:
                raise ValueError(f"Operator {step['operatorName']} not found")
            if cls not in self._operators:
                raise ValueError(f"Operator {step['operatorName']} is not a valid operator")
            try:
                node = cls(**step)
                nodes[node_id] = node
            except Exception as e:
                raise ValueError(f"Error creating node {node_id} of type {step['operatorName']}: {e}") from e

        # 2. Set dependencies
        for node_id, node in nodes.items():
            if not node.input:
                continue
            inputs = []
            for dependency_id in node.input:
                downstream_dependencies[dependency_id] = downstream_dependencies.get(dependency_id, []) + [node]
                inputs += [nodes.get(dependency_id)]
            # pylint: disable=protected-access
            node._dependencies = inputs

        # 3. Set downstream nodes
        for node_id, node in nodes.items():
            if node_id in downstream_dependencies.keys():
                # pylint: disable=protected-access
                node._downstream_nodes = downstream_dependencies[node_id]

        # pylint: disable=protected-access
        resultNodes = list(filter(lambda n: n._downstream_nodes is None, nodes.values()))
        if len(resultNodes) == 0:
            raise Exception("Invalid plan: Plan requires at least one terminal node")
        return resultNodes[0], nodes

    def plan(self, question: str) -> LogicalPlan:
        """Given a question from the user, generate a logical query plan."""
        openai_plan = self.generate_from_openai(question)
        result_node, nodes = self.process_llm_json_plan(openai_plan)
        plan = LogicalPlan(result_node=result_node, nodes=nodes, query=question, openai_plan=openai_plan)
        return plan
