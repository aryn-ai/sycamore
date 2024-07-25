from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch

from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.count import Count
from sycamore.query.operators.llmfilter import LlmFilter
from sycamore.query.operators.filter import Filter
from sycamore.query.operators.llmgenerate import LlmGenerate
from sycamore.query.operators.loaddata import LoadData
from sycamore.query.operators.llmextract import LlmExtract
from sycamore.query.operators.math import Math
from sycamore.query.operators.sort import Sort
from sycamore.query.operators.topk import TopK
from sycamore.query.operators.limit import Limit
from sycamore.query.operators.logical_operator import LogicalOperator
from sycamore.utils.extract_json import extract_json

OPERATORS = [LoadData, Filter, LlmFilter, LlmExtract, Count, LlmGenerate, Math, Sort, TopK, Limit]


class LlmPlanner:

    def __init__(
        self,
        index: str,
        data_schema: dict[str, Any],
        os_config: dict[str, str],
        os_client: OpenSearch,
        operators: Optional[List[LogicalOperator]] = None,
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

    def generate_prompt(self, query):
        prompt = """
        1. Return your answer as a standard JSON list of operators. Make sure to include each 
            operation as a separate step.
        2. Do not return any information except the standard JSON objects.
        3. Only use operators described below.
        4. Only use EXACT field names from the DATA_SCHEMA described below and fields created 
            from *LlmExtract*. Any new fields created by *LlmExtract* will be nested in properties. 
            e.g. if a new field called "state" is added, when referencing it in another operation, 
            you should use "properties.state". A database returned from *TopK* operation only has 
            "properties.key" or "properties.count"; you can only reference one of those fields. 
        5. If an optional field does not have a value in the query plan, return null in its place.
        6. If you cannot generate a plan to answer a question, return an empty list.
        7. The first step of each plan MUST be a **LoadData** operation that returns a database.
        8. The last step of each plan MUST be a **LlmGenerate** operation to generate an English 
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
        for operator in OPERATORS:
            prompt += f"""
            {operator.description()}
            
            Definition:
            {operator.input_schema()}
            ------------------
            
            """

        # examples
        if self._use_examples:
            prompt += """
            EXAMPLE 1:

            Data description: Database of aircraft incidents
            Question: Were there any incidents in Georgia?
            Answer:
            [
                {
                    "operatorName": "LoadData",
                    "description": "Get all the incident reports",
                    "index": "ntsb",
                    "query": "aircraft incident reports"
                    "id": 0
                },
                {
                    "operatorName": "LlmFilter",
                    "description": "Filter to only include incidents in Georgia",
                    "question": "Did this incident occur in Georgia?",
                    "field": "properties.entity.location",
                    "input": [0],
                    "id": 1
                },
                {
                    "operatorName": "LlmGenerate",
                    "description": "Generate an English response to the original question. 
                        Input 1 is a database that contains incidents in Georgia.",
                    "question": "Were there any incidents in Georgia?",
                    "input": [1],
                    "id": 2
                }
            ]
            
            EXAMPLE 2:
            Data description: Database of aircraft incidents
            Question: How many cities did Cessna aircrafts have incidents in?
            Answer:
            [
                {
                    "operatorName": "LoadData",
                    "description": "Get all the incident reports",
                    "index": "ntsb",
                    "query": "aircraft incident reports",
                    "id": 0
                },
                {
                    "operatorName": "LlmFilter",
                    "description": "Filter to only include Cessna aircraft incidents",
                    "question": "Did this incident occur in a Cessna aircraft?",
                    "field": "properties.entity.aircraft",
                    "input": [0],
                    "id": 1
                },
                {
                    "operatorName": "Count",
                    "description": "Count the number of cities that accidents occured in",
                    "field": "properties.entity.city",
                    "primaryField": "properties.entity.accidentNumber",
                    "input": [1],
                    "id": 2
                },
                {
                    "operatorName": "LlmGenerate",
                    "description": "description": "Generate an English response to the 
                        question. Input 1 is a number that corresponds to the number of 
                        cities that accidents occurred in.",
                    "question": "How many cities did Cessna aircrafts have incidents in?",
                    "input": [2],
                    "id": 3
                }
            ]

            EXAMPLE 3:
            Data description: Database of financial documents for different law firms
            Question: Which 2 law firms had the highest revenue in 2022?
            Answer:
            [
                {
                    "operatorName": "LoadData",
                    "description": "Get all the financial documents",
                    "index": "finance",
                    "query": "law firm financial documents",
                    "id": 0
                },
                {
                    "operatorName": "Filter",
                    "description": "Filter to only include documents in 2022",
                    "rangeFilter": true,
                    "query": null,
                    "start": "01-01-2022",
                    "end": "12-31-2022",
                    "field": "properties.entity.date",
                    "date": true,
                    "input": [0],
                    "id": 1,
                },
                    "operatorName": "Sort",
                    "description": "Sort in descending order by revenue",
                    "descending": true,
                    "field": "properties.entity.revenue",
                    "defaultValue": 0
                    "input": [1],
                    "id": 2,
                },
                {
                    "operatorName": "Limit",
                    "description": "Get the 2 law firms with highest revenue",
                    "K": 2
                    "input": [2],
                    "id": 3,
                }
                {
                    "operatorName": "LlmGenerate",
                    "description": "description": "Generate an English response to 
                        the question. Input 1 is a database that contains information 
                        about the 2 law firms with the highest revenue.",
                    "question": "Which 2 law firms had the highest revenue in 2022?",
                    "input": [3],
                    "id": 4
                }
            ]

            EXAMPLE 4:
            Data description: Database of shipwreck records and their respective properties
            Question: Which 5 countries were responsible for the most shipwrecks?
            Answer:
            [
                {
                    "operatorName": "LoadData",
                    "description": "Get all the shipwreck records",
                    "index": "shipwrecks",
                    "query": "shipwreck records",
                    "id": 0
                },
                {
                    "operatorName": "LlmExtract",
                    "description": "Extract the country",
                    "question": "What country was responsible for this ship?",
                    "field": "text_representation",
                    "newField": "country",
                    "format": "string",
                    "discrete": true,
                    "input": [0],
                    "id": 1
                },
                {
                    "operatorName": '"TopK"',
                    "description": "Gets top 5 water bodies based on shipwrecks",
                    "field": "properties.country",
                    "primaryField": "properties.entity.shipwreck_id",
                    "K": 5,
                    "descending": true,
                    "useLLM": false,
                    "input": [1],
                    "id": 2,
                },
                {
                    "operatorName": "LlmGenerate",
                    "description": "description": "Generate an English response to the 
                        question. Input 1 is a database that the top 5 water bodies shipwrecks 
                        occurred in and their corresponding frequency counts.",
                    "question": "Which 5 countries were responsible for the most shipwrecks?",
                    "input": [2],
                    "id": 3
                }
            ]

            EXAMPLE 5:
            Data description: Database of shipwreck records and their respective properties
            Question: What percent of shipwrecks occurred in 2023?
            Answer:
            [
                {
                    "operatorName": "LoadData",
                    "description": "Get all the shipwreck records",
                    "index": "shipwrecks",
                    "query": "shipwreck records",
                    "id": 0
                },
                {
                    "operatorName": "Count",
                    "description": "Count the number of total shipwrecks",
                    "field": null,
                    "primaryField": "properties.entity.shipwreck_id",
                    "input": [0],
                    "id": 1
                },
                {
                    "operatorName": "Filter",
                    "description": "Filter to only include documents in 2023",
                    "rangeFilter": true,
                    "query": null,
                    "start": "01-01-2023",
                    "end": "12-31-2023",
                    "field": "properties.entity.date",
                    "date": true,
                    "input": [0],
                    "id": 2,
                },
                {
                    "operatorName": "Count",
                    "description": "Count the number of shipwrecks in 2023",
                    "field": null,
                    "primaryField": "properties.entity.shipwreck_id",
                    "input": [2],
                    "id": 3
                },
                {
                    "operatorName": "Math", 
                    "description": "Divide the number of shipwrecks in 2023 by the total number",
                    "type": "divide", 
                    "input": [3, 1], 
                    "id": 4
                }
                {
                    "operatorName": "LlmGenerate",
                    "description": "Generate an English response to the question. Input 1 is a 
                        number that is the fraction of shipwrecks that occurred in 2023.",
                    "question": "What percent of shipwrecks occurred in 2023?",
                    "input": [4],
                    "id": 5
                }
            ]
            """

        # input
        prompt += f"""
        USER QUESTION: {query}
        """
        return prompt

    def generate_from_openai(self, question):
        messages = [
            {
                "role": "user",
                "content": self.generate_prompt(question),
            }
        ]

        prompt_kwargs = {"messages": messages}

        # call to LLM
        chat_completion = self._openai_client.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={})

        # for chunk in stream:
        #     print(chunk.choices[0].delta.content or "", end="")

        return chat_completion

    def process_llm_json_plan(self, llm_json_plan: str):
        # ugly
        classes = globals()

        # parse string as json
        parsed_plan = extract_json(llm_json_plan)
        assert isinstance(parsed_plan, list)

        nodes: Dict[str, LogicalOperator] = dict()
        downstream_dependencies: Dict[str, List[int]] = dict()

        # 1. Build nodes
        for step in parsed_plan:
            node_id = step["id"]
            cls = classes.get(step["operatorName"])
            if cls is None:
                raise ValueError(f"Operator {step['operatorName']} not found")
            node = cls(node_id, step)
            node.description = step.get("description", "")
            nodes[node_id] = node

        # 2. Set dependencies
        for node_id, node in nodes.items():
            if "input" not in node.data:
                continue
            inputs = []
            for dependency_id in node.data["input"]:
                downstream_dependencies[dependency_id] = downstream_dependencies.get(dependency_id, list()) + [node_id]
                inputs += [nodes.get(dependency_id)]
            node.dependencies = inputs

        for node_id, node in nodes.items():
            if node_id in downstream_dependencies:
                node.downstream_nodes = downstream_dependencies[node_id]

        resultNodes = list(filter(lambda n: n.downstream_nodes is None, nodes.values()))
        if len(resultNodes) == 0:
            raise Exception("Invalid plan: Plan requires at least one LlmGenerate result node")
        return resultNodes[0], nodes

    def plan(self, question: str) -> LogicalPlan:
        openai_plan = self.generate_from_openai(question)
        result_node, nodes = self.process_llm_json_plan(openai_plan)
        plan = LogicalPlan(result_node=result_node, nodes=nodes, query=question, openai_plan=openai_plan)
        return plan
