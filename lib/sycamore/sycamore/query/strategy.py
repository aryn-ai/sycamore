import logging
from typing import Type, Optional, Any

from sycamore.llms import LLM
from sycamore.query.logical_plan import Node
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.basic_filter import BasicFilter
from sycamore.query.operators.count import Count
from sycamore.query.operators.limit import Limit
from sycamore.query.operators.llm_extract_entity import LlmExtractEntity
from sycamore.query.operators.llm_filter import LlmFilter
from sycamore.query.operators.math import Math
from sycamore.query.operators.query_database import QueryDatabase, QueryVectorDatabase
from sycamore.query.operators.sort import Sort
from sycamore.query.operators.summarize_data import SummarizeData
from sycamore.query.operators.top_k import TopK

ALL_OPERATORS: list[type[Node]] = [
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


class LogicalPlanProcessor:
    def __call__(self, plan: LogicalPlan) -> LogicalPlan:
        """Given the LogicalPlan query plan, postprocess it using a set of rules that modify the plan for
        optimizing or other purposes."""
        return plan


class RemoveVectorSearchForAnalytics(LogicalPlanProcessor):

    def __init__(self, llm: LLM) -> None:
        super().__init__()
        self.llm = llm

    def __call__(self, plan: LogicalPlan) -> LogicalPlan:
        logging.info("Executing DefaultLogicalPlanProcessor")

        # Rule: If the plan has a vector search in the beginning followed by a count or nothing or extract_entity,
        # we replace that by removing the vector search and adding an LLM Filter before the count

        if plan.nodes[0].node_type == "QueryVectorDatabase" and (
            len(plan.nodes) == 1 or plan.nodes[1].node_type in ["Count", "LlmExtractEntity", "SummarizeData"]
        ):
            logging.info("Plan eligible for rewrite, executing...")
            # If the first operator has an "opensearch_filter", we will convert it to a QueryDatabase with
            #        that opensearch_filter as "query"
            # If not, we do a QueryDatabase with "match_all" instead
            op: Any = plan.nodes[0]

            modified_description = self.postprocess_llm_helper(
                f"""
                            The following is the description of a Python function. I am modifying the function code
                            to remove any functionality that specifically has to do with "{op.query_phrase}", thereby 
                            generalizing the description to be more flexible.
                            Return only the modified description. Do not make assumptions
                            about the intent of the question that are not explicitly specified.
                            {op.description}
                            """,
            )

            new_op = QueryDatabase.model_validate(
                {
                    "node_id": op.node_id,
                    "node_type": "QueryDatabase",
                    "description": modified_description,
                    "index": op.index,
                }
            )

            if op.opensearch_filter:
                new_op.query = op.opensearch_filter
            else:
                new_op.query = {"match_all": {}}

            plan.replace_node(0, new_op)

            # Add an LLM Filter as the second operator
            llm_op_description = self.postprocess_llm_helper(
                f"""
                        Generate a one-line description for a python function whose goal is to filter the input
                        records based on whether they contain {op.query_phrase}.
                        Here are two example outputs: 
                        (1) Filter to records involving wildfires.
                        (2) Filter to records that occurred in Northwest USA.
                        """,
            )
            llm_op_question = self.postprocess_llm_helper(
                f"""
                        Generate a one-line true/false question that is appropriate to check whether an input document
                        satisfies {op.query_phrase}. Keep it as generic and short as possible. Do not make assumptions
                        about the intent of the question that are not explicitly specified.

                        Here are two examples:
                        (1) Was this incident caused by an environmental condition?
                        (2) Did this incident occur in Georgia?
                        """,
            )
            llm_op = LlmFilter.model_validate(
                {
                    "node_id": 1,
                    "node_type": "LlmFilter",
                    "description": llm_op_description,
                    "inputs": [0],
                    "field": "text_representation",
                    "question": llm_op_question,
                }
            )

            plan.insert_node(1, llm_op)

        return plan

    def postprocess_llm_helper(self, user_message: str) -> str:
        messages = [
            {
                "role": "system",
                "content": """You are a helpful agent that assists in small transformations
                        of input text as per the instructions. You should make minimal changes
                        to the provided input and keep your response short""",
            },
            {"role": "user", "content": user_message},
        ]

        prompt_kwargs = {"messages": messages}
        chat_completion = self.llm.generate(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0, "seed": 42})
        return chat_completion


class QueryPlanStrategy:
    """
    A strategy for generating a query plan. This strategy is responsible for providing available Operators and
    processing steps on LogicalPlans.
    """

    def __init__(
        self, operators: Optional[list[Type[Node]]] = None, post_processors: Optional[list[LogicalPlanProcessor]] = None
    ) -> None:
        super().__init__()
        self.operators: list[Type[Node]] = operators or []
        self.post_processors: list[LogicalPlanProcessor] = post_processors or []


class DefaultQueryPlanStrategy(QueryPlanStrategy):
    """
    Default strategy that uses all available tools and optimizes result correctness.
    """

    def __init__(self, llm: LLM, additional_post_processors: Optional[list[LogicalPlanProcessor]] = None) -> None:
        post_processors: list[LogicalPlanProcessor] = [RemoveVectorSearchForAnalytics(llm)]
        if additional_post_processors:
            post_processors.extend(additional_post_processors)
        super().__init__(ALL_OPERATORS, post_processors)


class VectorSearchOnlyStrategy(QueryPlanStrategy):
    """
    Strategy that only uses Vector Search for query plans to optimize for speed. This is useful when the data size
    is smaller and vector search retrievals are sufficient to provide answers.
    """

    def __init__(self, post_processors: Optional[list[LogicalPlanProcessor]] = None) -> None:
        super().__init__([op for op in ALL_OPERATORS if op not in {QueryDatabase}], post_processors or [])
