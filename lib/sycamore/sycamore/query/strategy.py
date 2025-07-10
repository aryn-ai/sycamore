import logging
from typing import Type, Optional, Any

from sycamore.llms import LLM
from sycamore.llms.prompts.prompts import SycamorePrompt
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
from sycamore.query.planner_prompt import PlannerPrompt

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
    def __call__(self, plan: LogicalPlan, **kwargs) -> LogicalPlan:
        """Given the LogicalPlan query plan, postprocess it using a set of rules that modify the plan for
        optimizing or other purposes."""
        return plan


class PlannerPromptProcessor:
    def __call__(self, prompt: PlannerPrompt, **kwargs) -> PlannerPrompt:
        """Apply code to change the prompt used by the planner"""
        return prompt


class DefaultPlanValidator(LogicalPlanProcessor):
    """
    Performs a sequence of checks on the LogicalPlan to ensure it is valid and can be executed. These include:
    1. Type validation: ensure inputs to nodes are of valid types (e.g. DocSet, int, None, etc.)
    """

    def __call__(self, plan: LogicalPlan, **kwargs) -> LogicalPlan:
        logging.info("Executing DefaultPlanValidator processor")

        # type validation
        type_errors: list[str] = []
        for node_id, node in plan.nodes.items():
            for input_node in node.input_nodes():
                if not any(issubclass(input_node.output_type, input_type) for input_type in node.input_types):
                    type_errors += [
                        f"Node {node_id} ({node.node_type}): Invalid input type {input_node.output_type} "
                        f"from input {input_node.node_id} ({input_node.node_type}). "
                        f"Supported types: {node.input_types}"
                    ]
        if type_errors:
            errors_message = "\n".join(type_errors)
            raise TypeError(f"Invalid plan: \n{errors_message}")
        return plan


class RemoveVectorSearchForAnalytics(LogicalPlanProcessor):

    def __init__(self, llm: LLM) -> None:
        super().__init__()
        self.llm = llm

    def __call__(self, plan: LogicalPlan, **kwargs) -> LogicalPlan:
        logging.info("Executing RemoveVectorSearchForAnalytics processor")

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
        chat_completion = self.llm.generate_old(prompt_kwargs=prompt_kwargs, llm_kwargs={"temperature": 0})
        return chat_completion


class AlwaysSummarize(LogicalPlanProcessor):
    """
    A logical plan processor that makes sure every plan ends with a
    summarize data. If the plan already ends with a summarize data,
    nothing happens. If the plan ends with a sort, we drop the sort as it
    does not matter when summarizing. Then we add a summarize node to the
    end of the plan.
    """

    def __call__(self, plan: LogicalPlan, **kwargs) -> LogicalPlan:
        n = plan.nodes[plan.result_node]
        if n.node_type == "SummarizeData":
            return plan

        if n.node_type == "Sort":
            assert len(n.inputs) == 1, "Trailing sort had multiple or zero inputs?"
            penultimate = n.inputs[0]
            del plan.nodes[plan.result_node]
            plan.result_node = penultimate

        from sycamore.query.operators.summarize_data import SummarizeData

        question = plan.query
        prev_result = plan.result_node
        plan.result_node += 1
        plan.nodes[plan.result_node] = SummarizeData(
            node_type="SummarizeData",
            node_id=plan.result_node,
            description=f"Summarize the answer to the question {question}",
            inputs=[prev_result],
            question=question,
        )
        return LogicalPlan.model_validate(plan)


class OnlyRetrieval(LogicalPlanProcessor):
    """
    Remove nodes from the end of the plan that do not change the
    documents that are retrieved. These nodes are:
    - Sort
    - SummarizeData
    - LlmExtractEntity
    - TopK
    This processor is useful for efficiently computing retrieval metrics.
    """

    def __call__(self, plan: LogicalPlan, **kwargs) -> LogicalPlan:
        n = plan.nodes[plan.result_node]
        while n.node_type in ("Sort", "SummarizeData", "LlmExtractEntity", "TopK"):
            # SummarizeData is bad, it can fail to execute the entire pipeline so we don't get
            # a list of documents back because there is not materialize success dir
            # LlmExtractEntity without anything after it is just slow
            # TopK can sometimes fail if the LLM returns garbage and doesn't help with getting
            # the list of documents -- we just have to ignore it
            assert len(n.inputs) == 1, f"Trailing {n.node_type} node must have exactly one input"
            penultimate = n.inputs[0]
            del plan.nodes[plan.result_node]
            plan.result_node = penultimate
            n = plan.nodes[plan.result_node]
        return LogicalPlan.model_validate(plan)


class LimitLlmOperations(LogicalPlanProcessor):
    """
    Add a limit node before certain operators. This is useful to make some queries faster, although
    less accurate.

    Args:
        types: List of node types to limit; e.g. "LlmFilter", "SummarizeData"
        limit: How many documents to limit to
        message: Optional message to add to the description of the added limit node. Default is
            "Limit the number of documents going through the following llm operation for interactivity"
    """

    def __init__(
        self,
        types: list[str],
        limit: int,
        message: str = "Limit the number of documents going through the following llm operation for interactivity",
    ):
        self._types = types
        self._limit = limit
        self._message = message

    def __call__(self, plan: LogicalPlan, **kwargs) -> LogicalPlan:
        for i in sorted(plan.nodes.keys(), reverse=True):
            n = plan.nodes[i]
            if n.node_type not in self._types:
                continue
            inputs = plan.get_node_inputs(i)
            if len(inputs) != 1:
                logging.warning(f"Cannot add limit before {n.node_type} node because it has multiple or zero inputs")
                continue
            input = inputs[0]
            if isinstance(input, Limit):
                input.num_records = min(input.num_records, self._limit)
            else:
                limit = Limit(
                    node_type="Limit",
                    num_records=self._limit,
                    description=self._message,
                    node_id=n.node_id,
                    inputs=n.inputs,
                )
                plan.insert_node(i, limit)
        return plan


class RequireQueryDatabase(LogicalPlanProcessor):
    def __call__(self, plan: LogicalPlan, **kwargs) -> LogicalPlan:
        assert (
            plan.nodes[0].node_type == "QueryDatabase"
        ), "Found non-QueryDatabase start node, but QueryDatabase is required"
        return plan


class PrintPlan(LogicalPlanProcessor):
    """
    Print the plan nodes. Useful for debugging.

    Args:
        pre_message: An optional message to print before the plan
        post_message: An optional message to print after the plan
        quiet: Silence all output. Default is False
    """

    def __init__(self, pre_message: Optional[str] = None, post_message: Optional[str] = None, quiet: bool = False):
        self._pre = pre_message
        self._post = post_message
        self._quiet = quiet

    def __call__(self, plan: LogicalPlan, **kwargs) -> LogicalPlan:
        if self._quiet:
            return plan
        if self._pre is not None:
            print(self._pre)
        from sycamore.utils.jupyter import slow_pprint

        slow_pprint(plan.nodes)
        if self._post is not None:
            print(self._post)
        return plan


class LLMRewriteQuestion(PlannerPromptProcessor):
    """
    A prompt processor that uses an LLM to rewrite the question.

    Args:
        prompt: A SycamorePrompt that supports prompt.render_any(question=str)
        llm: The llm to use to rewrite the question
    """

    def __init__(self, prompt: SycamorePrompt, llm: LLM):
        # Make sure the prompt supports the render_any interface
        _ = prompt.render_any(question="dummy")
        self._rewrite_prompt = prompt
        self._llm = llm

    def __call__(self, prompt: PlannerPrompt, **kwargs) -> PlannerPrompt:
        q = prompt.query
        rendered = self._rewrite_prompt.render_any(question=q)
        rewritten = self._llm.generate(prompt=rendered)
        prompt.query = rewritten
        return prompt


class QueryPlanStrategy:
    """
    A strategy for generating a query plan. This strategy is responsible for providing available Operators and
    processing steps on LogicalPlans.
    """

    def __init__(
        self,
        operators: Optional[list[Type[Node]]] = None,
        plan_processors: Optional[list[LogicalPlanProcessor]] = None,
        prompt_processors: Optional[list[PlannerPromptProcessor]] = None,
    ) -> None:
        super().__init__()
        self.operators: list[Type[Node]] = operators or []
        self.plan_processors: list[LogicalPlanProcessor] = plan_processors or []
        self.prompt_processors: list[PlannerPromptProcessor] = prompt_processors or []


class DefaultQueryPlanStrategy(QueryPlanStrategy):
    """
    Default strategy that uses all available tools and optimizes result correctness.
    """

    def __init__(
        self,
        plan_processors: Optional[list[LogicalPlanProcessor]] = None,
        prompt_processors: Optional[list[PlannerPromptProcessor]] = None,
    ) -> None:
        super().__init__(ALL_OPERATORS, plan_processors, prompt_processors)


class VectorSearchOnlyStrategy(QueryPlanStrategy):
    """
    Strategy that only uses Vector Search for query plans to optimize for speed. This is useful when the data size
    is smaller and vector search retrievals are sufficient to provide answers.
    """

    def __init__(
        self,
        plan_processors: Optional[list[LogicalPlanProcessor]] = None,
        prompt_processors: Optional[list[PlannerPromptProcessor]] = None,
    ) -> None:
        super().__init__(
            [op for op in ALL_OPERATORS if op not in {QueryDatabase}], plan_processors or [], prompt_processors
        )
