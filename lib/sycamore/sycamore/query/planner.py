import logging
import typing
from abc import abstractmethod
from typing import List, Optional, Union

from sycamore.schema import Schema

from sycamore.llms.llms import LLM
from sycamore.llms.prompts.prompts import RenderedPrompt
from sycamore.llms.openai import OpenAI, OpenAIModels
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.planner_prompt import PlannerPrompt, PlannerExample, PLANNER_EXAMPLES
from sycamore.query.schema import OpenSearchSchema
from sycamore.query.strategy import QueryPlanStrategy, ALL_OPERATORS
from sycamore.utils.extract_json import extract_json

if typing.TYPE_CHECKING:
    from opensearchpy import OpenSearch


def process_json_plan(json_plan: str) -> LogicalPlan:
    """Deserialize the query plan returned by the LLM."""

    parsed_plan = extract_json(json_plan)
    if not isinstance(parsed_plan, dict):
        raise ValueError(f"Expected LLM query plan to contain a dict, got f{type(parsed_plan)}")
    return LogicalPlan.model_validate(parsed_plan)


class Planner:
    @abstractmethod
    def plan(self, question: str) -> LogicalPlan:
        raise NotImplementedError


class LlmPlanner(Planner):
    """The top-level query planner for SycamoreQuery. This class is responsible for generating
    a logical query plan from a user query using the OpenAI LLM.

    Args:
        index: The name of the index to query.
        data_schema: A dictionary mapping field names to their types.
        os_config: The OpenSearch configuration.
        os_client: The OpenSearch client.
        strategy: Strategy to use for planning, can be used to balance cost vs speed.
        llm_client: The LLM client.
        examples: Query examples to assist the LLM planner in few-shot learning.
            You may override this to customize the few-shot examples provided to the planner.
        natural_language_response: Whether to generate a natural language response. If False,
            the response will be raw data.
    """

    def __init__(
        self,
        index: str,
        data_schema: Union[OpenSearchSchema, Schema],
        os_config: Optional[dict[str, str]] = None,
        os_client: Optional["OpenSearch"] = None,
        strategy: QueryPlanStrategy = QueryPlanStrategy(ALL_OPERATORS, []),
        llm_client: Optional[LLM] = None,
        examples: Optional[List[PlannerExample]] = None,
        natural_language_response: bool = False,
        prompt: Optional[PlannerPrompt] = None,
    ) -> None:
        super().__init__()
        self._index = index
        self._strategy = strategy
        self._os_config = os_config
        self._os_client = os_client
        self._llm_client = llm_client or OpenAI(OpenAIModels.GPT_4O.value)
        self._examples = PLANNER_EXAMPLES if examples is None else examples
        self._natural_language_response = natural_language_response
        self._prompt = prompt

        self._data_schema: Schema = (
            data_schema.to_schema() if isinstance(data_schema, OpenSearchSchema) else data_schema
        )

    def generate_prompt(self, question: str, **kwargs) -> RenderedPrompt:
        if self._prompt is None:
            prompt = PlannerPrompt(
                query=question,
                examples=self._examples,
                natural_language_response=self._natural_language_response,
                operators=self._strategy.operators,
                index=self._index,
                data_schema=self._data_schema,
            )
        else:
            prompt = self._prompt.fork(query=question)
            if prompt.data_schema is None:
                prompt = prompt.fork(data_schema=self._data_schema)

        for preprocessor in self._strategy.prompt_processors:
            prompt = preprocessor(prompt, **kwargs)
        return prompt.render()

    def parse_llm_output(self, llm_output: str) -> LogicalPlan:
        return process_json_plan(llm_output)

    def plan(self, question: str, **kwargs) -> LogicalPlan:
        """Given a question from the user, generate a logical query plan."""
        llm_prompt = self.generate_prompt(question, **kwargs)
        llm_plan = self._llm_client.generate(prompt=llm_prompt, llm_kwargs={"temperature": 0})
        try:
            plan = self.parse_llm_output(llm_plan)
            for processor in self._strategy.plan_processors:
                plan = processor(plan, **kwargs)
        except Exception as e:
            logging.error(f"Error processing LLM-generated query plan: {e}\nPlan is:\n{llm_plan}")
            raise

        plan.query = question
        plan.llm_prompt = llm_prompt
        plan.llm_plan = llm_plan

        logging.debug(f"Query plan: {plan}")
        return plan
