import unittest
import pytest
from typing import Optional
import copy

from sycamore.llms import LLM
from sycamore.llms.llms import LLMMode
from sycamore.llms.prompts import RenderedPrompt
from sycamore.query.logical_plan import LogicalPlan
from sycamore.query.operators.query_database import QueryDatabase
from sycamore.query.strategy import (
    LimitLlmOperations,
    RemoveVectorSearchForAnalytics,
    VectorSearchOnlyStrategy,
    AlwaysSummarize,
    OnlyRetrieval,
    DefaultQueryPlanStrategy,
    ALL_OPERATORS,
    DefaultPlanValidator,
)

PLANS = [
    LogicalPlan.model_validate(
        {
            "query": "Question 1",
            "nodes": {
                "0": {
                    "node_type": "QueryDatabase",
                    "node_id": 0,
                    "description": "qdb",
                    "index": "dummy",
                    "inputs": [],
                    "query": {"match_all": {}},
                },
                "1": {
                    "node_type": "SummarizeData",
                    "node_id": 1,
                    "description": "sum",
                    "question": "wha",
                    "inputs": [0],
                },
            },
            "result_node": 1,
            "llm_prompt": None,
            "llm_plan": None,
        }
    ),
    LogicalPlan.model_validate(
        {
            "query": "Question 2",
            "nodes": {
                "0": {
                    "node_type": "QueryDatabase",
                    "node_id": 0,
                    "description": "qdb",
                    "index": "dummy",
                    "inputs": [],
                    "query": {"match_all": {}},
                },
                "1": {
                    "node_type": "Sort",
                    "node_id": 1,
                    "description": "sort",
                    "field": "properties.entity.is.stupid",
                    "inputs": [0],
                },
            },
            "result_node": 1,
            "llm_prompt": None,
            "llm_plan": None,
        }
    ),
    LogicalPlan.model_validate(
        {
            "query": "Question 3",
            "nodes": {
                "0": {
                    "node_type": "QueryDatabase",
                    "node_id": 0,
                    "description": "qdb",
                    "index": "dummy",
                    "inputs": [],
                    "query": {"match_all": {}},
                },
                "1": {
                    "node_type": "Sort",
                    "node_id": 1,
                    "description": "sort",
                    "field": "properties.entity.is.stupid",
                    "inputs": [0],
                },
                "2": {"node_type": "Limit", "node_id": 2, "description": "lmt", "num_records": 2, "inputs": [1]},
                "3": {
                    "node_type": "LlmExtractEntity",
                    "node_id": 3,
                    "description": "llex",
                    "question": "extraction question",
                    "field": "properties.entity.is.stupid",
                    "new_field": "properties.entity.is.dumb",
                    "new_field_type": "string",
                    "inputs": [2],
                },
            },
            "result_node": 3,
            "llm_prompt": None,
            "llm_plan": None,
        }
    ),
]


class DummyLLMClient(LLM):
    def is_chat_mode(self) -> bool:
        return False

    def generate(self, *, prompt: RenderedPrompt, llm_kwargs: Optional[dict] = None) -> str:
        return "Dummy response from an LLM Client"


class TestStrategies(unittest.TestCase):
    def test_default(self):
        processor = RemoveVectorSearchForAnalytics(DummyLLMClient("test_model", LLMMode.SYNC))
        strategy = DefaultQueryPlanStrategy(plan_processors=[processor])

        assert len(strategy.plan_processors) == 1
        assert isinstance(strategy.plan_processors[0], RemoveVectorSearchForAnalytics)
        assert strategy.operators == ALL_OPERATORS

    def test_vector_search_only(self):
        processor = RemoveVectorSearchForAnalytics(DummyLLMClient("test_model", LLMMode.SYNC))

        strategy = VectorSearchOnlyStrategy(plan_processors=[])
        assert strategy.plan_processors == []
        assert QueryDatabase not in strategy.operators

        strategy = VectorSearchOnlyStrategy(plan_processors=[processor])
        assert strategy.plan_processors == [processor]
        assert QueryDatabase not in strategy.operators


class TestPlanProcessors:
    @pytest.mark.parametrize("plan", PLANS)
    def test_always_summarize(self, plan):
        assert isinstance(plan, LogicalPlan)
        proc = AlwaysSummarize()
        new_plan = proc(copy.deepcopy(plan))
        assert new_plan.nodes[new_plan.result_node].node_type == "SummarizeData"

    @pytest.mark.parametrize("plan", PLANS)
    def test_only_retrieval(self, plan):
        assert isinstance(plan, LogicalPlan)
        proc = OnlyRetrieval()
        new_plan = proc(copy.deepcopy(plan))
        if new_plan.query == "Question 3":
            assert new_plan.nodes[new_plan.result_node].node_type == "Limit"
            assert len(new_plan.nodes) == 3
        else:
            assert new_plan.nodes[new_plan.result_node].node_type == "QueryDatabase"
            assert len(new_plan.nodes) == 1

    @pytest.mark.parametrize("plan", PLANS)
    def test_limit_llm_ops(self, plan):
        assert isinstance(plan, LogicalPlan)
        proc = LimitLlmOperations(["SummarizeData", "LlmExtractEntity"], 10)
        new_plan = proc(copy.deepcopy(plan))
        if new_plan.query == "Question 1":
            assert new_plan.nodes[1].node_type == "Limit"
            assert new_plan.nodes[new_plan.result_node].node_type == "SummarizeData"
            assert new_plan.result_node == 2
        else:
            assert new_plan == plan


class TestRemoveVectorSearchForAnalytics(unittest.TestCase):

    processor = RemoveVectorSearchForAnalytics(DummyLLMClient("test_model", default_mode=LLMMode.SYNC))

    def test_operators(self):
        processor = RemoveVectorSearchForAnalytics(DummyLLMClient("test_model", default_mode=LLMMode.SYNC))
        strategy = VectorSearchOnlyStrategy(plan_processors=[processor])
        assert strategy.plan_processors == [processor]
        assert QueryDatabase not in strategy.operators

    @staticmethod
    def vector_search_filter_plan_single_operator():
        json_plan = {
            "query": "List all incidents involving Piper Aircrafts in California",
            "nodes": {
                "0": {
                    "node_type": "QueryVectorDatabase",
                    "node_id": 0,
                    "description": "Get all the airplane incidents in California",
                    "index": "ntsb",
                    "inputs": [],
                    "query_phrase": "Get all the airplane incidents",
                    "opensearch_filter": {"match": {"properties.entity.location": "California"}},
                }
            },
            "result_node": 0,
            "llm_prompt": None,
            "llm_plan": None,
        }
        return LogicalPlan.model_validate(json_plan)

    @staticmethod
    def vector_search_filter_plan_with_opensearch_filter():
        json_plan = {
            "query": "How many incidents involving Piper Aircrafts in California",
            "nodes": {
                "0": {
                    "node_type": "QueryVectorDatabase",
                    "node_id": 0,
                    "description": "Get all the airplane incidents in California",
                    "index": "ntsb",
                    "inputs": [],
                    "query_phrase": "Get all the airplane incidents",
                    "opensearch_filter": {"match": {"properties.entity.location": "California"}},
                },
                "1": {
                    "node_type": "Count",
                    "description": "Determine how many incidents occurred in Piper aircrafts",
                    "countUnique": False,
                    "field": None,
                    "inputs": [0],
                    "node_id": 1,
                },
                "2": {
                    "node_type": "SummarizeData",
                    "description": "Generate an English response to the question",
                    "question": "How many Piper aircrafts were involved in accidents?",
                    "inputs": [1],
                    "node_id": 2,
                },
            },
            "result_node": 2,
            "llm_prompt": None,
            "llm_plan": None,
        }
        return LogicalPlan.model_validate(json_plan)

    @staticmethod
    def vector_search_filter_plan_without_opensearch_filter():
        json_plan = {
            "query": "How many incidents involving Piper Aircrafts",
            "nodes": {
                "0": {
                    "node_type": "QueryVectorDatabase",
                    "node_id": 0,
                    "description": "Get all the airplane incidents involving Piper Aircrafts",
                    "index": "ntsb",
                    "inputs": [],
                    "query_phrase": "piper aircrafts",
                },
                "1": {
                    "node_type": "Count",
                    "description": "Count the number of incidents",
                    "node_id": 1,
                    "inputs": [0],
                },
                "2": {
                    "node_type": "SummarizeData",
                    "description": "Generate an English response to the question",
                    "question": "How many Piper aircrafts were involved in accidents?",
                    "node_id": 2,
                    "inputs": [1],
                },
            },
            "result_node": 2,
        }
        return LogicalPlan.model_validate(json_plan)

    def test_postprocess_plan(self):
        plan1 = self.vector_search_filter_plan_with_opensearch_filter()
        plan2 = self.vector_search_filter_plan_without_opensearch_filter()
        plan3 = self.vector_search_filter_plan_single_operator()

        for index, plan in enumerate([plan1, plan2, plan3]):
            print(plan.nodes)
            copy_plan = plan.model_copy()
            modified_plan = self.processor(copy_plan)

            print(modified_plan.nodes)

            if index in [0, 1]:
                assert len(modified_plan.nodes) == 4
            else:
                assert len(modified_plan.nodes) == 2

            assert modified_plan.nodes[0].node_type == "QueryDatabase"
            if index in [0, 2]:
                assert "match" in modified_plan.nodes[0].query
            elif index == 1:
                assert "match_all" in modified_plan.nodes[0].query

            assert modified_plan.nodes[1].node_type == "LlmFilter"
            assert modified_plan.nodes[1].node_id == 1
            assert modified_plan.nodes[1].field == "text_representation"
            assert modified_plan.nodes[1].inputs[0] == 0


class TestDefaultPlanValidator(unittest.TestCase):

    processor = DefaultPlanValidator()

    def test_input_type_mismatch(self):
        json_plan = {
            "query": "How many incidents involving Piper Aircrafts",
            "nodes": {
                "0": {
                    "node_type": "QueryVectorDatabase",
                    "node_id": 0,
                    "description": "Get all the airplane incidents involving Piper Aircrafts",
                    "index": "ntsb",
                    "inputs": [],
                    "query_phrase": "piper aircrafts",
                },
                "1": {
                    "node_type": "Math",
                    "description": "Count the number of incidents",
                    "node_id": 1,
                    "operation": "add",
                    "inputs": [0],
                },
            },
            "result_node": 2,
        }
        plan = LogicalPlan.model_validate(json_plan)
        with self.assertRaises(TypeError):
            self.processor(plan)
