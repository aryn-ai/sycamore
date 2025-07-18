import pytest

from sycamore import DocSet
from sycamore.query.operators.groupby import AggregateCount, AggregateCollect
from sycamore.query.operators.clustering import KMeanClustering, LLMClustering
from sycamore.query.operators.groupby import GroupBy
from sycamore.query.strategy import QueryPlanStrategy

from sycamore.query.client import SycamoreQueryClient
from sycamore.query.operators.query_database import QueryVectorDatabase, QueryDatabase


class TestSycamoreQuery:

    @pytest.mark.parametrize("codegen_mode", [True, False])
    def test_simple(self, query_integration_test_index: str, codegen_mode: bool):
        """
        Simple test that ensures we can run a query end to end and get a text response.
        """
        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema(query_integration_test_index)
        plan = client.generate_plan(
            "How many incidents happened in california?",
            query_integration_test_index,
            schema,
            natural_language_response=True,
        )
        result = client.run_plan(plan, codegen_mode=codegen_mode)
        assert isinstance(result.result, str)
        assert len(result.result) > 0

    @pytest.mark.parametrize("codegen_mode", [True, False])
    def test_forked(self, query_integration_test_index: str, codegen_mode: bool):
        """
        Test that has a fork in the DAG, ensures we can support multiple execution paths.
        """
        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema(query_integration_test_index)
        plan = client.generate_plan(
            "What fraction of all incidents happened in california?",
            query_integration_test_index,
            schema,
            natural_language_response=True,
        )
        result = client.run_plan(plan, codegen_mode=codegen_mode)
        assert isinstance(result.result, str)
        assert len(result.result) > 0

    @pytest.mark.parametrize("dry_run", [True, False])
    def test_dry_run(self, query_integration_test_index: str, dry_run: bool):
        """
        Test that asserts dry_run mode doesn't execute the query. We use ray initialization as a proxy here,
        we can also look at LLM calls but this is a little more reliable in the current state.
        """
        import ray

        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema(query_integration_test_index)
        plan = client.generate_plan(
            "What fraction of all incidents happened in california?",
            query_integration_test_index,
            schema,
            natural_language_response=True,
        )
        ray.shutdown()
        result = client.run_plan(plan, dry_run=dry_run)
        if dry_run:
            assert isinstance(result.code, str)
            assert len(result.code) > 0
            assert not ray.is_initialized()
        else:
            assert isinstance(result.result, str)
            assert len(result.result) > 0
            assert ray.is_initialized()

    @pytest.mark.parametrize("codegen_mode", [False])
    def test_vector_search(self, query_integration_test_index, codegen_mode: bool):
        """ """

        client = SycamoreQueryClient(query_plan_strategy=QueryPlanStrategy())
        schema = client.get_opensearch_schema(query_integration_test_index)
        plan = client.generate_plan(
            "give me some wind related incidents",
            query_integration_test_index,
            schema,
            natural_language_response=False,
        )
        assert len(plan.nodes) == 2
        assert isinstance(plan.nodes[0], QueryVectorDatabase)
        result = client.run_plan(plan, codegen_mode=codegen_mode)
        assert isinstance(result.result, DocSet)
        docs = result.result.take_all()
        assert len(docs) > 0

    def test_vector_search_2(self):

        client = SycamoreQueryClient(query_plan_strategy=QueryPlanStrategy())
        schema = client.get_opensearch_schema("ntsb-accident-cause")
        plan = client.generate_plan(
            "What was the most common cause of accidents in the NTSB incident reports?",
            "ntsb-accident-cause",
            schema,
            natural_language_response=False,
        )
        assert len(plan.nodes) == 2
        assert isinstance(plan.nodes[0], QueryDatabase)
        plan.nodes[1] = KMeanClustering(
            node_type="KMeanClustering",
            node_id=1,
            description="Find the most common cause of accidents",
            inputs=[0],
            field="properties.cause",
            new_field="centroids",
            K=5,
        )
        plan.nodes[2] = GroupBy(
            node_type="GroupBy",
            node_id=2,
            description="Find the most common cause of accidents",
            inputs=[1],
            field="centroids",
        )
        plan.nodes[3] = AggregateCount(
            node_type="AggregateCount",
            node_id=3,
            description="Find the most common cause of accidents",
            inputs=[2],
            llm_summary=True,
            llm_summary_instruction="The cause of accident for this group",
        )
        plan.result_node = 3
        result = client.run_plan(plan, codegen_mode=False)
        assert isinstance(result.result, DocSet)
        docs = result.result.take_all()
        assert len(docs) > 0

    @pytest.mark.parametrize("codegen_mode", [False])
    def test_vector_search_with_result_filter(self, query_integration_test_index2, codegen_mode: bool):
        """
        Running with 3 documents ingested.
        1 document with page_numbers 1,2
        2 documents with page_numbers 1, 2, 3,4
        """

        client = SycamoreQueryClient(query_plan_strategy=QueryPlanStrategy())
        schema = client.get_opensearch_schema(query_integration_test_index2)
        plan = client.generate_plan(
            "give me some wind related incidents",
            query_integration_test_index2,
            schema,
            natural_language_response=False,
        )
        assert len(plan.nodes) == 2
        assert isinstance(plan.nodes[0], QueryVectorDatabase)
        result = client.run_plan(plan, codegen_mode=codegen_mode)
        assert isinstance(result.result, DocSet)
        docs = result.result.take_all()
        print(f"{len(docs)} docs found")
        expected_count = len(docs)
        assert expected_count > 0
        for doc in docs:
            print(f"{doc.doc_id}: {doc.properties}")

        db_node: QueryVectorDatabase = plan.nodes[0]
        db_node.result_filter = {"properties.page_numbers": [1, 2]}
        result = client.run_plan(plan, codegen_mode=codegen_mode)
        filtered_docs = result.result.take_all()
        assert len(filtered_docs) == expected_count

        db_node.result_filter = {"properties.page_numbers": [3, 4]}
        result = client.run_plan(plan, codegen_mode=codegen_mode)
        filtered_docs = result.result.take_all()
        assert len(filtered_docs) == 2

        db_node.opensearch_filter = {"bool": {"must": [{"terms": {"languages": ["eng"]}}]}}
        db_node.result_filter = {"properties.page_numbers": [3, 4]}
        result = client.run_plan(plan, codegen_mode=codegen_mode)
        filtered_docs = result.result.take_all()
        assert len(filtered_docs) == 2

    @pytest.mark.parametrize("codegen_mode", [False])
    def test_simple_with_result_filter(self, query_integration_test_index2: str, codegen_mode: bool):
        """
        Running with 3 documents ingested.
        1 document with page_numbers 1,2
        2 documents with page_numbers 1, 2, 3,4
        """

        client = SycamoreQueryClient()
        schema = client.get_opensearch_schema(query_integration_test_index2)
        plan = client.generate_plan(
            "List all documents",
            query_integration_test_index2,
            schema,
            natural_language_response=False,
        )

        assert len(plan.nodes) == 1
        assert isinstance(plan.nodes[0], QueryDatabase)

        result = client.run_plan(plan, codegen_mode=codegen_mode)
        assert isinstance(result.result, DocSet)

        docs = result.result.take_all()
        print(f"{len(docs)} docs found")
        expected_count = len(docs)
        assert expected_count > 0
        for doc in docs:
            print(f"{doc.doc_id}: {doc.properties}")

        db_node: QueryDatabase = plan.nodes[0]
        db_node.result_filter = {"properties.page_numbers": [1, 2]}
        result = client.run_plan(plan, codegen_mode=codegen_mode)
        filtered_docs = result.result.take_all()
        assert len(filtered_docs) == expected_count

        db_node.result_filter = {"properties.page_numbers": [3, 4]}
        result = client.run_plan(plan, codegen_mode=codegen_mode)
        filtered_docs = result.result.take_all()
        assert len(filtered_docs) == 2

    def test_vector_search_3(self):
        client = SycamoreQueryClient(query_plan_strategy=QueryPlanStrategy())
        schema = client.get_opensearch_schema("ntsb-accident-cause")
        plan = client.generate_plan(
            "What was the most common cause of accidents in the NTSB incident reports?",
            "ntsb-accident-cause",
            schema,
            natural_language_response=False,
        )
        assert len(plan.nodes) == 2
        assert isinstance(plan.nodes[0], QueryDatabase)
        plan.nodes[1] = LLMClustering(
            node_type="LLMClustering",
            node_id=1,
            description="Find the most common cause of accidents",
            inputs=[0],
            field="properties.cause",
            llm_group_instruction="Form groups of different cause of accidents",
        )
        plan.nodes[2] = GroupBy(
            node_type="GroupBy",
            node_id=2,
            description="Find the most common cause of accidents",
            inputs=[1],
        )
        plan.nodes[3] = AggregateCollect(
            node_type="AggregateCollect",
            node_id=3,
            description="Find the most common cause of accidents",
            inputs=[2],
            llm_summary=False,
        )
        plan.result_node = 3
        result = client.run_plan(plan, codegen_mode=False)
        assert isinstance(result.result, DocSet)
        docs = result.result.take_all()
        assert len(docs) > 0
