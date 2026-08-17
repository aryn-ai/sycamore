import pytest

from sycamore import DocSet
from sycamore.query.operators.groupby import AggregateCount, AggregateCollect
from sycamore.query.operators.clustering import KMeanClustering, LLMClustering
from sycamore.query.operators.groupby import GroupBy
from sycamore.query.strategy import QueryPlanStrategy

from sycamore.query.client import SycamoreQueryClient
from sycamore.query.operators.query_database import QueryVectorDatabase, QueryDatabase


class TestSycamoreQuery:

    @pytest.mark.parametrize("codegen_mode", [False])
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
        assert isinstance(result.result, (str, int))
        if isinstance(result.result, str):
            assert len(result.result) > 0

    @pytest.mark.parametrize("codegen_mode", [False])
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
            assert result.code is None
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

    def test_bug(self):
        import sycamore
        from sycamore.connectors.opensearch.utils import OpenSearchClientWithLogging
        from sycamore.data import Document
        from sycamore.functions import HuggingFaceTokenizer
        from sycamore.tests.config import TEST_DIR
        from sycamore.transforms.embed import SentenceTransformerEmbedder
        from sycamore.transforms.merge_elements import GreedyTextElementMerger
        from sycamore.transforms.partition import ArynPartitioner

        paths = str(TEST_DIR / "resources/data/pdfs/ntsb-report.pdf")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = HuggingFaceTokenizer(model_name)

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = HuggingFaceTokenizer(model_name)

        context = sycamore.init()

        def set_page_numbers(d: Document) -> Document:
            d.properties["page_numbers"] = d.elements[0].properties["page_numbers"]
            return d

        ds = (
            context.read.binary(paths, binary_format="pdf")
            .partition(partitioner=ArynPartitioner())
            .merge(GreedyTextElementMerger(tokenizer=tokenizer, max_tokens=1000))
            .map(set_page_numbers)
            .spread_properties(["path"])
            .explode()
            .embed(
                embedder=SentenceTransformerEmbedder(
                    batch_size=100, model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            )
        )

        docs = ds.take_all()
        print(f"{len(docs)} docs found")
        for d in docs:
            print(d.doc_id, d.properties.keys())
