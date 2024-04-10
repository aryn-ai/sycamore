# ruff: noqa: F401
from integration.containers.running import container_handles, container_urls, opensearch_client
from integration.ingests.index import ingested_index, ingest_profile
from integration.queries.queries import DEFAULT_OPTIONS, QueryConfigGenerator, query_generator
from integration.containers.stack import stack


QUERY_FIXTURE_NAME = "os_query"


def pytest_generate_tests(metafunc):
    """
    Generate all a test for every query configuration in the Query Config Generator
    """
    if QUERY_FIXTURE_NAME in metafunc.fixturenames:
        metafunc.parametrize(QUERY_FIXTURE_NAME, list(QueryConfigGenerator(DEFAULT_OPTIONS)))
