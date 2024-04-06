from integration.containers.running import container_handles, container_urls, opensearch_client
from integration.ingests import ingested_index
from integration.queries.queries import DEFAULT_OPTIONS, QueryConfigGenerator, query_generator


QUERY_FIXTURE_NAME = "os_query"


def pytest_generate_tests(metafunc):
    """
    Generate all a test for every query configuration in the Query Config Generator
    """
    if QUERY_FIXTURE_NAME in metafunc.fixturenames:
        metafunc.parametrize(QUERY_FIXTURE_NAME, list(QueryConfigGenerator(DEFAULT_OPTIONS)))


__all__ = [
    "container_handles",
    "container_urls",
    "opensearch_client",
    "ingested_index",
    "query_generator",
]
