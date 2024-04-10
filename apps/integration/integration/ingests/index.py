import pytest
from integration.ingests.crawler import HttpCrawlerIndex


INGEST_PROFILES = ["crawler-http-one", "crawler-http-all"]


@pytest.fixture(scope="session", params=INGEST_PROFILES)
def ingest_profile(request):
    return request.param


@pytest.fixture(scope="session")
def ingested_index(opensearch_client, container_handles, ingest_profile):
    """
    Ingest an index. Parametrize by pre-defined ingestion setting
    :return: the index name and number of ingested documents
    """
    index_ctx = None
    if ingest_profile == "crawler-http-one":
        index_ctx = HttpCrawlerIndex(
            profile="sort-one", opensearch=opensearch_client, importer=container_handles["importer"]
        )
    elif ingest_profile == "crawler-http-all":
        index_ctx = HttpCrawlerIndex(
            profile="sort-all", opensearch=opensearch_client, importer=container_handles["importer"]
        )
    with index_ctx as index_info:
        yield index_info
    print("left context")
