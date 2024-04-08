import pytest
from integration.ingests.crawler import HttpCrawlerIndex


@pytest.fixture(scope="session", params=["crawler-http-one", "crawler-http-all"])
def ingested_index(request, opensearch_client, container_handles):
    """
    Ingest an index. Parametrize by pre-defined ingestion setting
    :return: the index name and number of ingested documents
    """
    ingest_profile = request.param
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
