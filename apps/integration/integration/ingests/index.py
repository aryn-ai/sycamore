import pytest
from integration.ingests.crawler import HttpCrawlerIndex
from integration.ingests.jupyter import JupyterIndex

INGEST_PROFILES = ["crawler-http-one", "crawler-http-all", "jupyter-default-prep", "jupyter-dev-example"]
FLAKY_PROFILES = ["jupyter-dev-example"]


@pytest.fixture(scope="session", params=[p for p in INGEST_PROFILES if p not in FLAKY_PROFILES])
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
    elif ingest_profile == "jupyter-default-prep":
        index_ctx = JupyterIndex(
            nb_name="default-prep-script.ipynb", opensearch=opensearch_client, jupyter=container_handles["jupyter"]
        )
    elif ingest_profile == "jupyter-dev-example":
        index_ctx = JupyterIndex(
            nb_name="jupyter_dev_example.ipynb", opensearch=opensearch_client, jupyter=container_handles["jupyter"]
        )
    with index_ctx as index_info:
        yield index_info
