import pytest
from pathlib import Path


@pytest.mark.processor_name("debug")
def test_debug(opensearch_client, upload_jsonl_index, singleton_pipeline):
    opensearch_client.search(
        index=upload_jsonl_index,
        params={"search_pipeline": singleton_pipeline},
        body={"query": {"match_all": {}}}
    )
