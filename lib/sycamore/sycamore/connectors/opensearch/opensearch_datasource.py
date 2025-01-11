import logging
from typing import Any

from opensearchpy import OpenSearch
from ray.data import Datasource, ReadTask
from ray.data.block import BlockMetadata


logger = logging.getLogger(__name__)


class OpenSearchClientArgs:
    def __init__(self, host: str, port: int, username: str, password: str):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

    def get_client_args_as_dict(self):
        return {
            "host": self.host,
            "port": self.port,
            "http_auth": (self.username, self.password)
        }


class OpenSearchDatasource(Datasource):
    def __init__(self, client_args: OpenSearchClientArgs, index: str, query: dict[str, Any]):
        self.client_args = client_args
        self.index = index
        self.client = OpenSearch(**client_args.get_client_args_as_dict())

        res = self.client.search(index=index, body=query, size=0)

        # create a PIT

        def get_slice_count():
            return 0

        def get_slice_doc_count():
            return 0

        num_slices = get_slice_count()

        self.slices = []
        for i in range(num_slices):
            slice = {}
            count = get_slice_doc_count()
            metadata = BlockMetadata(
                num_rows=count,
                size_bytes=None,
                schema=None,
                input_files=None,
                exec_stats=None,
            )
            slice["metadata"] = metadata


    def estimate_inmemory_data_size(self) -> int:
        # If we're doing a full scan, we can get the index size via /_cat/indices
        return 0

    def get_read_tasks(self, parallelism: int):
        assert parallelism > 0, f"Invalid parallelism {parallelism}"

        return [ReadTask(read_fn=read_slice, slice["metadata"]) for slice in self.slices]