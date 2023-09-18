import logging
from typing import Optional, Iterable

from opensearchpy import OpenSearch
from opensearchpy.helpers import parallel_bulk
from ray.data import Datasource, Dataset
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor
from ray.data.datasource import WriteResult
from ray.data._internal.execution.interfaces import TaskContext
from sycamore.execution.basics import Node, Write

log = logging.getLogger(__name__)


class OpenSearchWriter(Write):
    def __init__(
        self,
        plan: Node,
        index_name: str,
        *,
        os_client_args: dict,
        index_settings: Optional[dict] = None,
        number_of_allowed_failures_per_block: int = 100,
        collect_failures_file_path: str = "failures.txt",
        **ray_remote_args,
    ):
        super().__init__(plan, **ray_remote_args)
        self.index_name = index_name
        self.index_settings = index_settings
        self.os_client_args = os_client_args
        self.number_of_allowed_failures_per_block = number_of_allowed_failures_per_block
        self.collect_failures_file_path = collect_failures_file_path

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        try:
            client = OpenSearch(**self.os_client_args)
            if not client.indices.exists(self.index_name):
                if self.index_settings is not None:
                    client.indices.create(self.index_name, **self.index_settings)
                else:
                    client.indices.create(self.index_name)

        except Exception as e:
            raise RuntimeError("Exception occurred while creating an index", e)

        dataset.write_datasource(
            OSDataSource(),
            index_name=self.index_name,
            os_client_args=self.os_client_args,
            number_of_allowed_failures_per_block=self.number_of_allowed_failures_per_block,
            collect_failures_file_path=self.collect_failures_file_path,
        )

        return dataset


class OSDataSource(Datasource):
    # todo: make this type specific to extract properties
    @staticmethod
    def extract_os_document(data):
        result = dict()
        default = {
            "doc_id": None,
            "type": None,
            "text_representation": None,
            "elements": {"array": []},
            "embedding": None,
            "parent_id": None,
            "properties": {},
        }
        for k, v in default.items():
            if k in data:
                result[k] = data[k]
            else:
                result[k] = v
        return result

    # The type: ignore is required for the ctx paramter, which is not part of the Datasource
    # API spec, but is passed at runtime by Ray. This can be removed once this commit is
    # included in Ray's release:
    #
    # https://github.com/ray-project/ray/commit/dae1d1f4a0f531fd8d0fbfca5e5cd2d1f21b551e
    def write(self, blocks: Iterable[Block], ctx: TaskContext, **write_args) -> WriteResult:  # type: ignore
        builder = DelegatingBlockBuilder()
        for block in blocks:
            builder.add_block(block)
        block = builder.build()

        self.write_block(block, **write_args)

        return "ok"

    @staticmethod
    def write_block(
        block: Block,
        *,
        os_client_args: dict,
        index_name: str,
        collect_failures_file_path: str,
        number_of_allowed_failures_per_block: int,
    ):
        client = OpenSearch(**os_client_args)

        block = BlockAccessor.for_block(block).to_arrow().to_pylist()

        def create_actions():
            for i, row in enumerate(block):
                doc = OSDataSource.extract_os_document(row)
                action = {"_index": index_name, "_id": doc["doc_id"], "_source": doc}
                yield action

        failures = []
        for success, info in parallel_bulk(client, create_actions()):
            if not success:
                log.error("A Document failed to upload", info)
                failures.append(info)

                if len(failures) > number_of_allowed_failures_per_block:
                    with open(collect_failures_file_path, "a") as f:
                        for doc in failures:
                            f.write(f"{doc}\n")
                    raise RuntimeError(
                        f"{number_of_allowed_failures_per_block} documents failed to index. "
                        f"Refer to {collect_failures_file_path}."
                    )

        log.info("All the documents have been ingested!")
