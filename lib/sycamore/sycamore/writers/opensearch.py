import logging
from typing import Any, Optional, Iterable

from opensearchpy import OpenSearch
from opensearchpy.helpers import parallel_bulk
from ray.data import Datasink, Dataset
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor
from ray.data._internal.execution.interfaces import TaskContext

from sycamore.data import Document
from sycamore.plan_nodes import Node, Write
from sycamore.utils.time_trace import timetrace

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

    @timetrace("OsrchWrite")
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

        dataset.write_datasink(
            OSDataSink(
                index_name=self.index_name,
                os_client_args=self.os_client_args,
                number_of_allowed_failures_per_block=self.number_of_allowed_failures_per_block,
                collect_failures_file_path=self.collect_failures_file_path,
            ),
            ray_remote_args=self.resource_args,
        )

        return dataset


class OSDataSink(Datasink):
    def __init__(self, index_name, os_client_args, number_of_allowed_failures_per_block, collect_failures_file_path):
        self.index_name = index_name
        self.os_client_args = os_client_args
        self.number_of_allowed_failures_per_block = number_of_allowed_failures_per_block
        self.collect_failures_file_path = collect_failures_file_path

    # todo: make this type specific to extract properties
    @staticmethod
    def extract_os_document(data):
        result = dict()
        default = {
            "doc_id": None,
            "type": None,
            "text_representation": None,
            "elements": [],
            "embedding": None,
            "parent_id": None,
            "properties": {},
            "bbox": None,
            "shingles": None,
        }
        for k, v in default.items():
            if k in data:
                result[k] = data[k]
            else:
                result[k] = v
        return result

    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> Any:
        builder = DelegatingBlockBuilder()
        for block in blocks:
            builder.add_block(block)
        block = builder.build()

        self.write_block(
            block,
            os_client_args=self.os_client_args,
            index_name=self.index_name,
            collect_failures_file_path=self.collect_failures_file_path,
            number_of_allowed_failures_per_block=self.number_of_allowed_failures_per_block,
        )

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
                doc = OSDataSink.extract_os_document(Document.from_row(row).data)
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
