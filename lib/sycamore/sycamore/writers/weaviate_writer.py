import logging
from typing import Any, Iterable, Optional

from ray.data import Dataset, Datasink
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.block import Block, BlockAccessor
from sycamore.data.document import Document
from sycamore.plan_nodes import Node, Write
from weaviate import WeaviateClient


class WeaviateWriter(Write):
    def __init__(
        self,
        plan: Node,
        collection_name: str,
        client_params: dict,
        collection_config: Optional[dict],
        **ray_remote_args,
    ):
        super().__init__(plan, **ray_remote_args)
        self.collection_name = collection_name
        self.client_params = client_params
        self.collection_config = collection_config

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        datasink = WeaviateDatasink(self.collection_name, self.client_params, self.collection_config)
        dataset.write_datasink(datasink, ray_remote_args=self.resource_args)
        return dataset


class WeaviateDatasink(Datasink):
    def __init__(self, collection_name: str, client_params: dict, collection_config: Optional[dict]):
        self._collection_name = collection_name
        self._client_params = client_params
        self._collection_config = collection_config

    def on_write_start(self):
        client = WeaviateClient(**self._client_params)
        try:
            client.connect()
            if client.collections.exists(self._collection_name):
                client.collections.get(self._collection_name)
            elif self._collection_config is not None:
                client.collections.create(**self._collection_config)
            else:
                pass
                # raise ValueError(f"Collection {self._collection_name} does not
                # exist and no collection config was provided")
        finally:
            client.close()

    def on_write_failed(self, error: Exception) -> None:
        logging.error(error)
        print(error)

    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> Any:
        try:
            builder = DelegatingBlockBuilder()
            for block in blocks:
                builder.add_block(block)
            block = builder.build()
            client = WeaviateClient(**self._client_params)
            with client:
                client.collections.get(self._collection_name)
                with client.collections.get(self._collection_name).batch.dynamic() as batch:

                    for obj in self._extract_weaviate_objects(block):
                        print(f"id: {obj['uuid']}\ntitle: {obj['properties']['properties']['title']}\n")
                        # vec = obj.pop('vector')
                        # print(json.dumps(obj, indent=2))
                        # if vec:
                        #     obj['vector'] = vec
                        batch.add_object(**obj)
                    batch.flush()
        except Exception as e:
            print(e)

    @staticmethod
    def _extract_weaviate_objects(block):
        import json

        records = BlockAccessor.for_block(block).to_arrow().to_pylist()
        print(len(records))

        # print([Document.from_row(r) for r in records])
        def record_to_object(record):
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
            doc = Document.from_row(record)
            uuid = doc.doc_id
            parent = doc.parent_id
            data = doc.data
            object = {
                "uuid": uuid,
                "properties": {
                    "properties": data.get("properties", default["properties"]),
                    "type": data.get("type", default["type"]),
                    "text_representation": data.get("text_representation", default["text_representation"]),
                    "bbox": data.get("bbox", default["bbox"]),
                    "shingles": data.get("shingles", default["shingles"]),
                },
            }
            print(json.dumps(object, indent=2))
            if "embedding" in data:
                object["vector"] = {"embedding": data.get("embedding")}
            if parent:
                object["references"] = {"parent": parent}
            return object

        return [record_to_object(r) for r in records]
