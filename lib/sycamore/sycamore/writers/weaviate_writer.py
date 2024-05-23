import logging
from typing import Any, Iterable, Optional

from ray.data import Dataset, Datasink
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.block import Block, BlockAccessor
from sycamore.data.document import Document
from sycamore.plan_nodes import Node, Write


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
        from weaviate import WeaviateClient

        with WeaviateClient(**self._client_params) as client:
            if self._collection_config is not None:
                if client.collections.exists(self._collection_name):
                    logging.warning(
                        f"Collection config was provided, but collection {self._collection_name} "
                        "already exists, so ignoring provided config."
                    )
                else:
                    client.collections.create(**self._collection_config)

    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> Any:
        from weaviate import WeaviateClient
        from weaviate.collections.classes.data import DataReference

        builder = DelegatingBlockBuilder()
        for block in blocks:
            builder.add_block(block)
        block = builder.build()
        client = WeaviateClient(**self._client_params)
        with client:
            collection = client.collections.get(self._collection_name)
            with collection.batch.dynamic() as batch:
                objects = self._extract_weaviate_objects(block)
                refs = []
                for obj in objects:
                    obj_refs = obj.pop("references", None)
                    if obj_refs is not None:
                        for k, v in obj_refs.items():
                            refs.append(DataReference(from_uuid=obj["uuid"], from_property=k, to_uuid=v))
                    batch.add_object(**obj)
                # Flush the objects first so that references can know about them
                batch.flush()
                for ref in refs:
                    batch.add_reference(from_uuid=ref.from_uuid, from_property=ref.from_property, to=ref.to_uuid)

    @staticmethod
    def _extract_weaviate_objects(block):
        # Weaviate doesn't like explicitly null values
        def not_none(x):
            return x is not None

        # Weaviate defaults empty lists to text[], which is often incorrect
        def not_empty_list(x):
            return not isinstance(x, list) or len(x) > 0

        def _trim_properties(props):
            if isinstance(props, dict):
                return _trim_properties_in_dict(props)
            if isinstance(props, list):
                return _trim_properties_in_list(props)
            return props

        def _trim_properties_in_dict(props: dict) -> dict:
            trim_children = ((k, _trim_properties(v)) for k, v in props.items())
            trim_nones = filter(lambda pair: not_none(pair[1]), trim_children)
            trim_lists = filter(lambda pair: not_empty_list(pair[1]), trim_nones)
            return dict(trim_lists)

        def _trim_properties_in_list(props: list) -> list:
            trim_children = (_trim_properties(x) for x in props)
            trim_nones = filter(not_none, trim_children)
            trim_lists = filter(not_empty_list, trim_nones)
            return list(trim_lists)

        def record_to_object(record):
            default = {
                "doc_id": None,
                "type": "",
                "text_representation": "",
                "elements": [],
                "embedding": None,
                "parent_id": None,
                "properties": {},
                "bbox": [],
                "shingles": [],
            }
            doc = Document.from_row(record)
            uuid = doc.doc_id
            parent = doc.parent_id
            data = doc.data
            properties = {
                "properties": data.get("properties", default["properties"]),
                "type": data.get("type", default["type"]),
                "text_representation": data.get("text_representation", default["text_representation"]),
                "bbox": data.get("bbox", default["bbox"]),
                "shingles": data.get("shingles", default["shingles"]),
            }
            object = {
                "uuid": uuid,
                "properties": _trim_properties(properties),
            }
            embedding = data.get("embedding", None)
            if embedding is not None:
                object["vector"] = {"embedding": embedding}
            if parent:
                object["references"] = {"parent": parent}
            return object

        records = BlockAccessor.for_block(block).to_arrow().to_pylist()
        return [record_to_object(r) for r in records]
