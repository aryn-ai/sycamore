from typing import Iterable, Any, Optional, Union, Tuple
import itertools
import os

from pinecone import PodSpec, ServerlessSpec
from pinecone.grpc import Vector
from pinecone.grpc.vector_factory_grpc import VectorFactoryGRPC
from ray.data import Dataset, Datasink
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor
from ray.data._internal.execution.interfaces import TaskContext
from sycamore.data.document import Document
from sycamore.plan_nodes import Node, Write


class PineconeWriter(Write):
    def __init__(
        self,
        plan: Node,
        index_name: str,
        index_spec: Union[None, dict, ServerlessSpec, PodSpec] = None,
        namespace: str = "",
        dimensions: Optional[int] = None,
        distance_metric: str = "cosine",
        api_key: Optional[str] = None,
        **ray_remote_args,
    ):
        super().__init__(plan, **ray_remote_args)
        self._api_key = api_key or os.environ["PINECONE_API_KEY"]
        self._index_name = index_name
        self._index_spec = index_spec
        self._namespace = namespace
        self._dimensions = dimensions
        self._distance_metric = distance_metric

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        datasink = PineconeDatasink(
            self._index_name,
            self._index_spec,
            self._namespace,
            self._dimensions,
            self._distance_metric,
            self._api_key,
        )
        dataset.write_datasink(datasink=datasink, ray_remote_args=self.resource_args)
        return dataset


class PineconeDatasink(Datasink):
    def __init__(
        self,
        index_name: str,
        index_spec: Union[None, dict, ServerlessSpec, PodSpec],
        namespace: str,
        dimensions: Optional[int],
        distance_metric: str,
        api_key: str,
        batch_size: int = 100,
    ):
        self._index_name = index_name
        self._index_spec = index_spec
        self._namespace = namespace
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._batch_size = batch_size
        self._api_key = api_key

    def on_write_start(self) -> None:
        from pinecone import Pinecone
        from pinecone.core.client.exceptions import NotFoundException

        pc = Pinecone(api_key=self._api_key)
        try:
            pc.describe_index(self._index_name)
        except NotFoundException:
            if self._dimensions is None:
                raise ValueError(f"dimensions must be specified in order to create a new index {self._index_name}")
            if self._index_spec is None:
                raise ValueError(f"index spec must be supplied in order to create a new index {self._index_name}")
            pc.create_index(
                name=self._index_name,
                dimension=self._dimensions,
                spec=self._index_spec,
                metric=self._distance_metric,
                timeout=None,
            )

    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> Any:
        from pinecone.grpc import PineconeGRPC

        builder = DelegatingBlockBuilder()
        for block in blocks:
            builder.add_block(block)
        block = builder.build()

        pc = PineconeGRPC(api_key=self._api_key)
        index = pc.Index(self._index_name)

        objects = self._extract_pinecone_objects(block)
        obj_it = iter(objects)
        batch = list(itertools.islice(obj_it, self._batch_size))
        async_results = []
        while len(batch) > 0:
            async_results.append(index.upsert(vectors=batch, namespace=self._namespace, async_req=True))
            batch = list(itertools.islice(obj_it, self._batch_size))
        for r in async_results:
            # Force async completion. Any errors are here.
            r.result()

    @staticmethod
    def _extract_pinecone_objects(block: Block) -> Iterable[Vector]:
        # TODO: https://arynai-my.sharepoint.com/:w:/g/personal/henry_aryn_ai/EXAzugsI3MZNt1d4AjP3k_ABTHW9NYG0wkA_8ifuGhxOJA?e=DQBjEB
        def _add_key_to_prefix(prefix, key):
            if len(prefix) == 0:
                return str(key)
            else:
                return f"{prefix}.{key}"

        def _flatten_metadata(data: Union[dict, list, tuple], prefix="") -> Iterable[Tuple[Any, Any]]:
            # Pinecone requires metadata to be flat (no nested objects) or a
            # list of strings so here's a traversal
            iterator = []  # type: ignore
            if isinstance(data, dict):
                iterator = data.items()  # type: ignore
            if isinstance(data, (list, tuple)):
                iterator = enumerate(data)  # type: ignore
            items = []
            for k, v in iterator:
                if isinstance(v, (dict, list, tuple)):
                    if isinstance(v, (list, tuple)) and all(isinstance(innerv, str) for innerv in v):
                        # Lists of strings are allowed
                        items.append((_add_key_to_prefix(prefix, k), v))
                    else:
                        inner_values = _flatten_metadata(v, prefix=(_add_key_to_prefix(prefix, k)))
                        items.extend([(innerk, innerv) for innerk, innerv in inner_values])
                elif v is not None:
                    items.append((_add_key_to_prefix(prefix, k), v))
            return items

        def _extract_metadata(doc):
            # Extract out specific metadata fields
            metadata = {
                "properties": doc.properties,
                "type": doc.type or "",
                "text_representation": doc.text_representation or "",
            }
            # represent bbox with coord names
            bbox = doc.bbox
            if bbox:
                metadata["bbox"] = {
                    "x1": bbox.x1,
                    "y1": bbox.y1,
                    "x2": bbox.x2,
                    "y2": bbox.y2,
                }
            # represent shingles as list of str
            shingles = doc.shingles
            if shingles:
                metadata["shingles"] = [str(s) for s in shingles]
            return metadata

        def _record_to_object(record):
            doc = Document.from_row(record)
            id = doc.doc_id
            # Use id prefixing
            parent_id = doc.parent_id
            if parent_id:
                id = f"{parent_id}#{id}"
            metadata = _extract_metadata(doc)
            # Get embedding (values in pinecone verbiage)
            # If there is no embedding this doc cannot be indexed.
            values = doc.embedding
            if values is None:
                return None
            # Create sparse vector representation from TF table if it exists
            sparse_vector = None
            tf_table = metadata.get("properties", {}).pop("term_frequency", None)
            if tf_table:
                sparse_indices = list(tf_table.keys())
                if not all(isinstance(index, int) for index in sparse_indices):
                    raise ValueError(
                        "Found non-integer terms in term frequency table. "
                        "Please use `docset.term_frequency(tokenizer, with_token_ids=True)` for pinecone hybrid search"
                    )
                sparse_values = [float(v) for v in tf_table.values()]
                sparse_vector = {"indices": sparse_indices, "values": sparse_values}
            # Flatten metadata and put it all together
            metadata = _flatten_metadata(metadata)
            vector_dict = {"id": id, "values": values, "metadata": dict(metadata)}
            if sparse_vector:
                vector_dict["sparse_values"] = sparse_vector
            return VectorFactoryGRPC.build(vector_dict)

        records = BlockAccessor.for_block(block).to_arrow().to_pylist()
        return iter(filter(lambda x: x is not None, (_record_to_object(r) for r in records)))  # type: ignore
