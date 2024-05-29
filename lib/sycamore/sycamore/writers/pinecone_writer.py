from typing import Iterable, Any, Optional, Union, Tuple
import itertools
from base64 import b64encode
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
            self._index_name, self._index_spec, self._namespace, self._dimensions, self._distance_metric, self._api_key
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
            r.result()

    @staticmethod
    def _extract_pinecone_objects(block: Block) -> Iterable[Vector]:

        def _flatten_metadata(data: Union[dict, list, tuple], prefix="") -> Iterable[Tuple[Any, Any]]:
            iterator = []  # type: ignore
            if isinstance(data, dict):
                iterator = data.items()  # type: ignore
            if isinstance(data, (list, tuple)):
                iterator = enumerate(data)  # type: ignore
            items = []
            for k, v in iterator:
                if isinstance(v, (dict, list, tuple)):
                    if isinstance(v, (list, tuple)) and all(isinstance(innerv, str) for innerv in v):
                        items.append(((str(k) if len(prefix) == 0 else f"{prefix}.{k}"), v))
                    else:
                        inner_values = _flatten_metadata(v, prefix=(str(k) if len(prefix) == 0 else f"{prefix}.{k}"))
                        items.extend([(innerk, innerv) for innerk, innerv in inner_values])
                elif v is not None:
                    items.append(((str(k) if len(prefix) == 0 else f"{prefix}.{k}"), v))
            return items

        def _metadata_special_cases(data: dict) -> dict:
            binary = data.get("binary_representation", None)
            if binary:
                b64binary = b64encode(binary)
                strbinary = b64binary.decode("UTF-8")
                data["binary_representation"] = strbinary
            shingles = data.get("shingles", None)
            if shingles:
                strshingles = [str(s) for s in shingles]
                data["shingles"] = strshingles
            bbox = data.get("bbox", None)
            if bbox:
                bbox_as_dict = {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                }
                data["bbox"] = bbox_as_dict
            return data

        def _record_to_object(record):
            doc = Document.from_row(record)
            data = doc.data
            id = data.pop("doc_id")
            parent_id = data.pop("parent_id", None)
            if parent_id:
                id = f"{parent_id}#{id}"
            values = data.pop("embedding", None)
            if values is None:
                return None
            sparse_vector = None
            tf_table = data.get("properties", {}).pop("term_frequency", None)
            if tf_table:
                sparse_indices = list(tf_table.keys())
                if not all(isinstance(index, int) for index in sparse_indices):
                    raise ValueError(
                        "Found non-integer terms in term frequency table. "
                        "Please use `docset.term_frequency(tokenizer, with_token_ids=True)` for pinecone hybrid search"
                    )
                sparse_values = [float(v) for v in tf_table.values()]
                sparse_vector = {"indices": sparse_indices, "values": sparse_values}
            metadata = _flatten_metadata(_metadata_special_cases(data))
            vector_dict = {"id": id, "values": values, "metadata": dict(metadata)}
            if sparse_vector:
                vector_dict["sparse_values"] = sparse_vector
            return VectorFactoryGRPC.build(vector_dict)

        records = BlockAccessor.for_block(block).to_arrow().to_pylist()
        return iter(filter(lambda x: x is not None, (_record_to_object(r) for r in records)))  # type: ignore
