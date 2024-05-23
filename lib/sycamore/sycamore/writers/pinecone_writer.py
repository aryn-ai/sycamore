from typing import Iterable, Any, Optional, Union, Tuple
import itertools

from pinecone import PodSpec, ServerlessSpec
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
        **ray_remote_args,
    ):
        super().__init__(plan, **ray_remote_args)
        self._index_name = index_name
        self._index_spec = index_spec
        self._namespace = namespace
        self._dimensions = dimensions
        self._distance_metric = distance_metric

    def execute(self) -> Dataset:
        dataset = self.child().execute()
        datasink = PineconeDatasink(
            self._index_name, self._index_spec, self._namespace, self._dimensions, self._distance_metric
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
        batch_size: int = 100,
    ):
        self._index_name = index_name
        self._index_spec = index_spec
        self._namespace = namespace
        self._dimensions = dimensions
        self._distance_metric = distance_metric
        self._batch_size = batch_size

    def on_write_start(self) -> None:
        from pinecone import Pinecone
        from pinecone.core.client.exceptions import NotFoundException

        pc = Pinecone()
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

        pc = PineconeGRPC()
        pc.Index(self._index_name)

        objects = self._extract_pinecone_objects(block)
        obj_it = iter(objects)
        batch = list(itertools.islice(obj_it, self._batch_size))
        while len(batch) > 0:
            # index.upsert(vectors=batch, async_req=True)
            print(f"\n{batch}\n{'-'*80}")
            batch = list(itertools.islice(obj_it, self._batch_size))

    @staticmethod
    def _extract_pinecone_objects(block: Block):

        def _flatten_metadata(data: Union[dict, list], prefix="") -> Iterable[Tuple[Any, Any]]:
            iterator = []  # type: ignore
            if isinstance(data, dict):
                iterator = data.items()  # type: ignore
            if isinstance(data, (list, tuple)):
                iterator = enumerate(data)  # type: ignore
            items = []
            for k, v in iterator:
                if isinstance(v, (dict, list)):
                    inner_values = _flatten_metadata(v, prefix=(str(k) if len(prefix) == 0 else f"{prefix}.{k}"))
                    items.extend([(innerk, innerv) for innerk, innerv in inner_values])
                elif v is not None:
                    items.append(((str(k) if len(prefix) == 0 else f"{prefix}.{k}"), v))
            return items

        def _record_to_object(record):
            doc = Document.from_row(record)
            data = doc.data
            id = data.pop("doc_id")
            values = data.pop("embedding", None)
            if values is None:
                return None
            return {"id": id, "values": values, "metadata": dict(_flatten_metadata(data))}

        records = BlockAccessor.for_block(block).to_arrow().to_pylist()
        return iter(filter(lambda x: x is not None, (_record_to_object(r) for r in records)))
