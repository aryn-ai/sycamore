from dataclasses import dataclass, asdict
from typing import Optional, TypedDict, Union
import json
from sycamore.utils import batched
from typing_extensions import TypeGuard

from pinecone import PineconeException, PineconeApiException, PodSpec, ServerlessSpec
from pinecone.grpc import PineconeGRPC, Vector
from pinecone.grpc.vector_factory_grpc import VectorFactoryGRPC
from sycamore.data.document import Document
from sycamore.connectors.writer_utils.base import BaseDBWriter
from sycamore.connectors.writer_utils.common import flatten_data
import time


@dataclass
class PineconeTargetParams(BaseDBWriter.TargetParams):
    index_name: str
    namespace: str = ""
    index_spec: Union[None, dict, ServerlessSpec, PodSpec] = None
    dimensions: Optional[int] = None
    distance_metric: str = "cosine"

    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        if not isinstance(other, PineconeTargetParams):
            return False
        if self.index_spec is not None and other.index_spec is not None:
            my_is = self.index_spec if isinstance(self.index_spec, dict) else self.index_spec.asdict()
            ot_is = other.index_spec if isinstance(other.index_spec, dict) else other.index_spec.asdict()
            if my_is != ot_is:
                return False
        if self.dimensions is not None and other.dimensions is not None:
            if self.dimensions != other.dimensions:
                return False
        return self.index_name == other.index_name and self.distance_metric == other.distance_metric


@dataclass
class PineconeClientParams(BaseDBWriter.ClientParams):
    api_key: str
    batch_size: int = 100


class PineconeClient(BaseDBWriter.Client):
    def __init__(self, api_key: str, batch_size: int):
        self._client = PineconeGRPC(api_key=api_key)
        self._batch_size = batch_size

    @classmethod
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "PineconeClient":
        assert isinstance(params, PineconeClientParams)
        return PineconeClient(params.api_key, params.batch_size)

    def write_many_records(self, records: list["BaseDBWriter.Record"], target_params: "BaseDBWriter.TargetParams"):
        assert isinstance(target_params, PineconeTargetParams)
        assert _narrow_list_of_pinecone_records(records), f"Found bad records in {records}"
        index = self._client.Index(target_params.index_name)
        async_results = []
        wait_on_index(self._client, target_params.index_name)
        for batch in batched(records, self._batch_size):
            vectors = [r.to_grpc_vector() for r in batch if r.values is not None]
            res = index.upsert(vectors=vectors, namespace=target_params.namespace, async_req=True)
            async_results.append(res)
            time.sleep(1)
        for res in async_results:
            # Force async completion. Errors are here
            try:
                res.result()
            except PineconeException as e:
                print(e)
                print("ERROR CHECK")
                print(res)

    def create_target_idempotent(self, target_params: "BaseDBWriter.TargetParams"):
        assert isinstance(target_params, PineconeTargetParams)
        if target_params.dimensions and target_params.index_spec:
            try:
                self._client.create_index(
                    name=target_params.index_name,
                    dimension=target_params.dimensions,
                    spec=target_params.index_spec,
                    metric=target_params.distance_metric,
                )
            except PineconeApiException as e:
                if e.status == 409 and json.loads(str(e.body)).get("error", {}).get("code", {}) == "ALREADY_EXISTS":
                    return
                raise e

    def get_existing_target_params(self, target_params: "BaseDBWriter.TargetParams") -> PineconeTargetParams:
        assert isinstance(target_params, PineconeTargetParams)
        index_dict = self._client.describe_index(target_params.index_name).to_dict()
        return PineconeTargetParams(
            index_name=index_dict["name"],
            dimensions=index_dict["dimension"],
            index_spec=index_dict["spec"],
            namespace=target_params.namespace,
            distance_metric=index_dict["metric"],
        )


@dataclass
class PineconeRecord(BaseDBWriter.Record):
    id: str
    values: Optional[list[float]]
    metadata: dict[str, Union[list[str], str, bool, int, float]]
    sparse_values: Optional["PineconeRecord.SparseVector"]

    class SparseVector(TypedDict):
        indices: list[int]
        values: list[float]

    @classmethod
    def from_doc(cls, document: Document, target_params: "BaseDBWriter.TargetParams") -> "PineconeRecord":
        assert isinstance(target_params, PineconeTargetParams)
        assert document.doc_id is not None, f"Document found with null id: {document}"
        if document.parent_id is None:
            id = document.doc_id
        else:
            id = f"{document.parent_id}#{document.doc_id}"
        values = document.embedding
        metadata = {
            "type": document.type,
            "text_representation": document.text_representation,
            "bbox": document.bbox.to_dict() if document.bbox else None,
            "shingles": [str(s) for s in document.shingles] if document.shingles else None,
        }
        sparse_vector = None
        tf_table = document.properties.pop("term_frequency", None)
        metadata["properties"] = document.properties
        if tf_table:
            sparse_indices = list(tf_table.keys())
            if not all(isinstance(index, int) for index in sparse_indices):
                raise ValueError(
                    "Found non-integer terms in term frequency table. "
                    "Please use `docset.term_frequency(tokenizer, with_token_ids=True)` for pinecone hybrid search"
                )
            sparse_values = [float(v) for v in tf_table.values()]
            sparse_vector = PineconeRecord.SparseVector(indices=sparse_indices, values=sparse_values)
        metadata = dict(flatten_data(metadata, allowed_list_types=[str]))
        assert PineconeRecord._validate_metadata(metadata)
        return PineconeRecord(id, values, metadata, sparse_vector)

    def to_grpc_vector(self) -> Vector:
        if self.sparse_values:
            return VectorFactoryGRPC.build(asdict(self))
        else:
            return VectorFactoryGRPC.build({"id": self.id, "values": self.values, "metadata": self.metadata})

    def to_http_vector(self) -> dict:
        if self.sparse_values:
            return asdict(self)
        else:
            return {"id": self.id, "values": self.values, "metadata": self.metadata}

    @staticmethod
    def _validate_metadata(metadata: dict) -> TypeGuard[dict[str, Union[list[str], str, bool, int, float]]]:
        for k, v in metadata.items():
            if not isinstance(k, str):
                return False
            if isinstance(v, list) and all(isinstance(inner, str) for inner in v):
                continue
            if not isinstance(v, (str, bool, int, float)):
                return False
        return True


def _narrow_list_of_pinecone_records(records: list[BaseDBWriter.Record]) -> TypeGuard[PineconeRecord]:
    return all(isinstance(r, PineconeRecord) for r in records)


def wait_on_index(client: PineconeGRPC, index: str):
    """
    Takes the name of the index to wait for and blocks until it's available and ready.
    """
    ready = False
    while not ready:
        try:
            desc = client.describe_index(index)
            if desc.get("status")["ready"]:
                return True
        except PineconeException:
            # NotFoundException means the index is created yet.
            pass
        time.sleep(1)


class PineconeWriter(BaseDBWriter):
    Client = PineconeClient
    Record = PineconeRecord
    TargetParams = PineconeTargetParams
    ClientParams = PineconeClientParams
