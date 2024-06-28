from dataclasses import dataclass, asdict
from typing import Optional, Union, Any
from sycamore.connectors.writers.common import drop_types, flatten_data
from typing_extensions import TypeGuard, TypeAlias

from sycamore.data.document import Document
from sycamore.connectors.writers.base import BaseDBWriter
from weaviate.classes.config import DataType, ReferenceProperty
from weaviate.client import (
    AdditionalConfig,
    AuthCredentials,
    ConnectionParams,
    EmbeddedOptions,
    UnexpectedStatusCodeError,
)
from weaviate.client import WeaviateClient as _WeaviateClient
from weaviate.collections.classes.config import (
    _CollectionConfigCreate,
    CollectionConfig,
)
from weaviate.util import WeaviateInvalidInputError


@dataclass
class WeaviateClientParams(BaseDBWriter.ClientParams):
    connection_params: Optional[ConnectionParams] = None
    embedded_options: Optional[EmbeddedOptions] = None
    auth_client_secret: Optional[AuthCredentials] = None
    additional_headers: Optional[dict] = None
    additional_config: Optional[AdditionalConfig] = None
    skip_init_checks: bool = False


# This is mainsly so people don't feel weird about importing
# this class. This kinda stuff is all over the wv8 codebase
CollectionConfigCreate: TypeAlias = _CollectionConfigCreate


@dataclass
class WeaviateTargetParams(BaseDBWriter.TargetParams):
    name: str
    collection_config: Union[CollectionConfigCreate, CollectionConfig]
    flatten_properties: bool = False

    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        if not isinstance(other, WeaviateTargetParams):
            return False
        if self.name != other.name:
            return False
        my_flat_dict = self._as_flattened_dict()
        other_flat_dict = other._as_flattened_dict()
        for k in my_flat_dict:
            if k not in other_flat_dict:
                if "nestedProperties" in k:
                    # Nested properties seem to not be handled
                    # correctly by .to_dict(), so for now we'll
                    # just ignore them.
                    continue
                return False
            # Convert DataType.OBJECT_ARRAY to "object[]" (or the
            # other enum values)
            my_v = my_flat_dict[k]
            other_v = other_flat_dict[k]
            if isinstance(my_v, DataType):
                my_v = my_v.value
            if isinstance(other_v, DataType):
                other_v = other_v.value
            if my_v != other_v:
                return False
        return True

    def _as_flattened_dict(self) -> dict[str, Any]:
        if isinstance(self.collection_config, _CollectionConfigCreate):
            my_dict = self.collection_config._to_dict()
        else:
            my_dict = self.collection_config.to_dict()
        my_dict["properties"] = {p.get("name", str(i)): p for i, p in enumerate(my_dict.get("properties", []))}
        my_flat_dict = dict(flatten_data(my_dict))
        return my_flat_dict

    def __repr__(self) -> str:
        my_flat_dict = self._as_flattened_dict()
        s = "=" * 80 + "\n"
        return s + "\n".join(f"{k: <80}{v}" for k, v in my_flat_dict.items())


class WeaviateClient(BaseDBWriter.Client):
    def __init__(self, client: _WeaviateClient):
        self._client = client

    @classmethod
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "WeaviateClient":
        assert isinstance(params, WeaviateClientParams)
        client = _WeaviateClient(**asdict(params))
        return WeaviateClient(client)

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, WeaviateTargetParams)
        assert _narrow_list_of_doc_records(records), f"Found a bad record in {records}"
        with self._client:
            with self._client.collections.get(target_params.name).batch.dynamic() as batch:
                for r in records:
                    if r.vector:
                        batch.add_object(**asdict(r))
                    else:
                        batch.add_object(properties=r.properties, uuid=r.uuid)

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, WeaviateTargetParams)
        try:
            with self._client:
                if isinstance(target_params.collection_config, CollectionConfig):
                    self._client.collections.create_from_config(target_params.collection_config)
                else:
                    cfg_crt = target_params.collection_config
                    self._client.collections.create(
                        name=target_params.name,
                        description=cfg_crt.description,
                        generative_config=cfg_crt.generativeSearch,
                        inverted_index_config=cfg_crt.invertedIndexConfig,
                        multi_tenancy_config=cfg_crt.multiTenancyConfig,
                        properties=cfg_crt.properties,
                        references=cfg_crt.references,
                        replication_config=cfg_crt.replicationConfig,
                        reranker_config=cfg_crt.rerankerConfig,
                        sharding_config=cfg_crt.shardingConfig,
                        vector_index_config=cfg_crt.vectorIndexConfig,
                        vectorizer_config=cfg_crt.vectorizerConfig,
                    )
        except UnexpectedStatusCodeError as e:
            if e.status_code == 422 and "already exists" in e.message:
                return
            raise e

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> "WeaviateTargetParams":
        assert isinstance(target_params, WeaviateTargetParams)
        with self._client:
            collection = self._client.collections.get(target_params.name)
            ccfg = collection.config.get(simple=False)
            return WeaviateTargetParams(
                name=target_params.name, collection_config=ccfg, flatten_properties=target_params.flatten_properties
            )


class WeaviateCrossReferenceClient(WeaviateClient):
    @classmethod
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "WeaviateCrossReferenceClient":
        assert isinstance(params, WeaviateClientParams)
        client = _WeaviateClient(**asdict(params))
        return WeaviateCrossReferenceClient(client)

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, WeaviateTargetParams)
        with self._client:
            try:
                collection = self._client.collections.get(target_params.name)
                collection.config.add_reference(
                    ref=ReferenceProperty(name="parent", target_collection=target_params.name)
                )
            except WeaviateInvalidInputError as e:
                if "already exists" in e.message:
                    return
                raise e

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, WeaviateTargetParams)
        assert _narrow_list_of_cr_records(records)
        with self._client:
            with self._client.collections.get(target_params.name).batch.dynamic() as batch:
                for r in records:
                    if r.to is not None:
                        batch.add_reference(**asdict(r))


@dataclass
class WeaviateDocumentRecord(BaseDBWriter.Record):
    uuid: str
    properties: dict
    vector: Optional[dict[str, list[float]]] = None

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "WeaviateDocumentRecord":
        assert isinstance(target_params, WeaviateTargetParams)
        uuid = document.doc_id
        if uuid is None:
            raise ValueError(f"Cannot write documents without a doc_id. Found {document}")
        properties = {
            "properties": document.properties,
            "type": document.type,
            "text_representation": document.text_representation,
            "bbox": document.bbox.coordinates if document.bbox else None,
            "shingles": document.shingles,
        }
        droperties = drop_types(properties, drop_empty_lists=True)
        if target_params.flatten_properties:
            # Property names must be [a-zA-Z][_0-9a-zA-Z]{0,230}, so use __ as a separator rather than .
            droperties = dict(flatten_data(droperties, allowed_list_types=[int, str, bool, float], separator="__"))
            droperties = {k.replace("-", "_"): v for k, v in droperties.items()}  # e.g. properties__y-index is invalid
        assert isinstance(droperties, dict)
        embedding = document.embedding
        if embedding is not None:
            return WeaviateDocumentRecord(uuid=uuid, properties=droperties, vector={"embedding": embedding})
        else:
            return WeaviateDocumentRecord(uuid=uuid, properties=droperties)


def _narrow_list_of_doc_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[WeaviateDocumentRecord]]:
    return all(isinstance(r, WeaviateDocumentRecord) for r in records)


@dataclass
class WeaviateCrossReferenceRecord(BaseDBWriter.Record):
    from_uuid: str
    from_property: str
    to: Optional[str]  # If this is None, then we don't write the cross-reference

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "WeaviateCrossReferenceRecord":
        assert isinstance(target_params, WeaviateTargetParams)
        from_uuid = document.doc_id
        assert from_uuid is not None, f"Found a document with no doc_id: {document}"
        to_uuid = document.parent_id
        from_prop = "parent"
        return WeaviateCrossReferenceRecord(from_uuid=from_uuid, to=to_uuid, from_property=from_prop)


def _narrow_list_of_cr_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[WeaviateCrossReferenceRecord]]:
    return all(isinstance(r, WeaviateCrossReferenceRecord) for r in records)


class WeaviateDocumentWriter(BaseDBWriter):
    Client = WeaviateClient
    Record = WeaviateDocumentRecord
    ClientParams = WeaviateClientParams
    TargetParams = WeaviateTargetParams


class WeaviateCrossReferenceWriter(BaseDBWriter):
    Client = WeaviateCrossReferenceClient
    Record = WeaviateCrossReferenceRecord
    ClientParams = WeaviateClientParams
    TargetParams = WeaviateTargetParams
