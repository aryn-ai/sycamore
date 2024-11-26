from dataclasses import dataclass, asdict
import typing
from typing import Optional, Union, Any

from sycamore.connectors.common import drop_types, flatten_data
from typing_extensions import TypeAlias, TypeGuard

from sycamore.data.docid import docid_to_uuid
from sycamore.data.document import Document
from sycamore.connectors.base_writer import BaseDBWriter
from sycamore.utils.import_utils import requires_modules

if typing.TYPE_CHECKING:
    from weaviate import WeaviateClient
    from weaviate.client import (
        AdditionalConfig,
        AuthCredentials,
        ConnectionParams,
        EmbeddedOptions,
    )
    from weaviate.collections.classes.config import (
        _CollectionConfigCreate,
        CollectionConfig,
    )


# This is a convenience so that you can use the alias CollectionConfigCreate
# when weaviate is available. In Python >= 3.11 we could use a quoted type
# to get around needing the import, but 3.9/3.10 don't support quoted types
# for type aliases.
try:
    from weaviate.collections.classes.config import _CollectionConfigCreate

    CollectionConfigCreate: TypeAlias = _CollectionConfigCreate
except ImportError:
    pass


@dataclass
class WeaviateClientParams(BaseDBWriter.ClientParams):
    connection_params: Optional["ConnectionParams"] = None
    embedded_options: Optional["EmbeddedOptions"] = None
    auth_client_secret: Optional["AuthCredentials"] = None
    additional_headers: Optional[dict] = None
    additional_config: Optional["AdditionalConfig"] = None
    skip_init_checks: bool = False


@dataclass
class WeaviateWriterTargetParams(BaseDBWriter.TargetParams):
    name: str
    collection_config: Union["_CollectionConfigCreate", "CollectionConfig"]
    flatten_properties: bool = False

    @requires_modules("weaviate.classes.config", extra="weaviate")
    def compatible_with(self, other: BaseDBWriter.TargetParams) -> bool:
        from weaviate.classes.config import DataType

        if not isinstance(other, WeaviateWriterTargetParams):
            return False
        if self.name != other.name:
            return False
        if self.flatten_properties != other.flatten_properties:
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

    @requires_modules("weaviate.collections.classes.config", extra="weaviate")
    def _as_flattened_dict(self) -> dict[str, Any]:
        from weaviate.collections.classes.config import _CollectionConfigCreate

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


class WeaviateWriterClient(BaseDBWriter.Client):
    def __init__(self, client: "WeaviateClient"):
        self._client = client

    @classmethod
    @requires_modules(["weaviate", "weaviate.client", "weaviate.collections.classes.config"], extra="weaviate")
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "WeaviateWriterClient":
        from weaviate import WeaviateClient

        assert isinstance(params, WeaviateClientParams)
        client = WeaviateClient(**params.__dict__)
        return WeaviateWriterClient(client)

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, WeaviateWriterTargetParams)
        assert _narrow_list_of_doc_records(records), f"Found a bad record in {records}"
        with self._client:
            with self._client.collections.get(target_params.name).batch.dynamic() as batch:
                for r in records:
                    if r.vector:
                        batch.add_object(**asdict(r))
                    else:
                        batch.add_object(properties=r.properties, uuid=r.uuid)

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        from weaviate.client import UnexpectedStatusCodeError
        from weaviate.collections.classes.config import CollectionConfig

        assert isinstance(target_params, WeaviateWriterTargetParams)
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
        except UnexpectedStatusCodeError:
            return

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> "WeaviateWriterTargetParams":
        assert isinstance(target_params, WeaviateWriterTargetParams)
        with self._client:
            collection = self._client.collections.get(target_params.name)
            ccfg = collection.config.get(simple=False)
            return WeaviateWriterTargetParams(
                name=target_params.name, collection_config=ccfg, flatten_properties=target_params.flatten_properties
            )


class WeaviateCrossReferenceClient(WeaviateWriterClient):

    @classmethod
    @requires_modules(["weaviate", "weaviate.classes.config", "weaviate.util"], extra="weaviate")
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "WeaviateCrossReferenceClient":
        from weaviate import WeaviateClient

        assert isinstance(params, WeaviateClientParams)
        client = WeaviateClient(**params.__dict__)
        return WeaviateCrossReferenceClient(client)

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        from weaviate.classes.config import ReferenceProperty
        from weaviate.util import WeaviateInvalidInputError

        assert isinstance(target_params, WeaviateWriterTargetParams)
        with self._client:
            try:
                collection = self._client.collections.get(target_params.name)
                collection.config.add_reference(
                    ref=ReferenceProperty(name="parent", target_collection=target_params.name)
                )
            except Exception as e:
                if isinstance(e, WeaviateInvalidInputError) and "already exists" in e.message:
                    return
                raise e

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        assert isinstance(target_params, WeaviateWriterTargetParams)
        assert _narrow_list_of_cr_records(records)
        with self._client:
            with self._client.collections.get(target_params.name).batch.dynamic() as batch:
                for r in records:
                    if r.to is not None:
                        batch.add_reference(**asdict(r))


@dataclass
class WeaviateWriterDocumentRecord(BaseDBWriter.Record):
    uuid: str
    properties: dict
    vector: Optional[dict[str, list[float]]] = None

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "WeaviateWriterDocumentRecord":
        assert isinstance(target_params, WeaviateWriterTargetParams)
        uuid = docid_to_uuid(document.doc_id)
        if uuid is None:
            raise ValueError(f"Cannot write documents without a doc_id. Found {document}")
        properties = {
            "properties": document.properties,
            "type": document.type,
            "text_representation": document.text_representation,
            "parent_id": document.parent_id,
            "bbox": document.bbox.coordinates if document.bbox else None,
            "shingles": document.shingles,
        }
        droperties = drop_types(properties, drop_empty_lists=True, drop_empty_dicts=True)
        if target_params.flatten_properties:
            # Property names must be [a-zA-Z][_0-9a-zA-Z]{0,230}, so use __ as a separator rather than .
            droperties = dict(flatten_data(droperties, allowed_list_types=[int, str, bool, float], separator="__"))
            droperties = {k.replace("-", "_"): v for k, v in droperties.items()}  # e.g. properties__y-index is invalid
        assert isinstance(droperties, dict)
        embedding = document.embedding
        if embedding is not None:
            return WeaviateWriterDocumentRecord(uuid=uuid, properties=droperties, vector={"embedding": embedding})
        else:
            return WeaviateWriterDocumentRecord(uuid=uuid, properties=droperties)


def _narrow_list_of_doc_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[WeaviateWriterDocumentRecord]]:
    return all(isinstance(r, WeaviateWriterDocumentRecord) for r in records)


@dataclass
class WeaviateCrossReferenceRecord(BaseDBWriter.Record):
    from_uuid: str
    from_property: str
    to: Optional[str]  # If this is None, then we don't write the cross-reference

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "WeaviateCrossReferenceRecord":
        assert isinstance(target_params, WeaviateWriterTargetParams)
        from_uuid = docid_to_uuid(document.doc_id)
        assert from_uuid is not None, f"Found a document with no doc_id: {document}"
        to_uuid = docid_to_uuid(document.parent_id)
        from_prop = "parent"
        return WeaviateCrossReferenceRecord(from_uuid=from_uuid, to=to_uuid, from_property=from_prop)


def _narrow_list_of_cr_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[WeaviateCrossReferenceRecord]]:
    return all(isinstance(r, WeaviateCrossReferenceRecord) for r in records)


class WeaviateDocumentWriter(BaseDBWriter):
    Client = WeaviateWriterClient
    Record = WeaviateWriterDocumentRecord
    ClientParams = WeaviateClientParams
    TargetParams = WeaviateWriterTargetParams


class WeaviateCrossReferenceWriter(BaseDBWriter):
    Client = WeaviateCrossReferenceClient
    Record = WeaviateCrossReferenceRecord
    ClientParams = WeaviateClientParams
    TargetParams = WeaviateWriterTargetParams
