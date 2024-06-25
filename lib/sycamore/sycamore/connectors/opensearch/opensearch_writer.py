from dataclasses import asdict, dataclass, field
import logging
from typing import Any, Optional
from typing_extensions import TypeGuard

from opensearchpy import OpenSearch
from opensearchpy.exceptions import RequestError
from opensearchpy.helpers import parallel_bulk

from sycamore.data import Document
from sycamore.writers.base import BaseDBWriter
from sycamore.writers.common import HostAndPort, flatten_data

log = logging.getLogger(__name__)


@dataclass
class OpenSearchClientParams(BaseDBWriter.ClientParams):
    hosts: list[HostAndPort] = field(default_factory=lambda: [HostAndPort(host="localhost", port=9200)])
    http_compress: bool = True
    http_auth: tuple[str, str] = ("admin", "admin")
    use_ssl: bool = True
    verify_certs: bool = True
    ssl_assert_hostname: bool = True
    ssl_show_warn: bool = True
    timeout: Optional[int] = None


@dataclass
class OpenSearchTargetParams(BaseDBWriter.TargetParams):
    index_name: str
    settings: dict[str, Any] = field(default_factory=lambda: {"index.knn": True})
    mappings: dict[str, Any] = field(
        default_factory=lambda: {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {"name": "hnsw", "engine": "faiss"},
                },
            }
        }
    )

    def compatible_with(self, other: "BaseDBWriter.TargetParams") -> bool:
        """
        OpenSearchTargetParams A is compatible with OpenSearchTargetParams B if
        all the keys in A are also in B, and if the values of the intersecting
        keys are the same. We don't check symmetry here, because B might include
        a bunch of other stuff, like creation time or UUID, where we don't want to
        demand equality. We also flatten for consistency.
        """
        if not isinstance(other, OpenSearchTargetParams):
            return False
        if self.index_name != other.index_name:
            return False
        my_flat_settings = dict(flatten_data(self.settings))
        other_flat_settings = dict(flatten_data(other.settings))
        for k in my_flat_settings:
            other_k = k
            if k not in other_flat_settings:
                if "index." + k in other_flat_settings:
                    # You can specify index params without the "index" part and
                    # they'll come back with the "index" part
                    other_k = "index." + k
                else:
                    return False
            if my_flat_settings[k] != other_flat_settings[other_k]:
                return False
        my_flat_mappings = dict(flatten_data(self.mappings))
        other_flat_mappings = dict(flatten_data(other.mappings))
        for k in my_flat_mappings:
            if k not in other_flat_mappings:
                return False
            if my_flat_mappings[k] != other_flat_mappings[k]:
                return False
        return True


class OpenSearchClient(BaseDBWriter.Client):
    def __init__(self, os_client: OpenSearch):
        self._client = os_client

    @classmethod
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "OpenSearchClient":
        assert isinstance(
            params, OpenSearchClientParams
        ), f"Provided params was not of type OpenSearchClientParams:\n{params}"
        paramsdict = asdict(params)
        os_client = OpenSearch(**paramsdict)
        os_client.ping()
        return OpenSearchClient(os_client)

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        assert isinstance(
            target_params, OpenSearchTargetParams
        ), f"Provided target_params was not of type OpenSearchTargetParams:\n{target_params}"
        assert _narrow_list_of_os_records(records), f"A provided record was not of type OpenSearchRecord:\n{records}"

        for success, info in parallel_bulk(self._client, [asdict(r) for r in records]):
            if not success:
                log.error("A Document failed to upload", info)

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        assert isinstance(
            target_params, OpenSearchTargetParams
        ), f"Provided target_params was not of type OpenSearchTargetParams:\n{target_params}"
        index_name = target_params.index_name
        try:
            self._client.indices.create(
                index_name, body={"mappings": target_params.mappings, "settings": target_params.settings}
            )
        except RequestError as e:
            if e.error != "resource_already_exists_exception":
                raise e

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> OpenSearchTargetParams:
        def _string_values_to_python_types(obj: Any):
            if isinstance(obj, dict):
                for k in obj:
                    obj[k] = _string_values_to_python_types(obj[k])
                return obj
            if isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = _string_values_to_python_types(obj[i])
                return obj
            if isinstance(obj, str):
                if obj == "true":
                    return True
                elif obj == "false":
                    return False
                elif obj.isnumeric():
                    return int(obj)
                try:
                    return float(obj)
                except ValueError:
                    return obj
            return obj

        assert isinstance(
            target_params, OpenSearchTargetParams
        ), f"Provided target_params was not of type OpenSearchTargetParams:\n{target_params}"
        index_name = target_params.index_name
        response = self._client.indices.get(index_name)
        mappings = _string_values_to_python_types(response.get(index_name, {}).get("mappings", {}))
        assert isinstance(mappings, dict)
        settings = _string_values_to_python_types(response.get(index_name, {}).get("settings", {}))
        assert isinstance(settings, dict)
        return OpenSearchTargetParams(index_name=index_name, mappings=mappings, settings=settings)


DEFAULT_OPENSEARCH_RECORD_PROPERTIES: dict[str, Any] = {
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


@dataclass
class OpenSearchRecord(BaseDBWriter.Record):
    _source: dict[str, Any]
    _index: str
    _id: str

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "OpenSearchRecord":
        assert isinstance(
            target_params, OpenSearchTargetParams
        ), f"Provided target_params was not of type OpenSearchTargetParams:\n{target_params}"
        assert (
            document.doc_id is not None
        ), f"Cannot create opensearch record from Document without a doc_id:\n{document}"
        result = dict()

        data = document.data
        for k, v in DEFAULT_OPENSEARCH_RECORD_PROPERTIES.items():
            if k in data:
                result[k] = data[k]
            else:
                result[k] = v
        return OpenSearchRecord(_index=target_params.index_name, _id=document.doc_id, _source=result)


def _narrow_list_of_os_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[OpenSearchRecord]]:
    return all(isinstance(r, OpenSearchRecord) for r in records)


class OpenSearchWriter(BaseDBWriter):
    Client = OpenSearchClient
    ClientParams = OpenSearchClientParams
    Record = OpenSearchRecord
    TargetParams = OpenSearchTargetParams
