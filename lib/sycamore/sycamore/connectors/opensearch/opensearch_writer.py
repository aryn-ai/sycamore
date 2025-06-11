import os
from dataclasses import asdict, dataclass, field
import logging
import typing
from typing import Any, Optional
from typing_extensions import TypeGuard
import random
import time

from sycamore.data import Document
from sycamore.connectors.base_writer import BaseDBWriter
from sycamore.connectors.common import (
    HostAndPort,
    flatten_data,
    check_dictionary_compatibility,
    DEFAULT_RECORD_PROPERTIES,
)
from sycamore.utils.import_utils import requires_modules
from sycamore.plan_nodes import Node
from sycamore.docset import DocSet
from sycamore.context import Context

if typing.TYPE_CHECKING:
    from opensearchpy import OpenSearch

log = logging.getLogger(__name__)

OS_ADMIN_PASSWORD = os.getenv("OS_ADMIN_PASSWORD", "admin")
MAX_RETRIES = 5  # Number of times to retry a failed request
INITIAL_BACKOFF = 1  # Initial backoff time in seconds


@dataclass
class OpenSearchWriterClientParams(BaseDBWriter.ClientParams):
    hosts: list[HostAndPort] = field(default_factory=lambda: [HostAndPort(host="localhost", port=9200)])
    http_compress: bool = True
    http_auth: tuple[str, str] = ("admin", OS_ADMIN_PASSWORD)
    use_ssl: bool = True
    verify_certs: bool = True
    ssl_assert_hostname: bool = True
    ssl_show_warn: bool = True
    timeout: Optional[int] = None


@dataclass
class OpenSearchWriterTargetParams(BaseDBWriter.TargetParams):
    index_name: str
    _doc_count: int = 0
    settings: dict[str, Any] = field(default_factory=lambda: {"index.knn": True})
    mappings: dict[str, Any] = field(
        default_factory=lambda: {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {"name": "hnsw", "engine": "faiss"},
                }
            }
        }
    )
    insert_settings: dict[str, Any] = field(
        default_factory=lambda: {
            "raise_on_error": False,
            "raise_on_exception": False,
            "chunk_size": 100,
            "thread_count": 3,
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
        if not isinstance(other, OpenSearchWriterTargetParams):
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
        return check_dictionary_compatibility(my_flat_mappings, other_flat_mappings)

    @classmethod
    def from_write_args(
        cls,
        index_name: str,
        plan: Node,
        context: Context,
        reliability_rewriter: bool,
        execute: bool,
        insert_settings: Optional[dict] = None,
        index_settings: Optional[dict] = None,
    ) -> "OpenSearchWriterTargetParams":
        """
        Build OpenSearchWriterTargetParams from write operation arguments.

        Args:
            index_name: Name of the OpenSearch index
            plan: The execution plan Node
            context: The execution Context
            reliability_rewriter: Whether to enable reliability rewriter mode
            execute: Whether to execute the pipeline immediately
            insert_settings: Optional settings for data insertion
            index_settings: Optional index configuration settings

        Returns:
            OpenSearchWriterTargetParams configured with the provided settings

        Raises:
            AssertionError: If reliability_rewriter conditions are not met
        """
        target_params_dict: dict[str, Any] = {
            "index_name": index_name,
            "_doc_count": 0,
        }

        if reliability_rewriter:
            from sycamore.materialize import Materialize

            assert execute, "Reliability rewriter requires execute to be True"
            assert isinstance(
                plan, Materialize
            ), "The first node must be a materialize node for reliability rewriter to work"
            assert not plan.children[
                0
            ], "Pipeline should only have read materialize and write nodes for reliability rewriter to work"
            target_params_dict["_doc_count"] = DocSet(context, plan).count()

        if insert_settings:
            target_params_dict["insert_settings"] = insert_settings

        if index_settings:
            target_params_dict["settings"] = index_settings.get("body", {}).get("settings", {})
            target_params_dict["mappings"] = index_settings.get("body", {}).get("mappings", {})

        return cls(**target_params_dict)


class OpenSearchWriterClient(BaseDBWriter.Client):
    def __init__(self, os_client: "OpenSearch"):
        self._client = os_client

    @classmethod
    @requires_modules(["opensearchpy", "opensearchpy.helpers"], extra="opensearch")
    def from_client_params(cls, params: BaseDBWriter.ClientParams) -> "OpenSearchWriterClient":
        from sycamore.connectors.opensearch.utils import OpenSearchClientWithLogging

        assert isinstance(
            params, OpenSearchWriterClientParams
        ), f"Provided params was not of type OpenSearchWriterClientParams:\n{params}"
        paramsdict = asdict(params)
        os_client = OpenSearchClientWithLogging(**paramsdict)
        os_client.ping()
        return OpenSearchWriterClient(os_client)

    def write_many_records(self, records: list[BaseDBWriter.Record], target_params: BaseDBWriter.TargetParams):
        from opensearchpy.helpers import parallel_bulk

        assert isinstance(
            target_params, OpenSearchWriterTargetParams
        ), f"Provided target_params was not of type OpenSearchWriterTargetParams:\n{target_params}"
        assert _narrow_list_of_os_records(records), f"A provided record was not of type OpenSearchRecord:\n{records}"
        retry_count = 0

        def generate_records(records):
            for r in records:
                yield asdict(r)

        requests = records
        while requests:
            failed_requests = []
            for success, item in parallel_bulk(
                self._client, generate_records(requests), **target_params.insert_settings
            ):
                if not success:
                    if item["index"]["status"] == 429:
                        if retry_count >= MAX_RETRIES:
                            msg = f"Max retries ({MAX_RETRIES}) exceeded"
                            log.error(msg)
                            raise Exception(msg)
                        failed_requests.append(item["index"]["data"])
                    else:
                        msg = f"Failed to upload document: {item}"
                        log.error(msg)
                        raise Exception(msg)
            if failed_requests:
                # Calculate backoff time with exponential increase and jitter
                backoff = INITIAL_BACKOFF * (2**retry_count)
                jitter = random.uniform(0, 0.1 * backoff)
                sleep_time = backoff + jitter
                log.warning(f"Received 429, backing off for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                retry_count += 1
            requests = failed_requests

    def create_target_idempotent(self, target_params: BaseDBWriter.TargetParams):
        from opensearchpy.exceptions import RequestError

        assert isinstance(
            target_params, OpenSearchWriterTargetParams
        ), f"Provided target_params was not of type OpenSearchWriterTargetParams:\n{target_params}"
        index_name = target_params.index_name
        try:
            self._client.indices.create(
                index_name, body={"mappings": target_params.mappings, "settings": target_params.settings}
            )
        except RequestError:
            return

    def get_existing_target_params(self, target_params: BaseDBWriter.TargetParams) -> OpenSearchWriterTargetParams:
        return get_existing_target_params(self._client, target_params)

    def reliability_assertor(self, target_params: BaseDBWriter.TargetParams):
        assert isinstance(
            target_params, OpenSearchWriterTargetParams
        ), f"Provided target_params was not of type OpenSearchWriterTargetParams:\n{target_params}"
        log.info("Flushing index...")
        self._client.indices.flush(index=target_params.index_name, params={"timeout": 300})
        log.info("Done flushing index.")
        indices = self._client.cat.indices(index=target_params.index_name, format="json")
        assert len(indices) == 1, f"Expected 1 index, found {len(indices)}"
        num_docs = int(indices[0]["docs.count"])
        log.info(f"{num_docs} chunks written in index {target_params.index_name}")
        assert num_docs == target_params._doc_count, f"Expected {target_params._doc_count} docs, found {num_docs}"


def get_existing_target_params(
    os_client: "OpenSearch", target_params: BaseDBWriter.TargetParams
) -> OpenSearchWriterTargetParams:
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

    # TODO: Convert OpenSearchWriterTargetParams to pydantic model

    assert isinstance(
        target_params, OpenSearchWriterTargetParams
    ), f"Provided target_params was not of type OpenSearchWriterTargetParams:\n{target_params}"
    index_name = target_params.index_name
    response = os_client.indices.get(index_name)
    mappings = _string_values_to_python_types(response.get(index_name, {}).get("mappings", {}))
    assert isinstance(mappings, dict)
    settings = _string_values_to_python_types(response.get(index_name, {}).get("settings", {}))
    assert isinstance(settings, dict)
    _doc_count = target_params._doc_count
    assert isinstance(_doc_count, int)
    return OpenSearchWriterTargetParams(
        index_name=index_name,
        mappings=mappings,
        settings=settings,
        _doc_count=_doc_count,
    )


@dataclass
class OpenSearchWriterRecord(BaseDBWriter.Record):
    _source: dict[str, Any]
    _index: str
    _id: str

    @classmethod
    def from_doc(cls, document: Document, target_params: BaseDBWriter.TargetParams) -> "OpenSearchWriterRecord":
        assert isinstance(
            target_params, OpenSearchWriterTargetParams
        ), f"Provided target_params was not of type OpenSearchWriterTargetParams:\n{target_params}"
        assert (
            document.doc_id is not None
        ), f"Cannot create opensearch record from Document without a doc_id:\n{document}"
        result = dict()

        data = document.data
        for k, v in DEFAULT_RECORD_PROPERTIES.items():
            if k in data:
                result[k] = data[k]
            else:
                result[k] = v
        return OpenSearchWriterRecord(_index=target_params.index_name, _id=document.doc_id, _source=result)


def _narrow_list_of_os_records(records: list[BaseDBWriter.Record]) -> TypeGuard[list[OpenSearchWriterRecord]]:
    return all(isinstance(r, OpenSearchWriterRecord) for r in records)


class OpenSearchWriter(BaseDBWriter):
    Client = OpenSearchWriterClient
    ClientParams = OpenSearchWriterClientParams
    Record = OpenSearchWriterRecord
    TargetParams = OpenSearchWriterTargetParams
