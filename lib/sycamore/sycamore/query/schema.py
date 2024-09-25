import logging
import typing
from typing import Dict, Set

from pydantic import BaseModel, field_serializer

from sycamore.transforms.query import OpenSearchQueryExecutor
from sycamore.data import OpenSearchQuery
from sycamore.utils.nested import dotted_lookup

if typing.TYPE_CHECKING:
    from opensearchpy.client.indices import IndicesClient


class OpenSearchSchemaField(BaseModel):
    """Represents the schema for a single field in an OpenSearch index."""

    type: str
    samples: Set[str]

    @field_serializer("samples")
    def serialize_samples(self, samples: Set[str]):
        return list(samples)


OpenSearchSchema = Dict[str, OpenSearchSchemaField]
"""Represents a mapping from field name to field type and a set of example values."""

logger = logging.getLogger(__name__)


class OpenSearchSchemaFetcher:
    """Responsible for retrieving the schema associated with a given OpenSearch index."""

    # Size of random samples for each field.
    NUM_EXAMPLES = 1000
    NUM_EXAMPLE_VALUES = 5

    def __init__(self, client: "IndicesClient", index: str, query_executor: OpenSearchQueryExecutor) -> None:
        super().__init__()
        self._client = client
        self._index = index
        self._query_executor = query_executor

    def get_schema(self) -> OpenSearchSchema:
        """Return a mapping from schema field name to a tuple (type, examples) where "type"
        is the type of the field and "examples" is a list of example values for that field."""

        schema = self._client.get_field_mapping(fields=["*"], index=self._index)
        query = OpenSearchQuery()

        logger.debug(f"Getting schema for index {self._index}")
        # Fetch example values.
        query["index"] = self._index
        query["query"] = {"query": {"match_all": {}}, "size": self.NUM_EXAMPLES}
        random_sample = self._query_executor.query(query)["result"]["hits"]["hits"]
        result: OpenSearchSchema = {}
        result["text_representation"] = OpenSearchSchemaField(
            type="str", samples={"Can be assumed to have all other details"}
        )

        # Get type and example values for each field.
        for key in schema[self._index]["mappings"].keys():
            if key.endswith(".keyword") or key.endswith(".type"):
                logger.debug(f"  Ignoring match key {key}")
                continue
            if not key.startswith("properties."):
                logger.debug(f"  Ignoring non-properties key {key} as they are likely not from sycamore")
                continue
            try:
                logger.debug(f"  Attempting to get samples for key {key}")
                samples: set[str] = set()
                sample_type = None
                for sample in random_sample:
                    if len(samples) >= self.NUM_EXAMPLE_VALUES:
                        break
                    sample_value = dotted_lookup(sample["_source"], key)
                    if sample_value is not None and sample_value != "":
                        if not sample_type:
                            sample_type = type(sample_value)
                        else:
                            t = type(sample_value)
                            if str(t) != str(sample_type):
                                if t.__name__ == "int" and sample_type.__name__ == "float":
                                    pass  # compatible
                                elif sample_type.__name__ == "int" and t.__name__ == "float":
                                    sample_type = t  # upgrade
                                else:
                                    logger.warning(
                                        "Got multiple sample types for key"
                                        + f" {key}: {sample_type} and {t}. Keeping the former"
                                    )

                        samples.add(str(sample_value))
                if len(samples) > 0:
                    logger.debug(f"  Got samples for {key} of type {sample_type}")
                    result[key] = OpenSearchSchemaField(
                        type=sample_type.__name__, samples={str(example) for example in samples}
                    )
                else:
                    logger.debug(f"  No samples for {key}; ignoring key")
            except KeyError:
                logger.debug(f"  Error retrieving samples for {key}")
                # This can happen if there are mappings that have no corresponding values.
                # Skip them.
                continue
        return result
