import typing
from typing import Dict, Set, Tuple

from sycamore.transforms.query import OpenSearchQueryExecutor
from sycamore.data import OpenSearchQuery

if typing.TYPE_CHECKING:
    from opensearchpy.client.indices import IndicesClient

OpenSearchSchema = Dict[str, Tuple[str, Set[str]]]
"""Represents a mapping from field name to field type and a set of example values."""


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

        # Fetch example values.
        query["index"] = self._index
        query["query"] = {"query": {"match_all": {}}, "size": self.NUM_EXAMPLES}
        random_sample = self._query_executor.query(query)["result"]["hits"]["hits"]
        result: OpenSearchSchema = {}
        result["text_representation"] = ("<class 'str'>", {"Can be assumed to have all other details"})

        # Get type and example values for each field.
        for key in schema[self._index]["mappings"].keys():
            # We only care about fields with the "properties.entity" prefix.
            if not key.startswith("properties.entity") or ".keyword" in key:
                continue
            try:
                samples = set()
                sample_type = None
                for sample in random_sample:
                    if len(samples) >= self.NUM_EXAMPLE_VALUES:
                        break
                    sample_value = sample["_source"]["properties"]["entity"].get(key[18:], None)
                    if sample_value is not None:
                        if not sample_type:
                            sample_type = type(sample_value)
                        samples.add(str(sample_value))
                if len(samples) > 0:
                    result[key] = (str(sample_type), {str(example) for example in samples})
            except KeyError:
                # This can happen if there are mappings that have no corresponding values.
                # Skip them.
                continue
        return result
