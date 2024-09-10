from typing import Dict, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class QueryDatabase(LogicalOperator):
    """Loads data from a specified OpenSearch index."""

    index: str
    """The index to load data from."""

    query: Optional[Dict] = None
    """A query in OpenSearch Query DSL format. This can be used to perform full-text queries,
    term-level queries for specific fields, and more. Here is an example of a query that
    retrieves all documents that have a properties.entity.location field containing the word
    "Georgia" and an isoDateTime field between July 1, 2023, and September 30, 2024:

    {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "properties.entity.isoDateTime": {
                            "gte": "2023-07-01T00:00:00",
                            "lte": "2024-09-30T23:59:59",
                            "format": "strict_date_optional_time"
                            }
                        }
                    },
                    {
                        "match": {
                            "properties.entity.location": "Georgia"
                        }
                    }
                ]
            }
        }
    }

    The full range of OpenSearch Query DSL parameters are supported.
    """


