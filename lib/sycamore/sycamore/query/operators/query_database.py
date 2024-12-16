from typing import Dict, Optional

from pydantic import Field

from sycamore import DocSet
from sycamore.query.logical_plan import Node


class QueryDatabase(Node):
    """Queries OpenSearch for data from a specified index."""

    index: str
    """The index to load data from."""

    query: Optional[Dict] = None
    """A query in OpenSearch Query DSL format. This can be used to perform full-text queries,
    term-level queries for specific fields, and more. Here is an example of a query that
    retrieves all documents where the "properties.path" field matches the wildcard
    expression "/path/to/data/*.pdf", and that have a properties.entity.location field containing
    the word "Georgia" and an isoDateTime field between July 1, 2023, and September 30, 2024.

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
                            "properties.path.keyword": "/path/to/data/*.pdf"
                        }
                    },
                    {
                        "match_phrase": {
                            "properties.entity.location": "Georgia"
                        }
                    }
                ]
            }
        }
    }

    Use the ".keyword" subfield for "properties.path", as this field represents the filename of the
    original document, and is generally accessed as a keyword field.

    The full range of OpenSearch Query DSL parameters are supported.
    Whenever possible, use the query parameter to filter data at the source, as this is more
    efficient than filtering data in subsequent data filtering operators.
    
    If filtering on a proper noun, e.g. "New York", use "match_phrase" instead of "match". 
    If filtering on a phrase that doesn't have exact spellings or forms, e.g. "Color" or "Colour", prefer "match".
    """

    @property
    def input_types(self) -> set[type]:
        return set()

    @property
    def output_type(self) -> type:
        return DocSet


class QueryVectorDatabase(Node):
    """Vector search to load records that are similar to a given query string. Returns the top k records
    according to the vector similarity between the query string and record's text content.

    You should only use QueryVectorDatabase for the following query types:
    1. "Is there *any* <record> similar to <query>?" - yes or no questions about inclusion of any record.
    2. "Give me *some* <record>s similar to <query>" - sample selection of records similar to a query.

    Important: Unless the query is asking for any or some records, DO NOT use QueryVectorDatabase.

    Do not use QueryVectorDatabase for questions that which require all results that match a criteria, even if it is a
    similarity criteria, e.g. "What incidents involved tigers in 2022" as that requires all possible records that match.

    Since vector search is approximate, QueryVectorDatabase must always be followed by an LLMFilter to ensure that the
    final results are accurate.
    """

    index: str
    """The index to load data from."""

    query_phrase: str = Field(..., json_schema_extra={"exclude_from_comparison": True})
    """The string to convert to a vector and perform vector search"""

    opensearch_filter: Optional[Dict] = None
    """
    A filter query in OpenSearch Query DSL format which represents a filter in an OpenSearch knn query. 
    
    Here is an example of a filter to get records between July 1, 2023, and September 30, 2024. 

    {
      "bool": {
        "must":[
            {
                "range": {
                    "properties.entity.isoDateTime": {
                    "gte": "2023-07-01T00:00:00",
                    "lte": "2024-09-30T23:59:59",
                    "format": "strict_date_optional_time"
                    }
                }
            },
        ]
      }
    }


    The full range of OpenSearch Query DSL parameters for a filter query are supported.
    """

    @property
    def input_types(self) -> set[type]:
        return set()
