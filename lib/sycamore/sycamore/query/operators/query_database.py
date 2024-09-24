from typing import Dict, Optional

from sycamore.query.operators.logical_operator import LogicalOperator


class QueryDatabase(LogicalOperator):
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
                            "properties.path.keyword": "/path/to/data/*.pdf",
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
    """


class QueryVectorDatabase(LogicalOperator):
    """Vector search to load records that are similar to a given query string. Returns the top k records
    according to the vector similarity between the query string and record's text content.
    Use this if you need to filter on a field that can't suffice with an exact match, but also isn't complex enough to
    require a llm_filter, i.e. a record level LLM call.

    Here are some instructions about using QueryVectorDatabase instead of QueryDatabase:
    1. Use QueryVectorDatabase if the query plan uses an LLM operator (LLMFilter or LLMExtractEntity),
        and the final answer doesn't require you to scan the full dataset (e.g. find me an instance of a transcript
        that talked about AI).
    2. Use QueryVectorDatabase for RAG-style questions where you need to retrieve a set of candidate records where the
        answer might exist.
    3. Do not use QueryVectorDatabase when examining each record is essential (e.g. did all transcripts talk about AI?).
    4. Do not use QueryVectorDatabase where you need to potentially return large datasets (e.g. give me all transcripts
        that talked about AI).

    """

    index: str
    """The index to load data from."""

    query_phrase: str
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
