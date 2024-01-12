# Hybrid Search

You can use an OpenSearch client version 2.12+ to query your Sycamore stack, and you can run direct hybrid searches (vector + keyword) on your data.

Hybrid search is implemented an [OpenSearch search processor](https://opensearch.org/docs/latest/search-plugins/hybrid-search/) that enables relevancy score normalization and combination of search results from both semantic and keyword search approaches. This allows you to make the best of both keyword and semantic (neural) search, giving higher-quality results. You can use Sycamore's default hybrid search configuration, or you can customize the way your search relevancy is calculated. 

## Default pipeline

By default, Sycamore includes a hybrid search pipeline named 'hybrid_pipeline' with default settings for weighting across vector and keyword retreival and other parameters. When using hybrid search, you must also create vector embeddings for your question using the same AI model that you used when indexing your data. For more information, visit the [OpenSearch Neural Query documentation](https://opensearch.org/docs/latest/query-dsl/specialized/neural/).


Example hybrid search query:

```javascript
GET <index-name>/_search?search_pipeline=hybrid_pipeline
{
  "query": {
    "hybrid": {
      "queries": [
        {
          "match": {
            "text_representation": "Who wrote the book of love?"
          }
        },
        {
          "neural": {
            "embedding": {
              "query_text": "Who wrote the book of love",
              "model_id": "<embedding model id>",
              "k": 100
            }
          }
        }
      ]
    }
  }
}
```

## Customize your hybrid search

You can also customize your hybrid search settings by creating your own hybrid search procesor. For instance, Sycamore's default settings use a `min_max` normalization with an `arithmetic_mean` (weighted `[0.111,0.889]` towards the neural vs. keyword score), but you may find that your use case works better with diffrent wieghting. 

For an example to create a new hybrid search pipeline with different configuration (weighted 0.2 for keyword, 0.8 for nerual):

```javascript
PUT /_search/pipeline/my_hybrid_search_pipeline
{
  "description": "My Hybrid Search Pipeline",
  "phase_results_processors": [
    {
      "normalization-processor": {
        "normalization": {
          "technique": "min_max"
        },
        "combination": {
          "technique": "arithmetic_mean",
          "parameters": {
            "weights": [0.2, 0.8]
          }
        }
      }
    }
  ]
}
```

To use this hybrid processor, specify it in your hybrid query request:

```javascript
GET <index-name>/_search?search_pipeline=my_hybrid_search_pipeline
{
  "query": {
    "hybrid": {
      "queries": [
        {
          "match": {
            "text_representation": "Who wrote the book of love?"
          }
        },
        {
          "neural": {
            "embedding": {
              "query_text": "Who wrote the book of love",
              "model_id": "<embedding model id>",
              "k": 100
            }
          }
        }
      ]
    }
  }
}
```
