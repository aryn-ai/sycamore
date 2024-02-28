# Reranking

Second-stage reranking is a technique to use AI to improve search relevancy for your queries. Sycamore uses OpenSearch's [Rerank processor](https://opensearch.org/docs/latest/search-plugins/search-pipelines/rerank-processor/), contributed by Aryn, to perform this operation. For more information, visit the [OpenSearch reranking documentation](https://opensearch.org/docs/latest/search-plugins/search-relevance/reranking-search-results/) and [search pipelines documentation](https://opensearch.org/docs/latest/search-plugins/search-pipelines/index/).

## Theoretical motivation

The standard way of using the recent advances in AI for search is to embed documents as vectors and then perform an approximate nearest-neighbor search over the vectors. It turns out that you can do better, by giving the language model both the query and the passages you want to search over at search-time. This kind of model is called a cross-encoder (TEXT_SIMILARITY in OpenSearch). The cross-encoder can then output a single number representing the similarity of the query and passage, generating an ordering over all the passages. As you might expect, this is computationally very expensive, since for every query you need to run an language model inference over each document in your index.

The solution is to use more traditional vector / keyword search to gather the top 50 or so documents, and then use the cross-encoder to re-rank only the top documents returned by faster search algorithms, henc the name "second-stage reranking".

## Usage

In order to use the rerank processor, you first need to register and deploy a [Text Similarity Model](https://opensearch.org/docs/latest/ml-commons-plugin/custom-local-models/#cross-encoder-models). Sycamore includes a quantized version of [bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base), and you will need to get its unique id in OpenSearch to use it. You can get its id (let's call this `reranker_id` for the remainder of this tutorial) with

```javascript
POST /_plugins/_ml/models/_search
{
  "query": {
    "bool": {
      "must_not": [
        { "exists": { "field": "chunk_number" } }
      ],
      "must": [
        { "term": { "function_name": "TEXT_SIMILARITY" } }
      ]
    }
  }
}
```

Now, create a new [search pipeline](https://opensearch.org/docs/latest/search-plugins/search-pipelines/creating-search-pipeline/) with the rerank processor in it:

```javascript
PUT /_search/pipeline/rerank-pipeline
{
  "response_processors": [
    {
      "rerank": {
        "ml_opensearch": {
          "model_id": "<reranker_id>"
        },
        "context": {
          "document_fields": [ "text_representation" ]
        }
      }
    }
  ]
}
```

You can compose processors in search pipelines. For example, to create a search pipeline that performs hybrid search, reranking, and RAG:

```javascript
PUT /_search/pipeline/mega-relevance-pipeline
{
  "description": "Pipeline with hybrid search, reranking, and RAG",
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
  ],
  "response_processors": [
    {
      "rerank": {
        "ml_opensearch": {
          "model_id": "<reranker_id>"
        },
        "context": {
          "document_fields": [ "text_representation" ]
        }
      }
    },
    {
      "retrieval_augmented_generation": {
        "tag": "openai_pipeline_demo",
        "model_id": "<remote_model_id>",
        "context_field_list": [
          "text_representation"
        ],
        "llm_model": "gpt-4"
      }
    }
  ]
}
```

The blocks of processors are ordered. We recommend adding the reranking processor before the RAG processor since it's important for RAG that the LLM gets the best results.

Now you can search over your data with reranking.

```javascript
POST /my-index/_search?search_pipeline=rerank-pipeline
{
  "query": {
    "match": {
      "text_representation": "Who wrote the book of love?"
    }
  },
  "ext": {
    "rerank": {
      "query_context": {
        "query_text": "Who wrote the book of love?"
      }
    }
  }
}
```

> Note: reranking can add latency to your query - the processing required grows linearly with the number of search results. We recommend using this feature only in a high-resource environment.
