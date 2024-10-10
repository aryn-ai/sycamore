# Elasticsearch

[Elasticsearch](https://www.elastic.co/elasticsearch) is a full-featured, multitenant-capable full-text search engine. Elasticsearch makes it easy to build hybrid search applications with extensive in-built functionality and support for RAG tooling.

## Configuration for Elasticsearch

*Please see Elasticsearch's [installation](https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html) page for more in-depth information on installing, configuring, and running Elasticsearch. We specify the setup required to run a simple demo app.*

For local development and testing, we recommend running Elasticsearch through docker compose. The provided `compose.yml` file runs Elasticsearch, which has an associated low-level Python library that makes querying easier. 

<details>
  <summary><i>compose.yml</i></summary>

  ```yaml
  version: "3.8"
  services:
    elasticsearch:
      image: docker.elastic.co/elasticsearch/elasticsearch:8.14.2
      ports:
        - 9200:9200
      restart: on-failure
      environment:
        - discovery.type=single-node
        - xpack.security.enabled=false
        - ES_JAVA_OPTS=-Xms4g -Xmx4g
      ulimits:
        memlock:
          soft: -1
          hard: -1
  ```
</details>

With this you can run Elasticsearch with a simple `docker compose up`.

## Writing to Elasticsearch

To write a DocSet to a Elasticsearch index from Sycamore, use the `docset.write.elasticsearch(...)` function. The Elasticsearch writer takes the following arguments:

- `url`: Connection endpoint for the Elasticsearch instance. Note that this must be paired with the necessary client arguments below
-  `index_name`: Index name to write to in the Elasticsearch instance
- `es_client_args`: (optional) Authentication arguments to be specified (if needed). See more information at [here](https://elasticsearch-py.readthedocs.io/en/v8.14.0/api/elasticsearch.html).
- `wait_for_completion`: (optional, default=`"false"`) Whether to wait for completion of the write before proceeding with next steps. See more information and valid values [here] (https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-refresh.html).
- `mappings`: (optional) Mappings of the Elasticsearch index, can be optionally specified
- `settings`:(optional) Settings of the Elasticsearch index, can be optionally specified
- `execute`: (optional, default=`True`) Whether to execute this sycamore pipeline now, or return a docset to add more transforms.

To write a docset to a local Elasticsearch index run by the Docker compose above, we can write the following:

```python
url = "http://localhost:9200"
index_name = "test_index-write"
wait_for_completion = "wait_for"

ds.write.elasticsearch(url=url, index_name=index_name, wait_for_completion=wait_for_completion)
```

More information can be found in the {doc}`API documentation <../APIs/docsetwriter>`. A demo of the writer can also be found in the [demo notebook](https://github.com/aryn-ai/sycamore/blob/main/notebooks/elasticsearch-writer.ipynb).


## Reading from Elasticsearch

Reading from an Elasticsearch index takes in the `index_name`, `url`,  and `es_client_args` arguments, with the same specification and defaults as above. It paginates and reads from all search results. It also takes in the arguments below:

- `query`: (Optional) Query to perform on the index. Note that this must be specified in the Elasticsearch
    Query DSL as a dictionary. Otherwise, it defaults to a full scan of the table.
    See more information [here] (https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html).
- `keep_alive`: (Optional) Keep alive time for the search context point in time. Defaults to 1 minute if not specified
- `kwargs`: (Optional) Parameters to configure the underlying Elasticsearch search query.
    See more information [here]
    (https://elasticsearch-py.readthedocs.io/en/v8.14.0/api/elasticsearch.html#elasticsearch.Elasticsearch.search).

To read from a Elasticsearch index into a Sycamore DocSet, use the following code:

```python
ctx = sycamore.init()
url = "http://localhost:9200"
index_name = "test_index-read"
target_doc_id = "target"
query_params = {"term": {"_id": target_doc_id}}
query_docs = ctx.read.elasticsearch(url=url, index_name=index_name, query=query_params).take_all()
```

More information can be found in the {doc}`API documentation <../APIs/docsetreader>`.
