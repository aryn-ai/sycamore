# OpenSearch

[OpenSearch](https://opensearch.org/) is an open-source flexible, scalable full-text search engine that is based off a 2021 fork of Elasticsearch. OpenSearch makes it easy to build hybrid search applications with clear in-built functionality and strucutre.

## Configuration for OpenSearch

*Please see OpenSearch's [installation](https://opensearch.org/docs/latest/install-and-configure/install-opensearch/index/) page for more in-depth information on installing, configuring, and running OpenSearch. We specify the setup required to run a simple demo app.*

For local development and testing, we recommend running OpenSearch through docker compose. The provided `compose.yml` file runs OpenSearch, which has an associated low-level Python library that makes querying easier.

<details>
  <summary><i>compose.yml</i></summary>

  ```yaml
  version: '3'
  services:
    opensearch:
      image: opensearchproject/opensearch:2.10.0
      container_name: opensearch
      environment:
        - discovery.type=single-node
        - bootstrap.memory_lock=true # Disable JVM heap memory swapping
      ulimits:
        memlock:
          soft: -1 # Set memlock to unlimited (no soft or hard limit)
          hard: -1
      ports:
        - 9200:9200 # REST API
  ```
</details>

With this you can run OpenSearch with a simple `docker compose up`.

## Writing to OpenSearch

To write a DocSet to a OpenSearch index from Sycamore, use the `docset.write.opensearch(...)` function. The OpenSearch writer takes the following arguments:

- `os_client_args`: Keyword parameters that are passed to the opensearch-py OpenSearch client constructor.
- `index_name`: The name of the OpenSearch index into which to load this DocSet.
- `index_settings`: Settings and mappings to pass when creating a new index. Specified as a Python dict corresponding to the JSON paramters taken by the OpenSearch CreateIndex API: https://opensearch.org/docs/latest/api-reference/index-apis/create-index/
- `execute`: (optional, default=`True`) Whether to execute this sycamore pipeline now, or return a docset to add more transforms.

To write a docset to the OpenSearch index run by the Docker compose above, we can write the following:

```python
index_name = "test_index-other"

os_client_args = {
    "hosts": [{"host": "localhost", "port": 9200}],
    "http_auth": ("user", "password"),
}

index_settings = {
    "body": {
        "settings": {
            "index.knn": True,
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384,
                    "method": {"name": "hnsw", "engine": "faiss"},
                },
            },
        },
    },
}
docset.write.opensearch(
    os_client_args=os_client_args,
    index_name=index_name,
    index_settings=index_settings,
)
```
More information can be found in the {doc}`API documentation </APIs/data_preparation/docsetwriter>`.


## Reading from OpenSearch

In addition to the `os_client_args` and `index_name` arguments above, reading from OpenSearch takes in an optional `query` parameter,
which takes in a dictionary using the OpenSearch query DSL (further information is given here: https://opensearch.org/docs/latest/query-dsl/).
Note that if the parameter is not specified, the function will return a full scan of all documents in the index.

```
ctx = sycamore.init()
ctx.read.opensearch(os_client_args=os_client_args, index_name=index_name, query={"query": {"term": {"_id": "SAMPLE-DOC-ID"}}})
```

More information can be found in the {doc}`API documentation </APIs/data_preparation/docsetreader>`.
