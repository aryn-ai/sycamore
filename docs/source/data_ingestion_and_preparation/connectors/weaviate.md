# Weaviate

[Weaviate](https://weaviate.io/) is a full-featured, open-source, AI-native vector database and search engine. Weaviate makes it easy to build semantic search and RAG applications through the use of modular, configurable connectors to many popular AI services.

## Configuration for Weaviate

*Please see Weaviate's [installation](https://weaviate.io/developers/weaviate/installation) page for more in-depth information on installing, configuring, and running Weaviate. We specify the setup required to run a simple demo app.*

We recommend running Weaviate through docker compose. The provided `compose.yml` file runs Weaviate along with a sidecar local embedding service to make querying easier.

<details>
  <summary><i>compose.yml</i></summary>

  ```yaml
version: "3.4"
services:
  weaviate:
    command:
      - --host
      - 0.0.0.0
      - --port
      - "8080"
      - --scheme
      - http
    image: cr.weaviate.io/semitechnologies/weaviate:1.25.0
    ports:
      - 8080:8080
      - 50051:50051
    volumes:
      - weaviate_data:/var/lib/weaviate
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: "/var/lib/weaviate"
      DEFAULT_VECTORIZER_MODULE: "text2vec-transformers"
      ENABLE_MODULES: "text2vec-transformers"
      TRANSFORMERS_INFERENCE_API: http://t2v-transformers:8080
      CLUSTER_HOSTNAME: "node1"
  t2v-transformers:
    image: cr.weaviate.io/semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2
    environment:
      ENABLE_CUDA: 0
volumes:
  weaviate_data:
  ```

  Note the choice of embedding model specified in the compose file.
</details>

With this you can run Weaviate with a simple `docker compose up`.

## Writing to Weaviate

To write a DocSet to a Weaviate Collection from Sycamore, use the `docset.write.weaviate(...)` function. The Weaviate writer takes the following arguments:

- `wv_client_args`: A dictionary of arguments to pass to the Weaviate client constructor, as in an [explicit conection](https://weaviate.io/developers/weaviate/client-libraries/python#python-client-v4-explicit-connection).
- `collection_name`: The name of the collection to write to.
- `collection_config`: (optional) A dictionary of keyword parameters passed to the Weaviate client's [`collection.create(...)`](https://weaviate.io/developers/weaviate/client-libraries/python#instantiate-a-collection) method. A name specified here must match the `collection_name` argument.
- `flatten_properties`: (optional, default=`False`) Whether to flatten nested property objects during the write. Weaviate can store flattened and nested properties, but will only filter and aggregate top-level properties, meaning they need to be flattened.
- `execute`: (optional, default=`True`) Whether to execute this sycamore pipeline now, or return a docset to add more transforms.

To write a docset to the weaviate instance run by the docker compose above, we can write the following:

```python
from weaviate.client import ConnectionParams
from weaviate.collections.classes.config import Configure

collection_name = "MyCollection"
client_args = {
    "connection_params": ConnectionParams.from_params(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
    )
}
collection_config = {
    "name": collection_name,
    "description": "A collection to demo data-prep with Sycamore",
    "vectorizer_config": [Configure.NamedVectors.text2vec_transformers(name="embedding", source_properties=['text_representation'])],
}

docset.write.weaviate(
    wv_client_args=client_args,
    collection_name=collection_name,
    collection_config=collection_config,
    flatten_properties=True
)
```

More information can be found in the {doc}`API documentation </APIs/data_preparation/docsetwriter>`.

## Reading from Weaviate

Reading from a Weaviate collection takes in the `wv_client_args` and `collection_name` arguments, with the same specification and defaults as above. It also takes in the arguments below:

- kwargs: (Optional) Search queries to pass into Weaviate. Note each keyword method argument must have its parameters specified
as a dictionary. Will default to a full scan if not specified..

To read from a Weaviate collection into a Sycamore DocSet, use the following code:

```python
from weaviate.client import ConnectionParams

ctx = sycamore.init()

collection_name = "MyCollection"
client_args = {
    "connection_params": ConnectionParams.from_params(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
    )
}
target_doc_id = "target"
fetch_object_dict = {"filters": Filter.by_id().equal(target_doc_id)}
query_docs = ctx.read.weaviate(
        wv_client_args=wv_client_args, collection_name=collection, fetch_objects=fetch_object_dict
    ).take_all()
```

More information can be found in the {doc}`API documentation </APIs/data_preparation/docsetreader>`.
